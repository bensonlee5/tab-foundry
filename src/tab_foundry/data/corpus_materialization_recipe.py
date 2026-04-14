"""Recipe staging and promotion helpers for corpus materialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, cast

from tab_realdata_hub.dagzoo_handoff import load_dagzoo_handoff_info
from tab_realdata_hub.manifest import build_manifest

from tab_foundry.hashing import sha256_path
from tab_foundry.timestamps import utc_now

from .corpus_loading import (
    CORPUS_RECORD_SCHEMA,
    CorpusRecipe,
    CorpusRecipeStorageContext,
    _latest_pointer_path,
    _recipe_storage_context,
    _repo_root,
    _write_latest_pointer,
    build_dagzoo_provenance_summary,
    corpus_id_for_manifest,
    corpus_outputs_root,
    load_corpus_recipe,
)
from .corpus_lookup import _load_reusable_corpus_record, _record_matches_recipe
from .corpus_materialization_shared import (
    _STAGED_VERIFY_MODES,
    _drop_none_values,
    _git_info,
    _int_or_none,
    _materialization_usable_cpu_budget,
    _resolve_materialize_processes,
    _read_json_mapping,
    _snapshot_tree,
)
from .manifest_characteristics import inspect_manifest_summary
from . import corpus_materialization_invocation as invocation_module

_BUILD_MANIFEST_SUPPORTS_WORKERS = (
    "manifest_workers" in inspect.signature(build_manifest).parameters
)
_AUTO_MANIFEST_WORKER_CAP = 32


@dataclass(slots=True)
class _PendingCorpusMaterialization:
    recipe: CorpusRecipe
    storage: CorpusRecipeStorageContext
    dagzoo_root: Path
    repo_root: Path
    recipe_root: Path
    stage_root: Path
    sweep_id: str | None
    sweeps_root: Path | None


def _elapsed_seconds_since(start_time: float) -> float:
    return max(0.0, float(time.perf_counter() - start_time))


def _build_manifest_compat(
    *,
    data_roots: list[Path],
    out_path: Path,
    train_ratio: float,
    val_ratio: float,
    filter_policy: str,
    missing_value_policy: str,
    dagzoo_handoff_manifest_path: Path | None,
    manifest_workers: int | None = None,
):
    kwargs: dict[str, Any] = {
        "data_roots": data_roots,
        "out_path": out_path,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "filter_policy": filter_policy,
        "missing_value_policy": missing_value_policy,
        "dagzoo_handoff_manifest_path": dagzoo_handoff_manifest_path,
    }
    if _BUILD_MANIFEST_SUPPORTS_WORKERS:
        kwargs["manifest_workers"] = manifest_workers
    return build_manifest(**kwargs)


def _load_reusable_recipe_record(
    *,
    recipe: CorpusRecipe,
    storage: CorpusRecipeStorageContext,
    force: bool,
    repo_root: Path,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> dict[str, Any] | None:
    if force:
        return None
    existing_record = _load_reusable_corpus_record(
        recipe.recipe_id,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    if existing_record is None or not _record_matches_recipe(
        existing_record,
        recipe,
        storage=storage,
    ):
        return None
    return existing_record


def _load_reusable_recipe_record_for_request(
    *,
    recipe_id: str,
    force: bool,
    repo_root: Path,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> dict[str, Any] | None:
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    storage = _recipe_storage_context(recipe, repo_root=repo_root)
    return _load_reusable_recipe_record(
        recipe=recipe,
        storage=storage,
        force=force,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )


def _prepare_recipe_materialization(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool,
    repo_root: Path,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> dict[str, Any] | _PendingCorpusMaterialization:
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    storage = _recipe_storage_context(recipe, repo_root=repo_root)
    existing_record = _load_reusable_recipe_record(
        recipe=recipe,
        storage=storage,
        force=force,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    if existing_record is not None:
        return existing_record

    recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe.recipe_id
    stage_root = recipe_root / ".staging"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    return _PendingCorpusMaterialization(
        recipe=recipe,
        storage=storage,
        dagzoo_root=dagzoo_root,
        repo_root=repo_root,
        recipe_root=recipe_root,
        stage_root=stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )


def _pending_recipe_materialization_from_existing_stage(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    repo_root: Path,
    stage_root: Path | None,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> _PendingCorpusMaterialization:
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    storage = _recipe_storage_context(recipe, repo_root=repo_root)
    recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe.recipe_id
    resolved_stage_root = (
        (recipe_root / ".staging") if stage_root is None else stage_root.expanduser().resolve()
    )
    if not resolved_stage_root.exists():
        raise RuntimeError(
            "staged corpus root does not exist: "
            f"recipe_id={recipe.recipe_id!r} stage_root={resolved_stage_root}"
        )
    return _PendingCorpusMaterialization(
        recipe=recipe,
        storage=storage,
        dagzoo_root=dagzoo_root,
        repo_root=repo_root,
        recipe_root=recipe_root,
        stage_root=resolved_stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )


def _normalize_staged_verify_mode(verify: str) -> str:
    normalized = str(verify).strip().lower()
    if normalized not in _STAGED_VERIFY_MODES:
        expected = ", ".join(sorted(_STAGED_VERIFY_MODES))
        raise ValueError(f"verify must be one of {expected}, got {verify!r}")
    return normalized


def _manifest_inputs_for_pending(
    pending: _PendingCorpusMaterialization,
) -> tuple[list[Path], Path | None]:
    recipe = pending.recipe
    stage_root = pending.stage_root

    filter_policy = str(recipe.manifest_policy.filter_policy)
    if filter_policy == "accepted_only":
        generated_roots = [
            invocation_module._invocation_curated_root(
                corpus_root=stage_root,
                invocation_id=spec.invocation_id,
            )
            for spec in recipe.invocations
        ]
        dagzoo_handoff_manifest_path = None
    elif len(recipe.invocations) == 1:
        single_handoff = load_dagzoo_handoff_info(
            invocation_module._invocation_paths(
                corpus_root=stage_root,
                invocation_id=recipe.invocations[0].invocation_id,
            )[1]
        )
        generated_roots = [
            invocation_module._manifest_source_root(
                handoff=single_handoff,
                filter_policy=str(recipe.manifest_policy.filter_policy),
            )
        ]
        dagzoo_handoff_manifest_path = single_handoff.handoff_manifest_path
    else:
        verified_handoffs = [
            invocation_module._verified_invocation_handoff(
                corpus_root=stage_root,
                spec=spec,
            )
            for spec in recipe.invocations
        ]
        generated_roots = [
            invocation_module._manifest_source_root(
                handoff=handoff,
                filter_policy=str(recipe.manifest_policy.filter_policy),
            )
            for handoff in verified_handoffs
        ]
        dagzoo_handoff_manifest_path = None
    return generated_roots, dagzoo_handoff_manifest_path


def _build_staged_manifest(
    pending: _PendingCorpusMaterialization,
    *,
    manifest_workers: int | None = None,
) -> Path:
    recipe = pending.recipe
    stage_root = pending.stage_root
    generated_roots, dagzoo_handoff_manifest_path = _manifest_inputs_for_pending(pending)

    manifest_path = stage_root / "manifest.parquet"
    _ = _build_manifest_compat(
        data_roots=generated_roots,
        out_path=manifest_path,
        train_ratio=float(recipe.manifest_policy.train_ratio),
        val_ratio=float(recipe.manifest_policy.val_ratio),
        filter_policy=str(recipe.manifest_policy.filter_policy),
        missing_value_policy=str(recipe.manifest_policy.missing_value_policy),
        dagzoo_handoff_manifest_path=dagzoo_handoff_manifest_path,
        manifest_workers=manifest_workers,
    )
    return manifest_path


def build_staged_corpus_manifest(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    out_manifest_path: Path,
    stage_root: Path | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
    manifest_workers: int | None = None,
) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    pending = _pending_recipe_materialization_from_existing_stage(
        recipe_id=recipe_id,
        dagzoo_root=resolved_dagzoo_root,
        repo_root=resolved_repo_root,
        stage_root=stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    generated_roots, dagzoo_handoff_manifest_path = _manifest_inputs_for_pending(pending)
    resolved_out_manifest_path = out_manifest_path.expanduser().resolve()
    resolved_out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _ = _build_manifest_compat(
        data_roots=generated_roots,
        out_path=resolved_out_manifest_path,
        train_ratio=float(pending.recipe.manifest_policy.train_ratio),
        val_ratio=float(pending.recipe.manifest_policy.val_ratio),
        filter_policy=str(pending.recipe.manifest_policy.filter_policy),
        missing_value_policy=str(pending.recipe.manifest_policy.missing_value_policy),
        dagzoo_handoff_manifest_path=dagzoo_handoff_manifest_path,
        manifest_workers=manifest_workers,
    )
    return resolved_out_manifest_path


def _manifest_characteristics_sidecar_path(*, corpus_root: Path) -> Path:
    return corpus_root / "manifest_characteristics.json"


def _resolved_compact_workers(
    *,
    compact_workers: int | None,
    invocation_count: int,
) -> int:
    if invocation_count <= 0:
        return 1
    if compact_workers is not None:
        return max(1, min(int(compact_workers), int(invocation_count)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(invocation_count), cpu_count, 32))


def _estimated_manifest_work_units(pending: _PendingCorpusMaterialization) -> int:
    invocation_count = len(pending.recipe.invocations)
    if invocation_count <= 0:
        return 1
    if str(pending.recipe.manifest_policy.filter_policy).strip() != "accepted_only":
        generated_roots, _dagzoo_handoff_manifest_path = _manifest_inputs_for_pending(pending)
        return max(1, len(generated_roots))
    output_shard_count = 0
    for spec in pending.recipe.invocations:
        summary_path = invocation_module._invocation_materialization_summary_path(
            corpus_root=pending.stage_root,
            invocation_id=str(spec.invocation_id),
        )
        if summary_path.exists():
            summary_payload = _read_json_mapping(
                summary_path,
                context=f"materialization summary for invocation {spec.invocation_id!r}",
            )
            curated_compaction = _curated_compaction_payload(summary_payload)
            resolved_output_shards = _int_or_none(curated_compaction.get("output_shard_count"))
            if resolved_output_shards is not None:
                output_shard_count += max(0, int(resolved_output_shards))
                continue
        output_shard_count += max(
            1,
            int(
                math.ceil(
                    float(int(spec.num_datasets))
                    / float(invocation_module.CURATED_COMPACTION_TARGET_DATASETS_PER_SHARD)
                )
            ),
        )
    return max(invocation_count, output_shard_count)


def _auto_manifest_workers_for_pending(
    pending: _PendingCorpusMaterialization,
) -> int | None:
    if not _BUILD_MANIFEST_SUPPORTS_WORKERS:
        return None
    estimated_work_units = _estimated_manifest_work_units(pending)
    if estimated_work_units <= 1:
        return None
    return max(
        1,
        min(
            int(estimated_work_units),
            int(_materialization_usable_cpu_budget()),
            _AUTO_MANIFEST_WORKER_CAP,
        ),
    )


def _resolved_staged_compact_shard_workers(
    *,
    compact_workers: int,
    compact_shard_workers: int | None,
) -> int | None:
    if compact_shard_workers is not None:
        return int(compact_shard_workers)
    if int(compact_workers) > 1:
        return 1
    return None


def _curated_compaction_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_payload = payload.get("curated_compaction")
    return cast(Mapping[str, Any], raw_payload) if isinstance(raw_payload, Mapping) else {}


def _curated_compaction_int(payload: Mapping[str, Any], key: str) -> int:
    value = _int_or_none(_curated_compaction_payload(payload).get(key))
    return 0 if value is None else int(value)


def _staged_compaction_result(
    *,
    recipe_id: str,
    stage_root: Path,
    resolved_compact_workers: int,
    requested_compact_workers: int | None,
    requested_compact_shard_workers: int | None,
    invocation_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    compacted_invocation_count = sum(
        1 for summary in invocation_summaries if str(summary.get("status")) == "compacted"
    )
    already_compacted_invocation_count = sum(
        1 for summary in invocation_summaries if str(summary.get("status")) == "already_compacted"
    )
    compact_shard_worker_values = [
        resolved
        for resolved in (
            _int_or_none(_curated_compaction_payload(summary).get("compact_shard_workers"))
            for summary in invocation_summaries
        )
        if resolved is not None
    ]
    summary = {
        "invocation_count": len(invocation_summaries),
        "compacted_invocation_count": compacted_invocation_count,
        "already_compacted_invocation_count": already_compacted_invocation_count,
        "source_shard_count": sum(
            _curated_compaction_int(invocation_summary, "source_shard_count")
            for invocation_summary in invocation_summaries
        ),
        "output_shard_count": sum(
            _curated_compaction_int(invocation_summary, "output_shard_count")
            for invocation_summary in invocation_summaries
        ),
        "dataset_count": sum(
            _curated_compaction_int(invocation_summary, "dataset_count")
            for invocation_summary in invocation_summaries
        ),
    }
    resolved_compact_shard_workers = (
        max(compact_shard_worker_values) if compact_shard_worker_values else None
    )
    if compacted_invocation_count == 0 and already_compacted_invocation_count > 0:
        status = "already_compacted"
    elif compacted_invocation_count > 0 and already_compacted_invocation_count > 0:
        status = "mixed"
    else:
        status = "compacted"
    return {
        "recipe_id": recipe_id,
        "stage_root": str(stage_root.resolve()),
        "status": status,
        "requested_compact_workers": (
            None if requested_compact_workers is None else int(requested_compact_workers)
        ),
        "compact_workers": int(resolved_compact_workers),
        "requested_compact_shard_workers": (
            None
            if requested_compact_shard_workers is None
            else int(requested_compact_shard_workers)
        ),
        "compact_shard_workers": (
            None
            if resolved_compact_shard_workers is None
            else int(resolved_compact_shard_workers)
        ),
        "summary": summary,
        "invocations": invocation_summaries,
    }


def _compaction_timing_payload(
    *,
    compaction: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    summary = (
        cast(Mapping[str, Any], compaction.get("summary"))
        if isinstance(compaction.get("summary"), Mapping)
        else {}
    )
    compacted_invocation_count = _int_or_none(summary.get("compacted_invocation_count"))
    already_compacted_invocation_count = _int_or_none(
        summary.get("already_compacted_invocation_count")
    )
    source_shard_count = _int_or_none(summary.get("source_shard_count"))
    output_shard_count = _int_or_none(summary.get("output_shard_count"))
    dataset_count = _int_or_none(summary.get("dataset_count"))
    return _drop_none_values(
        {
            "staged_compaction_elapsed_seconds": float(elapsed_seconds),
            "staged_compaction_status": (
                str(compaction.get("status")).strip()
                if isinstance(compaction.get("status"), str)
                and str(compaction.get("status")).strip()
                else "completed"
            ),
            "compact_workers": _int_or_none(compaction.get("compact_workers")),
            "compact_shard_workers": _int_or_none(compaction.get("compact_shard_workers")),
            "staged_compaction_invocation_count": _int_or_none(summary.get("invocation_count")),
            "staged_compaction_compacted_invocation_count": compacted_invocation_count,
            "staged_compaction_reused_invocation_count": already_compacted_invocation_count,
            "staged_compaction_reused": (
                True
                if compacted_invocation_count == 0 and already_compacted_invocation_count is not None
                else False if compacted_invocation_count is not None else None
            ),
            "staged_compaction_source_shard_count": source_shard_count,
            "staged_compaction_output_shard_count": output_shard_count,
            "staged_compaction_dataset_count": dataset_count,
            "staged_compaction_datasets_per_minute": (
                None
                if dataset_count is None or float(elapsed_seconds) <= 0.0
                else float(dataset_count) / float(elapsed_seconds) * 60.0
            ),
        }
    )


def _fresh_materialization_timing() -> dict[str, Any]:
    return {
        "materialization_mode": "fresh",
        "materialization_reused": False,
    }

def _existing_compacted_summary(curated_root: Path) -> dict[str, Any] | None:
    if not curated_root.exists():
        return None
    shard_dirs = sorted(path for path in curated_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        return {
            "target_datasets_per_shard": int(
                invocation_module.CURATED_COMPACTION_TARGET_DATASETS_PER_SHARD
            ),
            "source_shard_count": 0,
            "output_shard_count": 0,
            "dataset_count": 0,
        }
    dataset_count = 0
    for shard_dir in shard_dirs:
        resolved_shard_dir = invocation_module._resolved_public_shard_dir(shard_dir)
        catalog_path = invocation_module._public_catalog_path_for_shard(resolved_shard_dir)
        if catalog_path.name != invocation_module.HUB_DATASET_CATALOG_FILENAME:
            return None
        dataset_count += len(invocation_module._load_public_catalog_records(catalog_path))
    return {
        "target_datasets_per_shard": int(
            invocation_module.CURATED_COMPACTION_TARGET_DATASETS_PER_SHARD
        ),
        "source_shard_count": len(shard_dirs),
        "output_shard_count": len(shard_dirs),
        "dataset_count": dataset_count,
    }


def _compact_staged_invocation(
    *,
    stage_root: Path,
    invocation_id: str,
    force: bool,
    compact_shard_workers: int | None = None,
) -> dict[str, Any]:
    compaction_start_time = time.perf_counter()
    curated_root = invocation_module._invocation_curated_root(
        corpus_root=stage_root,
        invocation_id=invocation_id,
    )
    existing_compacted = None if force else _existing_compacted_summary(curated_root)
    if existing_compacted is not None:
        summary_path = invocation_module._invocation_materialization_summary_path(
            corpus_root=stage_root,
            invocation_id=invocation_id,
        )
        summary_payload = (
            _read_json_mapping(
                summary_path,
                context=f"staged materialization summary for invocation {invocation_id!r}",
            )
            if summary_path.exists()
            else {}
        )
        prior_curated_compaction = dict(_curated_compaction_payload(summary_payload))
        summary_payload["curated_compaction"] = {
            **prior_curated_compaction,
            **existing_compacted,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "invocation_id": invocation_id,
            "curated_root": str(curated_root.resolve()),
            "curated_compaction": existing_compacted,
            "compact_shard_workers": _int_or_none(existing_compacted.get("compact_shard_workers")),
            "elapsed_seconds": _elapsed_seconds_since(compaction_start_time),
            "status": "already_compacted",
        }
    compacting_root = curated_root.parent / f"{curated_root.name}.compacting"
    backup_root = curated_root.parent / f"{curated_root.name}.precompact"
    if compacting_root.exists():
        shutil.rmtree(compacting_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)
    summary = invocation_module.compact_curated_root(
        source_curated_dir=curated_root,
        output_curated_dir=compacting_root,
        compact_shard_workers=compact_shard_workers,
    )
    curated_root.rename(backup_root)
    compacting_root.rename(curated_root)
    shutil.rmtree(backup_root)

    summary_path = invocation_module._invocation_materialization_summary_path(
        corpus_root=stage_root,
        invocation_id=invocation_id,
    )
    summary_payload = (
        _read_json_mapping(
            summary_path,
            context=f"staged materialization summary for invocation {invocation_id!r}",
        )
        if summary_path.exists()
        else {}
    )
    summary_payload["curated_compaction"] = {
        "target_datasets_per_shard": int(summary["target_datasets_per_shard"]),
        "source_shard_count": int(summary["source_shard_count"]),
        "output_shard_count": int(summary["output_shard_count"]),
        "dataset_count": int(summary["dataset_count"]),
        "compact_shard_workers": int(summary["compact_shard_workers"]),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "invocation_id": invocation_id,
        "curated_root": str(curated_root.resolve()),
        "curated_compaction": summary_payload["curated_compaction"],
        "compact_shard_workers": int(summary["compact_shard_workers"]),
        "elapsed_seconds": _elapsed_seconds_since(compaction_start_time),
        "status": "compacted",
    }


def compact_staged_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    stage_root: Path | None = None,
    force: bool = False,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    compaction_start_time = time.perf_counter()
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    pending = _pending_recipe_materialization_from_existing_stage(
        recipe_id=recipe_id,
        dagzoo_root=resolved_dagzoo_root,
        repo_root=resolved_repo_root,
        stage_root=stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    if str(pending.recipe.manifest_policy.filter_policy).strip() != "accepted_only":
        raise RuntimeError(
            "compact-staged currently supports only accepted_only staged corpora: "
            f"recipe_id={pending.recipe.recipe_id!r}"
        )

    invocation_count = len(pending.recipe.invocations)
    resolved_compact_workers = _resolved_compact_workers(
        compact_workers=compact_workers,
        invocation_count=invocation_count,
    )
    resolved_compact_shard_workers = _resolved_staged_compact_shard_workers(
        compact_workers=resolved_compact_workers,
        compact_shard_workers=compact_shard_workers,
    )
    completed_by_invocation_id: dict[str, dict[str, Any]] = {}
    if resolved_compact_workers <= 1 or invocation_count <= 1:
        for index, spec in enumerate(pending.recipe.invocations, start=1):
            summary = _compact_staged_invocation(
                stage_root=pending.stage_root,
                invocation_id=str(spec.invocation_id),
                force=force,
                compact_shard_workers=resolved_compact_shard_workers,
            )
            completed_by_invocation_id[str(spec.invocation_id)] = summary
            if progress_callback is not None:
                progress_callback(
                    {
                        "index": index,
                        "total": invocation_count,
                        **summary,
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=resolved_compact_workers) as executor:
            futures = {
                executor.submit(
                    _compact_staged_invocation,
                    stage_root=pending.stage_root,
                    invocation_id=str(spec.invocation_id),
                    force=force,
                    compact_shard_workers=resolved_compact_shard_workers,
                ): (index, str(spec.invocation_id))
                for index, spec in enumerate(pending.recipe.invocations, start=1)
            }
            for future in as_completed(futures):
                index, invocation_id = futures[future]
                summary = future.result()
                completed_by_invocation_id[invocation_id] = summary
                if progress_callback is not None:
                    progress_callback(
                        {
                            "index": index,
                            "total": invocation_count,
                            **summary,
                        }
                    )
    invocation_summaries = [
        completed_by_invocation_id[str(spec.invocation_id)] for spec in pending.recipe.invocations
    ]
    result = _staged_compaction_result(
        recipe_id=pending.recipe.recipe_id,
        stage_root=pending.stage_root,
        resolved_compact_workers=resolved_compact_workers,
        requested_compact_workers=compact_workers,
        requested_compact_shard_workers=compact_shard_workers,
        invocation_summaries=invocation_summaries,
    )
    result["elapsed_seconds"] = _elapsed_seconds_since(compaction_start_time)
    return result


def _ensure_staged_accepted_only_compaction(
    pending: _PendingCorpusMaterialization,
    *,
    force: bool = False,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if str(pending.recipe.manifest_policy.filter_policy).strip() != "accepted_only":
        return {
            "recipe_id": pending.recipe.recipe_id,
            "stage_root": str(pending.stage_root.resolve()),
            "status": "skipped",
            "reason": "filter_policy_not_accepted_only",
            "invocations": [],
        }
    return compact_staged_corpus_recipe(
        recipe_id=pending.recipe.recipe_id,
        dagzoo_root=pending.dagzoo_root,
        stage_root=pending.stage_root,
        force=force,
        compact_workers=compact_workers,
        compact_shard_workers=compact_shard_workers,
        progress_callback=progress_callback,
        repo_root=pending.repo_root,
        sweep_id=pending.sweep_id,
        sweeps_root=pending.sweeps_root,
    )


def _promote_materialized_recipe(
    pending: _PendingCorpusMaterialization,
    *,
    force: bool,
    materialization_timing: dict[str, Any] | None = None,
    recipe_start_time: float | None = None,
) -> dict[str, Any]:
    promotion_start_time = time.perf_counter()
    recipe = pending.recipe
    storage = pending.storage
    resolved_repo_root = pending.repo_root
    resolved_dagzoo_root = pending.dagzoo_root
    recipe_root = pending.recipe_root
    stage_root = pending.stage_root
    sweep_id = pending.sweep_id
    sweeps_root = pending.sweeps_root

    manifest_path = stage_root / "manifest.parquet"
    if not manifest_path.exists():
        raise RuntimeError(
            "cannot promote staged corpus without a manifest.parquet: "
            f"recipe_id={recipe.recipe_id!r} stage_root={stage_root}"
        )
    manifest_sha256 = sha256_path(manifest_path)
    corpus_id = corpus_id_for_manifest(
        recipe_id=recipe.recipe_id,
        manifest_sha256=manifest_sha256,
        recipe_identity=(storage.recipe_identity if storage.uses_scoped_identity else None),
    )
    corpus_ref = f"{recipe.recipe_id}/{corpus_id}"
    final_root = recipe_root / corpus_id
    if final_root.exists():
        if force:
            shutil.rmtree(final_root)
        else:
            existing_record = _load_reusable_corpus_record(
                corpus_ref,
                repo_root=resolved_repo_root,
                sweep_id=sweep_id,
                sweeps_root=sweeps_root,
            )
            if existing_record is not None and _record_matches_recipe(
                existing_record,
                recipe,
                storage=storage,
            ):
                return existing_record
            shutil.rmtree(final_root)
    try:
        _snapshot_tree(stage_root, final_root)
    except Exception:
        if final_root.exists():
            shutil.rmtree(final_root)
        raise
    resolved_manifest_path = final_root / "manifest.parquet"
    invocation_payloads = [
        invocation_module._invocation_record_payload(
            dagzoo_root=resolved_dagzoo_root,
            corpus_root=final_root,
            spec=spec,
        )
        for spec in recipe.invocations
    ]
    resolved_materialization_timing = (
        {}
        if materialization_timing is None
        else {str(key): value for key, value in materialization_timing.items()}
    )
    resolved_materialization_timing["promotion_elapsed_seconds"] = _elapsed_seconds_since(
        promotion_start_time
    )
    if recipe_start_time is not None:
        resolved_materialization_timing["recipe_elapsed_seconds"] = _elapsed_seconds_since(
            recipe_start_time
        )
    dagzoo_provenance_summary = build_dagzoo_provenance_summary(
        recipe=recipe,
        corpus_ref=corpus_ref,
        corpus_id=corpus_id,
        provenance={
            "invocations": invocation_payloads,
            "materialization_timing": resolved_materialization_timing,
        },
    )
    invocation_filter_payloads = invocation_module._invocation_filter_payloads(invocation_payloads)
    dagzoo_provenance = _drop_none_values(
        {
            "corpus_ref": corpus_ref,
            "recipe_id": recipe.recipe_id,
            "corpus_id": corpus_id,
            "recipe_kind": recipe.kind,
            "corpus_variant": recipe.provenance_labels.get("corpus_variant", recipe.surface_label),
            "comparator_role": recipe.provenance_labels.get("comparator_role"),
            "config_refs": sorted(
                {
                    invocation_module._invocation_requested_config_ref(invocation)
                    for invocation in recipe.invocations
                }
            ),
            "commands": [
                command
                for payload in invocation_payloads
                for command in (
                    cast(list[Any], payload.get("commands"))
                    if isinstance(payload.get("commands"), list)
                    else ([payload.get("command")] if payload.get("command") is not None else [])
                )
                if isinstance(command, str) and command.strip()
            ],
            "filter_policy": dagzoo_provenance_summary.get("filter_policy"),
            "accepted_datasets": dagzoo_provenance_summary.get("accepted_datasets"),
            "rejected_datasets": dagzoo_provenance_summary.get("rejected_datasets"),
            "curated_accepted_datasets": dagzoo_provenance_summary.get("curated_accepted_datasets"),
            "acceptance_rate": dagzoo_provenance_summary.get("acceptance_rate"),
            "filter_manifest_paths": [
                filter_payload.get("filter_manifest_path")
                for filter_payload in invocation_filter_payloads
                if filter_payload.get("filter_manifest_path") is not None
            ],
            "filter_summary_paths": [
                filter_payload.get("filter_summary_path")
                for filter_payload in invocation_filter_payloads
                if filter_payload.get("filter_summary_path") is not None
            ],
            "curated_root_lineage": [
                filter_payload.get("curated_dir")
                for filter_payload in invocation_filter_payloads
                if filter_payload.get("curated_dir") is not None
            ],
            "invocations": invocation_payloads,
            "dagzoo_git": _git_info(resolved_dagzoo_root),
            "target_derivation": dagzoo_provenance_summary.get("target_derivation"),
            "target_relevant_feature_count_range": dagzoo_provenance_summary.get(
                "target_relevant_feature_count_range"
            ),
            "target_relevant_feature_fraction_range": dagzoo_provenance_summary.get(
                "target_relevant_feature_fraction_range"
            ),
            "materialization_timing": dagzoo_provenance_summary.get("materialization_timing"),
        }
    )
    manifest_inspection = inspect_manifest_summary(resolved_manifest_path)
    manifest_persisted_summary = manifest_inspection.get("persisted_summary")
    characteristics_sidecar_path = _manifest_characteristics_sidecar_path(corpus_root=final_root)
    record: dict[str, Any] = {
        "schema": CORPUS_RECORD_SCHEMA,
        "generated_at_utc": utc_now(),
        "recipe_id": recipe.recipe_id,
        "corpus_id": corpus_id,
        "corpus_ref": corpus_ref,
        "recipe_path": str(recipe.recipe_path),
        "recipe_identity": storage.recipe_identity,
        "recipe_relative_path": storage.recipe_relative_path,
        "surface_label": recipe.surface_label,
        "surface_label_recommendation": recipe.surface_label,
        "recipe": recipe.to_dict(),
        "artifacts": {
            "corpus_root": str(final_root.resolve()),
            "manifest_path": str(resolved_manifest_path.resolve()),
            "latest_pointer_path": str(
                _latest_pointer_path(
                    recipe_id=recipe.recipe_id,
                    repo_root=resolved_repo_root,
                    recipe_identity=(
                        storage.recipe_identity if storage.uses_scoped_identity else None
                    ),
                )
            ),
        },
        "manifest": {
            "manifest_path": str(resolved_manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "inspection": manifest_inspection,
            "characteristics": {
                "persisted_summary": manifest_persisted_summary,
                "sidecar_path": str(characteristics_sidecar_path.resolve()),
                "cache_status": "deferred",
            },
        },
        "dagzoo_provenance": dagzoo_provenance,
        "dagzoo_provenance_summary": dagzoo_provenance_summary,
    }
    record_path = final_root / "corpus_record.json"
    record["corpus_record_path"] = str(record_path.resolve())
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_path = _write_latest_pointer(
        recipe_id=recipe.recipe_id,
        corpus_id=corpus_id,
        corpus_ref=corpus_ref,
        record_path=record_path,
        recipe_path=recipe.recipe_path,
        recipe_identity=storage.recipe_identity,
        repo_root=resolved_repo_root,
        scoped_recipe_identity=(storage.recipe_identity if storage.uses_scoped_identity else None),
    )
    record["artifacts"]["latest_pointer_path"] = str(latest_path.resolve())
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _finalize_materialized_recipe(
    pending: _PendingCorpusMaterialization,
    *,
    force: bool,
    materialization_timing: dict[str, Any] | None = None,
    recipe_start_time: float | None = None,
    manifest_workers: int | None = None,
) -> dict[str, Any]:
    resolved_materialization_timing = (
        {}
        if materialization_timing is None
        else {str(key): value for key, value in materialization_timing.items()}
    )
    resolved_manifest_workers = (
        manifest_workers
        if manifest_workers is not None
        else _auto_manifest_workers_for_pending(pending)
    )
    manifest_build_start_time = time.perf_counter()
    _ = _build_staged_manifest(pending, manifest_workers=resolved_manifest_workers)
    resolved_materialization_timing["manifest_build_elapsed_seconds"] = _elapsed_seconds_since(
        manifest_build_start_time
    )
    if resolved_manifest_workers is not None:
        resolved_materialization_timing["manifest_workers"] = int(resolved_manifest_workers)
    return _promote_materialized_recipe(
        pending,
        force=force,
        materialization_timing=resolved_materialization_timing,
        recipe_start_time=recipe_start_time,
    )


def _verify_staged_materialization(
    pending: _PendingCorpusMaterialization,
    *,
    verify: str,
) -> dict[str, Any]:
    resolved_verify = _normalize_staged_verify_mode(verify)
    recipe = pending.recipe
    stage_root = pending.stage_root
    filter_policy = str(recipe.manifest_policy.filter_policy).strip()
    errors: list[str] = []
    verified_invocations = 0
    target_accepted_datasets = 0
    accepted_datasets = 0
    curated_accepted_datasets = 0

    for spec in recipe.invocations:
        invocation_id = str(spec.invocation_id)
        invocation_root, handoff_manifest_path = invocation_module._invocation_paths(
            corpus_root=stage_root,
            invocation_id=invocation_id,
        )
        invocation_error_count = len(errors)
        if not invocation_root.exists():
            errors.append(f"invocation {invocation_id!r} is missing {invocation_root}")
            continue
        if filter_policy == "accepted_only":
            summary_path = invocation_module._invocation_materialization_summary_path(
                corpus_root=stage_root,
                invocation_id=invocation_id,
            )
            if not summary_path.exists():
                errors.append(
                    f"invocation {invocation_id!r} is missing materialization_summary.json"
                )
                continue
            try:
                summary_payload = _read_json_mapping(
                    summary_path,
                    context=f"staged materialization summary for invocation {invocation_id!r}",
                )
            except Exception as exc:
                errors.append(
                    f"invocation {invocation_id!r} materialization_summary.json is unreadable: {exc}"
                )
                continue

            filter_manifest_path = (
                invocation_module._invocation_filter_root(
                    corpus_root=stage_root,
                    invocation_id=invocation_id,
                )
                / "filter_manifest.parquet"
            )
            filter_summary_path = (
                invocation_module._invocation_filter_root(
                    corpus_root=stage_root,
                    invocation_id=invocation_id,
                )
                / "filter_summary.json"
            )
            curated_root = invocation_module._invocation_curated_root(
                corpus_root=stage_root,
                invocation_id=invocation_id,
            )
            for label, required_path in (
                ("filter_manifest.parquet", filter_manifest_path),
                ("filter_summary.json", filter_summary_path),
                ("curated root", curated_root),
            ):
                if not required_path.exists():
                    errors.append(
                        f"invocation {invocation_id!r} is missing {label}: {required_path}"
                    )

            requested_datasets = int(spec.num_datasets)
            target_accepted_datasets += requested_datasets
            resolved_accepted_datasets = _int_or_none(summary_payload.get("accepted_datasets"))
            resolved_curated_datasets = _int_or_none(
                summary_payload.get("curated_accepted_datasets")
            )
            if (
                resolved_accepted_datasets is None
                or resolved_accepted_datasets < requested_datasets
            ):
                errors.append(
                    f"invocation {invocation_id!r} accepted_datasets must be at least "
                    f"{requested_datasets}, got {resolved_accepted_datasets!r}"
                )
            if resolved_curated_datasets != requested_datasets:
                errors.append(
                    f"invocation {invocation_id!r} curated_accepted_datasets must equal "
                    f"{requested_datasets}, got {resolved_curated_datasets!r}"
                )
            if resolved_verify == "full":
                try:
                    _ = invocation_module._invocation_record_payload(
                        dagzoo_root=pending.dagzoo_root,
                        corpus_root=stage_root,
                        spec=spec,
                    )
                except Exception as exc:
                    errors.append(
                        f"invocation {invocation_id!r} failed full staged provenance verification: {exc}"
                    )
            accepted_datasets += (
                0 if resolved_accepted_datasets is None else int(resolved_accepted_datasets)
            )
            curated_accepted_datasets += (
                0 if resolved_curated_datasets is None else int(resolved_curated_datasets)
            )
        else:
            if not handoff_manifest_path.exists():
                errors.append(
                    f"invocation {invocation_id!r} is missing handoff manifest: {handoff_manifest_path}"
                )
                continue
            try:
                if resolved_verify == "full":
                    _ = invocation_module._verified_invocation_handoff(
                        corpus_root=stage_root,
                        spec=spec,
                    )
                else:
                    _ = load_dagzoo_handoff_info(handoff_manifest_path)
            except Exception as exc:
                errors.append(
                    f"invocation {invocation_id!r} failed staged handoff verification: {exc}"
                )
        if len(errors) == invocation_error_count:
            verified_invocations += 1

    if errors:
        raise RuntimeError("staged corpus verification failed:\n- " + "\n- ".join(errors))

    verification = {
        "mode": resolved_verify,
        "stage_root": str(stage_root.resolve()),
        "filter_policy": filter_policy,
        "expected_invocations": len(recipe.invocations),
        "verified_invocations": verified_invocations,
    }
    if filter_policy == "accepted_only":
        verification["accepted_only"] = {
            "target_accepted_datasets": target_accepted_datasets,
            "accepted_datasets": accepted_datasets,
            "curated_accepted_datasets": curated_accepted_datasets,
        }
    return verification


def finalize_staged_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    verify: str = "fast",
    stage_root: Path | None = None,
    force: bool = False,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    manifest_workers: int | None = None,
) -> dict[str, Any]:
    recipe_start_time = time.perf_counter()
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    pending = _pending_recipe_materialization_from_existing_stage(
        recipe_id=recipe_id,
        dagzoo_root=resolved_dagzoo_root,
        repo_root=resolved_repo_root,
        stage_root=stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    verification = _verify_staged_materialization(
        pending,
        verify=verify,
    )
    compaction_start_time = time.perf_counter()
    compaction = _ensure_staged_accepted_only_compaction(
        pending,
        compact_workers=compact_workers,
        compact_shard_workers=compact_shard_workers,
    )
    record = _finalize_materialized_recipe(
        pending,
        force=force,
        recipe_start_time=recipe_start_time,
        materialization_timing=_drop_none_values(
            {
                **_fresh_materialization_timing(),
                **_compaction_timing_payload(
                    compaction=compaction,
                    elapsed_seconds=_elapsed_seconds_since(compaction_start_time),
                ),
            }
        ),
        manifest_workers=manifest_workers,
    )
    return {
        "record": record,
        "verification": verification,
        "compaction": compaction,
    }


def load_staged_corpus_recipe_preview(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    stage_root: Path | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    pending = _pending_recipe_materialization_from_existing_stage(
        recipe_id=recipe_id,
        dagzoo_root=resolved_dagzoo_root,
        repo_root=resolved_repo_root,
        stage_root=stage_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    return {
        "recipe_id": pending.recipe.recipe_id,
        "surface_label": pending.recipe.surface_label,
        "stage_root": str(pending.stage_root.resolve()),
        "recipe": pending.recipe.to_dict(),
        "invocations": [
            invocation_module._invocation_record_payload(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=pending.stage_root,
                spec=spec,
            )
            for spec in pending.recipe.invocations
        ],
    }


def materialize_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    manifest_workers: int | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    recipe_start_time = time.perf_counter()
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    prepared = _prepare_recipe_materialization(
        recipe_id=recipe_id,
        dagzoo_root=resolved_dagzoo_root,
        force=force,
        repo_root=resolved_repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    if isinstance(prepared, dict):
        return prepared

    try:
        resolved_materialize_processes = max(
            1,
            min(
                _resolve_materialize_processes(materialize_processes),
                len(prepared.recipe.invocations),
            ),
        )
        invocation_fanout_start_time = time.perf_counter()
        invocation_module._materialize_invocations_with_subprocess_fanout(
            recipe_id=prepared.recipe.recipe_id,
            invocations=prepared.recipe.invocations,
            dagzoo_root=prepared.dagzoo_root,
            corpus_root=prepared.stage_root,
            repo_root=prepared.repo_root,
            sweep_id=prepared.sweep_id,
            sweeps_root=prepared.sweeps_root,
            materialize_processes=materialize_processes,
            materialize_worker_threads=materialize_worker_threads,
            compact_shard_workers=compact_shard_workers,
        )
        compaction_start_time = time.perf_counter()
        compaction = _ensure_staged_accepted_only_compaction(
            prepared,
            compact_workers=compact_workers,
            compact_shard_workers=compact_shard_workers,
        )
        return _finalize_materialized_recipe(
            prepared,
            force=force,
            materialization_timing=_drop_none_values(
                {
                    **_fresh_materialization_timing(),
                    "materialize_processes": int(resolved_materialize_processes),
                    "materialize_worker_threads": (
                        None
                        if materialize_worker_threads is None
                        else int(materialize_worker_threads)
                    ),
                    **_compaction_timing_payload(
                        compaction=compaction,
                        elapsed_seconds=_elapsed_seconds_since(compaction_start_time),
                    ),
                    "invocation_fanout_elapsed_seconds": _elapsed_seconds_since(
                        invocation_fanout_start_time
                    ),
                }
            ),
            recipe_start_time=recipe_start_time,
            manifest_workers=manifest_workers,
        )
    finally:
        if prepared.stage_root.exists():
            shutil.rmtree(prepared.stage_root)
