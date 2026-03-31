"""Corpus materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import yaml

from tab_realdata_hub.manifest import build_manifest

from tab_foundry.hashing import sha256_path
from tab_foundry.timestamps import utc_now

from .corpus_loading import (
    CORPUS_RECORD_SCHEMA,
    DagzooInvocationRecipe,
    _copy_jsonable,
    _deep_merge_payload,
    _ensure_non_empty_string,
    _load_yaml_mapping,
    _latest_pointer_path,
    _recipe_storage_context,
    _repo_root,
    _resolve_from_root,
    _write_latest_pointer,
    corpus_id_for_manifest,
    corpus_outputs_root,
    load_corpus_recipe,
)
from .corpus_lookup import _load_reusable_corpus_record, _record_matches_recipe
from .manifest_characteristics import inspect_manifest_summary
from .dagzoo_handoff import (
    DagzooGeneratedIdentityAccumulator,
    DagzooHandoffInfo,
    load_dagzoo_handoff_info,
    verify_dagzoo_handoff_matches_generated_corpus,
)
from .dagzoo_workflow import DagzooGenerateConfig, build_dagzoo_generate_argv, run_dagzoo_generate


def _git_info(root: Path) -> dict[str, Any] | None:
    if not root.exists():
        return None

    def _capture(*argv: str) -> str | None:
        completed = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    head = _capture("git", "rev-parse", "HEAD")
    if head is None:
        return None
    describe = _capture("git", "describe", "--always", "--dirty", "--tags")
    status = _capture("git", "status", "--short")
    return {
        "head": head,
        "describe": describe,
        "dirty": bool(status),
    }


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _scan_dagzoo_generated_identity(generated_dir: Path) -> DagzooGeneratedIdentityAccumulator:
    resolved_generated_dir = generated_dir.expanduser().resolve()
    metadata_paths = sorted(resolved_generated_dir.rglob("metadata.ndjson"))
    if not metadata_paths:
        raise RuntimeError(
            "dagzoo generated directory does not contain metadata.ndjson: "
            f"{resolved_generated_dir}"
        )
    scanned_identity = DagzooGeneratedIdentityAccumulator()
    for metadata_path in metadata_paths:
        for line_number, raw_line in enumerate(
            metadata_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "failed to parse dagzoo metadata record while verifying handoff: "
                    f"path={metadata_path}, line={line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    "dagzoo metadata NDJSON record must decode to an object: "
                    f"path={metadata_path}, line={line_number}"
                )
            dataset_index = payload.get("dataset_index")
            metadata = payload.get("metadata")
            if dataset_index is None:
                raise RuntimeError(
                    "dagzoo metadata record missing dataset_index while verifying handoff: "
                    f"path={metadata_path}, line={line_number}"
                )
            if not isinstance(metadata, Mapping):
                raise RuntimeError(
                    "dagzoo metadata record missing object payload at key 'metadata' "
                    f"while verifying handoff: path={metadata_path}, line={line_number}"
                )
            scanned_identity.add_metadata(
                metadata,
                metadata_path=metadata_path,
                dataset_index=int(dataset_index),
            )
    return scanned_identity


def _verified_invocation_handoff(
    *,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> DagzooHandoffInfo:
    _invocation_root, handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    handoff = load_dagzoo_handoff_info(handoff_manifest_path)
    verify_dagzoo_handoff_matches_generated_corpus(
        handoff,
        scanned_identity=_scan_dagzoo_generated_identity(handoff.generated_dir),
    )
    return handoff


def _invocation_paths(*, corpus_root: Path, invocation_id: str) -> tuple[Path, Path]:
    invocation_root = corpus_root / "invocations" / invocation_id
    return invocation_root, invocation_root / "handoff_manifest.json"


def _invocation_rendered_config_path(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / "dagzoo_config.yaml"


def _invocation_requested_config_ref(spec: DagzooInvocationRecipe) -> str:
    config_ref = spec.config_ref if spec.config_ref is not None else spec.base_config_ref
    if config_ref is None:
        raise RuntimeError(f"invocation {spec.invocation_id!r} does not define a dagzoo config")
    return str(config_ref)


def _invocation_dagzoo_config_path(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    write_rendered_config: bool,
) -> Path:
    if spec.config_ref is not None:
        return Path(str(spec.config_ref))
    if spec.base_config_ref is None:
        raise RuntimeError(f"invocation {spec.invocation_id!r} does not define a dagzoo config")
    rendered_config_path = _invocation_rendered_config_path(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    if write_rendered_config:
        base_config_path = _resolve_from_root(dagzoo_root, Path(spec.base_config_ref))
        merged_payload = _deep_merge_payload(
            _load_yaml_mapping(
                base_config_path,
                context=f"dagzoo base config for invocation {spec.invocation_id!r}",
            ),
            spec.config_overrides,
        )
        rendered_config_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_config_path.write_text(
            yaml.safe_dump(merged_payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    return rendered_config_path.resolve()


def _dagzoo_generate_config(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    write_rendered_config: bool,
) -> DagzooGenerateConfig:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    return DagzooGenerateConfig(
        dagzoo_root=dagzoo_root,
        dagzoo_config=_invocation_dagzoo_config_path(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=write_rendered_config,
        ),
        handoff_root=invocation_root,
        num_datasets=int(spec.num_datasets),
        seed=spec.seed,
        rows=spec.rows,
        device=spec.device,
        hardware_policy=str(spec.hardware_policy),
        diagnostics=bool(spec.diagnostics),
        diagnostics_out_dir=(
            None if spec.diagnostics_out_dir is None else Path(str(spec.diagnostics_out_dir))
        ),
        missing_rate=spec.missing_rate,
        missing_mechanism=spec.missing_mechanism,
        missing_mar_observed_fraction=spec.missing_mar_observed_fraction,
        missing_mar_logit_scale=spec.missing_mar_logit_scale,
        missing_mnar_logit_scale=spec.missing_mnar_logit_scale,
    )


def materialize_corpus_ref(
    *,
    corpus_ref: str,
    dagzoo_root: Path,
    force: bool = False,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    normalized_corpus_ref = _ensure_non_empty_string(corpus_ref, context="corpus_ref")
    recipe_id, separator, corpus_id = normalized_corpus_ref.partition("/")
    if not separator:
        return materialize_corpus_recipe(
            recipe_id=recipe_id,
            dagzoo_root=dagzoo_root,
            force=force,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )

    if not force:
        existing_record = _load_reusable_corpus_record(
            normalized_corpus_ref,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
        if existing_record is not None:
            return existing_record

    record = materialize_corpus_recipe(
        recipe_id=recipe_id,
        dagzoo_root=dagzoo_root,
        force=force,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    materialized_corpus_ref = _ensure_non_empty_string(
        record.get("corpus_ref"),
        context="materialized corpus record corpus_ref",
    )
    if materialized_corpus_ref != normalized_corpus_ref:
        raise RuntimeError(
            f"requested corpus_ref {normalized_corpus_ref!r} is pinned to an exact corpus id, "
            f"but materializing recipe {recipe_id!r} produced {materialized_corpus_ref!r}"
        )
    return record


def _materialize_invocation(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> None:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    invocation_root.mkdir(parents=True, exist_ok=True)
    run_dagzoo_generate(
        _dagzoo_generate_config(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=True,
        )
    )


def _invocation_record_payload(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> dict[str, Any]:
    invocation_root, handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    handoff = load_dagzoo_handoff_info(handoff_manifest_path)
    generate_config = _dagzoo_generate_config(
        dagzoo_root=dagzoo_root,
        corpus_root=corpus_root,
        spec=spec,
        write_rendered_config=False,
    )
    resolved_config_path = _resolve_from_root(dagzoo_root, generate_config.dagzoo_config)
    payload = {
        "invocation_id": str(spec.invocation_id),
        "requested_config_ref": _invocation_requested_config_ref(spec),
        "num_datasets": int(spec.num_datasets),
        "seed": None if spec.seed is None else int(spec.seed),
        "rows": spec.rows,
        "device": spec.device,
        "hardware_policy": str(spec.hardware_policy),
        "command": " ".join(build_dagzoo_generate_argv(generate_config)),
        "resolved_config_path": str(resolved_config_path),
        "invocation_root": str(invocation_root.resolve()),
        "handoff": handoff.to_summary_dict(),
    }
    if spec.config_ref is not None:
        payload["config_ref"] = str(spec.config_ref)
    if spec.base_config_ref is not None:
        rendered_config_path = _invocation_rendered_config_path(
            corpus_root=corpus_root,
            invocation_id=spec.invocation_id,
        )
        payload["base_config_ref"] = str(spec.base_config_ref)
        payload["config_overrides"] = _copy_jsonable(spec.config_overrides)
        payload["rendered_config_path"] = str(rendered_config_path.resolve())
        payload["rendered_config_sha256"] = sha256_path(rendered_config_path)
    return payload


def _manifest_characteristics_sidecar_path(*, corpus_root: Path) -> Path:
    return corpus_root / "manifest_characteristics.json"


def materialize_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool = False,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=resolved_repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    storage = _recipe_storage_context(recipe, repo_root=resolved_repo_root)
    if not force:
        existing_record = _load_reusable_corpus_record(
            recipe.recipe_id,
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

    recipe_root = corpus_outputs_root(repo_root=resolved_repo_root) / recipe.recipe_id
    stage_root = recipe_root / ".staging"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        for spec in recipe.invocations:
            _materialize_invocation(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=stage_root,
                spec=spec,
            )
        if len(recipe.invocations) == 1:
            single_handoff = load_dagzoo_handoff_info(
                _invocation_paths(
                    corpus_root=stage_root,
                    invocation_id=recipe.invocations[0].invocation_id,
                )[1]
            )
            generated_roots = [single_handoff.generated_dir]
            dagzoo_handoff_manifest_path = single_handoff.handoff_manifest_path
        else:
            verified_handoffs = [
                _verified_invocation_handoff(
                    corpus_root=stage_root,
                    spec=spec,
                )
                for spec in recipe.invocations
            ]
            generated_roots = [handoff.generated_dir for handoff in verified_handoffs]
            dagzoo_handoff_manifest_path = None
        manifest_path = stage_root / "manifest.parquet"
        _ = build_manifest(
            data_roots=generated_roots,
            out_path=manifest_path,
            train_ratio=float(recipe.manifest_policy.train_ratio),
            val_ratio=float(recipe.manifest_policy.val_ratio),
            filter_policy=str(recipe.manifest_policy.filter_policy),
            missing_value_policy=str(recipe.manifest_policy.missing_value_policy),
            dagzoo_handoff_manifest_path=dagzoo_handoff_manifest_path,
        )
        manifest_sha256 = sha256_path(manifest_path)
        corpus_id = corpus_id_for_manifest(
            recipe_id=recipe.recipe_id,
            manifest_sha256=manifest_sha256,
            recipe_identity=(
                storage.recipe_identity if storage.uses_scoped_identity else None
            ),
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
                    shutil.rmtree(stage_root)
                    return existing_record
                shutil.rmtree(final_root)
        shutil.move(str(stage_root), str(final_root))
        resolved_manifest_path = final_root / "manifest.parquet"
        invocation_payloads = [
            _invocation_record_payload(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=final_root,
                spec=spec,
            )
            for spec in recipe.invocations
        ]
        dagzoo_provenance = _drop_none_values(
            {
                "corpus_ref": corpus_ref,
                "recipe_id": recipe.recipe_id,
                "corpus_id": corpus_id,
                "recipe_kind": recipe.kind,
                "corpus_variant": recipe.provenance_labels.get("corpus_variant", recipe.surface_label),
                "comparator_role": recipe.provenance_labels.get("comparator_role"),
                "config_refs": sorted(
                    {_invocation_requested_config_ref(invocation) for invocation in recipe.invocations}
                ),
                "commands": [payload["command"] for payload in invocation_payloads],
                "curated_root_lineage": [],
                "invocations": invocation_payloads,
                "dagzoo_git": _git_info(resolved_dagzoo_root),
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
            scoped_recipe_identity=(
                storage.recipe_identity if storage.uses_scoped_identity else None
            ),
        )
        record["artifacts"]["latest_pointer_path"] = str(latest_path.resolve())
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
