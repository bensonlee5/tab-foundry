"""Concrete sweep corpus materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, cast

from tab_foundry.data.corpus_lookup import _load_reusable_corpus_record
from tab_foundry.data.corpus_materialization import materialize_corpus_refs_batch
from tab_foundry.data.corpus_materialization_recipe import _load_reusable_recipe_record_for_request

from .configuration import _effective_queue_corpus_ref, _resolved_repo_root
from .queue_loading import load_system_delta_queue, write_resolved_system_delta_queue


def _record_materialization_timing(record: Mapping[str, Any]) -> Mapping[str, Any]:
    dagzoo_provenance_summary = record.get("dagzoo_provenance_summary")
    if not isinstance(dagzoo_provenance_summary, Mapping):
        return {}
    materialization_timing = dagzoo_provenance_summary.get("materialization_timing")
    return (
        cast(Mapping[str, Any], materialization_timing)
        if isinstance(materialization_timing, Mapping)
        else {}
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_materialization_float(
    records: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        value
        for value in (
            _float_or_none(_record_materialization_timing(record).get(key))
            for record in records
        )
        if value is not None
    ]
    if not values:
        return None
    return float(sum(values))


def _sum_materialization_int(
    records: list[dict[str, Any]],
    key: str,
) -> int | None:
    values = [
        value
        for value in (
            _int_or_none(_record_materialization_timing(record).get(key))
            for record in records
        )
        if value is not None
    ]
    if not values:
        return None
    return int(sum(values))


def _slowest_materialized_records(
    records: list[dict[str, Any]],
    *,
    key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    sortable: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        timing = _record_materialization_timing(record)
        elapsed_seconds = _float_or_none(timing.get(key))
        if elapsed_seconds is None:
            continue
        sortable.append((elapsed_seconds, record))
    sortable.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("recipe_id", "")),
            str(item[1].get("corpus_ref", "")),
        ),
        reverse=True,
    )
    return [
        {
            "recipe_id": str(record.get("recipe_id")),
            "corpus_ref": str(record.get("corpus_ref")),
            key: float(elapsed_seconds),
        }
        for elapsed_seconds, record in sortable[: max(1, int(limit))]
    ]


def _requested_recipe_reuse_summary(
    *,
    requested_corpus_refs: list[str],
    force: bool,
    repo_root: Path | None,
    sweep_id: str,
    sweeps_root: Path | None,
) -> dict[str, int]:
    per_recipe_reused: dict[str, bool] = {}
    for corpus_ref in requested_corpus_refs:
        recipe_id, separator, _corpus_id = corpus_ref.partition("/")
        if force:
            is_reused = False
        elif separator:
            try:
                is_reused = (
                    _load_reusable_corpus_record(
                        corpus_ref,
                        repo_root=repo_root,
                        sweep_id=sweep_id,
                        sweeps_root=sweeps_root,
                    )
                    is not None
                )
            except (FileNotFoundError, RuntimeError):
                is_reused = False
        else:
            if repo_root is None:
                is_reused = False
                prior_state = per_recipe_reused.get(recipe_id)
                per_recipe_reused[recipe_id] = (
                    is_reused if prior_state is None else prior_state and is_reused
                )
                continue
            try:
                is_reused = (
                    _load_reusable_recipe_record_for_request(
                        recipe_id=recipe_id,
                        force=False,
                        repo_root=repo_root,
                        sweep_id=sweep_id,
                        sweeps_root=sweeps_root,
                    )
                    is not None
                )
            except (FileNotFoundError, RuntimeError):
                is_reused = False
        prior_state = per_recipe_reused.get(recipe_id)
        per_recipe_reused[recipe_id] = is_reused if prior_state is None else prior_state and is_reused
    requested_recipe_count = len(per_recipe_reused)
    reused_recipe_count = sum(1 for reused in per_recipe_reused.values() if reused)
    return {
        "requested_recipe_count": requested_recipe_count,
        "reused_recipe_count": reused_recipe_count,
        "newly_materialized_recipe_count": requested_recipe_count - reused_recipe_count,
    }


def _sweep_materialization_summary_payload(
    *,
    requested_corpus_refs: list[str],
    records: list[dict[str, Any]],
    force: bool,
    repo_root: Path | None,
    sweep_id: str,
    sweeps_root: Path | None,
    batch_elapsed_seconds: float,
) -> dict[str, Any]:
    reuse_summary = _requested_recipe_reuse_summary(
        requested_corpus_refs=requested_corpus_refs,
        force=force,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    summed_materialization_timing = {
        key: value
        for key, value in {
            "invocation_fanout_elapsed_seconds": _sum_materialization_float(
                records, "invocation_fanout_elapsed_seconds"
            ),
            "cumulative_generate_elapsed_seconds": _sum_materialization_float(
                records, "cumulative_generate_elapsed_seconds"
            ),
            "cumulative_filter_elapsed_seconds": _sum_materialization_float(
                records, "cumulative_filter_elapsed_seconds"
            ),
            "cumulative_copy_elapsed_seconds": _sum_materialization_float(
                records, "cumulative_copy_elapsed_seconds"
            ),
            "staged_compaction_elapsed_seconds": _sum_materialization_float(
                records, "staged_compaction_elapsed_seconds"
            ),
            "manifest_build_elapsed_seconds": _sum_materialization_float(
                records, "manifest_build_elapsed_seconds"
            ),
            "promotion_elapsed_seconds": _sum_materialization_float(
                records, "promotion_elapsed_seconds"
            ),
            "recipe_elapsed_seconds": _sum_materialization_float(
                records, "recipe_elapsed_seconds"
            ),
            "invocation_count": _sum_materialization_int(records, "invocation_count"),
            "cumulative_round_count": _sum_materialization_int(
                records, "cumulative_round_count"
            ),
            "cumulative_generated_datasets": _sum_materialization_int(
                records, "cumulative_generated_datasets"
            ),
            "cumulative_accepted_datasets": _sum_materialization_int(
                records, "cumulative_accepted_datasets"
            ),
            "cumulative_rejected_datasets": _sum_materialization_int(
                records, "cumulative_rejected_datasets"
            ),
            "cumulative_curated_accepted_datasets": _sum_materialization_int(
                records, "cumulative_curated_accepted_datasets"
            ),
            "cumulative_source_shard_count": _sum_materialization_int(
                records, "cumulative_source_shard_count"
            ),
            "cumulative_output_shard_count": _sum_materialization_int(
                records, "cumulative_output_shard_count"
            ),
        }.items()
        if value is not None
    }
    return {
        **reuse_summary,
        "requested_corpus_ref_count": len(requested_corpus_refs),
        "batch_elapsed_seconds": float(batch_elapsed_seconds),
        "summed_materialization_timing": summed_materialization_timing,
        "slowest_by_recipe_elapsed_seconds": _slowest_materialized_records(
            records,
            key="recipe_elapsed_seconds",
        ),
        "slowest_by_manifest_build_elapsed_seconds": _slowest_materialized_records(
            records,
            key="manifest_build_elapsed_seconds",
        ),
    }


def materialize_sweep_corpora(
    *,
    dagzoo_root: Path,
    sweep_id: str | None = None,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    compact_workers: int | None = None,
    compact_shard_workers: int | None = None,
    manifest_workers: int | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    batch_start_time = time.perf_counter()
    try:
        _ = write_resolved_system_delta_queue(
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    except FileNotFoundError:
        pass
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    resolved_repo_root = _resolved_repo_root(catalog_path=catalog_path, sweeps_root=sweeps_root)
    resolved_sweep_id = str(queue["sweep_id"])
    requested_corpus_refs: list[str] = []
    seen_corpus_refs: set[str] = set()
    raw_rows = queue.get("rows")
    if not isinstance(raw_rows, list):
        raise RuntimeError("materialized system delta queue must include rows")
    ordered_rows_for_corpora = sorted(
        [cast(Mapping[str, Any], row) for row in raw_rows if isinstance(row, Mapping)],
        key=lambda row: (
            int(row.get("order", 0)),
            str(row.get("delta_id", row.get("delta_ref", ""))),
        ),
    )
    for row in ordered_rows_for_corpora:
        raw_data = row.get("data")
        data_payload = cast(Mapping[str, Any], raw_data) if isinstance(raw_data, Mapping) else {}
        normalized_corpus_ref = _effective_queue_corpus_ref(data_payload)
        if normalized_corpus_ref is None:
            continue
        if normalized_corpus_ref in seen_corpus_refs:
            continue
        seen_corpus_refs.add(normalized_corpus_ref)
        requested_corpus_refs.append(normalized_corpus_ref)
    records = materialize_corpus_refs_batch(
        corpus_refs=requested_corpus_refs,
        dagzoo_root=dagzoo_root,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        compact_workers=compact_workers,
        compact_shard_workers=compact_shard_workers,
        manifest_workers=manifest_workers,
        repo_root=resolved_repo_root,
        sweep_id=resolved_sweep_id,
        sweeps_root=sweeps_root,
    )
    telemetry_summary = _sweep_materialization_summary_payload(
        requested_corpus_refs=requested_corpus_refs,
        records=records,
        force=force,
        repo_root=resolved_repo_root,
        sweep_id=resolved_sweep_id,
        sweeps_root=sweeps_root,
        batch_elapsed_seconds=time.perf_counter() - batch_start_time,
    )
    summary_root = (resolved_repo_root or Path.cwd()).expanduser().resolve()
    telemetry_summary_path = (
        summary_root
        / "outputs"
        / "staged_ladder"
        / "research"
        / resolved_sweep_id
        / "corpus_materialization_summary.json"
    )
    telemetry_summary_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_summary_path.write_text(
        json.dumps(telemetry_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "sweep_id": resolved_sweep_id,
        "recipe_count": len(records),
        "requested_corpus_refs": requested_corpus_refs,
        "requested_recipe_ids": [
            corpus_ref.split("/", 1)[0]
            for corpus_ref in requested_corpus_refs
        ],
        "corpus_refs": [str(record["corpus_ref"]) for record in records],
        "telemetry_summary": telemetry_summary,
        "telemetry_summary_path": str(telemetry_summary_path.resolve()),
        "records": records,
    }
