"""Shared sweep inspection target resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from .inspection_artifacts import (
    anchor_has_training_artifacts,
    anchor_run_artifacts,
    find_row,
    inspection_run_dir,
    load_inspection_queue,
    load_json_mapping,
    queue_anchor_row,
    queue_metadata_payload,
    row_artifacts,
)
from .paths_io import _copy_jsonable, default_registry_path
from .surface_resolution import (
    anchor_row_payload,
    inspection_spec_and_record,
    merge_model_fallback,
    resolved_surface_payload,
    resolve_anchor_originating_queue_row,
    resolve_anchor_model_spec,
    resolve_anchor_training_surface_record,
    training_surface_record_model_spec,
)


def _anchor_metrics_payload(registry_run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(registry_run, Mapping):
        return None
    payload: dict[str, Any] = {}
    tab_foundry_metrics = registry_run.get("tab_foundry_metrics")
    if isinstance(tab_foundry_metrics, Mapping):
        payload.update(dict(cast(Mapping[str, Any], tab_foundry_metrics)))
    training_diagnostics = registry_run.get("training_diagnostics")
    if isinstance(training_diagnostics, Mapping):
        payload.update(dict(cast(Mapping[str, Any], training_diagnostics)))
    return payload or None


def resolve_row_target(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_path: Path,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    queue_metadata = queue_metadata_payload(queue)
    artifacts = row_artifacts(queue=queue, row=row, registry_path=registry_path)
    training_surface_entry = artifacts.get("training_surface_record_json")
    if isinstance(training_surface_entry, Mapping) and bool(training_surface_entry.get("exists")):
        training_surface_path = Path(str(training_surface_entry["path"]))
        spec = training_surface_record_model_spec(training_surface_path)
        training_surface_record = load_json_mapping(
            training_surface_path,
            context=f"row {int(row['order']):02d} training surface record",
        )
    else:
        run_dir_entry = artifacts.get("run_dir")
        run_dir = (
            Path(str(run_dir_entry["path"]))
            if isinstance(run_dir_entry, Mapping)
            else inspection_run_dir(
                sweep_id=str(queue_metadata["sweep_id"]),
                target_kind="row",
                target_id=f"{int(row['order']):02d}_{str(row['delta_id'])}",
            )
        )
        spec, fallback_training_surface_record = inspection_spec_and_record(
            row=row,
            run_dir=run_dir,
            training_experiment=str(queue_metadata["training_experiment"]),
            sweep_id=str(queue_metadata["sweep_id"]),
            sweeps_root=sweeps_root,
        )
        persisted_resolved_surface = row.get("resolved_surface")
        training_surface_record = (
            dict(cast(Mapping[str, Any], persisted_resolved_surface))
            if isinstance(persisted_resolved_surface, Mapping)
            else fallback_training_surface_record
        )
    metrics = row.get("benchmark_metrics")
    if not isinstance(metrics, Mapping):
        metrics = row.get("screen_metrics")
    resolved_payload = resolved_surface_payload(
        spec=spec,
        training_surface_record=training_surface_record,
    )
    return {
        "kind": "row",
        "identity": {
            "order": int(row["order"]),
            "delta_id": str(row["delta_id"]),
            "status": str(row["status"]),
            "decision": None if row.get("decision") is None else str(row["decision"]),
            "run_id": None if row.get("run_id") is None else str(row["run_id"]),
        },
        "artifacts": artifacts,
        "resolved": merge_model_fallback(
            resolved=resolved_payload,
            fallback_model=cast(Mapping[str, Any] | None, row.get("model")),
        ),
        "metrics": None if not isinstance(metrics, Mapping) else dict(cast(Mapping[str, Any], metrics)),
    }


def resolve_anchor_target(
    *,
    queue: Mapping[str, Any],
    registry_path: Path,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    queue_metadata = queue_metadata_payload(queue)
    artifacts, registry_run = anchor_run_artifacts(queue=queue, registry_path=registry_path)
    anchor_row = anchor_row_payload(queue)
    try:
        spec, metadata = resolve_anchor_model_spec(
            queue=queue,
            registry_path=registry_path,
            index_path=index_path,
            sweeps_root=sweeps_root,
            load_registry=load_benchmark_run_registry,
            resolve_registry_path=resolve_registry_path_value,
        )
        if anchor_has_training_artifacts(artifacts):
            training_surface_record = resolve_anchor_training_surface_record(queue=queue, artifacts=artifacts)
        else:
            source_row: dict[str, Any] | None = None
            source_training_experiment: str | None = None
            if str(metadata["source"]) == "queue_row":
                source_row = queue_anchor_row(queue)
                source_training_experiment = str(queue_metadata["training_experiment"])
            elif str(metadata["source"]) == "originating_sweep_row":
                originating_row = resolve_anchor_originating_queue_row(
                    queue=queue,
                    registry_path=registry_path,
                    index_path=index_path,
                    sweeps_root=sweeps_root,
                    load_registry=load_benchmark_run_registry,
                )
                if originating_row is not None:
                    source_row, originating_metadata = originating_row
                    source_training_experiment = str(originating_metadata["training_experiment"])

            if source_row is None or source_training_experiment is None:
                training_surface_record = resolve_anchor_training_surface_record(queue=queue, artifacts=artifacts)
            else:
                run_dir = inspection_run_dir(
                    sweep_id=str(queue_metadata["sweep_id"]),
                    target_kind="anchor",
                    target_id="anchor",
                )
                _, training_surface_record = inspection_spec_and_record(
                    row=source_row,
                    run_dir=run_dir,
                    training_experiment=source_training_experiment,
                    sweep_id=str(queue_metadata["sweep_id"]),
                    sweeps_root=sweeps_root,
                )
    except RuntimeError:
        run_dir = inspection_run_dir(
            sweep_id=str(queue_metadata["sweep_id"]),
            target_kind="anchor",
            target_id="anchor",
        )
        spec, training_surface_record = inspection_spec_and_record(
            row=anchor_row,
            run_dir=run_dir,
            training_experiment=str(queue_metadata["training_experiment"]),
            sweep_id=str(queue_metadata["sweep_id"]),
            sweeps_root=sweeps_root,
        )
        metadata = {"source": "anchor_context"}
    return {
        "kind": "anchor",
        "identity": {
            "run_id": queue_metadata["anchor_run_id"],
            "source": str(metadata["source"]),
        },
        "artifacts": artifacts,
        "resolved": resolved_surface_payload(spec=spec, training_surface_record=training_surface_record),
        "metrics": _anchor_metrics_payload(registry_run),
    }


def inspect_sweep_row(
    *,
    order: int,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    queue = load_inspection_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    row = find_row(queue, order=int(order))
    resolved_registry_path = registry_path or default_registry_path()
    target = resolve_row_target(
        queue=queue,
        row=row,
        registry_path=resolved_registry_path,
        sweeps_root=sweeps_root,
    )
    queue_metadata = queue_metadata_payload(queue)
    return {
        "queue": queue_metadata,
        "row": dict(cast(dict[str, Any], _copy_jsonable(row))),
        "target": target,
    }
