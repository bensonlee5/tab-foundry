"""Shared sweep inspection target resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import torch

from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.model.inspection import model_surface_payload, parameter_counts_from_model_spec

from .anchor import anchor_training_surface_label
from .inspection_artifacts import (
    anchor_has_training_artifacts,
    anchor_run_artifacts,
    inspection_run_dir,
    load_json_mapping,
    optional_string,
    row_artifacts,
)
from .inspection_queue import (
    find_row,
    load_inspection_queue,
    queue_anchor_row,
    queue_metadata_payload,
)
from .paths_io import _copy_jsonable, default_registry_path
from .surface_resolution import (
    build_lightweight_training_surface_record,
    inspection_spec_and_record,
    resolve_anchor_originating_queue_row,
    resolve_anchor_model_spec,
    training_surface_record_model_spec,
)


def _surface_payload(
    *,
    spec: Any,
    training_surface_record: Mapping[str, Any],
) -> dict[str, Any]:
    record_payload = dict(cast(dict[str, Any], _copy_jsonable(training_surface_record)))
    labels = record_payload.get("labels")
    return {
        "surface_labels": (
            dict(cast(Mapping[str, Any], labels))
            if isinstance(labels, Mapping)
            else None
        ),
        "model": {
            **model_surface_payload(spec),
            "parameter_counts": parameter_counts_from_model_spec(spec),
        },
        "data": (
            dict(cast(Mapping[str, Any], record_payload["data"]))
            if isinstance(record_payload.get("data"), Mapping)
            else None
        ),
        "preprocessing": (
            dict(cast(Mapping[str, Any], record_payload["preprocessing"]))
            if isinstance(record_payload.get("preprocessing"), Mapping)
            else None
        ),
        "training": (
            dict(cast(Mapping[str, Any], record_payload["training"]))
            if isinstance(record_payload.get("training"), Mapping)
            else None
        ),
    }


def _merge_model_fallback(
    *,
    resolved: dict[str, Any],
    fallback_model: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(fallback_model, Mapping):
        return resolved
    resolved_model = resolved.get("model")
    if not isinstance(resolved_model, dict):
        return resolved
    for key in ("arch", "stage", "stage_label"):
        if resolved_model.get(key) in (None, "") and fallback_model.get(key) not in (None, ""):
            resolved_model[key] = fallback_model[key]
    return resolved


def _anchor_row_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    anchor_context = queue.get("anchor_context")
    raw_model = cast(Mapping[str, Any], anchor_context.get("model", {})) if isinstance(anchor_context, Mapping) else {}
    raw_module_selection = raw_model.get("module_selection")
    module_overrides = (
        dict(cast(Mapping[str, Any], raw_module_selection))
        if isinstance(raw_module_selection, Mapping)
        else {}
    )
    surface_labels = (
        cast(Mapping[str, Any], anchor_context.get("surface_labels"))
        if isinstance(anchor_context, Mapping) and isinstance(anchor_context.get("surface_labels"), Mapping)
        else {}
    )
    model_payload: dict[str, Any] = {}
    for key in (
        "arch",
        "stage",
        "stage_label",
        "d_icl",
        "input_normalization",
        "feature_group_size",
        "many_class_base",
        "tficl_n_heads",
        "tficl_n_layers",
        "head_hidden_dim",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tfrow_norm",
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
    ):
        if key in raw_model:
            model_payload[key] = raw_model[key]
    if module_overrides:
        model_payload["module_overrides"] = module_overrides
    return {
        "order": 0,
        "delta_id": "anchor",
        "status": "anchor",
        "model": model_payload,
        "data": (
            {"surface_label": str(surface_labels["data"])}
            if isinstance(surface_labels.get("data"), str) and str(surface_labels["data"]).strip()
            else {}
        ),
        "preprocessing": (
            {"surface_label": str(surface_labels["preprocessing"])}
            if isinstance(surface_labels.get("preprocessing"), str) and str(surface_labels["preprocessing"]).strip()
            else {}
        ),
        "training": {
            "surface_label": anchor_training_surface_label(
                cast(Mapping[str, Any], anchor_context) if isinstance(anchor_context, Mapping) else {}
            ),
            "overrides": {},
        },
        "run_id": queue.get("anchor_run_id"),
    }


def _anchor_training_surface_record(
    *,
    queue: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_context = queue.get("anchor_context")
    if isinstance(anchor_context, Mapping):
        raw_surface_labels = anchor_context.get("surface_labels")
        surface_labels = (
            dict(cast(Mapping[str, Any], raw_surface_labels))
            if isinstance(raw_surface_labels, Mapping)
            else {}
        )
    else:
        surface_labels = {}

    training_surface_entry = artifacts.get("training_surface_record_json")
    if isinstance(training_surface_entry, Mapping) and bool(training_surface_entry.get("exists")):
        return load_json_mapping(
            Path(str(training_surface_entry["path"])),
            context="anchor training surface record",
        )

    run_dir_entry = artifacts.get("run_dir")
    checkpoint_entry = artifacts.get("best_checkpoint_path")
    if isinstance(run_dir_entry, Mapping) and isinstance(checkpoint_entry, Mapping) and bool(
        checkpoint_entry.get("exists")
    ):
        checkpoint_payload = torch.load(
            Path(str(checkpoint_entry["path"])),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise RuntimeError("anchor checkpoint payload must be a mapping")
        raw_cfg = checkpoint_payload.get("config")
        if not isinstance(raw_cfg, Mapping):
            raise RuntimeError("anchor checkpoint payload omitted config")
        raw_state_dict = checkpoint_payload.get("model")
        state_dict = raw_state_dict if isinstance(raw_state_dict, Mapping) else None
        record = build_lightweight_training_surface_record(
            raw_cfg={str(key): value for key, value in raw_cfg.items()},
            run_dir=Path(str(run_dir_entry["path"])),
            state_dict=state_dict,
        )
    else:
        record = {
            "labels": dict(surface_labels),
            "data": {
                "surface_label": surface_labels.get("data"),
            },
            "preprocessing": {
                "surface_label": surface_labels.get("preprocessing"),
            },
            "training": {
                "surface_label": surface_labels.get("training"),
            },
        }
    labels = record.get("labels")
    if not isinstance(labels, Mapping):
        record["labels"] = dict(surface_labels)
    else:
        merged_labels = dict(surface_labels)
        merged_labels.update(dict(cast(Mapping[str, Any], labels)))
        record["labels"] = merged_labels

    if not isinstance(record.get("training"), Mapping) and surface_labels.get("training") is not None:
        record["training"] = {"surface_label": surface_labels.get("training")}
    return record


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
                sweep_id=str(queue["sweep_id"]),
                target_kind="row",
                target_id=f"{int(row['order']):02d}_{str(row['delta_id'])}",
            )
        )
        spec, fallback_training_surface_record = inspection_spec_and_record(
            row=row,
            run_dir=run_dir,
            training_experiment=str(queue["training_experiment"]),
            sweep_id=str(queue["sweep_id"]),
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
    resolved_payload = _surface_payload(
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
        "resolved": _merge_model_fallback(
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
    artifacts, registry_run = anchor_run_artifacts(queue=queue, registry_path=registry_path)
    anchor_row = _anchor_row_payload(queue)
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
            training_surface_record = _anchor_training_surface_record(queue=queue, artifacts=artifacts)
        else:
            source_row: dict[str, Any] | None = None
            source_training_experiment: str | None = None
            if str(metadata["source"]) == "queue_row":
                source_row = queue_anchor_row(queue)
                source_training_experiment = str(queue["training_experiment"])
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
                training_surface_record = _anchor_training_surface_record(queue=queue, artifacts=artifacts)
            else:
                run_dir = inspection_run_dir(
                    sweep_id=str(queue["sweep_id"]),
                    target_kind="anchor",
                    target_id="anchor",
                )
                _, training_surface_record = inspection_spec_and_record(
                    row=source_row,
                    run_dir=run_dir,
                    training_experiment=source_training_experiment,
                    sweep_id=str(queue["sweep_id"]),
                    sweeps_root=sweeps_root,
                )
    except RuntimeError:
        run_dir = inspection_run_dir(
            sweep_id=str(queue["sweep_id"]),
            target_kind="anchor",
            target_id="anchor",
        )
        spec, training_surface_record = inspection_spec_and_record(
            row=anchor_row,
            run_dir=run_dir,
            training_experiment=str(queue["training_experiment"]),
            sweep_id=str(queue["sweep_id"]),
            sweeps_root=sweeps_root,
        )
        metadata = {"source": "anchor_context"}
    return {
        "kind": "anchor",
        "identity": {
            "run_id": optional_string(queue.get("anchor_run_id")),
            "source": str(metadata["source"]),
        },
        "artifacts": artifacts,
        "resolved": _surface_payload(spec=spec, training_surface_record=training_surface_record),
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
    return {
        "queue": queue_metadata_payload(queue),
        "row": dict(cast(dict[str, Any], _copy_jsonable(row))),
        "target": target,
    }
