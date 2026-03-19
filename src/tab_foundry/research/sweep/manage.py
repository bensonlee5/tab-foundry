"""Sweep lifecycle helpers for system-delta tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .anchor import anchor_context_from_registry_run, anchor_training_surface_label, build_anchor_surface
from .catalog import (
    SWEEP_QUEUE_SCHEMA,
    SWEEP_SCHEMA,
    load_system_delta_catalog,
    load_system_delta_index,
    load_system_delta_sweep,
)
from .materialize import guarded_initial_state, load_system_delta_queue, materialize_system_delta_queue
from .matrix import render_and_write_system_delta_matrix
from .paths_io import _copy_jsonable, _write_yaml, default_matrix_path, default_queue_path, default_sweep_index_path, sweep_metadata_path, sweep_queue_path
from .validation import ensure_mapping, ensure_non_empty_string


DEFAULT_SWEEP_STATUS = "draft"


def instantiate_queue_row(
    *,
    sweep_id: str,
    anchor_run_id: str,
    order: int,
    delta_id: str,
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    status, interpretation_status, next_action_override = guarded_initial_state(
        delta_entry=delta_entry,
        anchor_context=anchor_context,
    )
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    parameter_policy = cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))
    return {
        "order": int(order),
        "delta_ref": str(delta_id),
        "status": status,
        "rationale": f"Contextualize `{delta_id}` against anchor `{anchor_run_id}` for sweep `{sweep_id}`.",
        "hypothesis": "",
        "anchor_delta": f"Delta description pending for `{delta_id}` against locked anchor `{anchor_run_id}`.",
        "model": cast(dict[str, Any], _copy_jsonable(default_effective_surface.get("model", {}))),
        "data": cast(dict[str, Any], _copy_jsonable(default_effective_surface.get("data", {}))),
        "preprocessing": cast(dict[str, Any], _copy_jsonable(default_effective_surface.get("preprocessing", {}))),
        "training": cast(
            dict[str, Any],
            _copy_jsonable(
                default_effective_surface.get(
                    "training",
                    {
                        "surface_label": anchor_training_surface_label(anchor_context),
                        "overrides": {},
                    },
                )
            ),
        ),
        "parameter_adequacy_plan": cast(list[Any], _copy_jsonable(parameter_policy.get("default_plan", []))),
        "run_id": None,
        "followup_run_ids": [],
        "decision": None,
        "interpretation_status": interpretation_status,
        "confounders": [],
        "next_action": str(next_action_override or delta_entry.get("default_next_action", "")),
        "notes": [],
    }


def create_sweep(
    *,
    sweep_id: str,
    anchor_run_id: str,
    parent_sweep_id: str | None,
    complexity_level: str,
    benchmark_bundle_path: str,
    control_baseline_id: str,
    delta_refs: Sequence[str] | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    normalized_sweep_id = ensure_non_empty_string(sweep_id, context="sweep_id")
    normalized_anchor_run_id = ensure_non_empty_string(anchor_run_id, context="anchor_run_id")
    normalized_complexity_level = ensure_non_empty_string(complexity_level, context="complexity_level")
    normalized_benchmark_bundle_path = ensure_non_empty_string(benchmark_bundle_path, context="benchmark_bundle_path")
    normalized_control_baseline_id = ensure_non_empty_string(control_baseline_id, context="control_baseline_id")
    resolved_index_path = (index_path or default_sweep_index_path()).expanduser().resolve()
    resolved_sweeps_root = sweeps_root or resolved_index_path.parent
    index = load_system_delta_index(resolved_index_path)
    sweeps = ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    if normalized_sweep_id in sweeps:
        raise RuntimeError(f"sweep_id {normalized_sweep_id!r} already exists")

    catalog = load_system_delta_catalog(catalog_path)
    active_sweep_id = ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")
    template_sweep = load_system_delta_sweep(
        parent_sweep_id or active_sweep_id,
        index_path=resolved_index_path,
        sweeps_root=resolved_sweeps_root,
    )
    anchor_context = anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=registry_path,
    )
    sweep_status = DEFAULT_SWEEP_STATUS
    if not sweeps:
        sweep_status = "active"
        index["active_sweep_id"] = normalized_sweep_id

    sweep_payload = {
        "schema": SWEEP_SCHEMA,
        "sweep_id": normalized_sweep_id,
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": sweep_status,
        "complexity_level": normalized_complexity_level,
        "anchor_run_id": normalized_anchor_run_id,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
        "comparison_policy": str(template_sweep.get("comparison_policy", "anchor_only")),
        "upstream_reference": cast(
            dict[str, Any],
            _copy_jsonable(cast(dict[str, Any], template_sweep.get("upstream_reference", {}))),
        ),
        "anchor_surface": build_anchor_surface(
            anchor_run_id=normalized_anchor_run_id,
            benchmark_bundle_path=normalized_benchmark_bundle_path,
            anchor_context=anchor_context,
        ),
        "anchor_context": anchor_context,
    }

    deltas = ensure_mapping(catalog.get("deltas"), context="catalog deltas")
    if delta_refs is None:
        selected_delta_ids = list(deltas)
    else:
        selected_delta_ids = [
            ensure_non_empty_string(delta_ref, context="delta_refs[]") for delta_ref in delta_refs
        ]
        if not selected_delta_ids:
            raise RuntimeError("delta_refs must include at least one delta id when provided")
        if len(set(selected_delta_ids)) != len(selected_delta_ids):
            raise RuntimeError("delta_refs must not contain duplicates")
        unknown_delta_ids = [delta_id for delta_id in selected_delta_ids if delta_id not in deltas]
        if unknown_delta_ids:
            raise RuntimeError(f"unknown delta_refs for sweep {normalized_sweep_id!r}: {unknown_delta_ids}")
    queue_rows = [
        instantiate_queue_row(
            sweep_id=normalized_sweep_id,
            anchor_run_id=normalized_anchor_run_id,
            order=order,
            delta_id=delta_id,
            delta_entry=cast(dict[str, Any], deltas[delta_id]),
            anchor_context=anchor_context,
        )
        for order, delta_id in enumerate(selected_delta_ids, start=1)
    ]
    queue_payload = {
        "schema": SWEEP_QUEUE_SCHEMA,
        "sweep_id": normalized_sweep_id,
        "rows": queue_rows,
    }

    sweep_info = {
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": sweep_status,
        "anchor_run_id": normalized_anchor_run_id,
        "complexity_level": normalized_complexity_level,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
    }
    sweeps[normalized_sweep_id] = sweep_info

    _write_yaml(sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root), sweep_payload)
    _write_yaml(sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root), queue_payload)
    _write_yaml(resolved_index_path, index)

    queue = materialize_system_delta_queue(
        catalog=catalog,
        sweep=sweep_payload,
        queue_instance=queue_payload,
        catalog_path=catalog_path,
        sweeps_root=resolved_sweeps_root,
    )
    matrix_path = render_and_write_system_delta_matrix(
        sweep_id=normalized_sweep_id,
        queue=queue,
        registry_path=registry_path,
        sweeps_root=resolved_sweeps_root,
    )
    if ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id") == normalized_sweep_id:
        sync_active_sweep_aliases(
            sweep_id=normalized_sweep_id,
            index_path=resolved_index_path,
            catalog_path=catalog_path,
            registry_path=registry_path,
            sweeps_root=resolved_sweeps_root,
        )

    return {
        "sweep_path": str(sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "queue_path": str(sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "matrix_path": str(matrix_path),
        "index_path": str(resolved_index_path),
    }


def set_active_sweep(
    sweep_id: str,
    *,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    normalized_sweep_id = ensure_non_empty_string(sweep_id, context="sweep_id")
    resolved_index_path = (index_path or default_sweep_index_path()).expanduser().resolve()
    index = load_system_delta_index(resolved_index_path)
    sweeps = ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    if normalized_sweep_id not in sweeps:
        raise RuntimeError(f"unknown sweep_id: {normalized_sweep_id}")
    index["active_sweep_id"] = normalized_sweep_id
    _write_yaml(resolved_index_path, index)
    return sync_active_sweep_aliases(
        sweep_id=normalized_sweep_id,
        index_path=resolved_index_path,
        catalog_path=catalog_path,
        registry_path=registry_path,
        sweeps_root=sweeps_root,
    )


def sync_active_sweep_aliases(
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    alias_queue_path = default_queue_path()
    alias_matrix_path = default_matrix_path()
    _write_yaml(alias_queue_path, queue)
    _ = render_and_write_system_delta_matrix(
        sweep_id=str(queue["sweep_id"]),
        queue=queue,
        registry_path=registry_path,
        out_path=alias_matrix_path,
    )
    return {
        "queue_alias_path": str(alias_queue_path.resolve()),
        "matrix_alias_path": str(alias_matrix_path.resolve()),
    }


def list_sweeps(*, index_path: Path | None = None) -> list[dict[str, Any]]:
    index = load_system_delta_index(index_path)
    active_sweep_id = ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")
    sweeps = ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    ordered = sorted(sweeps.items(), key=lambda item: str(item[0]))
    return [
        {
            "sweep_id": sweep_id,
            "is_active": sweep_id == active_sweep_id,
            **cast(dict[str, Any], _copy_jsonable(sweep_info)),
        }
        for sweep_id, sweep_info in ordered
    ]
