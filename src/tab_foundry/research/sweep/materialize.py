"""Queue materialization helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from .anchor import anchor_training_surface_label
from .catalog import (
    SWEEP_QUEUE_SCHEMA,
    load_system_delta_catalog,
    load_system_delta_queue_instance,
    load_system_delta_sweep,
)
from .paths_io import (
    _copy_jsonable,
    _load_yaml_mapping,
    _render_path,
    default_catalog_path,
    default_sweeps_root,
    sweep_matrix_path,
    sweep_metadata_path,
    sweep_queue_path,
)
from .validation import ensure_non_empty_string, ensure_rows, ensure_string_list, ensure_mapping, validate_prose_fields


MATERIALIZED_QUEUE_SCHEMA = "tab-foundry-system-delta-queue-v1"


def evaluate_applicability_guard(
    guard: Mapping[str, Any],
    *,
    anchor_context: Mapping[str, Any],
) -> tuple[bool, str | None]:
    kind = ensure_non_empty_string(guard.get("kind"), context="applicability guard kind")
    if kind != "requires_anchor_model_selection":
        raise RuntimeError(f"Unsupported applicability guard kind: {kind!r}")
    key = ensure_non_empty_string(guard.get("key"), context="applicability guard key")
    any_of_raw = guard.get("any_of")
    if not isinstance(any_of_raw, list) or not any_of_raw:
        raise RuntimeError("applicability guard any_of must be a non-empty list")
    any_of = {str(item) for item in any_of_raw}
    anchor_model = cast(dict[str, Any], anchor_context.get("model", {}))
    module_selection = anchor_model.get("module_selection")
    if not isinstance(module_selection, dict):
        return False, None
    current_value = module_selection.get(key)
    if current_value is None:
        return False, None
    current_value_str = str(current_value)
    return current_value_str in any_of, current_value_str


def guarded_initial_state(
    *,
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    status = str(delta_entry.get("default_initial_status", "ready"))
    interpretation_status = str(delta_entry.get("default_initial_interpretation_status", "pending"))
    next_action_override: str | None = None
    guards = delta_entry.get("applicability_guards")
    if not isinstance(guards, list):
        return status, interpretation_status, next_action_override
    for raw_guard in guards:
        if not isinstance(raw_guard, dict):
            raise RuntimeError("applicability_guards entries must be mappings")
        matched, _value = evaluate_applicability_guard(raw_guard, anchor_context=anchor_context)
        if matched:
            continue
        status = str(raw_guard.get("failure_status", status))
        interpretation_status = str(
            raw_guard.get("failure_interpretation_status", interpretation_status)
        )
        failure_next_action = raw_guard.get("failure_next_action")
        if isinstance(failure_next_action, str) and failure_next_action.strip():
            next_action_override = str(failure_next_action)
        break
    return status, interpretation_status, next_action_override


def materialize_row(
    *,
    queue_row: Mapping[str, Any],
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    parameter_policy = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))),
    )
    validate_prose_fields(
        queue_row,
        context=f"queue row {queue_row.get('delta_ref', '<missing>')!r}",
        field_names=("notes", "confounders", "parameter_adequacy_plan"),
    )
    validate_prose_fields(
        delta_entry,
        context=f"delta entry {queue_row.get('delta_ref', '<missing>')!r}",
        field_names=("adequacy_knobs",),
    )
    parameter_plan = queue_row.get("parameter_adequacy_plan")
    if not isinstance(parameter_plan, list):
        parameter_plan = parameter_policy.get("default_plan", [])
    _ = ensure_string_list(
        parameter_plan,
        context=f"queue row {queue_row.get('delta_ref', '<missing>')!r}.parameter_adequacy_plan",
    )
    return {
        "order": int(queue_row["order"]),
        "delta_id": ensure_non_empty_string(queue_row.get("delta_ref"), context="queue row delta_ref"),
        "status": str(queue_row["status"]),
        "dimension_family": str(delta_entry["dimension_family"]),
        "family": str(delta_entry["family"]),
        "binary_applicable": bool(delta_entry.get("binary_applicable", False)),
        "description": str(delta_entry["description"]),
        "rationale": str(queue_row.get("rationale", "")),
        "hypothesis": str(queue_row.get("hypothesis", "")),
        "upstream_delta": str(delta_entry["upstream_delta"]),
        "anchor_delta": str(queue_row.get("anchor_delta", "")),
        "entangled_legacy_stage": str(delta_entry.get("legacy_stage_alias", "none")),
        "expected_effect": str(delta_entry["expected_effect"]),
        "adequacy_knobs": cast(list[Any], _copy_jsonable(delta_entry.get("adequacy_knobs", []))),
        "parameter_adequacy_policy": parameter_policy,
        "applicability_guards": cast(
            list[Any],
            _copy_jsonable(delta_entry.get("applicability_guards", [])),
        ),
        "model": cast(
            dict[str, Any],
            _copy_jsonable(queue_row.get("model", default_effective_surface.get("model", {}))),
        ),
        "data": cast(
            dict[str, Any],
            _copy_jsonable(queue_row.get("data", default_effective_surface.get("data", {}))),
        ),
        "preprocessing": cast(
            dict[str, Any],
            _copy_jsonable(
                queue_row.get("preprocessing", default_effective_surface.get("preprocessing", {}))
            ),
        ),
        "training": cast(
            dict[str, Any],
            _copy_jsonable(
                queue_row.get(
                    "training",
                    default_effective_surface.get(
                        "training",
                        {
                            "surface_label": anchor_training_surface_label(anchor_context),
                            "overrides": {},
                        },
                    ),
                )
            ),
        ),
        "parameter_adequacy_plan": cast(list[Any], _copy_jsonable(parameter_plan)),
        "run_id": queue_row.get("run_id"),
        "followup_run_ids": cast(list[Any], _copy_jsonable(queue_row.get("followup_run_ids", []))),
        "decision": queue_row.get("decision"),
        "interpretation_status": str(queue_row.get("interpretation_status", "pending")),
        "confounders": cast(list[Any], _copy_jsonable(queue_row.get("confounders", []))),
        "next_action": str(queue_row.get("next_action", "")),
        "notes": cast(list[Any], _copy_jsonable(queue_row.get("notes", []))),
        "benchmark_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("benchmark_metrics")) if queue_row.get("benchmark_metrics") else None,
        ),
    }


def materialize_system_delta_queue(
    *,
    catalog: Mapping[str, Any],
    sweep: Mapping[str, Any],
    queue_instance: Mapping[str, Any],
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    deltas = ensure_mapping(catalog.get("deltas"), context="catalog deltas")
    sweep_id = ensure_non_empty_string(sweep.get("sweep_id"), context="sweep.sweep_id")
    rows_payload = ensure_rows(queue_instance.get("rows"), context="queue rows")
    rows: list[dict[str, Any]] = []
    for queue_row in sorted(rows_payload, key=lambda row: (int(row["order"]), str(row["delta_ref"]))):
        delta_ref = ensure_non_empty_string(queue_row.get("delta_ref"), context="queue row delta_ref")
        delta_entry = deltas.get(delta_ref)
        if not isinstance(delta_entry, dict):
            raise RuntimeError(f"unknown delta_ref {delta_ref!r} in sweep {sweep_id!r}")
        rows.append(
            materialize_row(
                queue_row=queue_row,
                delta_entry=delta_entry,
                anchor_context=cast(dict[str, Any], sweep.get("anchor_context", {})),
            )
        )
    for index, row in enumerate(rows):
        validate_prose_fields(row, context=f"materialized queue rows[{index}]")
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    return {
        "schema": MATERIALIZED_QUEUE_SCHEMA,
        "generated_from_sweep_id": sweep_id,
        "catalog_path": _render_path(catalog_path or default_catalog_path()),
        "canonical_sweep_path": _render_path(sweep_metadata_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "canonical_queue_path": _render_path(sweep_queue_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "canonical_matrix_path": _render_path(sweep_matrix_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "sweep_id": sweep_id,
        "parent_sweep_id": sweep.get("parent_sweep_id"),
        "sweep_status": sweep.get("status"),
        "complexity_level": sweep.get("complexity_level"),
        "anchor_run_id": sweep["anchor_run_id"],
        "benchmark_bundle_path": sweep["benchmark_bundle_path"],
        "control_baseline_id": sweep["control_baseline_id"],
        "comparison_policy": sweep["comparison_policy"],
        "upstream_reference": cast(dict[str, Any], _copy_jsonable(sweep["upstream_reference"])),
        "anchor_surface": cast(dict[str, Any], _copy_jsonable(sweep["anchor_surface"])),
        "anchor_context": cast(dict[str, Any], _copy_jsonable(sweep.get("anchor_context", {}))),
        "rows": rows,
    }


def load_system_delta_queue(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        catalog = load_system_delta_catalog(catalog_path)
        sweep = load_system_delta_sweep(sweep_id, index_path=index_path, sweeps_root=sweeps_root)
        queue_instance = load_system_delta_queue_instance(
            sweep_id or str(sweep["sweep_id"]),
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return materialize_system_delta_queue(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )

    payload = _load_yaml_mapping(path, context="system delta queue")
    schema = payload.get("schema")
    if schema == SWEEP_QUEUE_SCHEMA:
        queue_instance = payload
        resolved_sweep_id = ensure_non_empty_string(
            queue_instance.get("sweep_id"),
            context="system delta queue instance sweep_id",
        )
        catalog = load_system_delta_catalog(catalog_path)
        sweep = load_system_delta_sweep(
            resolved_sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return materialize_system_delta_queue(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("materialized system delta queue must include rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"materialized system delta queue rows[{index}] must be mappings")
        validate_prose_fields(row, context=f"materialized system delta queue rows[{index}]")
    return payload


def ordered_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = cast(list[dict[str, Any]], queue["rows"])
    return sorted(rows, key=lambda row: (int(row["order"]), str(row["delta_id"])))


def next_ready_row(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    for row in ordered_rows(queue):
        if str(row.get("status", "")).strip().lower() == "ready":
            return row
    return None
