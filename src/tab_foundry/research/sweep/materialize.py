"""Queue materialization helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, cast

from pydantic import ValidationError

from tab_foundry.data.corpus_materialization import materialize_corpus_ref
from tab_foundry.external_benchmarks import normalize_external_benchmarks
from tab_foundry.research.lane_contract import resolve_sweep_semantics

from .anchor import anchor_training_surface_label
from .catalog import (
    load_system_delta_catalog_payload,
    load_system_delta_queue_instance_payload,
    load_system_delta_sweep_payload,
)
from .models import (
    DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
    MATERIALIZED_QUEUE_SCHEMA,
    CatalogDeltaPayload,
    CatalogPayload,
    MaterializedQueuePayload,
    MaterializedQueueRowPayload,
    QueueRowPayload,
    SWEEP_QUEUE_SCHEMA,
    SweepPayload,
    SweepQueuePayload,
)
from .configuration import _effective_queue_corpus_ref, _resolved_repo_root, apply_synthetic_epoch_budget
from .paths_io import (
    _copy_jsonable,
    _render_path,
    default_catalog_path,
    default_sweeps_root,
    load_yaml_mapping,
    sweep_matrix_path,
    sweep_metadata_path,
    sweep_queue_path,
)


def _resolved_external_benchmarks(sweep: SweepPayload) -> list[str]:
    values = sweep.external_benchmarks
    return list(normalize_external_benchmarks(
        values,
        default=DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
        context="sweep.external_benchmarks",
        allow_empty=True,
    )
    )


def _materialized_queue_payload(
    *,
    sweep: SweepPayload,
    rows: list[MaterializedQueueRowPayload],
    catalog_path: Path | None,
    sweeps_root: Path,
) -> dict[str, Any]:
    semantics = resolve_sweep_semantics(sweep)
    return {
        "schema": MATERIALIZED_QUEUE_SCHEMA,
        "generated_from_sweep_id": sweep.sweep_id,
        "catalog_path": None if catalog_path is None else _render_path(catalog_path),
        "canonical_sweep_path": _render_path(sweep_metadata_path(sweep.sweep_id, sweeps_root=sweeps_root)),
        "canonical_queue_path": _render_path(sweep_queue_path(sweep.sweep_id, sweeps_root=sweeps_root)),
        "canonical_matrix_path": _render_path(sweep_matrix_path(sweep.sweep_id, sweeps_root=sweeps_root)),
        "sweep_id": sweep.sweep_id,
        "parent_sweep_id": sweep.parent_sweep_id,
        "sweep_status": sweep.status,
        "complexity_level": sweep.complexity_level,
        "anchor_run_id": sweep.anchor_run_id,
        "benchmark_manifest_path": sweep.benchmark_manifest_path,
        "control_baseline_id": sweep.control_baseline_id,
        "external_benchmarks": _resolved_external_benchmarks(sweep),
        **semantics.to_payload_dict(),
        "upstream_reference": cast(dict[str, Any], _copy_jsonable(sweep.upstream_reference)),
        "anchor_surface": cast(dict[str, Any], _copy_jsonable(sweep.anchor_surface)),
        "anchor_context": cast(dict[str, Any], _copy_jsonable(sweep.anchor_context)),
        "rows": [row.to_payload_dict() for row in rows],
    }


def _inspection_surface_payload(
    queue_row: QueueRowPayload | Mapping[str, Any],
    *,
    anchor_context: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    raw_payload = queue_row.get(key)
    if isinstance(raw_payload, Mapping):
        payload = cast(dict[str, Any], _copy_jsonable(raw_payload))
    else:
        payload = {}
    surface_labels = anchor_context.get("surface_labels")
    surface_label = surface_labels.get(key) if isinstance(surface_labels, Mapping) else None
    if not isinstance(payload.get("surface_label"), str) or not str(payload.get("surface_label")).strip():
        if not isinstance(surface_label, str) or not surface_label.strip():
            if key == "training":
                surface_label = anchor_training_surface_label(anchor_context)
            else:
                return payload
        payload["surface_label"] = str(surface_label)
    if key == "training" and not isinstance(payload.get("overrides"), Mapping):
        payload["overrides"] = {}
    return payload


def _surface_payload_or_default(
    queue_row: QueueRowPayload | Mapping[str, Any],
    *,
    field_name: str,
    value: Any,
    default: Mapping[str, Any],
) -> dict[str, Any]:
    field_present = (
        field_name in queue_row.model_fields_set
        if isinstance(queue_row, QueueRowPayload)
        else field_name in queue_row
    )
    merged = cast(dict[str, Any], _copy_jsonable(default))
    if field_present and isinstance(value, Mapping) and not value:
        return cast(dict[str, Any], _copy_jsonable(value))
    if field_present and not isinstance(value, Mapping):
        return {}
    if not isinstance(value, Mapping):
        return merged
    for key, item in cast(Mapping[str, Any], value).items():
        if isinstance(merged.get(str(key)), Mapping) and isinstance(item, Mapping):
            merged[str(key)] = _surface_payload_or_default(
                cast(Mapping[str, Any], value),
                field_name=str(key),
                value=item,
                default=cast(Mapping[str, Any], merged[str(key)]),
            )
        else:
            merged[str(key)] = _copy_jsonable(item) if isinstance(item, (dict, list)) else item
    return merged


def _optional_parent_delta_ref(queue_row: QueueRowPayload | Mapping[str, Any]) -> str | None:
    parent_delta_ref = queue_row.get("parent_delta_ref")
    if parent_delta_ref is None:
        return None
    normalized = str(parent_delta_ref).strip()
    if not normalized:
        raise RuntimeError(
            f"queue row {queue_row.get('delta_ref', '<missing>')!r}.parent_delta_ref must be a non-empty string"
        )
    return normalized


def inspection_row(
    *,
    queue_row: QueueRowPayload,
    anchor_context: Mapping[str, Any],
) -> MaterializedQueueRowPayload:
    parameter_plan = queue_row.parameter_adequacy_plan
    raw_model = queue_row.get("model")
    model_payload = (
        cast(dict[str, Any], _copy_jsonable(raw_model))
        if isinstance(raw_model, Mapping)
        else {}
    )
    payload = {
        "order": int(queue_row["order"]),
        "delta_id": str(queue_row.delta_ref),
        "status": str(queue_row.status),
        "rationale": str(queue_row.get("rationale", "")),
        "hypothesis": str(queue_row.get("hypothesis", "")),
        "anchor_delta": str(queue_row.get("anchor_delta", "")),
        "model": model_payload,
        "dynamic_model_overrides": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("dynamic_model_overrides"))
            if queue_row.get("dynamic_model_overrides")
            else None,
        ),
        "data": _inspection_surface_payload(queue_row, anchor_context=anchor_context, key="data"),
        "preprocessing": _inspection_surface_payload(
            queue_row,
            anchor_context=anchor_context,
            key="preprocessing",
        ),
        "training": _inspection_surface_payload(
            queue_row,
            anchor_context=anchor_context,
            key="training",
        ),
        "parameter_adequacy_plan": cast(list[Any], _copy_jsonable(parameter_plan)),
        "execution_policy": str(queue_row.get("execution_policy", "benchmark_full")),
        "run_id": queue_row.get("run_id"),
        "followup_run_ids": cast(list[Any], _copy_jsonable(queue_row.get("followup_run_ids", []))),
        "decision": queue_row.get("decision"),
        "interpretation_status": str(queue_row.get("interpretation_status", "pending")),
        "confounders": cast(list[Any], _copy_jsonable(queue_row.get("confounders", []))),
        "next_action": str(queue_row.get("next_action", "")),
        "notes": cast(list[Any], _copy_jsonable(queue_row.get("notes", []))),
        "screen_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("screen_metrics")) if queue_row.get("screen_metrics") else None,
        ),
        "benchmark_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("benchmark_metrics")) if queue_row.get("benchmark_metrics") else None,
        ),
    }
    parent_delta_ref = _optional_parent_delta_ref(queue_row)
    if parent_delta_ref is not None:
        payload["parent_delta_ref"] = parent_delta_ref
    return MaterializedQueueRowPayload.model_validate(payload)


def inspection_system_delta_queue(
    *,
    sweep: SweepPayload,
    queue_instance: SweepQueuePayload,
    sweeps_root: Path | None = None,
) -> MaterializedQueuePayload:
    anchor_context = cast(dict[str, Any], _copy_jsonable(sweep.anchor_context))
    rows = [
        inspection_row(queue_row=queue_row, anchor_context=anchor_context)
        for queue_row in sorted(queue_instance.rows, key=lambda row: (int(row.order), str(row.delta_ref)))
    ]
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    return MaterializedQueuePayload.model_validate(
        _materialized_queue_payload(
            sweep=sweep,
            rows=rows,
            catalog_path=None,
            sweeps_root=resolved_sweeps_root,
        )
    )


def evaluate_applicability_guard(
    guard: Mapping[str, Any],
    *,
    anchor_context: Mapping[str, Any],
) -> tuple[bool, str | None]:
    raw_kind = guard.get("kind")
    kind = str(raw_kind).strip() if raw_kind is not None else ""
    if not kind:
        raise RuntimeError("applicability guard kind must be a non-empty string")
    if kind != "requires_anchor_model_selection":
        raise RuntimeError(f"Unsupported applicability guard kind: {kind!r}")
    raw_key = guard.get("key")
    key = str(raw_key).strip() if raw_key is not None else ""
    if not key:
        raise RuntimeError("applicability guard key must be a non-empty string")
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
    delta_entry: CatalogDeltaPayload | Mapping[str, Any],
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
    queue_row: QueueRowPayload,
    delta_entry: CatalogDeltaPayload,
    anchor_context: Mapping[str, Any],
    sweep_id: str,
    repo_root: Path | None,
    sweeps_root: Path | None,
) -> MaterializedQueueRowPayload:
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    parameter_policy = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))),
    )
    parameter_plan = queue_row.parameter_adequacy_plan
    payload = {
        "order": int(queue_row["order"]),
        "delta_id": str(queue_row.delta_ref),
        "status": str(queue_row.status),
        "dimension_family": delta_entry.dimension_family,
        "family": delta_entry.family,
        "binary_applicable": bool(delta_entry.binary_applicable),
        "description": delta_entry.description,
        "rationale": str(queue_row.get("rationale", "")),
        "hypothesis": str(queue_row.get("hypothesis", "")),
        "upstream_delta": delta_entry.upstream_delta,
        "anchor_delta": str(queue_row.get("anchor_delta", "")),
        "entangled_legacy_stage": delta_entry.legacy_stage_alias or "none",
        "expected_effect": delta_entry.expected_effect,
        "adequacy_knobs": cast(list[Any], _copy_jsonable(delta_entry.get("adequacy_knobs", []))),
        "parameter_adequacy_policy": parameter_policy,
        "applicability_guards": cast(
            list[Any],
            _copy_jsonable(delta_entry.get("applicability_guards", [])),
        ),
        "model": cast(
            dict[str, Any],
            _surface_payload_or_default(
                queue_row,
                field_name="model",
                value=queue_row.get("model"),
                default=cast(Mapping[str, Any], default_effective_surface.get("model", {})),
            ),
        ),
        "dynamic_model_overrides": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("dynamic_model_overrides"))
            if queue_row.get("dynamic_model_overrides")
            else None,
        ),
        "data": cast(
            dict[str, Any],
            _surface_payload_or_default(
                queue_row,
                field_name="data",
                value=queue_row.get("data"),
                default=cast(Mapping[str, Any], default_effective_surface.get("data", {})),
            ),
        ),
        "preprocessing": cast(
            dict[str, Any],
            _surface_payload_or_default(
                queue_row,
                field_name="preprocessing",
                value=queue_row.get("preprocessing"),
                default=cast(Mapping[str, Any], default_effective_surface.get("preprocessing", {})),
            ),
        ),
        "training": cast(
            dict[str, Any],
            _surface_payload_or_default(
                queue_row,
                field_name="training",
                value=queue_row.get("training"),
                default=cast(
                    Mapping[str, Any],
                    default_effective_surface.get(
                        "training",
                        {
                            "surface_label": anchor_training_surface_label(anchor_context),
                            "overrides": {},
                        },
                    ),
                ),
            ),
        ),
        "parameter_adequacy_plan": cast(list[Any], _copy_jsonable(parameter_plan)),
        "execution_policy": str(queue_row.get("execution_policy", "benchmark_full")),
        "run_id": queue_row.get("run_id"),
        "followup_run_ids": cast(list[Any], _copy_jsonable(queue_row.get("followup_run_ids", []))),
        "decision": queue_row.get("decision"),
        "interpretation_status": str(queue_row.get("interpretation_status", "pending")),
        "confounders": cast(list[Any], _copy_jsonable(queue_row.get("confounders", []))),
        "next_action": str(queue_row.get("next_action", "")),
        "notes": cast(list[Any], _copy_jsonable(queue_row.get("notes", []))),
        "screen_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("screen_metrics")) if queue_row.get("screen_metrics") else None,
        ),
        "benchmark_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("benchmark_metrics")) if queue_row.get("benchmark_metrics") else None,
        ),
    }
    parent_delta_ref = _optional_parent_delta_ref(queue_row)
    if parent_delta_ref is not None:
        payload["parent_delta_ref"] = parent_delta_ref
    apply_synthetic_epoch_budget(
        row_payload=payload,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    return MaterializedQueueRowPayload.model_validate(payload)


def materialize_system_delta_queue(
    *,
    catalog: CatalogPayload,
    sweep: SweepPayload,
    queue_instance: SweepQueuePayload,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> MaterializedQueuePayload:
    sweep_id = sweep.sweep_id
    resolved_repo_root = _resolved_repo_root(catalog_path=catalog_path, sweeps_root=sweeps_root)
    rows: list[MaterializedQueueRowPayload] = []
    for queue_row in sorted(queue_instance.rows, key=lambda row: (int(row.order), str(row.delta_ref))):
        delta_entry = catalog.deltas.get(queue_row.delta_ref)
        if delta_entry is None:
            raise RuntimeError(f"unknown delta_ref {queue_row.delta_ref!r} in sweep {sweep_id!r}")
        rows.append(
            materialize_row(
                queue_row=queue_row,
                delta_entry=delta_entry,
                anchor_context=cast(dict[str, Any], sweep.anchor_context),
                sweep_id=sweep_id,
                repo_root=resolved_repo_root,
                sweeps_root=sweeps_root,
            )
        )
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    return MaterializedQueuePayload.model_validate(
        _materialized_queue_payload(
            sweep=sweep,
            rows=rows,
            catalog_path=catalog_path or default_catalog_path(),
            sweeps_root=resolved_sweeps_root,
        )
    )


def _load_materialized_queue_payload(path: Path) -> MaterializedQueuePayload:
    payload = load_yaml_mapping(path, context="system delta queue")
    return MaterializedQueuePayload.model_validate(payload)


def _load_system_delta_queue_common(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    mode: Literal["materialized", "inspection"],
) -> MaterializedQueuePayload:
    def _materialize_or_fallback(
        *,
        catalog: CatalogPayload,
        sweep: SweepPayload,
        queue_instance: SweepQueuePayload,
    ) -> MaterializedQueuePayload:
        try:
            return materialize_system_delta_queue(
                catalog=catalog,
                sweep=sweep,
                queue_instance=queue_instance,
                catalog_path=catalog_path,
                sweeps_root=sweeps_root,
            )
        except RuntimeError as exc:
            if mode != "inspection" or "unknown delta_ref" not in str(exc):
                raise
            return inspection_system_delta_queue(
                sweep=sweep,
                queue_instance=queue_instance,
                sweeps_root=sweeps_root,
            )

    if path is None:
        catalog = load_system_delta_catalog_payload(catalog_path)
        sweep = load_system_delta_sweep_payload(sweep_id, index_path=index_path, sweeps_root=sweeps_root)
        queue_instance = load_system_delta_queue_instance_payload(
            sweep_id or sweep.sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return _materialize_or_fallback(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
        )

    payload = load_yaml_mapping(path, context="system delta queue")
    schema = payload.get("schema")
    if schema == SWEEP_QUEUE_SCHEMA:
        try:
            queue_instance = SweepQueuePayload.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"system delta queue instance is invalid: {exc}") from exc
        catalog = load_system_delta_catalog_payload(catalog_path)
        sweep = load_system_delta_sweep_payload(
            queue_instance.sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return _materialize_or_fallback(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
        )
    try:
        materialized = MaterializedQueuePayload.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"system delta queue is invalid: {exc}") from exc
    return materialized


def load_system_delta_queue(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return _load_system_delta_queue_common(
        path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        mode="materialized",
    ).to_payload_dict()


def load_system_delta_queue_for_inspection(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return _load_system_delta_queue_common(
        path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        mode="inspection",
    ).to_payload_dict()


def _ordered_row_models(queue: MaterializedQueuePayload | Mapping[str, Any]) -> list[MaterializedQueueRowPayload]:
    if isinstance(queue, MaterializedQueuePayload):
        rows = queue.rows
    else:
        raw_rows = queue.get("rows")
        if not isinstance(raw_rows, list):
            raise RuntimeError("materialized system delta queue must include rows")
        rows = [MaterializedQueueRowPayload.model_validate(row) for row in raw_rows]
    return sorted(rows, key=lambda row: (int(row.order), str(row.delta_id)))


def ordered_rows(queue: MaterializedQueuePayload | Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row.to_payload_dict() for row in _ordered_row_models(queue)]


def next_ready_row(queue: MaterializedQueuePayload | Mapping[str, Any]) -> dict[str, Any] | None:
    for row in _ordered_row_models(queue):
        if str(row.status).strip().lower() == "ready":
            return row.to_payload_dict()
    return None


def materialize_sweep_corpora(
    *,
    dagzoo_root: Path,
    sweep_id: str | None = None,
    force: bool = False,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
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
    records = [
        materialize_corpus_ref(
            corpus_ref=corpus_ref,
            dagzoo_root=dagzoo_root,
            force=force,
            repo_root=resolved_repo_root,
            sweep_id=resolved_sweep_id,
            sweeps_root=sweeps_root,
        )
        for corpus_ref in requested_corpus_refs
    ]
    return {
        "sweep_id": resolved_sweep_id,
        "recipe_count": len(records),
        "requested_corpus_refs": requested_corpus_refs,
        "requested_recipe_ids": [
            corpus_ref.split("/", 1)[0]
            for corpus_ref in requested_corpus_refs
        ],
        "corpus_refs": [str(record["corpus_ref"]) for record in records],
        "records": records,
    }
