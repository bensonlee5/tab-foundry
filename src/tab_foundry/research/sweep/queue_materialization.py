"""Concrete queue materialization owners for system-delta sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from tab_foundry.external_benchmarks import normalize_external_benchmarks
from tab_foundry.hashing import sha256_text
from tab_foundry.research.lane_contract import resolve_sweep_semantics
from tab_foundry.training.surface import build_training_surface_record

from .anchor import anchor_training_surface_label
from .configuration import (
    _resolved_repo_root,
    apply_synthetic_epoch_budget,
    validate_one_epoch_contract,
)
from .models import (
    DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
    MATERIALIZED_QUEUE_SCHEMA,
    RESOLVED_QUEUE_SCHEMA,
    CatalogDeltaPayload,
    CatalogPayload,
    MaterializedQueuePayload,
    MaterializedQueueRowPayload,
    QueueRowPayload,
    ResolvedQueuePayload,
    SweepPayload,
    SweepQueuePayload,
)
from .paths_io import (
    _copy_jsonable,
    _render_path,
    default_catalog_path,
    default_sweeps_root,
    repo_root as shared_repo_root,
    sweep_matrix_path,
    sweep_metadata_path,
    sweep_queue_path,
    sweep_resolved_queue_path,
)
from .training_state import normalize_training_surface_record, training_surface_record_fingerprint


def _resolved_external_benchmarks(sweep: SweepPayload) -> list[str]:
    values = sweep.external_benchmarks
    return list(
        normalize_external_benchmarks(
            values,
            default=DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
            context="sweep.external_benchmarks",
            allow_empty=True,
        )
    )


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


def _reuse_train_artifact_payload(
    queue_row: QueueRowPayload | Mapping[str, Any],
) -> dict[str, Any] | None:
    raw_payload = queue_row.get("reuse_train_artifact")
    if raw_payload is None:
        return None
    if isinstance(raw_payload, Mapping):
        return cast(dict[str, Any], _copy_jsonable(raw_payload))
    if hasattr(raw_payload, "to_payload_dict"):
        return cast(dict[str, Any], raw_payload.to_payload_dict())
    return None


def _optional_extra_payload(
    queue_row: QueueRowPayload | Mapping[str, Any],
    key: str,
) -> Any:
    raw_payload = queue_row.get(key)
    if raw_payload is None:
        return None
    return _copy_jsonable(raw_payload) if isinstance(raw_payload, (dict, list)) else raw_payload


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(_copy_jsonable(payload), sort_keys=True, separators=(",", ":")))


def _resolved_queue_inputs_fingerprint(
    *,
    catalog: CatalogPayload,
    sweep: SweepPayload,
    queue_instance: SweepQueuePayload,
) -> str:
    return _json_fingerprint(
        {
            "catalog": catalog.to_payload_dict(),
            "sweep": sweep.to_payload_dict(),
            "queue": queue_instance.to_payload_dict(),
        }
    )


def _resolved_surface_run_dir(
    *,
    repo_root: Path | None,
    sweep_id: str,
    delta_id: str,
) -> Path:
    base_root = repo_root or shared_repo_root()
    return (
        base_root
        / "outputs"
        / ".resolved_queue"
        / "research"
        / str(sweep_id)
        / str(delta_id)
        / "train"
    )


def _resolved_surface_payload(
    *,
    row: Mapping[str, Any],
    sweep_id: str,
    training_experiment: str,
    repo_root: Path | None,
    sweeps_root: Path | None,
) -> tuple[dict[str, Any], str]:
    from .configuration import compose_cfg

    def _row_fallback_record() -> dict[str, Any]:
        labels: dict[str, Any] = {}
        raw_row_model = row.get("model")
        if isinstance(raw_row_model, Mapping):
            model_label = raw_row_model.get("stage_label", raw_row_model.get("arch"))
            if model_label is not None:
                labels["model"] = model_label
        for key in ("data", "preprocessing", "training"):
            raw_row_surface = row.get(key)
            if isinstance(raw_row_surface, Mapping):
                surface_label = raw_row_surface.get("surface_label")
                if surface_label is not None:
                    labels[key] = surface_label
        training_payload = (
            cast(Mapping[str, Any], row.get("training"))
            if isinstance(row.get("training"), Mapping)
            else {}
        )
        overrides = (
            cast(Mapping[str, Any], training_payload.get("overrides"))
            if isinstance(training_payload.get("overrides"), Mapping)
            else {}
        )
        runtime_payload = (
            {
                str(key): value
                for key, value in cast(Mapping[str, Any], overrides.get("runtime", {})).items()
                if str(key) not in {"device", "output_dir"}
            }
            if isinstance(overrides.get("runtime"), Mapping)
            else {}
        )
        return {
            "labels": labels,
            "model": (
                cast(dict[str, Any], _copy_jsonable(raw_row_model))
                if isinstance(raw_row_model, Mapping)
                else {}
            ),
            "data": (
                cast(dict[str, Any], _copy_jsonable(row.get("data")))
                if isinstance(row.get("data"), Mapping)
                else {}
            ),
            "preprocessing": (
                cast(dict[str, Any], _copy_jsonable(row.get("preprocessing")))
                if isinstance(row.get("preprocessing"), Mapping)
                else {}
            ),
            "training": (
                cast(dict[str, Any], _copy_jsonable(training_payload))
                if training_payload
                else {}
            ),
            "runtime": runtime_payload,
        }

    run_dir = _resolved_surface_run_dir(
        repo_root=repo_root,
        sweep_id=sweep_id,
        delta_id=str(row["delta_id"]),
    )
    try:
        cfg = compose_cfg(
            row=row,
            run_dir=run_dir,
            device="cpu",
            training_experiment=training_experiment,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
        raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        record = _row_fallback_record()
        return normalize_training_surface_record(record), training_surface_record_fingerprint(record)
    if not isinstance(raw_cfg, Mapping):
        record = _row_fallback_record()
        return normalize_training_surface_record(record), training_surface_record_fingerprint(record)
    try:
        record = build_training_surface_record(
            raw_cfg=cast(Mapping[str, Any], raw_cfg),
            run_dir=run_dir,
            include_manifest_characteristics=False,
            allow_unresolved_corpus_ref=True,
        )
    except Exception:
        record = _row_fallback_record()
    return normalize_training_surface_record(record), training_surface_record_fingerprint(record)


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
        "benchmark_checkpoint_selection": str(
            queue_row.get("benchmark_checkpoint_selection", "all")
        ),
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
    for extra_key in (
        "dynamic_training_overrides",
        "dynamic_reuse_train_artifact",
        "transfer_context",
        "transfer_resolution",
        "imported_baseline_provenance",
    ):
        extra_payload = _optional_extra_payload(queue_row, extra_key)
        if extra_payload is not None:
            payload[extra_key] = extra_payload
    reuse_train_artifact = _reuse_train_artifact_payload(queue_row)
    if reuse_train_artifact is not None:
        payload["reuse_train_artifact"] = reuse_train_artifact
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
    sweep_id = sweep.sweep_id
    semantics = resolve_sweep_semantics(sweep)
    training_surface = semantics.training_surface
    anchor_context = cast(dict[str, Any], _copy_jsonable(sweep.anchor_context))
    rows = [
        inspection_row(queue_row=queue_row, anchor_context=anchor_context)
        for queue_row in sorted(queue_instance.rows, key=lambda row: (int(row.order), str(row.delta_ref)))
    ]
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    return MaterializedQueuePayload.model_validate(
        {
            "schema": MATERIALIZED_QUEUE_SCHEMA,
            "generated_from_sweep_id": sweep_id,
            "catalog_path": None,
            "canonical_sweep_path": _render_path(sweep_metadata_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "canonical_queue_path": _render_path(sweep_queue_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "canonical_matrix_path": _render_path(sweep_matrix_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "sweep_id": sweep_id,
            "parent_sweep_id": sweep.parent_sweep_id,
            "sweep_status": sweep.status,
            "complexity_level": sweep.complexity_level,
            "anchor_run_id": sweep.anchor_run_id,
            "benchmark_manifest_path": sweep.benchmark_manifest_path,
            "control_baseline_id": sweep.control_baseline_id,
            "external_benchmarks": _resolved_external_benchmarks(sweep),
            "training_experiment": training_surface.training_experiment,
            "training_config_profile": training_surface.training_config_profile,
            "surface_role": training_surface.surface_role,
            "comparison_policy": semantics.comparison_policy,
            "upstream_reference": cast(dict[str, Any], _copy_jsonable(sweep.upstream_reference)),
            "anchor_surface": cast(dict[str, Any], _copy_jsonable(sweep.anchor_surface)),
            "anchor_context": anchor_context,
            "rows": [row.to_payload_dict() for row in rows],
        }
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
        "benchmark_checkpoint_selection": str(
            queue_row.get("benchmark_checkpoint_selection", "all")
        ),
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
    for extra_key in (
        "dynamic_training_overrides",
        "dynamic_reuse_train_artifact",
        "transfer_context",
        "transfer_resolution",
        "imported_baseline_provenance",
    ):
        extra_payload = _optional_extra_payload(queue_row, extra_key)
        if extra_payload is not None:
            payload[extra_key] = extra_payload
    reuse_train_artifact = _reuse_train_artifact_payload(queue_row)
    if reuse_train_artifact is not None:
        payload["reuse_train_artifact"] = reuse_train_artifact
    parent_delta_ref = _optional_parent_delta_ref(queue_row)
    if parent_delta_ref is not None:
        payload["parent_delta_ref"] = parent_delta_ref
    apply_synthetic_epoch_budget(
        row_payload=payload,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    validate_one_epoch_contract(
        payload,
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
    semantics = resolve_sweep_semantics(sweep)
    training_surface = semantics.training_surface
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
        {
            "schema": MATERIALIZED_QUEUE_SCHEMA,
            "generated_from_sweep_id": sweep_id,
            "catalog_path": _render_path(catalog_path or default_catalog_path()),
            "canonical_sweep_path": _render_path(sweep_metadata_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "canonical_queue_path": _render_path(sweep_queue_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "canonical_matrix_path": _render_path(sweep_matrix_path(sweep_id, sweeps_root=resolved_sweeps_root)),
            "sweep_id": sweep_id,
            "parent_sweep_id": sweep.parent_sweep_id,
            "sweep_status": sweep.status,
            "complexity_level": sweep.complexity_level,
            "anchor_run_id": sweep.anchor_run_id,
            "benchmark_manifest_path": sweep.benchmark_manifest_path,
            "control_baseline_id": sweep.control_baseline_id,
            "external_benchmarks": _resolved_external_benchmarks(sweep),
            "training_experiment": training_surface.training_experiment,
            "training_config_profile": training_surface.training_config_profile,
            "surface_role": training_surface.surface_role,
            "comparison_policy": semantics.comparison_policy,
            "upstream_reference": cast(dict[str, Any], _copy_jsonable(sweep.upstream_reference)),
            "anchor_surface": cast(dict[str, Any], _copy_jsonable(sweep.anchor_surface)),
            "anchor_context": cast(dict[str, Any], _copy_jsonable(sweep.anchor_context)),
            "rows": [row.to_payload_dict() for row in rows],
        }
    )


def materialize_resolved_system_delta_queue(
    *,
    catalog: CatalogPayload,
    sweep: SweepPayload,
    queue_instance: SweepQueuePayload,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> ResolvedQueuePayload:
    from . import row_dependencies as _row_dependencies

    materialized = materialize_system_delta_queue(
        catalog=catalog,
        sweep=sweep,
        queue_instance=queue_instance,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    resolved_repo_root = _resolved_repo_root(catalog_path=catalog_path, sweeps_root=sweeps_root)
    training_experiment = resolve_sweep_semantics(sweep).training_surface.training_experiment
    materialized_payload = materialized.to_payload_dict()
    queue_rows_payload = cast(list[dict[str, Any]], materialized_payload["rows"])
    queue_for_resolution = {
        "sweep_id": sweep.sweep_id,
        "training_experiment": training_experiment,
        "rows": queue_rows_payload,
    }
    resolved_rows: list[dict[str, Any]] = []
    for row_payload in queue_rows_payload:
        _row_dependencies.resolve_dynamic_model_overrides(
            queue=queue_for_resolution,
            queue_row=row_payload,
            materialized_row=row_payload,
        )
        _row_dependencies.resolve_dynamic_training_overrides(
            queue=queue_for_resolution,
            queue_row=row_payload,
            materialized_row=row_payload,
        )
        _row_dependencies.resolve_dynamic_reuse_train_artifact(
            queue=queue_for_resolution,
            queue_row=row_payload,
            materialized_row=row_payload,
        )
        validate_one_epoch_contract(
            row_payload,
            repo_root=resolved_repo_root,
            sweep_id=sweep.sweep_id,
            sweeps_root=sweeps_root,
        )
        resolved_surface, resolved_surface_fingerprint = _resolved_surface_payload(
            row=row_payload,
            sweep_id=sweep.sweep_id,
            training_experiment=training_experiment,
            repo_root=resolved_repo_root,
            sweeps_root=sweeps_root,
        )
        row_payload["resolved_surface"] = resolved_surface
        row_payload["resolved_surface_fingerprint"] = resolved_surface_fingerprint
        resolved_rows.append(dict(row_payload))
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    payload = materialized_payload
    payload.update(
        {
            "schema": RESOLVED_QUEUE_SCHEMA,
            "canonical_resolved_queue_path": _render_path(
                sweep_resolved_queue_path(sweep.sweep_id, sweeps_root=resolved_sweeps_root)
            ),
            "inputs_fingerprint": _resolved_queue_inputs_fingerprint(
                catalog=catalog,
                sweep=sweep,
                queue_instance=queue_instance,
            ),
            "rows": resolved_rows,
        }
    )
    return ResolvedQueuePayload.model_validate(payload)
