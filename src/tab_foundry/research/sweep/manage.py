"""Sweep lifecycle helpers for system-delta tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from tab_foundry.research.lane_contract import (
    PFN_CONTROL_SURFACES,
    resolve_surface_role,
    resolve_training_config_profile,
    resolve_training_experiment,
)

from .anchor import anchor_context_from_registry_run, anchor_training_surface_label, build_anchor_surface
from .catalog import (
    load_system_delta_catalog_payload,
    load_system_delta_index_payload,
    load_system_delta_sweep_payload,
)
from .materialize import guarded_initial_state, materialize_system_delta_queue
from .matrix import render_and_write_system_delta_matrix
from .models import SWEEP_INDEX_SCHEMA, SWEEP_QUEUE_SCHEMA, SWEEP_SCHEMA, QueueRowPayload, SweepIndexPayload, SweepPayload, SweepQueuePayload
from .paths_io import (
    _copy_jsonable,
    default_sweep_index_path,
    sweep_metadata_path,
    sweep_queue_path,
    write_yaml,
)
from .validation import (
    DEFAULT_NEW_SWEEP_EXTERNAL_BENCHMARKS,
    ensure_external_benchmarks,
)


DEFAULT_SWEEP_STATUS = "draft"
_STAGED_ONLY_MODEL_KEYS = ("arch", "stage", "stage_label", "module_overrides")


def _require_non_empty_string(value: str | None, *, context: str) -> str:
    if value is None:
        raise RuntimeError(f"{context} must be a non-empty string")
    normalized = str(value).strip()
    if not normalized:
        raise RuntimeError(f"{context} must be a non-empty string")
    return normalized


def _sanitize_model_payload_for_training_experiment(
    model_payload: Mapping[str, Any],
    *,
    training_experiment: str,
) -> dict[str, Any]:
    sanitized = cast(dict[str, Any], _copy_jsonable(cast(dict[str, Any], model_payload)))
    if training_experiment not in PFN_CONTROL_SURFACES:
        return sanitized
    for key in _STAGED_ONLY_MODEL_KEYS:
        sanitized.pop(key, None)
    return sanitized


def instantiate_queue_row(
    *,
    sweep_id: str,
    anchor_run_id: str,
    order: int,
    delta_id: str,
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
    training_experiment: str,
) -> dict[str, Any]:
    status, interpretation_status, next_action_override = guarded_initial_state(
        delta_entry=delta_entry,
        anchor_context=anchor_context,
    )
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    model_payload = _sanitize_model_payload_for_training_experiment(
        cast(dict[str, Any], default_effective_surface.get("model", {})),
        training_experiment=training_experiment,
    )
    parameter_policy = cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))
    return {
        "order": int(order),
        "delta_ref": str(delta_id),
        "status": status,
        "rationale": f"Contextualize `{delta_id}` against anchor `{anchor_run_id}` for sweep `{sweep_id}`.",
        "hypothesis": "",
        "anchor_delta": f"Delta description pending for `{delta_id}` against locked anchor `{anchor_run_id}`.",
        "model": model_payload,
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
    external_benchmarks: Sequence[str] | None = None,
    training_experiment: str | None = None,
    training_config_profile: str | None = None,
    surface_role: str | None = None,
    delta_refs: Sequence[str] | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    normalized_sweep_id = _require_non_empty_string(sweep_id, context="sweep_id")
    normalized_anchor_run_id = _require_non_empty_string(anchor_run_id, context="anchor_run_id")
    normalized_complexity_level = _require_non_empty_string(complexity_level, context="complexity_level")
    normalized_benchmark_bundle_path = _require_non_empty_string(benchmark_bundle_path, context="benchmark_bundle_path")
    normalized_control_baseline_id = _require_non_empty_string(control_baseline_id, context="control_baseline_id")
    resolved_index_path = (index_path or default_sweep_index_path()).expanduser().resolve()
    resolved_sweeps_root = sweeps_root or resolved_index_path.parent
    index = load_system_delta_index_payload(resolved_index_path)
    if normalized_sweep_id in index.sweeps:
        raise RuntimeError(f"sweep_id {normalized_sweep_id!r} already exists")

    catalog = load_system_delta_catalog_payload(catalog_path)
    template_sweep = (
        load_system_delta_sweep_payload(
            parent_sweep_id,
            index_path=resolved_index_path,
            sweeps_root=resolved_sweeps_root,
        )
        if parent_sweep_id is not None
        else None
    )
    anchor_context = anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=registry_path,
    )

    explicit_training_experiment = (
        None
        if training_experiment is None
        else _require_non_empty_string(training_experiment, context="training_experiment")
    )
    explicit_training_config_profile = (
        None
        if training_config_profile is None
        else _require_non_empty_string(training_config_profile, context="training_config_profile")
    )
    explicit_surface_role = (
        None if surface_role is None else _require_non_empty_string(surface_role, context="surface_role")
    )
    inherited_external_benchmarks = (
        template_sweep.get("external_benchmarks")
        if template_sweep is not None and external_benchmarks is None
        else None
    )
    resolved_external_benchmarks = ensure_external_benchmarks(
        list(external_benchmarks)
        if external_benchmarks is not None
        else (
            list(inherited_external_benchmarks)
            if isinstance(inherited_external_benchmarks, Sequence)
            else None
        ),
        context="external_benchmarks",
        default=DEFAULT_NEW_SWEEP_EXTERNAL_BENCHMARKS,
        allow_empty=True,
    )
    if (
        template_sweep is None
        and (
            explicit_training_experiment is None
            or explicit_training_config_profile is None
            or explicit_surface_role is None
        )
    ):
        raise RuntimeError(
            "create_sweep requires --parent-sweep-id or all of "
            "--training-experiment, --training-config-profile, and --surface-role"
        )
    if explicit_training_experiment is None:
        assert template_sweep is not None
        resolved_training_experiment = resolve_training_experiment(template_sweep)
    else:
        resolved_training_experiment = explicit_training_experiment
    if explicit_training_config_profile is None:
        if explicit_training_experiment is not None:
            resolved_training_config_profile = explicit_training_experiment
        else:
            assert template_sweep is not None
            resolved_training_config_profile = resolve_training_config_profile(template_sweep)
    else:
        resolved_training_config_profile = explicit_training_config_profile
    if explicit_surface_role is None:
        if explicit_training_experiment is not None:
            resolved_surface_role = resolve_surface_role(
                {"training_experiment": explicit_training_experiment}
            )
        else:
            assert template_sweep is not None
            resolved_surface_role = resolve_surface_role(template_sweep)
    else:
        resolved_surface_role = explicit_surface_role

    sweep_payload = SweepPayload.model_validate(
        {
        "schema": SWEEP_SCHEMA,
        "sweep_id": normalized_sweep_id,
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": DEFAULT_SWEEP_STATUS,
        "complexity_level": normalized_complexity_level,
        "anchor_run_id": normalized_anchor_run_id,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
        "external_benchmarks": resolved_external_benchmarks,
        "training_experiment": resolved_training_experiment,
        "training_config_profile": resolved_training_config_profile,
        "surface_role": resolved_surface_role,
        "comparison_policy": (
            str(template_sweep.get("comparison_policy", "anchor_only"))
            if template_sweep is not None
            else "anchor_only"
        ),
        "upstream_reference": cast(
            dict[str, Any],
            _copy_jsonable(
                cast(
                    dict[str, Any],
                    {} if template_sweep is None else template_sweep.get("upstream_reference", {}),
                )
            ),
        ),
        "anchor_surface": build_anchor_surface(
            anchor_run_id=normalized_anchor_run_id,
            benchmark_bundle_path=normalized_benchmark_bundle_path,
            anchor_context=anchor_context,
        ),
        "anchor_context": anchor_context,
        }
    )

    deltas = catalog.deltas
    if delta_refs is None:
        selected_delta_ids = list(deltas)
    else:
        selected_delta_ids = [
            _require_non_empty_string(delta_ref, context="delta_refs[]") for delta_ref in delta_refs
        ]
        if not selected_delta_ids:
            raise RuntimeError("delta_refs must include at least one delta id when provided")
        if len(set(selected_delta_ids)) != len(selected_delta_ids):
            raise RuntimeError("delta_refs must not contain duplicates")
        unknown_delta_ids = [delta_id for delta_id in selected_delta_ids if delta_id not in deltas]
        if unknown_delta_ids:
            raise RuntimeError(f"unknown delta_refs for sweep {normalized_sweep_id!r}: {unknown_delta_ids}")
    queue_rows = [
        QueueRowPayload.model_validate(
            instantiate_queue_row(
                sweep_id=normalized_sweep_id,
                anchor_run_id=normalized_anchor_run_id,
                order=order,
                delta_id=delta_id,
                delta_entry=deltas[delta_id].to_payload_dict(),
                anchor_context=anchor_context,
                training_experiment=resolved_training_experiment,
            )
        )
        for order, delta_id in enumerate(selected_delta_ids, start=1)
    ]
    queue_payload = SweepQueuePayload.model_validate(
        {
            "schema": SWEEP_QUEUE_SCHEMA,
            "sweep_id": normalized_sweep_id,
            "rows": [row.to_payload_dict() for row in queue_rows],
        }
    )

    sweep_info = {
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": DEFAULT_SWEEP_STATUS,
        "anchor_run_id": normalized_anchor_run_id,
        "complexity_level": normalized_complexity_level,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
        "external_benchmarks": resolved_external_benchmarks,
    }
    index_payload = SweepIndexPayload.model_validate(
        {
            "schema": SWEEP_INDEX_SCHEMA,
            "sweeps": {
                **{sweep_id: entry.to_payload_dict() for sweep_id, entry in index.sweeps.items()},
                normalized_sweep_id: sweep_info,
            },
        }
    )

    write_yaml(
        sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root),
        sweep_payload.to_payload_dict(),
    )
    write_yaml(
        sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root),
        queue_payload.to_payload_dict(),
    )
    write_yaml(resolved_index_path, index_payload.to_payload_dict())

    queue = materialize_system_delta_queue(
        catalog=catalog,
        sweep=sweep_payload,
        queue_instance=queue_payload,
        catalog_path=catalog_path,
        sweeps_root=resolved_sweeps_root,
    )
    matrix_path = render_and_write_system_delta_matrix(
        sweep_id=normalized_sweep_id,
        queue=queue.to_payload_dict(),
        registry_path=registry_path,
        sweeps_root=resolved_sweeps_root,
    )

    return {
        "sweep_path": str(sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "queue_path": str(sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "matrix_path": str(matrix_path),
        "index_path": str(resolved_index_path),
    }


def list_sweeps(*, index_path: Path | None = None) -> list[dict[str, Any]]:
    index = load_system_delta_index_payload(index_path)
    ordered = sorted(index.sweeps.items(), key=lambda item: str(item[0]))
    return [
        {
            "sweep_id": sweep_id,
            **cast(dict[str, Any], _copy_jsonable(sweep_info.to_payload_dict())),
        }
        for sweep_id, sweep_info in ordered
    ]
