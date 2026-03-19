"""Internal derivation and validation helpers for the benchmark run registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch

from tab_foundry.bench.artifacts import load_history
from tab_foundry.bench.nanotabpfn import resolve_tab_foundry_run_artifact_paths
from tab_foundry.bench.registry.record_helpers import (
    _count_parameters_from_cfg,
    _model_payload_from_cfg,
    _resolve_record_checkpoint_path,
    _training_diagnostics_from_history,
    _training_surface_record,
)
from tab_foundry.bench.registry.schema import (
    _BenchmarkRunEntryPayload,
    _BenchmarkRunRecordPayload,
    _TOP_LEVEL_KEYS,
    _validate_payload_model,
    ALLOWED_DECISIONS,
    REGISTRY_SCHEMA,
    REGISTRY_VERSION,
)
from tab_foundry.bench.registry.storage import load_versioned_registry_payload
from tab_foundry.bench.registry.summary_metrics import (
    benchmark_bundle_payload_from_summary,
    ensure_mapping,
    ensure_non_empty_string,
    ensure_optional_finite_number,
    ensure_optional_positive_int,
    ensure_optional_string,
    tab_foundry_metrics_from_summary,
)
from tab_foundry.data.surface import resolve_data_surface


def empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": {},
    }


def sweep_payload(
    *,
    sweep_id: str | None,
    delta_id: str | None,
    parent_sweep_id: str | None,
    queue_order: int | None,
    run_kind: str | None,
) -> dict[str, Any] | None:
    raw_values = (sweep_id, delta_id, parent_sweep_id, queue_order, run_kind)
    if all(value is None for value in raw_values):
        return None
    normalized_sweep_id = ensure_non_empty_string(sweep_id, context="sweep_id")
    normalized_delta_id = ensure_non_empty_string(delta_id, context="delta_id")
    normalized_parent_sweep_id = ensure_optional_string(
        parent_sweep_id,
        context="parent_sweep_id",
    )
    normalized_queue_order = ensure_optional_positive_int(queue_order, context="queue_order")
    normalized_run_kind = ensure_non_empty_string(run_kind, context="run_kind").lower()
    if normalized_run_kind not in {"primary", "followup"}:
        raise RuntimeError("run_kind must be 'primary' or 'followup'")
    return {
        "sweep_id": normalized_sweep_id,
        "delta_id": normalized_delta_id,
        "parent_sweep_id": normalized_parent_sweep_id,
        "queue_order": normalized_queue_order,
        "run_kind": normalized_run_kind,
    }


def validate_record_payload(payload: Any) -> None:
    _ = _validate_payload_model(
        _BenchmarkRunRecordPayload,
        payload,
        context="benchmark run record",
    )


def validate_run_entry(entry: Any, *, run_id: str) -> None:
    validated = _validate_payload_model(
        _BenchmarkRunEntryPayload,
        entry,
        context=f"benchmark run entry {run_id}",
    )
    if str(validated.run_id) != run_id:
        raise RuntimeError(
            "benchmark run entry run_id mismatch: "
            f"expected={run_id!r}, actual={validated.run_id!r}"
        )


def load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    return load_versioned_registry_payload(
        path,
        allow_missing=allow_missing,
        empty_payload=empty_registry(),
        top_level_keys=_TOP_LEVEL_KEYS,
        schema=REGISTRY_SCHEMA,
        version=REGISTRY_VERSION,
        entries_key="runs",
        registry_label="benchmark run registry",
        validate_entry_fn=validate_run_entry,
        entry_label="run_id",
    )


def comparison_delta(
    *,
    reference_run_id: str,
    current_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    def _metric_delta(metric_name: str) -> float | None:
        current_value = ensure_optional_finite_number(
            current_metrics.get(metric_name),
            context=f"current_metrics.{metric_name}",
        )
        reference_value = ensure_optional_finite_number(
            reference_metrics.get(metric_name),
            context=f"reference_metrics.{metric_name}",
        )
        if current_value is None or reference_value is None:
            return None
        return float(current_value) - float(reference_value)

    current_final_log_loss = ensure_optional_finite_number(
        current_metrics.get("final_log_loss"),
        context="current_metrics.final_log_loss",
    )
    reference_final_log_loss = ensure_optional_finite_number(
        reference_metrics.get("final_log_loss"),
        context="reference_metrics.final_log_loss",
    )
    return {
        "reference_run_id": str(reference_run_id),
        "best_roc_auc_delta": _metric_delta("best_roc_auc"),
        "final_roc_auc_delta": _metric_delta("final_roc_auc"),
        "final_log_loss_delta": None
        if current_final_log_loss is None or reference_final_log_loss is None
        else float(current_final_log_loss) - float(reference_final_log_loss),
        "final_brier_score_delta": _metric_delta("final_brier_score"),
        "final_crps_delta": _metric_delta("final_crps"),
        "final_avg_pinball_loss_delta": _metric_delta("final_avg_pinball_loss"),
        "final_picp_90_delta": _metric_delta("final_picp_90"),
        "best_training_time_delta": float(current_metrics["best_training_time"])
        - float(reference_metrics["best_training_time"]),
        "final_training_time_delta": float(current_metrics["final_training_time"])
        - float(reference_metrics["final_training_time"]),
    }


def derive_benchmark_run_record(
    *,
    run_dir: Path,
    comparison_summary_path: Path,
    prior_dir: Path | None = None,
    benchmark_run_record_path: Path | None = None,
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
    normalize_path_value_fn: Callable[[Path], str],
    resolve_registry_path_value_fn: Callable[[str], Path],
    resolve_config_path_fn: Callable[[Any], Path],
    utc_now_fn: Callable[[], str],
) -> dict[str, Any]:
    """Derive one machine-readable benchmark run record from current artifacts."""

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_summary_path = comparison_summary_path.expanduser().resolve()
    from tab_foundry.bench.registry_common import load_comparison_summary

    summary = load_comparison_summary(resolved_summary_path)
    tab_foundry = ensure_mapping(summary["tab_foundry"], context="comparison_summary.tab_foundry")
    summary_run_dir = resolve_registry_path_value_fn(
        ensure_non_empty_string(
            tab_foundry.get("run_dir"),
            context="comparison_summary.tab_foundry.run_dir",
        )
    )
    if summary_run_dir != resolved_run_dir:
        raise RuntimeError(
            "comparison summary run_dir does not match requested run dir: "
            f"summary={summary_run_dir}, requested={resolved_run_dir}"
        )

    history_path, _checkpoint_dir = resolve_tab_foundry_run_artifact_paths(resolved_run_dir)
    best_checkpoint_path = _resolve_record_checkpoint_path(
        resolved_run_dir,
        summary_tab_foundry=tab_foundry,
    )
    checkpoint_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {best_checkpoint_path}")
    raw_cfg = checkpoint_payload.get("config")
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"checkpoint config must be a mapping: {best_checkpoint_path}")
    raw_state_dict = checkpoint_payload.get("model")
    if raw_state_dict is not None and not isinstance(raw_state_dict, dict):
        raise RuntimeError(f"checkpoint model state_dict must be a mapping: {best_checkpoint_path}")
    data_cfg = raw_cfg.get("data")
    runtime_cfg = raw_cfg.get("runtime")
    if not isinstance(data_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise RuntimeError(f"checkpoint config must include data/runtime mappings: {best_checkpoint_path}")
    data_surface = resolve_data_surface(data_cfg)
    manifest_path_raw = (
        data_surface.overrides["manifest_path"]
        if "manifest_path" in data_surface.overrides
        else data_cfg.get("manifest_path")
    )
    if manifest_path_raw is None:
        raise RuntimeError(
            "checkpoint config must include a non-empty effective data.manifest_path"
        )
    manifest_path = resolve_config_path_fn(manifest_path_raw)
    seed_raw = runtime_cfg.get("seed")
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        raise RuntimeError(f"checkpoint runtime.seed must be an int: {best_checkpoint_path}")

    history = load_history(history_path)
    benchmark_bundle = ensure_mapping(summary["benchmark_bundle"], context="comparison_summary.benchmark_bundle")
    raw_artifacts = summary.get("artifacts")
    comparison_curve_path: Path | None = None
    if isinstance(raw_artifacts, dict):
        raw_curve = raw_artifacts.get("comparison_curve_png")
        if isinstance(raw_curve, str) and raw_curve.strip():
            comparison_curve_path = Path(str(raw_curve)).expanduser().resolve()
    if comparison_curve_path is None:
        comparison_curve_path = resolved_summary_path.parent / "comparison_curve.png"
    training_surface_payload, training_surface_path = _training_surface_record(
        run_dir=resolved_run_dir,
        raw_cfg=raw_cfg,
        raw_state_dict=raw_state_dict,
        benchmark_run_record_path=benchmark_run_record_path,
    )

    record = {
        "manifest_path": normalize_path_value_fn(manifest_path),
        "seed_set": [int(seed_raw)],
        "model": _model_payload_from_cfg(
            raw_cfg,
            state_dict=raw_state_dict,
            summary_tab_foundry=tab_foundry,
        ),
        "benchmark_bundle": benchmark_bundle_payload_from_summary(
            benchmark_bundle,
            source_context="comparison_summary.benchmark_bundle.source_path",
            normalize_path_value_fn=normalize_path_value_fn,
            resolve_registry_path_value_fn=resolve_registry_path_value_fn,
        ),
        "artifacts": {
            "run_dir": normalize_path_value_fn(resolved_run_dir),
            "benchmark_dir": normalize_path_value_fn(resolved_summary_path.parent),
            "prior_dir": None if prior_dir is None else normalize_path_value_fn(prior_dir),
            "history_path": normalize_path_value_fn(history_path),
            "best_checkpoint_path": normalize_path_value_fn(best_checkpoint_path),
            "comparison_summary_path": normalize_path_value_fn(resolved_summary_path),
            "comparison_curve_path": normalize_path_value_fn(comparison_curve_path),
            "benchmark_run_record_path": None
            if benchmark_run_record_path is None
            else normalize_path_value_fn(benchmark_run_record_path),
            "training_surface_record_path": None
            if training_surface_path is None
            else normalize_path_value_fn(training_surface_path),
        },
        "tab_foundry_metrics": tab_foundry_metrics_from_summary(tab_foundry),
        "training_diagnostics": _training_diagnostics_from_history(history, raw_cfg=raw_cfg),
        "model_size": _count_parameters_from_cfg(raw_cfg, state_dict=raw_state_dict),
        "surface_labels": None
        if training_surface_payload is None
        else dict(cast(dict[str, Any], training_surface_payload["labels"])),
        "sweep": sweep_payload(
            sweep_id=sweep_id,
            delta_id=delta_id,
            parent_sweep_id=parent_sweep_id,
            queue_order=queue_order,
            run_kind=run_kind,
        ),
        "generated_at_utc": utc_now_fn(),
    }
    validate_record_payload(record)
    return record


def derive_benchmark_run_entry(
    *,
    run_id: str,
    track: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    decision: str,
    conclusion: str,
    parent_run_id: str | None = None,
    anchor_run_id: str | None = None,
    prior_dir: Path | None = None,
    control_baseline_id: str | None = None,
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
    registry_path: Path | None = None,
    ensure_registry_payload_fn: Callable[[Path | None], tuple[Path, dict[str, Any]]] | Any = None,
    derive_benchmark_run_record_fn: Callable[..., dict[str, Any]] | Any = None,
    comparison_delta_fn: Callable[..., dict[str, Any]] = comparison_delta,
    validate_run_entry_fn: Callable[..., None] = validate_run_entry,
    utc_now_fn: Callable[[], str] | Any = None,
    write_json_fn: Callable[[Path, Mapping[str, Any]], Any] | Any = None,
) -> dict[str, Any]:
    """Derive one benchmark registry entry from benchmark artifacts and lineage."""

    normalized_run_id = ensure_non_empty_string(run_id, context="run_id")
    normalized_track = ensure_non_empty_string(track, context="track")
    normalized_experiment = ensure_non_empty_string(experiment, context="experiment")
    normalized_config_profile = ensure_non_empty_string(config_profile, context="config_profile")
    normalized_budget_class = ensure_non_empty_string(budget_class, context="budget_class")
    normalized_decision = ensure_non_empty_string(decision, context="decision").lower()
    if normalized_decision not in ALLOWED_DECISIONS:
        raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
    normalized_conclusion = ensure_non_empty_string(conclusion, context="conclusion")

    if ensure_registry_payload_fn is None or derive_benchmark_run_record_fn is None or utc_now_fn is None or write_json_fn is None:
        raise RuntimeError("registry derivation helpers must be provided")

    resolved_summary_path = comparison_summary_path.expanduser().resolve()
    resolved_record_path = resolved_summary_path.parent / "benchmark_run_record.json"
    record = derive_benchmark_run_record_fn(
        run_dir=run_dir,
        comparison_summary_path=resolved_summary_path,
        prior_dir=prior_dir,
        benchmark_run_record_path=resolved_record_path,
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
    )
    write_json_fn(resolved_record_path, record)

    _resolved_registry_path, payload = ensure_registry_payload_fn(registry_path)
    runs = cast(dict[str, Any], payload["runs"])
    if normalized_run_id in {str(parent_run_id), str(anchor_run_id)}:
        raise RuntimeError("run_id must not match parent_run_id or anchor_run_id")

    parent_entry = None
    if parent_run_id is not None:
        parent_entry = runs.get(str(parent_run_id))
        if parent_entry is None:
            raise RuntimeError(f"unknown parent_run_id: {parent_run_id}")
    anchor_entry = None
    if anchor_run_id is not None:
        anchor_entry = runs.get(str(anchor_run_id))
        if anchor_entry is None:
            raise RuntimeError(f"unknown anchor_run_id: {anchor_run_id}")

    entry = {
        "run_id": normalized_run_id,
        "track": normalized_track,
        "experiment": normalized_experiment,
        "config_profile": normalized_config_profile,
        "budget_class": normalized_budget_class,
        "model": record["model"],
        "lineage": {
            "parent_run_id": None if parent_run_id is None else str(parent_run_id),
            "anchor_run_id": None if anchor_run_id is None else str(anchor_run_id),
            "control_baseline_id": None if control_baseline_id is None else str(control_baseline_id),
        },
        "manifest_path": str(record["manifest_path"]),
        "seed_set": list(record["seed_set"]),
        "benchmark_bundle": record["benchmark_bundle"],
        "artifacts": record["artifacts"],
        "tab_foundry_metrics": record["tab_foundry_metrics"],
        "training_diagnostics": record["training_diagnostics"],
        "model_size": record["model_size"],
        "surface_labels": record.get("surface_labels"),
        "sweep": record.get("sweep"),
        "comparisons": {
            "vs_parent": None
            if parent_entry is None
            else comparison_delta_fn(
                reference_run_id=str(parent_run_id),
                current_metrics=cast(dict[str, Any], record["tab_foundry_metrics"]),
                reference_metrics=cast(dict[str, Any], parent_entry["tab_foundry_metrics"]),
            ),
            "vs_anchor": None
            if anchor_entry is None
            else comparison_delta_fn(
                reference_run_id=str(anchor_run_id),
                current_metrics=cast(dict[str, Any], record["tab_foundry_metrics"]),
                reference_metrics=cast(dict[str, Any], anchor_entry["tab_foundry_metrics"]),
            ),
        },
        "decision": normalized_decision,
        "conclusion": normalized_conclusion,
        "registered_at_utc": utc_now_fn(),
    }
    validate_run_entry_fn(entry, run_id=normalized_run_id)
    return entry
