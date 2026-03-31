"""Shared queue-state helpers for system-delta sweep execution."""

from __future__ import annotations

from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import resolve_registry_path_value

from .objective_metrics import objective_metric_from_run, preferred_drift_metric_keys
from .queue_updates import append_note, stage_local_telemetry_metrics


def _require_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value).strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def completed_queue_metrics_from_registry_run(run: Mapping[str, Any]) -> dict[str, float]:
    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    diagnostics = cast(dict[str, Any], run["training_diagnostics"])
    comparisons = cast(dict[str, Any], run.get("comparisons", {}))
    raw_vs_anchor = comparisons.get("vs_anchor", {})
    vs_anchor = cast(dict[str, Any], raw_vs_anchor if isinstance(raw_vs_anchor, Mapping) else {})
    expected: dict[str, float] = {}

    best_step = _optional_float(metrics.get("best_step"))
    if best_step is not None:
        expected["best_step"] = best_step
    max_grad_norm = _optional_float(diagnostics.get("max_grad_norm"))
    if max_grad_norm is not None:
        expected["max_grad_norm"] = max_grad_norm

    metric_suffixes = (
        "bpc",
        "bpf",
        "roc_auc",
        "log_loss",
        "brier_score",
        "crps",
        "avg_pinball_loss",
        "picp_90",
    )
    for prefix in ("best", "final"):
        for suffix in metric_suffixes:
            key = f"{prefix}_{suffix}"
            value = _optional_float(metrics.get(key))
            if value is not None:
                expected[key] = value

    for suffix in metric_suffixes:
        best_key = f"best_{suffix}"
        final_key = f"final_{suffix}"
        best_value = expected.get(best_key)
        final_value = expected.get(final_key)
        if best_value is None or final_value is None:
            continue
        expected[f"final_minus_best_{suffix}"] = final_value - best_value
    objective_metric = objective_metric_from_run(run)
    for drift_key in preferred_drift_metric_keys(objective_metric):
        drift_value = expected.get(drift_key)
        if drift_value is not None:
            expected["drift"] = drift_value
            break

    comparison_keys = {
        "final_bpc_delta": "delta_final_bpc",
        "final_bpf_delta": "delta_final_bpf",
        "final_log_loss_delta": "delta_final_log_loss",
        "final_brier_score_delta": "delta_final_brier_score",
        "final_roc_auc_delta": "delta_final_roc_auc",
        "final_crps_delta": "delta_final_crps",
        "final_avg_pinball_loss_delta": "delta_final_avg_pinball_loss",
        "final_picp_90_delta": "delta_final_picp_90",
    }
    for registry_key, queue_key in comparison_keys.items():
        value = _optional_float(vs_anchor.get(registry_key))
        if value is not None:
            expected[queue_key] = value
    artifacts = cast(dict[str, Any], run.get("artifacts", {}))
    run_dir_value = artifacts.get("run_dir")
    if isinstance(run_dir_value, str) and run_dir_value.strip():
        run_dir = resolve_registry_path_value(run_dir_value)
        expected.update(
            {
                key: float(value)
                for key, value in stage_local_telemetry_metrics(run_dir).items()
            }
        )
    return expected


def recover_completed_queue_row_from_registry_run(
    *,
    queue_row: dict[str, Any],
    run_id: str,
    run: Mapping[str, Any],
) -> None:
    decision = _require_non_empty_string(
        run.get("decision"),
        context=f"benchmark registry run {run_id!r}.decision",
    )
    conclusion = _require_non_empty_string(
        run.get("conclusion"),
        context=f"benchmark registry run {run_id!r}.conclusion",
    )
    original_run_id = queue_row.get("run_id")
    queue_row["status"] = "completed"
    queue_row["run_id"] = run_id
    queue_row["followup_run_ids"] = []
    queue_row["decision"] = decision
    queue_row["interpretation_status"] = "completed"
    queue_row["benchmark_metrics"] = completed_queue_metrics_from_registry_run(run)
    queue_row["confounders"] = []
    notes = cast(list[str], queue_row.get("notes", []))
    if isinstance(original_run_id, str) and original_run_id.strip() and original_run_id != run_id:
        notes = append_note(
            notes,
            f"Supersedes historical queue run `{original_run_id}`; that registry entry is retained as history only.",
        )
    notes = append_note(notes, f"Canonical rerun registered as `{run_id}`.")
    notes = append_note(notes, conclusion)
    queue_row["notes"] = notes
