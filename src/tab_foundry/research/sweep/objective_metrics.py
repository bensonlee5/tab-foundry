"""Objective-metric helpers for system-delta sweep reporting."""

from __future__ import annotations

from typing import Any, Mapping

from tab_foundry.training.instability import (
    CELL_BPC_OBJECTIVE_METRIC,
    CLASSIFICATION_OBJECTIVE_METRIC,
)


def _normalized_metric_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


def objective_metric_from_run(run: Mapping[str, Any] | None) -> str | None:
    if not isinstance(run, Mapping):
        return None
    raw_regime_budget = run.get("regime_budget")
    if not isinstance(raw_regime_budget, Mapping):
        return None
    return _normalized_metric_name(raw_regime_budget.get("objective_metric"))


def objective_metric_from_queue_metrics(queue_metrics: Mapping[str, Any] | None) -> str | None:
    if not isinstance(queue_metrics, Mapping):
        return None
    return _normalized_metric_name(queue_metrics.get("objective_metric"))


def is_classification_objective_metric(objective_metric: str | None) -> bool:
    return objective_metric == CLASSIFICATION_OBJECTIVE_METRIC


def is_legacy_feature_cell_metric_key(metric_key: str) -> bool:
    return metric_key in {
        "best_bpc",
        "final_bpc",
        "best_bpf",
        "final_bpf",
        "final_minus_best_bpc",
        "final_minus_best_bpf",
        "delta_final_bpc",
        "delta_final_bpf",
    }


def display_metric_label(
    label: str,
    *,
    metric_key: str,
    objective_metric: str | None,
) -> str:
    if (
        is_classification_objective_metric(objective_metric)
        and is_legacy_feature_cell_metric_key(metric_key)
    ):
        return f"{label} (legacy feature-cell diagnostic)"
    return label


def preferred_final_metric_keys(objective_metric: str | None) -> tuple[str, ...]:
    if objective_metric == CLASSIFICATION_OBJECTIVE_METRIC:
        return (
            "final_log_loss",
            "final_brier_score",
            "final_roc_auc",
            "final_bpc",
            "final_bpf",
            "final_crps",
            "final_avg_pinball_loss",
            "final_picp_90",
        )
    if objective_metric == CELL_BPC_OBJECTIVE_METRIC:
        return (
            "final_bpc",
            "final_bpf",
            "final_log_loss",
            "final_brier_score",
            "final_roc_auc",
            "final_crps",
            "final_avg_pinball_loss",
            "final_picp_90",
        )
    return (
        "final_bpc",
        "final_log_loss",
        "final_roc_auc",
        "final_bpf",
        "final_brier_score",
        "final_crps",
        "final_avg_pinball_loss",
        "final_picp_90",
    )


def preferred_drift_metric_keys(objective_metric: str | None) -> tuple[str, ...]:
    if objective_metric == CLASSIFICATION_OBJECTIVE_METRIC:
        return (
            "final_minus_best_log_loss",
            "final_minus_best_brier_score",
            "final_minus_best_roc_auc",
            "final_minus_best_bpc",
            "final_minus_best_bpf",
            "final_minus_best_crps",
            "final_minus_best_avg_pinball_loss",
            "final_minus_best_picp_90",
        )
    if objective_metric == CELL_BPC_OBJECTIVE_METRIC:
        return (
            "final_minus_best_bpc",
            "final_minus_best_bpf",
            "final_minus_best_log_loss",
            "final_minus_best_roc_auc",
            "final_minus_best_brier_score",
            "final_minus_best_crps",
            "final_minus_best_avg_pinball_loss",
            "final_minus_best_picp_90",
        )
    return (
        "final_minus_best_bpc",
        "final_minus_best_log_loss",
        "final_minus_best_roc_auc",
        "final_minus_best_bpf",
        "final_minus_best_brier_score",
        "final_minus_best_crps",
        "final_minus_best_avg_pinball_loss",
        "final_minus_best_picp_90",
    )


def first_present_metric_key(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        if payload.get(key) is not None:
            return key
    return None
