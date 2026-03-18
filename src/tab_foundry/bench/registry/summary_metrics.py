"""Shared comparison-summary extraction helpers for benchmark registries."""

from __future__ import annotations

import math
from typing import Any, Mapping, cast


def ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def ensure_optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, context=context)


def ensure_optional_positive_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"{context} must be a positive int or null")
    return int(value)


def ensure_optional_finite_number(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{context} must be a number or null")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise RuntimeError(f"{context} must be finite when present")
    return value_f


def ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be an object")
    return cast(dict[str, Any], value)


def benchmark_bundle_payload_from_summary(
    benchmark_bundle: Mapping[str, Any],
    *,
    source_context: str,
    normalize_path_value_fn: Any,
    resolve_registry_path_value_fn: Any,
) -> dict[str, Any]:
    benchmark_bundle_source = ensure_non_empty_string(
        benchmark_bundle.get("source_path"),
        context=source_context,
    )
    return {
        "name": str(benchmark_bundle["name"]),
        "version": int(benchmark_bundle["version"]),
        "source_path": normalize_path_value_fn(resolve_registry_path_value_fn(benchmark_bundle_source)),
        "task_count": int(benchmark_bundle["task_count"]),
        "task_ids": [int(task_id) for task_id in cast(list[Any], benchmark_bundle["task_ids"])],
    }


def tab_foundry_metrics_from_summary(tab_foundry: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "best_step": float(tab_foundry["best_step"]),
        "best_training_time": float(tab_foundry["best_training_time"]),
        "best_roc_auc": ensure_optional_finite_number(
            tab_foundry.get("best_roc_auc"),
            context="comparison_summary.tab_foundry.best_roc_auc",
        ),
        "best_log_loss": ensure_optional_finite_number(
            tab_foundry.get("best_log_loss"),
            context="comparison_summary.tab_foundry.best_log_loss",
        ),
        "best_brier_score": ensure_optional_finite_number(
            tab_foundry.get("best_brier_score"),
            context="comparison_summary.tab_foundry.best_brier_score",
        ),
        "best_crps": ensure_optional_finite_number(
            tab_foundry.get("best_crps"),
            context="comparison_summary.tab_foundry.best_crps",
        ),
        "best_avg_pinball_loss": ensure_optional_finite_number(
            tab_foundry.get("best_avg_pinball_loss"),
            context="comparison_summary.tab_foundry.best_avg_pinball_loss",
        ),
        "best_picp_90": ensure_optional_finite_number(
            tab_foundry.get("best_picp_90"),
            context="comparison_summary.tab_foundry.best_picp_90",
        ),
        "final_step": float(tab_foundry["final_step"]),
        "final_training_time": float(tab_foundry["final_training_time"]),
        "final_roc_auc": ensure_optional_finite_number(
            tab_foundry.get("final_roc_auc"),
            context="comparison_summary.tab_foundry.final_roc_auc",
        ),
        "final_log_loss": ensure_optional_finite_number(
            tab_foundry.get("final_log_loss"),
            context="comparison_summary.tab_foundry.final_log_loss",
        ),
        "final_brier_score": ensure_optional_finite_number(
            tab_foundry.get("final_brier_score"),
            context="comparison_summary.tab_foundry.final_brier_score",
        ),
        "final_crps": ensure_optional_finite_number(
            tab_foundry.get("final_crps"),
            context="comparison_summary.tab_foundry.final_crps",
        ),
        "final_avg_pinball_loss": ensure_optional_finite_number(
            tab_foundry.get("final_avg_pinball_loss"),
            context="comparison_summary.tab_foundry.final_avg_pinball_loss",
        ),
        "final_picp_90": ensure_optional_finite_number(
            tab_foundry.get("final_picp_90"),
            context="comparison_summary.tab_foundry.final_picp_90",
        ),
    }
