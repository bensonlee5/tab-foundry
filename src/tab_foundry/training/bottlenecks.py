"""Posthoc bottleneck summaries from existing timing/utilization payloads."""

from __future__ import annotations

import math
from typing import Any, Mapping


_STEP_TIMING_BUCKETS = (
    "data_wait",
    "batch_diagnostics",
    "h2d_transfer",
    "forward_backward",
    "activation_trace",
    "grad_diagnostics",
    "optimizer",
    "checkpoint",
)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _bucket_fraction(
    buckets: Mapping[str, Any],
    bucket: str,
) -> float | None:
    raw_payload = buckets.get(bucket)
    if not isinstance(raw_payload, Mapping):
        return None
    return _optional_float(raw_payload.get("fraction_of_profiled_step_time"))


def _bucket_mean_seconds(
    buckets: Mapping[str, Any],
    bucket: str,
) -> float | None:
    raw_payload = buckets.get(bucket)
    if not isinstance(raw_payload, Mapping):
        return None
    return _optional_float(raw_payload.get("mean_seconds"))


def _sum_optional(values: tuple[float | None, ...]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(sum(finite))


def build_bottleneck_summary(
    *,
    step_timing_summary: Mapping[str, Any] | None,
    utilization_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a neutral posthoc bottleneck summary from existing telemetry."""

    if not isinstance(step_timing_summary, Mapping):
        return None
    raw_buckets = step_timing_summary.get("buckets")
    if not isinstance(raw_buckets, Mapping):
        return None

    ranked_buckets: list[dict[str, Any]] = []
    for bucket in _STEP_TIMING_BUCKETS:
        mean_seconds = _bucket_mean_seconds(raw_buckets, bucket)
        fraction = _bucket_fraction(raw_buckets, bucket)
        if mean_seconds is None and fraction is None:
            continue
        ranked_buckets.append(
            {
                "name": bucket,
                "mean_seconds": mean_seconds,
                "fraction_of_profiled_step_time": fraction,
            }
        )
    ranked_buckets.sort(
        key=lambda item: (
            -float(item["fraction_of_profiled_step_time"] or 0.0),
            str(item["name"]),
        )
    )

    utilization_payload = utilization_summary if isinstance(utilization_summary, Mapping) else {}
    payload: dict[str, Any] = {
        "profiled_step_count": (
            None
            if step_timing_summary.get("profiled_step_count") is None
            else int(step_timing_summary["profiled_step_count"])
        ),
        "mean_profiled_step_seconds": _optional_float(
            step_timing_summary.get("mean_profiled_step_seconds")
        ),
        "dominant_bucket": None if not ranked_buckets else str(ranked_buckets[0]["name"]),
        "ranked_step_time_buckets": ranked_buckets,
        "host_pipeline_fraction": _sum_optional(
            (
                _bucket_fraction(raw_buckets, "data_wait"),
                _bucket_fraction(raw_buckets, "batch_diagnostics"),
            )
        ),
        "h2d_transfer_fraction": _bucket_fraction(raw_buckets, "h2d_transfer"),
        "forward_backward_fraction": _bucket_fraction(raw_buckets, "forward_backward"),
        "optimizer_fraction": _bucket_fraction(raw_buckets, "optimizer"),
        "checkpoint_fraction": _bucket_fraction(raw_buckets, "checkpoint"),
        "diagnostic_overhead_fraction": _sum_optional(
            (
                _bucket_fraction(raw_buckets, "activation_trace"),
                _bucket_fraction(raw_buckets, "grad_diagnostics"),
            )
        ),
        "achieved_train_tflops_per_second": _optional_float(
            utilization_payload.get("achieved_train_tflops_per_second")
        ),
        "theoretical_peak_tflops_per_second": _optional_float(
            utilization_payload.get("theoretical_peak_tflops_per_second")
        ),
        "compute_utilization_fraction": _optional_float(
            utilization_payload.get("compute_utilization_fraction")
        ),
    }
    return payload if any(value is not None for value in payload.values()) else None
