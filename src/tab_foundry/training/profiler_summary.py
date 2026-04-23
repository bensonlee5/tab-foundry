"""Structured summaries for opt-in torch profiler runs."""

from __future__ import annotations

import math
from typing import Any, Iterable


_COMPUTE_MARKERS = (
    "mm",
    "matmul",
    "bmm",
    "addmm",
    "linear",
    "convolution",
    "scaled_dot_product",
    "flash",
    "attention",
    "softmax",
    "layer_norm",
    "native_layer_norm",
    "gelu",
    "silu",
)
_MEMORY_MARKERS = (
    "copy",
    "memcpy",
    "to",
    "clone",
    "contiguous",
    "cat",
    "stack",
    "gather",
    "scatter",
    "index",
    "slice",
    "select",
    "view",
    "reshape",
)
_OPTIMIZER_MARKERS = (
    "optimizer",
    "adam",
    "sgd",
    "muon",
    "_foreach",
    "clip_grad",
)
_OPERATOR_CLASSES = ("compute", "memory_movement", "optimizer", "other")


def _finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _event_attr(event: Any, *names: str) -> Any:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return value
    return None


def _operator_class(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in _OPTIMIZER_MARKERS):
        return "optimizer"
    if any(marker in lowered for marker in _MEMORY_MARKERS):
        return "memory_movement"
    if any(marker in lowered for marker in _COMPUTE_MARKERS):
        return "compute"
    return "other"


def _operator_payload(event: Any) -> dict[str, Any]:
    name = str(_event_attr(event, "key", "name") or event)
    self_cpu_time_us = _finite_float(_event_attr(event, "self_cpu_time_total"))
    cpu_time_us = _finite_float(_event_attr(event, "cpu_time_total"))
    self_cuda_time_us = _finite_float(
        _event_attr(event, "self_cuda_time_total", "self_device_time_total")
    )
    cuda_time_us = _finite_float(_event_attr(event, "cuda_time_total", "device_time_total"))
    cpu_memory_usage = int(_finite_float(_event_attr(event, "cpu_memory_usage")))
    cuda_memory_usage = int(
        _finite_float(_event_attr(event, "cuda_memory_usage", "device_memory_usage"))
    )
    flops = _finite_float(_event_attr(event, "flops"))
    return {
        "name": name,
        "operator_class": _operator_class(name),
        "count": int(_finite_float(_event_attr(event, "count"))),
        "self_cpu_time_total_us": self_cpu_time_us,
        "cpu_time_total_us": cpu_time_us,
        "self_cuda_time_total_us": self_cuda_time_us,
        "cuda_time_total_us": cuda_time_us,
        "cpu_memory_usage_bytes": cpu_memory_usage,
        "cuda_memory_usage_bytes": cuda_memory_usage,
        "flops": flops if flops > 0.0 else None,
    }


def _profiled_step_count(operators: Iterable[dict[str, Any]]) -> int | None:
    profiled_steps = sum(
        int(operator["count"])
        for operator in operators
        if str(operator["name"]).startswith("ProfilerStep")
    )
    return profiled_steps if profiled_steps > 0 else None


def build_torch_profiler_summary(
    key_averages: Iterable[Any],
    *,
    trace_dir: str,
    summary_path: str,
    output_dir: str,
    activities: list[str],
    max_steps: int,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    top_operator_count: int = 32,
) -> dict[str, Any]:
    """Build a JSON-safe summary from torch profiler key averages."""

    operators = [_operator_payload(event) for event in key_averages]
    operators.sort(
        key=lambda item: (
            -float(item["self_cuda_time_total_us"] or 0.0),
            -float(item["self_cpu_time_total_us"] or 0.0),
            str(item["name"]),
        )
    )
    class_totals: dict[str, dict[str, Any]] = {
        operator_class: {
            "self_cpu_time_total_us": 0.0,
            "self_cuda_time_total_us": 0.0,
            "cpu_memory_allocated_bytes": 0,
            "cuda_memory_allocated_bytes": 0,
            "flops": 0.0,
            "operator_count": 0,
        }
        for operator_class in _OPERATOR_CLASSES
    }
    for operator in operators:
        class_payload = class_totals[str(operator["operator_class"])]
        class_payload["self_cpu_time_total_us"] += float(operator["self_cpu_time_total_us"])
        class_payload["self_cuda_time_total_us"] += float(operator["self_cuda_time_total_us"])
        class_payload["cpu_memory_allocated_bytes"] += max(
            0,
            int(operator["cpu_memory_usage_bytes"]),
        )
        class_payload["cuda_memory_allocated_bytes"] += max(
            0,
            int(operator["cuda_memory_usage_bytes"]),
        )
        if operator["flops"] is not None:
            class_payload["flops"] += float(operator["flops"])
        class_payload["operator_count"] += 1

    total_self_cpu_time_us = sum(float(item["self_cpu_time_total_us"]) for item in operators)
    total_self_cuda_time_us = sum(float(item["self_cuda_time_total_us"]) for item in operators)
    total_flops = sum(float(item["flops"] or 0.0) for item in operators)
    return {
        "schema": "tab-foundry-torch-profiler-summary-v1",
        "output_dir": output_dir,
        "trace_dir": trace_dir,
        "summary_path": summary_path,
        "activities": activities,
        "schedule": {
            "max_steps": int(max_steps),
            "wait": int(wait),
            "warmup": int(warmup),
            "active": int(active),
            "repeat": int(repeat),
            "expected_profiled_step_count": int(active) * int(repeat),
        },
        "profiled_step_count": _profiled_step_count(operators),
        "totals": {
            "self_cpu_time_total_us": total_self_cpu_time_us,
            "self_cuda_time_total_us": total_self_cuda_time_us,
            "cpu_memory_allocated_bytes": sum(
                max(0, int(item["cpu_memory_usage_bytes"])) for item in operators
            ),
            "cuda_memory_allocated_bytes": sum(
                max(0, int(item["cuda_memory_usage_bytes"])) for item in operators
            ),
            "flops": total_flops if total_flops > 0.0 else None,
        },
        "operator_class_totals": class_totals,
        "top_operators": operators[: int(top_operator_count)],
    }
