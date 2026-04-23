"""Hardware-profile helpers shared by training and benchmark registries."""

from __future__ import annotations

import re
from typing import Any


_GPU_CLASS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("h100", "h100"),
    ("h200", "h200"),
    ("a100", "a100"),
    ("a10g", "a10g"),
    ("a10", "a10"),
    ("l40s", "l40s"),
    ("l40", "l40"),
    ("l4", "l4"),
    ("v100", "v100"),
    ("t4", "t4"),
    ("quadro rtx 8000", "rtx8000"),
    ("rtx 6000 ada", "rtx6000ada"),
    ("rtx 6000", "rtx6000"),
    ("rtx 4090", "rtx4090"),
    ("rtx 3090", "rtx3090"),
)

_GPU_UTILIZATION_CAPABILITIES: dict[str, dict[str, Any]] = {
    "a100": {
        "theoretical_hbm_bandwidth_gbps": 2039.0,
        "precisions": {
            "bf16": {
                "theoretical_peak_tflops_per_second": 312.0,
                "peak_compute_basis": "tensorcore_bf16_dense",
            },
            "fp16": {
                "theoretical_peak_tflops_per_second": 312.0,
                "peak_compute_basis": "tensorcore_fp16_dense",
            },
        },
    },
    "h100": {
        "theoretical_hbm_bandwidth_gbps": 3350.0,
        "precisions": {
            "bf16": {
                "theoretical_peak_tflops_per_second": 989.0,
                "peak_compute_basis": "tensorcore_bf16_dense",
            },
            "fp16": {
                "theoretical_peak_tflops_per_second": 989.0,
                "peak_compute_basis": "tensorcore_fp16_dense",
            },
        },
    },
    "h200": {
        "theoretical_hbm_bandwidth_gbps": 4800.0,
        "precisions": {
            "bf16": {
                "theoretical_peak_tflops_per_second": 989.0,
                "peak_compute_basis": "tensorcore_bf16_dense",
            },
            "fp16": {
                "theoretical_peak_tflops_per_second": 989.0,
                "peak_compute_basis": "tensorcore_fp16_dense",
            },
        },
    },
}


def _load_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _slugify_name(raw_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(raw_name).strip().lower())
    normalized = normalized.strip("-")
    return normalized or "unknown"


def normalize_gpu_class(
    raw_device_name: str | None,
    *,
    device_type: str | None = None,
) -> str | None:
    """Collapse raw accelerator names into stable GPU-class ids."""

    normalized_device_type = None if device_type is None else str(device_type).strip().lower()
    if normalized_device_type in {"cpu", "mps"}:
        return normalized_device_type
    if raw_device_name is None or not str(raw_device_name).strip():
        return None
    normalized_name = str(raw_device_name).strip().lower()
    for needle, gpu_class in _GPU_CLASS_PATTERNS:
        if needle in normalized_name:
            return gpu_class
    return _slugify_name(normalized_name)


def normalize_vram_class_gb(total_device_vram_bytes: int | None) -> int | None:
    """Quantize raw device memory into a stable GiB class."""

    if total_device_vram_bytes is None:
        return None
    total_bytes = int(total_device_vram_bytes)
    if total_bytes <= 0:
        return None
    gib = float(total_bytes) / float(1024**3)
    return max(1, int(round(gib)))


def build_hardware_profile_id(
    *,
    gpu_class: str | None,
    vram_class_gb: int | None,
) -> str | None:
    """Build the canonical hardware-profile id."""

    normalized_gpu_class = None if gpu_class is None else str(gpu_class).strip().lower()
    if not normalized_gpu_class:
        return None
    if vram_class_gb is None:
        return normalized_gpu_class
    return f"{normalized_gpu_class}_{int(vram_class_gb)}gb"


def normalize_mixed_precision_mode(value: Any) -> str | None:
    """Normalize runtime mixed-precision modes for capability lookups."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    aliases = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "no": "no",
        "fp32": "no",
        "float32": "no",
        "full": "no",
    }
    return aliases.get(normalized, normalized)


def resolve_gpu_utilization_capability(
    *,
    gpu_class: str | None,
    mixed_precision: Any,
) -> dict[str, Any] | None:
    """Return roofline-adjacent peak capability data for supported GPU/precision pairs."""

    normalized_gpu_class = None if gpu_class is None else str(gpu_class).strip().lower() or None
    normalized_precision = normalize_mixed_precision_mode(mixed_precision)
    if normalized_gpu_class is None or normalized_precision is None:
        return None
    raw_capability = _GPU_UTILIZATION_CAPABILITIES.get(normalized_gpu_class)
    if raw_capability is None:
        return None
    raw_precisions = raw_capability.get("precisions")
    if not isinstance(raw_precisions, dict):
        return None
    precision_capability = raw_precisions.get(normalized_precision)
    if not isinstance(precision_capability, dict):
        return None
    bandwidth_gbps = raw_capability.get("theoretical_hbm_bandwidth_gbps")
    theoretical_peak_tflops = precision_capability.get("theoretical_peak_tflops_per_second")
    if bandwidth_gbps is None or theoretical_peak_tflops is None:
        return None
    bandwidth_gbps_f = float(bandwidth_gbps)
    theoretical_peak_tflops_f = float(theoretical_peak_tflops)
    if bandwidth_gbps_f <= 0.0 or theoretical_peak_tflops_f <= 0.0:
        return None
    return {
        "theoretical_peak_tflops_per_second": theoretical_peak_tflops_f,
        "theoretical_hbm_bandwidth_gbps": bandwidth_gbps_f,
        # Both values use decimal giga-units, so the knee is 1000 * TFLOP/s / GB/s.
        "roofline_knee_flops_per_byte": 1000.0 * theoretical_peak_tflops_f / bandwidth_gbps_f,
        "peak_compute_basis": precision_capability.get("peak_compute_basis"),
    }


def build_hardware_summary(device: Any | None) -> dict[str, Any] | None:
    """Build one normalized hardware summary for training telemetry."""

    if device is None:
        return None

    device_type = None
    device_index = None
    raw_device_name: str | None = None
    total_device_vram_bytes: int | None = None

    raw_type = getattr(device, "type", None)
    if raw_type is not None:
        device_type = str(raw_type).strip().lower() or None
        raw_index = getattr(device, "index", None)
        if raw_index is not None:
            device_index = int(raw_index)
    elif isinstance(device, str):
        normalized = str(device).strip().lower()
        if normalized:
            if ":" in normalized:
                device_type, raw_index = normalized.split(":", 1)
                if raw_index.isdigit():
                    device_index = int(raw_index)
            else:
                device_type = normalized

    if device_type is None:
        return None

    torch_module = _load_torch()
    if device_type == "cuda" and torch_module is not None and torch_module.cuda.is_available():
        if device_index is None:
            device_index = int(torch_module.cuda.current_device())
        properties = torch_module.cuda.get_device_properties(device_index)
        raw_device_name = str(getattr(properties, "name", "")).strip() or None
        total_memory = getattr(properties, "total_memory", None)
        if total_memory is not None:
            total_device_vram_bytes = int(total_memory)
    elif device_type == "cpu":
        raw_device_name = "cpu"
    elif device_type == "mps":
        raw_device_name = "mps"

    gpu_class = normalize_gpu_class(raw_device_name, device_type=device_type)
    vram_class_gb = normalize_vram_class_gb(total_device_vram_bytes)
    hardware_profile_id = build_hardware_profile_id(
        gpu_class=gpu_class,
        vram_class_gb=vram_class_gb,
    )

    payload = {
        "device_type": device_type,
        "raw_device_name": raw_device_name,
        "gpu_class": gpu_class,
        "total_device_vram_bytes": total_device_vram_bytes,
        "vram_class_gb": vram_class_gb,
        "hardware_profile_id": hardware_profile_id,
    }
    return payload if any(value is not None for value in payload.values()) else None
