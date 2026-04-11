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
