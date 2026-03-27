"""Sweep-specific device policy helpers."""

from __future__ import annotations

from typing import Any, Callable

from tab_foundry.device import resolve_device as _resolve_device
from tab_foundry.device import resolve_torch_device as _resolve_torch_device


_AUTO_DEVICE = "auto"
_CPU_DEVICE = "cpu"
_CUDA_DEVICE = "cuda"
_MPS_DEVICE = "mps"
_SUPPORTED_SWEEP_DEVICES = frozenset({_AUTO_DEVICE, _CPU_DEVICE, _CUDA_DEVICE})


def normalize_sweep_requested_device(requested_device: str) -> str:
    """Normalize one sweep device request and reject unsupported devices."""

    normalized = str(requested_device or _AUTO_DEVICE).strip().lower()
    if normalized == _MPS_DEVICE:
        raise RuntimeError(
            "research sweep execution does not support --device mps; "
            "use --device cuda, --device cpu, or --device auto"
        )
    if normalized not in _SUPPORTED_SWEEP_DEVICES:
        raise RuntimeError(
            "research sweep execution supports only 'cpu', 'cuda', or 'auto'; "
            f"got {requested_device!r}"
        )
    return normalized


def resolve_sweep_metadata_device(
    requested_device: str,
    *,
    auto_resolve_fn: Callable[[str], str] = _resolve_device,
) -> tuple[str, str]:
    """Resolve one sweep device request for recorded metadata."""

    normalized = normalize_sweep_requested_device(requested_device)
    if normalized != _AUTO_DEVICE:
        return normalized, normalized

    resolved = str(auto_resolve_fn(_AUTO_DEVICE)).strip().lower()
    if resolved == _CUDA_DEVICE:
        return normalized, _CUDA_DEVICE
    return normalized, _CPU_DEVICE


def resolve_sweep_execution_device(
    requested_device: str,
    *,
    auto_resolve_fn: Callable[[str], str] = _resolve_device,
    strict_resolve_torch_device_fn: Callable[[str], Any] = _resolve_torch_device,
) -> str:
    """Resolve one sweep device request to a concrete execution device."""

    _normalized, resolved = resolve_sweep_metadata_device(
        requested_device,
        auto_resolve_fn=auto_resolve_fn,
    )
    if resolved == _CUDA_DEVICE:
        _ = strict_resolve_torch_device_fn(_CUDA_DEVICE)
    return resolved

