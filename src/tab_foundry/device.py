"""Dependency-light device selection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def _load_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _mps_is_available(torch_module: object) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps_backend = None if backends is None else getattr(backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_device(device: str) -> str:
    """Resolve auto device selection to a concrete torch device string."""

    normalized = str(device).strip().lower()
    if normalized != "auto":
        return normalized
    torch_module = _load_torch()
    if torch_module is None:
        return "cpu"
    if torch_module.cuda.is_available():
        return "cuda"
    if _mps_is_available(torch_module):
        return "mps"
    return "cpu"


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve one requested device to a validated ``torch.device``."""

    torch_module = _load_torch()
    if torch_module is None:
        raise RuntimeError("torch is required for strict device validation")

    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch_module.device(resolve_device("auto"))
    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("requested --device cuda, but CUDA is not available")
        return torch_module.device("cuda")
    if normalized == "mps":
        if not _mps_is_available(torch_module):
            raise RuntimeError("requested --device mps, but MPS is not available")
        return torch_module.device("mps")
    if normalized == "cpu":
        return torch_module.device("cpu")
    raise RuntimeError(f"unsupported device: {requested!r}")
