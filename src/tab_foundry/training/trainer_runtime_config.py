"""Runtime config helpers for training entrypoints."""

from __future__ import annotations

from omegaconf import DictConfig


def _resolve_grad_accum_steps(cfg: DictConfig) -> int:
    value = int(getattr(cfg, "grad_accum_steps", 1))
    if value <= 0:
        raise ValueError(f"runtime.grad_accum_steps must be >= 1, got {value}")
    return value


def _checkpoint_every(cfg: DictConfig) -> int | None:
    raw_value = getattr(cfg, "checkpoint_every", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.checkpoint_every must be >= 1, got {value}")
    return value


def _resolve_max_steps(runtime_cfg: DictConfig) -> int | None:
    raw_value = getattr(runtime_cfg, "max_steps", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.max_steps must be >= 1, got {value}")
    return value


def _resolve_target_train_seconds(runtime_cfg: DictConfig) -> float | None:
    raw_value = getattr(runtime_cfg, "target_train_seconds", None)
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.target_train_seconds must be > 0, got {value}")
    return value
