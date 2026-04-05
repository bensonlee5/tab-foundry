"""Runtime config helpers for training entrypoints."""

from __future__ import annotations

from omegaconf import DictConfig


def _coerce_runtime_bool(*, raw_value: object, name: str) -> bool:
    if raw_value is None:
        return False
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int) and raw_value in {0, 1}:
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be boolean-compatible, got {raw_value!r}")


def _resolve_activation_checkpointing(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "activation_checkpointing", False),
        name="runtime.activation_checkpointing",
    )


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


def _resolve_val_batches(runtime_cfg: DictConfig) -> int:
    raw_value = getattr(runtime_cfg, "val_batches", 0)
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"runtime.val_batches must be >= 0, got {value}")
    return value


def _resolve_target_train_seconds(runtime_cfg: DictConfig) -> float | None:
    raw_value = getattr(runtime_cfg, "target_train_seconds", None)
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.target_train_seconds must be > 0, got {value}")
    return value


def _resolve_loader_pin_memory(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "loader_pin_memory", False),
        name="runtime.loader_pin_memory",
    )


def _resolve_loader_persistent_workers(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "loader_persistent_workers", False),
        name="runtime.loader_persistent_workers",
    )


def _resolve_loader_prefetch_factor(runtime_cfg: DictConfig) -> int | None:
    raw_value = getattr(runtime_cfg, "loader_prefetch_factor", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.loader_prefetch_factor must be >= 1, got {value}")
    return value


def _resolve_non_blocking_device_transfer(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "non_blocking_device_transfer", False),
        name="runtime.non_blocking_device_transfer",
    )
