"""Runtime config helpers for training entrypoints."""

from __future__ import annotations

from omegaconf import DictConfig

_VALID_COMPILE_BACKENDS = frozenset({"inductor", "aot_eager", "eager"})
_VALID_COMPILE_MODES = frozenset(
    {"max-autotune-no-cudagraphs", "default", "reduce-overhead"}
)


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


def _coerce_runtime_choice(
    *,
    raw_value: object,
    name: str,
    default: str,
    valid_values: frozenset[str],
) -> str:
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if not normalized:
        return default
    if normalized in valid_values:
        return normalized
    raise ValueError(f"{name} must be one of {sorted(valid_values)}, got {raw_value!r}")


def _resolve_activation_checkpointing(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "activation_checkpointing", False),
        name="runtime.activation_checkpointing",
    )


def _resolve_compile_model(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "compile_model", False),
        name="runtime.compile_model",
    )


def _resolve_compile_dynamic(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "compile_dynamic", False),
        name="runtime.compile_dynamic",
    )


def _resolve_compile_backend(runtime_cfg: DictConfig) -> str:
    return _coerce_runtime_choice(
        raw_value=getattr(runtime_cfg, "compile_backend", "inductor"),
        name="runtime.compile_backend",
        default="inductor",
        valid_values=_VALID_COMPILE_BACKENDS,
    )


def _resolve_compile_mode(runtime_cfg: DictConfig) -> str:
    return _coerce_runtime_choice(
        raw_value=getattr(runtime_cfg, "compile_mode", "max-autotune-no-cudagraphs"),
        name="runtime.compile_mode",
        default="max-autotune-no-cudagraphs",
        valid_values=_VALID_COMPILE_MODES,
    )


def _resolve_trace_activations(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "trace_activations", False),
        name="runtime.trace_activations",
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


def _resolve_loader_task_batch_cache(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "loader_task_batch_cache", False),
        name="runtime.loader_task_batch_cache",
    )


def _resolve_module_grad_norm_every(runtime_cfg: DictConfig) -> int:
    raw_value = getattr(runtime_cfg, "module_grad_norm_every", 1)
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.module_grad_norm_every must be >= 1, got {value}")
    return value


def _resolve_profile_step_timing(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "profile_step_timing", False),
        name="runtime.profile_step_timing",
    )


def _resolve_non_blocking_device_transfer(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "non_blocking_device_transfer", False),
        name="runtime.non_blocking_device_transfer",
    )
