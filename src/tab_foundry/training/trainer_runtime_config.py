"""Runtime config helpers for training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import os

from omegaconf import DictConfig

_VALID_COMPILE_BACKENDS = frozenset({"inductor", "aot_eager", "eager"})
_VALID_COMPILE_MODES = frozenset(
    {"max-autotune-no-cudagraphs", "default", "reduce-overhead"}
)
_VALID_COMPILE_SHAPE_DISPATCH_MODES = frozenset({"off", "signature_family"})
_AUTO_SENTINEL = "auto"
_VALID_TASK_BATCH_CACHE_MODES = frozenset({"off", "eager_full", "bounded_streaming"})
_RESERVED_CPU_LARGE_HOST_THRESHOLD = 8
_RESERVED_CPU_SMALL_HOST_COUNT = 1
_RESERVED_CPU_LARGE_HOST_COUNT = 2
_LOW_WORKER_USABLE_CPU_MAX = 4
_MEDIUM_WORKER_USABLE_CPU_MAX = 12
_MEDIUM_WORKER_COUNT = 2
_MAX_AUTO_WORKERS = 8
_WORKERS_PER_CPU_DIVISOR = 4
_LOW_PREFETCH_MAX_WORKERS = 2
_LOW_PREFETCH_FACTOR = 2
_HIGH_PREFETCH_FACTOR = 4


@dataclass(slots=True)
class LoaderOverlapRuntimeSettings:
    """Resolved loader overlap settings after runtime defaults and auto heuristics."""

    num_workers: int
    prefetch_factor: int | None
    num_workers_is_auto: bool
    prefetch_factor_is_auto: bool


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


def _resolve_compile_shape_dispatch_mode(runtime_cfg: DictConfig) -> str:
    raw_value = getattr(runtime_cfg, "compile_shape_dispatch_mode", "off")
    if isinstance(raw_value, bool):
        return "signature_family" if raw_value else "off"
    return _coerce_runtime_choice(
        raw_value=raw_value,
        name="runtime.compile_shape_dispatch_mode",
        default="off",
        valid_values=_VALID_COMPILE_SHAPE_DISPATCH_MODES,
    )


def _resolve_compile_shape_dispatch_max_families(runtime_cfg: DictConfig) -> int:
    raw_value = getattr(runtime_cfg, "compile_shape_dispatch_max_families", 16)
    value = int(raw_value)
    if value <= 0:
        raise ValueError(
            "runtime.compile_shape_dispatch_max_families must be >= 1, "
            f"got {value}"
        )
    return value


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


def _resolve_loader_train_shuffle(runtime_cfg: DictConfig) -> bool:
    raw_value = getattr(runtime_cfg, "loader_train_shuffle", True)
    if raw_value is None:
        return True
    return _coerce_runtime_bool(
        raw_value=raw_value,
        name="runtime.loader_train_shuffle",
    )


def _resolve_loader_persistent_workers(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "loader_persistent_workers", False),
        name="runtime.loader_persistent_workers",
    )


def _resolve_loader_num_workers(runtime_cfg: DictConfig) -> int | str:
    raw_value = getattr(runtime_cfg, "num_workers", 0)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized == _AUTO_SENTINEL:
            return _AUTO_SENTINEL
        if not normalized:
            raise ValueError("runtime.num_workers must be >= 0 or 'auto', got empty string")
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"runtime.num_workers must be >= 0 or 'auto', got {value}")
    return value


def _resolve_loader_prefetch_factor(runtime_cfg: DictConfig) -> int | None | str:
    raw_value = getattr(runtime_cfg, "loader_prefetch_factor", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized == _AUTO_SENTINEL:
            return _AUTO_SENTINEL
        if not normalized:
            return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(
            f"runtime.loader_prefetch_factor must be >= 1, null, or 'auto', got {value}"
        )
    return value


def _resolve_loader_task_batch_cache(runtime_cfg: DictConfig) -> bool:
    return _coerce_runtime_bool(
        raw_value=getattr(runtime_cfg, "loader_task_batch_cache", False),
        name="runtime.loader_task_batch_cache",
    )


def _resolve_loader_task_batch_cache_mode(runtime_cfg: DictConfig) -> str:
    raw_mode = getattr(runtime_cfg, "loader_task_batch_cache_mode", None)
    if raw_mode is not None:
        normalized_mode = str(raw_mode).strip().lower()
        if normalized_mode:
            if normalized_mode not in _VALID_TASK_BATCH_CACHE_MODES:
                raise ValueError(
                    "runtime.loader_task_batch_cache_mode must be one of "
                    f"{sorted(_VALID_TASK_BATCH_CACHE_MODES)}, got {raw_mode!r}"
                )
            return normalized_mode
    return "eager_full" if _resolve_loader_task_batch_cache(runtime_cfg) else "off"


def _resolve_signature_family_run_length(runtime_cfg: DictConfig) -> int:
    raw_value = getattr(runtime_cfg, "signature_family_run_length", 1)
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.signature_family_run_length must be >= 1, got {value}")
    return value


def _resolve_signature_family_optimizer_step_block_length(
    runtime_cfg: DictConfig,
) -> int | None:
    raw_value = getattr(runtime_cfg, "signature_family_optimizer_step_block_length", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(
            "runtime.signature_family_optimizer_step_block_length must be >= 1, "
            f"got {value}"
        )
    return value


def _resolved_cpu_count(*, cpu_count: int | None = None) -> int:
    resolved = os.cpu_count() if cpu_count is None else int(cpu_count)
    if resolved is None or resolved <= 0:
        return 1
    return int(resolved)


def default_loader_num_workers(*, cpu_count: int | None = None) -> int:
    """Return the CPU-count heuristic default for loader worker overlap."""

    resolved_cpu_count = _resolved_cpu_count(cpu_count=cpu_count)
    reserved = (
        _RESERVED_CPU_LARGE_HOST_COUNT
        if resolved_cpu_count >= _RESERVED_CPU_LARGE_HOST_THRESHOLD
        else _RESERVED_CPU_SMALL_HOST_COUNT
    )
    usable = max(1, resolved_cpu_count - reserved)
    if usable <= _LOW_WORKER_USABLE_CPU_MAX:
        return 1
    if usable <= _MEDIUM_WORKER_USABLE_CPU_MAX:
        return _MEDIUM_WORKER_COUNT
    return min(_MAX_AUTO_WORKERS, max(1, usable // _WORKERS_PER_CPU_DIVISOR))


def default_loader_prefetch_factor(*, num_workers: int) -> int | None:
    """Return the default prefetch factor for one resolved worker count."""

    resolved_workers = int(num_workers)
    if resolved_workers <= 0:
        return None
    return (
        _LOW_PREFETCH_FACTOR
        if resolved_workers <= _LOW_PREFETCH_MAX_WORKERS
        else _HIGH_PREFETCH_FACTOR
    )


def resolve_loader_overlap_runtime_settings(
    runtime_cfg: DictConfig,
) -> LoaderOverlapRuntimeSettings:
    """Resolve loader overlap settings, applying hardware-aware auto heuristics."""

    raw_num_workers = _resolve_loader_num_workers(runtime_cfg)
    raw_prefetch_factor = _resolve_loader_prefetch_factor(runtime_cfg)

    num_workers_is_auto = raw_num_workers == _AUTO_SENTINEL
    resolved_num_workers = (
        default_loader_num_workers() if num_workers_is_auto else int(raw_num_workers)
    )

    prefetch_factor_is_auto = raw_prefetch_factor == _AUTO_SENTINEL
    if resolved_num_workers <= 0:
        resolved_prefetch_factor = None
    elif prefetch_factor_is_auto:
        resolved_prefetch_factor = default_loader_prefetch_factor(
            num_workers=resolved_num_workers
        )
    else:
        resolved_prefetch_factor = (
            None if raw_prefetch_factor is None else int(raw_prefetch_factor)
        )

    return LoaderOverlapRuntimeSettings(
        num_workers=resolved_num_workers,
        prefetch_factor=resolved_prefetch_factor,
        num_workers_is_auto=num_workers_is_auto,
        prefetch_factor_is_auto=prefetch_factor_is_auto,
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
