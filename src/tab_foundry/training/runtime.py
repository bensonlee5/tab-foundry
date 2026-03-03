"""Runtime helpers shared by training and evaluation."""

from __future__ import annotations

from accelerate import Accelerator
from omegaconf import DictConfig


def resolve_cpu_mode(runtime_cfg: DictConfig) -> bool:
    """Return whether execution should be pinned to CPU."""

    return str(runtime_cfg.device).strip().lower() == "cpu"


def resolve_mixed_precision(runtime_cfg: DictConfig, *, override: str | None = None) -> str:
    """Resolve mixed precision mode from runtime config."""

    if override is not None:
        return str(override)
    return str(runtime_cfg.mixed_precision)


def resolve_grad_accum_steps(runtime_cfg: DictConfig, *, override: int | None = None) -> int:
    """Resolve gradient accumulation steps from runtime config."""

    if override is not None:
        steps = int(override)
    else:
        steps = int(getattr(runtime_cfg, "grad_accum_steps", 1))
    if steps <= 0:
        raise ValueError(f"runtime.grad_accum_steps must be >= 1, got {steps}")
    return steps


def build_accelerator_from_runtime(
    runtime_cfg: DictConfig,
    *,
    mixed_precision_override: str | None = None,
    grad_accum_steps_override: int | None = None,
) -> Accelerator:
    """Create an Accelerator honoring runtime device policy."""

    return Accelerator(
        mixed_precision=resolve_mixed_precision(runtime_cfg, override=mixed_precision_override),
        gradient_accumulation_steps=resolve_grad_accum_steps(
            runtime_cfg,
            override=grad_accum_steps_override,
        ),
        cpu=resolve_cpu_mode(runtime_cfg),
    )
