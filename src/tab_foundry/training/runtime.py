"""Runtime helpers shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from omegaconf import DictConfig
import torch

from tab_foundry.device import resolve_device

from .trainer_runtime_config import (
    _resolve_compile_backend,
    _resolve_compile_dynamic,
    _resolve_compile_mode,
    _resolve_compile_model,
    _resolve_trace_activations,
)


@dataclass(frozen=True, slots=True)
class CompilePolicy:
    """Resolved training-time torch.compile policy."""

    enabled: bool
    backend: str
    mode: str
    dynamic: bool

    def torch_compile_kwargs(self) -> dict[str, object]:
        """Return the torch.compile kwargs for this resolved policy."""

        if not self.enabled:
            return {}
        kwargs: dict[str, object] = {"backend": self.backend}
        if self.backend == "inductor":
            kwargs["mode"] = self.mode
        if self.dynamic:
            kwargs["dynamic"] = True
        return kwargs


def resolve_training_device_name(runtime_cfg: DictConfig) -> str:
    """Resolve the training/eval device and reject unsupported backends."""

    requested_device = str(getattr(runtime_cfg, "device", "auto") or "auto").strip()
    resolved_device = resolve_device(requested_device)
    if resolved_device == "mps":
        if requested_device.lower() == "mps":
            raise ValueError(
                "MPS is unsupported for training and checkpoint evaluation; "
                "got runtime.device='mps'. Use runtime.device='cuda' or 'cpu' instead."
            )
        raise ValueError(
            "MPS is unsupported for training and checkpoint evaluation; "
            f"runtime.device={requested_device!r} resolved to 'mps'. "
            "Use runtime.device='cuda' or 'cpu' instead."
        )
    return resolved_device


def resolve_cpu_mode(runtime_cfg: DictConfig) -> bool:
    """Return whether execution should be pinned to CPU."""

    return resolve_training_device_name(runtime_cfg) == "cpu"


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


def resolve_compile_model(runtime_cfg: DictConfig) -> bool:
    """Resolve training-time torch.compile policy from runtime config."""

    return resolve_compile_policy(runtime_cfg).enabled


def resolve_compile_policy(runtime_cfg: DictConfig) -> CompilePolicy:
    """Resolve and validate the training-time torch.compile policy."""

    compile_model = _resolve_compile_model(runtime_cfg)
    compile_dynamic = _resolve_compile_dynamic(runtime_cfg)
    compile_backend = _resolve_compile_backend(runtime_cfg)
    compile_mode = _resolve_compile_mode(runtime_cfg)
    if not compile_model:
        return CompilePolicy(
            enabled=False,
            backend=compile_backend,
            mode=compile_mode,
            dynamic=compile_dynamic,
        )
    if _resolve_trace_activations(runtime_cfg):
        raise ValueError("runtime.compile_model=true requires runtime.trace_activations=false")
    resolved_device = resolve_training_device_name(runtime_cfg)
    if resolved_device != "cuda":
        raise ValueError(
            "runtime.compile_model=true requires runtime.device to resolve to 'cuda', "
            f"got {resolved_device!r}"
        )
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError("runtime.compile_model=true requires torch.compile support")
    return CompilePolicy(
        enabled=True,
        backend=compile_backend,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )


def build_accelerator_from_runtime(
    runtime_cfg: DictConfig,
    *,
    mixed_precision_override: str | None = None,
    grad_accum_steps_override: int | None = None,
    dataloader_even_batches_override: bool | None = None,
) -> Accelerator:
    """Create an Accelerator honoring runtime device policy."""

    dataloader_config = None
    if dataloader_even_batches_override is not None:
        dataloader_config = DataLoaderConfiguration(
            even_batches=bool(dataloader_even_batches_override),
        )
    return Accelerator(
        mixed_precision=resolve_mixed_precision(runtime_cfg, override=mixed_precision_override),
        gradient_accumulation_steps=resolve_grad_accum_steps(
            runtime_cfg,
            override=grad_accum_steps_override,
        ),
        cpu=resolve_cpu_mode(runtime_cfg),
        dataloader_config=dataloader_config,
    )
