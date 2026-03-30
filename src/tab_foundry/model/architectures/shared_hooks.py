"""Cross-family helpers for activation checkpointing and tracing."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def enable_activation_checkpointing(model: Any) -> None:
    """Enable activation checkpointing on one model instance."""

    model._activation_checkpointing_enabled = True


def disable_activation_checkpointing(model: Any) -> None:
    """Disable activation checkpointing on one model instance."""

    model._activation_checkpointing_enabled = False


def apply_activation_checkpoint(
    model: Any,
    function: Callable[..., torch.Tensor],
    *args: torch.Tensor,
) -> torch.Tensor:
    """Apply activation checkpointing when the model surface enables it."""

    if not bool(getattr(model, "_activation_checkpointing_enabled", False)) or not bool(
        getattr(model, "training", False)
    ):
        return function(*args)
    if not any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args):
        return function(*args)
    return activation_checkpoint(function, *args, use_reentrant=False)


def enable_activation_trace(model: Any) -> None:
    """Enable activation trace collection on one model instance."""

    model._activation_trace = {}


def disable_activation_trace(model: Any) -> None:
    """Disable activation trace collection on one model instance."""

    model._activation_trace = None


def trace_activation(model: Any, name: str, tensor: torch.Tensor) -> None:
    """Accumulate squared activation statistics for one named tensor."""

    trace_state = getattr(model, "_activation_trace", None)
    if trace_state is None:
        return
    trace_tensor = tensor.detach().to(torch.float32)
    trace_sum_sq = float(trace_tensor.square().sum().item())
    trace_count = int(trace_tensor.numel())
    total_sum_sq, total_count = trace_state.get(name, (0.0, 0))
    trace_state[name] = (
        total_sum_sq + trace_sum_sq,
        total_count + trace_count,
    )


def flush_activation_trace_stats(model: Any) -> dict[str, tuple[float, int]] | None:
    """Return and reset raw activation trace statistics for one model."""

    trace_state = getattr(model, "_activation_trace", None)
    if trace_state is None:
        return None
    snapshot = {
        name: (float(total_sum_sq), int(total_count))
        for name, (total_sum_sq, total_count) in trace_state.items()
        if total_count > 0
    }
    model._activation_trace = {}
    return snapshot


def flush_activation_trace(model: Any) -> dict[str, float] | None:
    """Return and reset RMS activation trace statistics for one model."""

    snapshot = flush_activation_trace_stats(model)
    if snapshot is None:
        return None
    return {
        name: float(math.sqrt(total_sum_sq / float(total_count)))
        for name, (total_sum_sq, total_count) in snapshot.items()
        if total_count > 0
    }
