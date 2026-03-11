"""Distributed reduction helpers."""

from __future__ import annotations

import torch
from accelerate import Accelerator


def _reduction_float_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _reduce_sum_scalar(accelerator: Accelerator, value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=_reduction_float_dtype(device))
    reduced = accelerator.reduce(tensor, reduction="sum")
    return reduced.item()


def _reduce_count_scalar(accelerator: Accelerator, value: int, *, device: torch.device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.int64)
    reduced = accelerator.reduce(tensor, reduction="sum")
    return int(reduced.item())


def _global_mean_from_local(
    accelerator: Accelerator,
    *,
    local_sum: float,
    local_count: int,
    device: torch.device,
    default: float,
) -> float:
    global_sum = _reduce_sum_scalar(accelerator, local_sum, device=device)
    global_count = _reduce_count_scalar(accelerator, local_count, device=device)
    if global_count <= 0:
        return default
    return global_sum / global_count
