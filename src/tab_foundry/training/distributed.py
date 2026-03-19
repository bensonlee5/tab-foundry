"""Distributed reduction helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object


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


def _reduce_any_flag(
    accelerator: Accelerator,
    value: bool | int,
    *,
    device: torch.device,
) -> bool:
    tensor = torch.tensor(1 if bool(value) else 0, device=device, dtype=torch.int64)
    reduced = accelerator.reduce(tensor, reduction="sum")
    return int(reduced.item()) > 0


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


def _reduce_keyed_weighted_scalars(
    accelerator: Accelerator,
    *,
    weighted_sums: Mapping[str, float],
    weights: Mapping[str, float],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    local_keys = sorted({str(key) for key in weighted_sums} | {str(key) for key in weights})
    gathered_keys = gather_object(local_keys)
    ordered_keys: list[str] = []
    if getattr(accelerator, "is_main_process", True):
        key_union: set[str] = set()
        gathered_groups = gathered_keys if isinstance(gathered_keys, list) else [gathered_keys]
        for group in gathered_groups:
            if isinstance(group, (list, tuple, set)):
                key_union.update(str(key) for key in group)
            elif group is not None:
                key_union.add(str(group))
        ordered_keys = sorted(key_union)
    broadcast_payload = [ordered_keys]
    broadcast_object_list(broadcast_payload, from_process=0)
    resolved_keys = broadcast_payload[0]
    if not isinstance(resolved_keys, (list, tuple)):
        return {}, {}
    ordered_keys = [str(key) for key in resolved_keys]
    if not ordered_keys:
        return {}, {}

    packed = torch.zeros(
        2 * len(ordered_keys),
        device=device,
        dtype=_reduction_float_dtype(device),
    )
    for index, key in enumerate(ordered_keys):
        packed[2 * index] = float(weighted_sums.get(key, 0.0))
        packed[2 * index + 1] = float(weights.get(key, 0.0))
    reduced = accelerator.reduce(packed, reduction="sum")
    reduced_sums = {
        key: float(reduced[2 * index].item())
        for index, key in enumerate(ordered_keys)
    }
    reduced_weights = {
        key: float(reduced[2 * index + 1].item())
        for index, key in enumerate(ordered_keys)
    }
    return reduced_sums, reduced_weights
