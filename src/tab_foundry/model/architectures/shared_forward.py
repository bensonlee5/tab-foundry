"""Cross-family helpers for common batched-input preparation."""

from __future__ import annotations

from typing import cast

import torch

from tab_foundry.input_normalization import InputNormalizationMode, normalize_train_test_tensors
from tab_foundry.types import TaskBatch


_UNBATCHED_TASK_RANK = 2
_BATCHED_TASK_RANK = 3


def task_num_classes(batch: TaskBatch, *, arch_name: str) -> int:
    """Resolve the class count from explicit metadata or training labels."""

    if batch.num_classes is not None:
        return int(batch.num_classes)
    if batch.y_train.numel() == 0 and batch.y_test.numel() == 0:
        raise RuntimeError(f"{arch_name} requires at least one label")
    max_label = None
    if batch.y_train.numel() > 0:
        max_label = int(batch.y_train.max().item())
    if batch.y_test.numel() > 0:
        y_test_max = int(batch.y_test.max().item())
        max_label = y_test_max if max_label is None else max(max_label, y_test_max)
    if max_label is None:
        raise RuntimeError(f"{arch_name} requires at least one label")
    return max_label + 1


def prepare_task_inputs(
    batch: TaskBatch,
    *,
    arch_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Convert one task batch into the canonical [B, R, C] form."""

    if batch.x_train.ndim == _UNBATCHED_TASK_RANK:
        train_test_split_index = int(batch.x_train.shape[0])
        if train_test_split_index <= 0:
            raise RuntimeError(f"{arch_name} requires at least one training row")
        x_all = torch.cat([batch.x_train, batch.x_test], dim=0).to(torch.float32).unsqueeze(0)
        y_train = batch.y_train.to(torch.int64).unsqueeze(0)
        y_test = batch.y_test.to(torch.int64).unsqueeze(0)
        return x_all, y_train, y_test, train_test_split_index
    if batch.x_train.ndim != _BATCHED_TASK_RANK or batch.x_test.ndim != _BATCHED_TASK_RANK:
        raise RuntimeError(
            f"{arch_name} task batching requires x_train/x_test rank 2 or 3, "
            f"got x_train={tuple(int(dim) for dim in batch.x_train.shape)}, "
            f"x_test={tuple(int(dim) for dim in batch.x_test.shape)}"
        )
    if batch.y_train.ndim != _UNBATCHED_TASK_RANK or batch.y_test.ndim != _UNBATCHED_TASK_RANK:
        raise RuntimeError(
            f"{arch_name} task batching requires y_train/y_test rank 2 when batching, "
            f"got y_train={tuple(int(dim) for dim in batch.y_train.shape)}, "
            f"y_test={tuple(int(dim) for dim in batch.y_test.shape)}"
        )
    if int(batch.x_train.shape[0]) != int(batch.x_test.shape[0]):
        raise RuntimeError(f"{arch_name} batched train/test tensors must share a batch dimension")
    train_test_split_index = int(batch.x_train.shape[1])
    if train_test_split_index <= 0:
        raise RuntimeError(f"{arch_name} requires at least one training row")
    x_all = torch.cat([batch.x_train, batch.x_test], dim=1).to(torch.float32)
    y_train = batch.y_train.to(torch.int64)
    y_test = batch.y_test.to(torch.int64)
    return x_all, y_train, y_test, train_test_split_index


def validate_batched_inputs(
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    train_test_split_index: int,
) -> None:
    """Validate canonical [B, R, C] input tensors before the forward path."""

    if x_all.ndim != _BATCHED_TASK_RANK:
        raise ValueError(f"x_all must have shape [B, R, C], got {tuple(x_all.shape)}")
    if y_train.ndim != _UNBATCHED_TASK_RANK:
        raise ValueError(f"y_train must have shape [B, R_train], got {tuple(y_train.shape)}")
    if int(x_all.shape[0]) != int(y_train.shape[0]):
        raise ValueError("x_all and y_train must have matching batch dimensions")
    if train_test_split_index <= 0 or train_test_split_index >= int(x_all.shape[1]):
        raise ValueError(
            "train_test_split_index must satisfy 0 < split < num_rows, got "
            f"split={train_test_split_index}, num_rows={x_all.shape[1]}"
        )
    if int(y_train.shape[1]) != train_test_split_index:
        raise ValueError("y_train length must match train_test_split_index")


def normalize_x_all(
    x_all: torch.Tensor,
    *,
    train_test_split_index: int,
    input_normalization: str,
    preserve_non_finite: bool,
) -> torch.Tensor:
    """Apply train/test normalization using one shared implementation."""

    x_train = x_all[:, :train_test_split_index, :]
    x_test = x_all[:, train_test_split_index:, :]
    train_norm, test_norm = normalize_train_test_tensors(
        x_train,
        x_test,
        mode=cast(InputNormalizationMode, input_normalization),
        preserve_non_finite=preserve_non_finite,
    )
    return torch.cat([train_norm, test_norm], dim=1)
