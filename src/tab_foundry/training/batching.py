"""Shared task-batch collation and device movement helpers."""

from __future__ import annotations

import torch

from tab_foundry.types import TaskBatch


def collate_task_batch(items: list[TaskBatch]) -> TaskBatch:
    """Collate one task per batch."""

    if len(items) != 1:
        raise RuntimeError("Only batch_size=1 is supported for task-level batching")
    return items[0]


def move_batch(batch: TaskBatch, device: torch.device) -> TaskBatch:
    """Move tensors in a task batch to device."""

    return TaskBatch(
        x_train=batch.x_train.to(device),
        y_train=batch.y_train.to(device),
        x_test=batch.x_test.to(device),
        y_test=batch.y_test.to(device),
        metadata=batch.metadata,
        num_classes=batch.num_classes,
    )
