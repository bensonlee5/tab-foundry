"""Shared task-batching helpers used across data and training layers."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from tab_foundry.types import TaskBatch


def resolve_task_batch_size(training_cfg: Any | None) -> int:
    """Resolve the configured manifest task-batch size."""

    if training_cfg is None:
        return 1
    raw_value: Any
    if isinstance(training_cfg, Mapping):
        raw_value = training_cfg.get("task_batch_size", 1)
    else:
        raw_value = getattr(training_cfg, "task_batch_size", 1)
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"training.task_batch_size must be >= 1, got {value}")
    return value


def task_batch_signature(batch: TaskBatch) -> tuple[int, int, int, int | None]:
    """Return the exact-shape signature used for task batching."""

    if batch.x_train.ndim not in {2, 3}:
        raise RuntimeError(
            "TaskBatch.x_train must be rank 2 or 3, "
            f"got shape={tuple(int(dim) for dim in batch.x_train.shape)}"
        )
    if batch.x_test.ndim != batch.x_train.ndim:
        raise RuntimeError(
            "TaskBatch train/test rank mismatch: "
            f"x_train={tuple(int(dim) for dim in batch.x_train.shape)}, "
            f"x_test={tuple(int(dim) for dim in batch.x_test.shape)}"
        )
    if batch.x_train.ndim == 2:
        n_train = int(batch.x_train.shape[0])
        n_test = int(batch.x_test.shape[0])
        n_features = int(batch.x_train.shape[1])
    else:
        n_train = int(batch.x_train.shape[1])
        n_test = int(batch.x_test.shape[1])
        n_features = int(batch.x_train.shape[2])
    return (
        n_train,
        n_test,
        n_features,
        None if batch.num_classes is None else int(batch.num_classes),
    )


def task_batch_signature_text(signature: tuple[int, int, int, int | None]) -> str:
    """Render one task signature as a stable compact string."""

    n_train, n_test, n_features, num_classes = signature
    class_text = "na" if num_classes is None else str(int(num_classes))
    return f"{int(n_train)}x{int(n_test)}x{int(n_features)}x{class_text}"


def task_batch_diagnostics(batch: TaskBatch) -> dict[str, Any]:
    """Return normalized task-batching diagnostics for one batch."""

    signature_text = task_batch_signature_text(task_batch_signature(batch))
    metadata = batch.metadata
    requested = metadata.get("task_batch_size_requested")
    actual = metadata.get("task_batch_size_actual")
    if requested is None:
        requested = 1
    if actual is None:
        actual = int(batch.x_train.shape[0]) if batch.x_train.ndim == 3 else 1
    requested_value = int(requested)
    actual_value = int(actual)
    mode = metadata.get("task_batch_mode")
    if not isinstance(mode, str) or not mode.strip():
        if requested_value > 1 and actual_value == 1:
            mode = "singleton_fallback"
        elif actual_value > 1:
            mode = "batched"
        else:
            mode = "singleton"
    task_members = metadata.get("task_members")
    if not isinstance(task_members, list):
        task_members = [dict(metadata)]
    return {
        "task_batch_size_requested": requested_value,
        "task_batch_size_actual": actual_value,
        "task_batch_mode": str(mode),
        "task_batch_signature": str(metadata.get("task_batch_signature", signature_text)),
        "task_members": task_members,
    }


def _stack_task_tensors(items: list[TaskBatch]) -> TaskBatch:
    signatures = [task_batch_signature(item) for item in items]
    if len({signature for signature in signatures}) != 1:
        raise RuntimeError(
            "Only shape-compatible tasks can be tensor-batched, "
            f"got signatures={[task_batch_signature_text(signature) for signature in signatures]}"
        )
    num_classes = {item.num_classes for item in items}
    if len(num_classes) != 1:
        raise RuntimeError(
            "Only matching num_classes can be tensor-batched, "
            f"got {[None if value is None else int(value) for value in num_classes]}"
        )
    signature_text = task_batch_signature_text(signatures[0])
    return TaskBatch(
        x_train=torch.stack([item.x_train for item in items], dim=0),
        y_train=torch.stack([item.y_train for item in items], dim=0),
        x_test=torch.stack([item.x_test for item in items], dim=0),
        y_test=torch.stack([item.y_test for item in items], dim=0),
        metadata={
            "task_members": [dict(item.metadata) for item in items],
            "task_batch_size_requested": len(items),
            "task_batch_size_actual": len(items),
            "task_batch_signature": signature_text,
            "task_batch_mode": "batched",
        },
        num_classes=items[0].num_classes,
    )


def collate_task_batch(
    items: list[TaskBatch],
    *,
    requested_task_batch_size: int = 1,
) -> TaskBatch:
    """Collate one or more shape-compatible tasks."""

    if requested_task_batch_size <= 0:
        raise ValueError(
            f"requested_task_batch_size must be >= 1, got {requested_task_batch_size}"
        )
    if not items:
        raise RuntimeError("task batch collation requires at least one item")
    if requested_task_batch_size == 1:
        if len(items) != 1:
            raise RuntimeError("Only batch_size=1 is supported for task-level batching")
        return items[0]
    if len(items) == 1:
        item = items[0]
        signature_text = task_batch_signature_text(task_batch_signature(item))
        return TaskBatch(
            x_train=item.x_train,
            y_train=item.y_train,
            x_test=item.x_test,
            y_test=item.y_test,
            metadata={
                "task_members": [dict(item.metadata)],
                "task_batch_size_requested": int(requested_task_batch_size),
                "task_batch_size_actual": 1,
                "task_batch_signature": signature_text,
                "task_batch_mode": "singleton_fallback",
            },
            num_classes=item.num_classes,
        )
    if len(items) > int(requested_task_batch_size):
        raise RuntimeError(
            "collate_task_batch received more items than requested_task_batch_size: "
            f"len(items)={len(items)}, requested_task_batch_size={requested_task_batch_size}"
        )
    batch = _stack_task_tensors(items)
    batch.metadata["task_batch_size_requested"] = int(requested_task_batch_size)
    batch.metadata["task_batch_size_actual"] = len(items)
    return batch


def move_batch(
    batch: TaskBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> TaskBatch:
    """Move tensors in a task batch to device."""

    return TaskBatch(
        x_train=batch.x_train.to(device, non_blocking=non_blocking),
        y_train=batch.y_train.to(device, non_blocking=non_blocking),
        x_test=batch.x_test.to(device, non_blocking=non_blocking),
        y_test=batch.y_test.to(device, non_blocking=non_blocking),
        metadata=batch.metadata,
        num_classes=batch.num_classes,
    )
