from __future__ import annotations

import pytest
import torch

from tab_foundry.training.batching import collate_task_batch, move_batch
from tab_foundry.types import TaskBatch


def _sample_batch() -> TaskBatch:
    return TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"dataset_id": "d0"},
        num_classes=3,
    )


def test_collate_task_batch_single_item() -> None:
    batch = _sample_batch()
    out = collate_task_batch([batch])
    assert out is batch


def test_collate_task_batch_rejects_non_singleton() -> None:
    batch = _sample_batch()
    with pytest.raises(RuntimeError, match="Only batch_size=1 is supported for task-level batching"):
        _ = collate_task_batch([batch, batch])


def test_move_batch_moves_tensors_and_preserves_metadata() -> None:
    batch = _sample_batch()
    out = move_batch(batch, torch.device("cpu"))
    assert out.x_train.device.type == "cpu"
    assert out.y_train.device.type == "cpu"
    assert out.x_test.device.type == "cpu"
    assert out.y_test.device.type == "cpu"
    assert out.metadata == batch.metadata
    assert out.num_classes == batch.num_classes
