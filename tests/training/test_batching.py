from __future__ import annotations

import pytest
import torch

from tab_foundry.data.factory import _ManifestTaskBatchSampler
from tab_foundry.training.batching import (
    collate_task_batch,
    move_batch,
    task_batch_diagnostics,
)
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


class _StubManifestTaskDataset:
    def __len__(self) -> int:
        return 15

    def task_signature(self, index: int) -> tuple[int, int, int, int | None]:
        del index
        return (6, 3, 4, 3)


def test_collate_task_batch_single_item() -> None:
    batch = _sample_batch()
    out = collate_task_batch([batch])
    assert out is batch


def test_collate_task_batch_rejects_non_singleton() -> None:
    batch = _sample_batch()
    with pytest.raises(RuntimeError, match="Only batch_size=1 is supported for task-level batching"):
        _ = collate_task_batch([batch, batch])


def test_collate_task_batch_stacks_exact_shape_tasks() -> None:
    batch_a = _sample_batch()
    batch_b = _sample_batch()

    out = collate_task_batch([batch_a, batch_b], requested_task_batch_size=4)

    assert tuple(out.x_train.shape) == (2, 4, 3)
    assert tuple(out.y_train.shape) == (2, 4)
    assert tuple(out.x_test.shape) == (2, 2, 3)
    assert tuple(out.y_test.shape) == (2, 2)
    assert out.metadata == {
        "task_members": [batch_a.metadata, batch_b.metadata],
        "task_batch_size_requested": 4,
        "task_batch_size_actual": 2,
        "task_batch_signature": "4x2x3x3",
        "task_batch_mode": "batched",
    }
    assert task_batch_diagnostics(out) == {
        "task_batch_size_requested": 4,
        "task_batch_size_actual": 2,
        "task_batch_mode": "batched",
        "task_batch_signature": "4x2x3x3",
        "task_members": [batch_a.metadata, batch_b.metadata],
    }


def test_collate_task_batch_emits_singleton_fallback_metadata() -> None:
    batch = _sample_batch()

    out = collate_task_batch([batch], requested_task_batch_size=8)

    assert out.x_train is batch.x_train
    assert out.metadata == {
        "task_members": [batch.metadata],
        "task_batch_size_requested": 8,
        "task_batch_size_actual": 1,
        "task_batch_signature": "4x2x3x3",
        "task_batch_mode": "singleton_fallback",
    }
    assert task_batch_diagnostics(out)["task_batch_mode"] == "singleton_fallback"


def test_collate_task_batch_rejects_shape_mismatch() -> None:
    batch_a = _sample_batch()
    batch_b = TaskBatch(
        x_train=torch.randn(5, 3),
        y_train=torch.randint(0, 3, (5,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"dataset_id": "d1"},
        num_classes=3,
    )

    with pytest.raises(RuntimeError, match="Only shape-compatible tasks can be tensor-batched"):
        _ = collate_task_batch([batch_a, batch_b], requested_task_batch_size=2)


def test_manifest_task_batch_sampler_batches_multi_item_remainder_together() -> None:
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(),
        task_batch_size=8,
        shuffle=False,
        seed=0,
    )

    assert list(sampler) == [list(range(8)), list(range(8, 15))]
    assert len(sampler) == 2


def test_move_batch_moves_tensors_and_preserves_metadata() -> None:
    batch = _sample_batch()
    out = move_batch(batch, torch.device("cpu"))
    assert out.x_train.device.type == "cpu"
    assert out.y_train.device.type == "cpu"
    assert out.x_test.device.type == "cpu"
    assert out.y_test.device.type == "cpu"
    assert out.metadata == batch.metadata
    assert out.num_classes == batch.num_classes
