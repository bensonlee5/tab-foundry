from __future__ import annotations

import pytest
import torch

from tab_foundry.feature_state import TaskFeatureState
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


def test_move_batch_moves_feature_state() -> None:
    batch = _sample_batch()
    batch = TaskBatch(
        x_train=batch.x_train,
        y_train=batch.y_train,
        x_test=batch.x_test,
        y_test=batch.y_test,
        metadata=batch.metadata,
        num_classes=batch.num_classes,
        feature_state=TaskFeatureState(
            categorical_mask=torch.tensor([False, True, False], dtype=torch.bool),
            categorical_cardinalities=torch.tensor([0, 3, 0], dtype=torch.int64),
            x_train_categorical_ids=torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]],
                dtype=torch.int64,
            ),
            x_test_categorical_ids=torch.tensor([[0, 1, 0], [0, 3, 0]], dtype=torch.int64),
        ),
    )

    out = move_batch(batch, torch.device("cpu"))

    assert out.feature_state is not None
    assert out.feature_state.categorical_mask.device.type == "cpu"
    assert out.feature_state.categorical_cardinalities.device.type == "cpu"
    assert out.feature_state.x_train_categorical_ids.device.type == "cpu"
    assert out.feature_state.x_test_categorical_ids.device.type == "cpu"
