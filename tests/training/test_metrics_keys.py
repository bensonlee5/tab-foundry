from __future__ import annotations

import pytest
import torch

from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.training.trainer import _compute_loss_and_metrics
from tab_foundry.model.factory import build_model
from tab_foundry.types import TaskBatch


def _classification_batch(*, n_test: int = 6, num_classes: int = 32) -> TaskBatch:
    return TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, num_classes, (8,)),
        x_test=torch.randn(n_test, 4),
        y_test=torch.randint(0, num_classes, (n_test,)),
        metadata={},
        num_classes=num_classes,
    )


def test_metrics_reject_non_classification_task() -> None:
    output = ClassificationOutput(logits=torch.randn(6, 4), num_classes=4)
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, 4, (8,)),
        x_test=torch.randn(6, 4),
        y_test=torch.randint(0, 4, (6,)),
        metadata={},
        num_classes=4,
    )

    with pytest.raises(RuntimeError, match="classification"):
        _ = _compute_loss_and_metrics(output, batch, task="regression")


def test_manyclass_path_metrics_do_not_require_acc() -> None:
    output = ClassificationOutput(
        logits=None,
        num_classes=32,
        class_probs=None,
        path_logits=[torch.randn(4, 3), torch.randn(2, 2)],
        path_targets=[torch.randint(0, 3, (4,)), torch.randint(0, 2, (2,))],
        path_sample_counts=[4, 2],
        aux_metrics={"many_class_nodes_visited": 3.0},
    )
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, 32, (8,)),
        x_test=torch.randn(6, 4),
        y_test=torch.randint(0, 32, (6,)),
        metadata={},
        num_classes=32,
    )
    loss, metrics = _compute_loss_and_metrics(output, batch, task="classification")
    assert torch.isfinite(loss)
    assert "acc" not in metrics
    assert metrics["many_class_nodes_visited"] == 3.0


def test_manyclass_path_loss_is_finite_with_sparse_train_labels() -> None:
    model = build_model(task="classification", arch="tabfoundry_staged", stage="many_class")
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.tensor([7, 8, 3, 4, 9, 2, 1, 10], dtype=torch.int64),
        metadata={},
        num_classes=12,
    )
    output = model(batch)
    loss, metrics = _compute_loss_and_metrics(output, batch, task="classification")
    assert torch.isfinite(loss)
    assert metrics["many_class_empty_nodes"] >= 1.0


def test_classification_metrics_raise_for_empty_test_targets() -> None:
    output = ClassificationOutput(
        logits=torch.randn(0, 4),
        num_classes=4,
    )
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, 4, (8,)),
        x_test=torch.randn(0, 4),
        y_test=torch.empty(0, dtype=torch.int64),
        metadata={},
        num_classes=4,
    )
    with pytest.raises(RuntimeError, match="zero test labels"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")


def test_classification_metrics_raise_for_underwidth_logits() -> None:
    output = ClassificationOutput(
        logits=torch.randn(6, 2),
        num_classes=3,
    )
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, 3, (8,)),
        x_test=torch.randn(6, 4),
        y_test=torch.randint(0, 3, (6,)),
        metadata={},
        num_classes=3,
    )

    with pytest.raises(RuntimeError, match="logits width=2"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")


def test_classification_metrics_raise_for_underwidth_class_probs() -> None:
    output = ClassificationOutput(
        logits=None,
        class_probs=torch.full((6, 2), 0.5),
        num_classes=3,
    )
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randint(0, 3, (8,)),
        x_test=torch.randn(6, 4),
        y_test=torch.randint(0, 3, (6,)),
        metadata={},
        num_classes=3,
    )

    with pytest.raises(RuntimeError, match="class_probs width=2"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")


def test_manyclass_path_metrics_raise_for_underfull_path_counts() -> None:
    output = ClassificationOutput(
        logits=None,
        num_classes=32,
        class_probs=None,
        path_logits=[torch.randn(4, 3), torch.randn(1, 2)],
        path_targets=[torch.randint(0, 3, (4,)), torch.randint(0, 2, (1,))],
        path_sample_counts=[4, 1],
    )
    batch = _classification_batch(n_test=6)

    with pytest.raises(RuntimeError, match="path_sample_counts total=5, expected at least 6"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")


def test_manyclass_path_metrics_raise_for_count_logits_row_mismatch() -> None:
    output = ClassificationOutput(
        logits=None,
        num_classes=32,
        class_probs=None,
        path_logits=[torch.randn(4, 3)],
        path_targets=[torch.randint(0, 3, (4,))],
        path_sample_counts=[3],
    )
    batch = _classification_batch(n_test=4)

    with pytest.raises(RuntimeError, match=r"path_sample_counts\[0\]=3, but path_logits\[0\] rows=4"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")


def test_manyclass_path_metrics_raise_for_count_targets_row_mismatch() -> None:
    output = ClassificationOutput(
        logits=None,
        num_classes=32,
        class_probs=None,
        path_logits=[torch.randn(3, 3)],
        path_targets=[torch.randint(0, 3, (4,))],
        path_sample_counts=[3],
    )
    batch = _classification_batch(n_test=3)

    with pytest.raises(RuntimeError, match=r"path_sample_counts\[0\]=3, but path_targets\[0\] rows=4"):
        _ = _compute_loss_and_metrics(output, batch, task="classification")
