from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry import (
    ClassificationOutput,
    RegressionOutput,
    TabFoundryClassifier,
)
from tab_foundry.training.trainer import _compute_loss_and_metrics
from tab_foundry.types import TaskBatch


def test_regression_metrics_do_not_include_acc() -> None:
    output = RegressionOutput(quantiles=torch.randn(6, 999))
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randn(8),
        x_test=torch.randn(6, 4),
        y_test=torch.randn(6),
        metadata={},
        num_classes=None,
    )

    loss, metrics = _compute_loss_and_metrics(output, batch, task="regression")
    assert torch.isfinite(loss)
    assert "rmse" in metrics
    assert "acc" not in metrics


def test_regression_uses_output_quantile_levels() -> None:
    levels = torch.full((999,), 0.9, dtype=torch.float32)
    output = RegressionOutput(
        quantiles=torch.zeros(6, 999),
        quantile_levels=levels,
    )
    batch = TaskBatch(
        x_train=torch.randn(8, 4),
        y_train=torch.randn(8),
        x_test=torch.randn(6, 4),
        y_test=torch.ones(6),
        metadata={},
        num_classes=None,
    )

    loss, _ = _compute_loss_and_metrics(output, batch, task="regression")
    assert torch.isclose(loss, torch.tensor(0.9), atol=1e-4)


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
    model = TabFoundryClassifier()
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
