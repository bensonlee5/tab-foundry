from __future__ import annotations

import pytest
import torch

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.training.trainer import _compute_loss_and_metrics
from tab_foundry.types import TaskBatch


def _build_model(*, task: str = "classification", **model_overrides: object) -> torch.nn.Module:
    spec = model_build_spec_from_mappings(task=task, primary=model_overrides)
    return build_model_from_spec(spec)


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
    model = _build_model(task="classification", arch="tabfoundry_staged", stage="many_class")
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


def test_classification_metrics_flatten_batched_targets() -> None:
    output = ClassificationOutput(
        logits=torch.tensor(
            [
                [[6.0, -1.0, -2.0], [-2.0, 5.0, -1.0]],
                [[-3.0, -2.0, 4.0], [5.0, -1.0, -2.0]],
            ],
            dtype=torch.float32,
        ).reshape(4, 3),
        num_classes=3,
    )
    batch = TaskBatch(
        x_train=torch.randn(2, 8, 4),
        y_train=torch.randint(0, 3, (2, 8)),
        x_test=torch.randn(2, 2, 4),
        y_test=torch.tensor([[0, 1], [2, 0]], dtype=torch.int64),
        metadata={
            "task_batch_size_requested": 2,
            "task_batch_size_actual": 2,
            "task_batch_mode": "batched",
        },
        num_classes=3,
    )

    loss, metrics = _compute_loss_and_metrics(output, batch, task="classification")

    assert torch.isfinite(loss)
    assert metrics["acc"] == pytest.approx(1.0)


def test_classification_metrics_add_z_loss_only_when_requested() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    batch = TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata={},
        num_classes=3,
    )
    output = ClassificationOutput(logits=logits, num_classes=3)

    baseline_loss, baseline_metrics = _compute_loss_and_metrics(
        output,
        batch,
        task="classification",
    )
    z_loss, z_metrics = _compute_loss_and_metrics(
        output,
        batch,
        task="classification",
        classification_z_loss_coeff=1.0e-4,
    )

    assert "classification_z_loss" not in baseline_metrics
    assert z_loss.item() > baseline_loss.item()
    assert z_metrics["classification_ce_loss"] == pytest.approx(baseline_loss.item())
    assert z_metrics["classification_z_loss"] > 0.0
    assert z_metrics["classification_z_loss_coeff"] == pytest.approx(1.0e-4)


def test_classification_metrics_add_moe_aux_losses_only_when_requested() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    batch = TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata={},
        num_classes=3,
    )
    output = ClassificationOutput(
        logits=logits,
        num_classes=3,
        aux_losses={
            "moe_load_balance_loss": torch.tensor(2.0),
            "moe_router_z_loss": torch.tensor(3.0),
        },
        aux_metrics={"moe_router_entropy": 1.25},
    )

    baseline_loss, baseline_metrics = _compute_loss_and_metrics(
        output,
        batch,
        task="classification",
    )
    moe_loss, moe_metrics = _compute_loss_and_metrics(
        output,
        batch,
        task="classification",
        moe_load_balance_loss_coeff=0.1,
        moe_router_z_loss_coeff=0.01,
    )

    assert "moe_load_balance_loss" not in baseline_metrics
    assert "moe_router_z_loss" not in baseline_metrics
    assert baseline_metrics["moe_router_entropy"] == pytest.approx(1.25)
    assert moe_loss.item() == pytest.approx(baseline_loss.item() + 0.23)
    assert moe_metrics["moe_load_balance_loss"] == pytest.approx(2.0)
    assert moe_metrics["moe_load_balance_loss_coeff"] == pytest.approx(0.1)
    assert moe_metrics["moe_router_z_loss"] == pytest.approx(3.0)
    assert moe_metrics["moe_router_z_loss_coeff"] == pytest.approx(0.01)


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
