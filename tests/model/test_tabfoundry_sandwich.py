from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from tab_foundry.types import TaskBatch


def _batch(*, num_classes: int = 3) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, float("nan"), 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 2], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata={"source": "unit_test"},
        num_classes=num_classes,
    )


def _batched_inputs() -> tuple[torch.Tensor, torch.Tensor, int]:
    x_all = torch.tensor(
        [
            [
                [1.0, 2.0, float("nan"), 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0, -1.0],
                [0.5, 0.5, 0.5, 0.5],
                [3.0, -1.0, 2.0, 4.0],
            ],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=torch.int64,
    )
    return x_all, y_train, 3


def test_tabfoundry_sandwich_forward_shapes() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_row_latents=8,
        sandwich_col_latents=4,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )

    output = model(_batch())

    assert output.logits is not None
    assert output.class_probs is None
    assert output.num_classes == 3
    assert tuple(output.logits.shape) == (2, 4)


def test_tabfoundry_sandwich_forward_batched_shapes() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_row_latents=8,
        sandwich_col_latents=4,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )
    x_all, y_train, train_test_split_index = _batched_inputs()

    logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=train_test_split_index,
    )

    assert tuple(logits.shape) == (2, 2, 4)
    assert torch.isfinite(logits).all()


def test_tabfoundry_sandwich_exposes_activation_trace_hooks() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_row_latents=8,
        sandwich_col_latents=4,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )
    model.enable_activation_trace()

    _ = model(_batch())
    trace = model.flush_activation_trace_stats()

    assert trace is not None
    assert "post_feature_encoder" in trace
    assert "post_test_readout" in trace


def test_tabfoundry_sandwich_rejects_true_many_class_batches() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=3,
        head_hidden_dim=64,
        sandwich_row_latents=8,
        sandwich_col_latents=4,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )

    with pytest.raises(RuntimeError, match="small-class only"):
        _ = model(_batch(num_classes=5))
