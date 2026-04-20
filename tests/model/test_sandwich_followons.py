from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.grid_sandwich import GridSandwichClassifier
from tab_foundry.model.architectures.routed_sandwich import RoutedSandwichClassifier
from tab_foundry.types import TaskBatch


_FEATURE_TYPES = ["floating", "integer", "bool"]


def _task_batch(*, num_classes: int = 2) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 1.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [7.0, 8.0, 1.0],
                [9.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata={"source": "unit_test", "feature_types": list(_FEATURE_TYPES)},
        num_classes=num_classes,
    )


def _batched_inputs() -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    batch = _task_batch()
    x_all = torch.cat([batch.x_train, batch.x_test], dim=0).unsqueeze(0)
    y_train = batch.y_train.unsqueeze(0)
    return x_all, y_train, int(batch.x_train.shape[0]), list(_FEATURE_TYPES)


def _routed_model() -> RoutedSandwichClassifier:
    return RoutedSandwichClassifier(
        d_icl=32,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=8,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        routed_row_summary_tokens=2,
        routed_column_summary_tokens=1,
        routed_evidence_tokens=4,
    )


def _grid_model(
    *,
    sandwich_pre_row_attention_layers: int = 1,
    sandwich_pre_column_attention_layers: int = 1,
) -> GridSandwichClassifier:
    return GridSandwichClassifier(
        d_icl=32,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
        sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
        sandwich_pre_column_inducing_tokens=8,
    )


def test_routed_sandwich_forward_and_forward_batched_shapes_match() -> None:
    model = _routed_model()
    batch = _task_batch()
    x_all, y_train, split_index, feature_types = _batched_inputs()

    output = model(batch)
    batched_logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=split_index,
        feature_types=feature_types,
    )

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert output.num_classes == 2
    assert tuple(batched_logits.shape) == (1, 2, 4)
    assert torch.allclose(output.logits, batched_logits.squeeze(0), atol=1e-5, rtol=1e-5)
    assert model.test_query_seed.shape == (1, 1, 2, 32)
    assert model.deepnorm_residual_depth == 11
    assert model.deepnorm_alpha == pytest.approx((2.0 * 11.0) ** 0.25)
    assert model.deepnorm_beta == pytest.approx((8.0 * 11.0) ** (-0.25))


def test_routed_sandwich_requires_feature_types_for_forward_batched() -> None:
    model = _routed_model()
    x_all, y_train, split_index, _feature_types = _batched_inputs()

    with pytest.raises(ValueError, match="feature_types"):
        _ = model.forward_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=split_index,
            feature_types=None,  # type: ignore[arg-type]
        )


def test_routed_sandwich_rejects_non_classification_loss_surface() -> None:
    model = _routed_model()

    with pytest.raises(ValueError, match="classification"):
        model.set_loss_surface("cell_bpc")


def test_grid_sandwich_forward_and_forward_batched_shapes_match() -> None:
    model = _grid_model()
    batch = _task_batch()
    x_all, y_train, split_index, feature_types = _batched_inputs()

    output = model(batch)
    batched_logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=split_index,
        feature_types=feature_types,
    )

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert output.num_classes == 2
    assert tuple(batched_logits.shape) == (1, 2, 4)
    assert torch.allclose(output.logits, batched_logits.squeeze(0), atol=1e-5, rtol=1e-5)
    assert len(model.grid_layers) == 2


def test_grid_sandwich_honors_pre_perceiver_mixer_depths() -> None:
    baseline = _grid_model(
        sandwich_pre_row_attention_layers=0,
        sandwich_pre_column_attention_layers=0,
    )
    mixed = _grid_model(
        sandwich_pre_row_attention_layers=2,
        sandwich_pre_column_attention_layers=1,
    )

    baseline_params = sum(int(parameter.numel()) for parameter in baseline.parameters())
    mixed_params = sum(int(parameter.numel()) for parameter in mixed.parameters())

    assert len(baseline.pre_row_attention_blocks) == 0
    assert len(baseline.pre_column_attention_blocks) == 0
    assert len(mixed.pre_row_attention_blocks) == 2
    assert len(mixed.pre_column_attention_blocks) == 1
    assert mixed_params > baseline_params


def test_grid_sandwich_rejects_non_classification_loss_surface() -> None:
    model = _grid_model()

    with pytest.raises(ValueError, match="classification"):
        model.set_loss_surface("cell_bpc")
