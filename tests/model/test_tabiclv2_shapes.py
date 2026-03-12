from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabiclv2 import TabICLv2Classifier, TabICLv2Regressor
from tab_foundry.types import TaskBatch


def test_classifier_forward_shapes() -> None:
    model = TabICLv2Classifier()
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 5, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 5, (8,)),
        metadata={},
        num_classes=5,
    )
    out = model(batch)
    assert out.logits is not None
    assert out.logits.shape == (8, 10)


def test_classifier_manyclass_forward_shapes() -> None:
    model = TabICLv2Classifier()
    model.eval()
    # num_classes=32 should trigger _forward_many_class
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 32, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 32, (8,)),
        metadata={},
        num_classes=32,
    )
    out = model(batch)
    assert out.logits is None
    assert out.class_probs is not None
    assert out.class_probs.shape == (8, 32)
    # Check that probabilities sum to 1 (within reasonable precision)
    assert torch.allclose(out.class_probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_classifier_manyclass_train_path_outputs() -> None:
    model = TabICLv2Classifier()
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 32, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 32, (8,)),
        metadata={},
        num_classes=32,
    )
    out = model(batch)
    assert out.class_probs is None
    assert out.path_logits is not None
    assert out.path_targets is not None
    assert len(out.path_logits) == len(out.path_targets)
    assert out.aux_metrics is not None
    assert "many_class_nodes_visited" in out.aux_metrics
    assert "many_class_avg_path_depth" in out.aux_metrics
    assert "many_class_empty_nodes" in out.aux_metrics


def test_classifier_manyclass_eval_handles_sparse_train_labels() -> None:
    model = TabICLv2Classifier()
    model.eval()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 12, (8,)),
        metadata={},
        num_classes=12,
    )
    out = model(batch)
    assert out.class_probs is not None
    assert torch.isfinite(out.class_probs).all()
    assert torch.allclose(out.class_probs.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert out.aux_metrics is not None
    assert "many_class_empty_nodes" in out.aux_metrics


def test_classifier_manyclass_train_path_handles_sparse_train_labels() -> None:
    model = TabICLv2Classifier()
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.tensor([7, 8, 3, 4, 9, 2, 1, 10], dtype=torch.int64),
        metadata={},
        num_classes=12,
    )
    out = model(batch)
    assert out.path_logits is not None
    assert out.path_targets is not None
    for logits in out.path_logits:
        assert torch.isfinite(logits).all()
    assert out.aux_metrics is not None
    assert out.aux_metrics["many_class_empty_nodes"] >= 1.0


def test_classifier_manyclass_without_digit_position_embedding() -> None:
    model = TabICLv2Classifier(use_digit_position_embed=False)
    model.eval()
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 24, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 24, (8,)),
        metadata={},
        num_classes=24,
    )
    out = model(batch)
    assert model.digit_position_embed is None
    assert out.class_probs is not None
    assert out.class_probs.shape == (8, 24)


def test_classifier_respects_configured_many_class_base_for_logits_width() -> None:
    model = TabICLv2Classifier(many_class_base=12)
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 5, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 5, (8,)),
        metadata={},
        num_classes=5,
    )
    out = model(batch)
    assert out.logits is not None
    assert out.logits.shape == (8, 12)


def test_feature_grouping_reduces_token_count() -> None:
    model = TabICLv2Classifier(feature_group_size=32)
    x = torch.randn(5, 70)
    grouped, token_padding_mask = model._group_features(x)
    assert grouped.shape[:2] == (5, 3)
    assert token_padding_mask.shape == (3,)
    assert torch.equal(token_padding_mask, torch.zeros_like(token_padding_mask))


def test_feature_grouping_size_one_matches_feature_count() -> None:
    model = TabICLv2Classifier(feature_group_size=1)
    x = torch.randn(4, 13)
    grouped, token_padding_mask = model._group_features(x)
    assert grouped.shape[:2] == (4, 13)
    assert token_padding_mask.shape == (13,)
    assert torch.equal(token_padding_mask, torch.zeros_like(token_padding_mask))


def test_manyclass_mixed_radix_digit_limit() -> None:
    model = TabICLv2Classifier(max_mixed_radix_digits=1)
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randint(0, 100, (32,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 100, (8,)),
        metadata={},
        num_classes=100,
    )
    with pytest.raises(RuntimeError, match="mixed-radix depth exceeds"):
        _ = model(batch)


def test_regressor_forward_shapes() -> None:
    model = TabICLv2Regressor()
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.randn(32),
        x_test=torch.randn(8, 12),
        y_test=torch.randn(8),
        metadata={},
        num_classes=None,
    )
    out = model(batch)
    assert out.quantiles.shape == (8, 999)
    assert out.quantile_levels is not None
    assert out.quantile_levels.shape == (999,)
