from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry import (
    TabFoundryClassifier,
    TabFoundryRegressor,
)
from tab_foundry.model.components.many_class import balanced_bases, encode_mixed_radix
from tab_foundry.types import TaskBatch


def test_classifier_forward_shapes() -> None:
    model = TabFoundryClassifier()
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
    model = TabFoundryClassifier()
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
    model = TabFoundryClassifier()
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
    model = TabFoundryClassifier()
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
    out = model(batch)
    assert out.path_logits is not None
    assert out.path_targets is not None
    for logits in out.path_logits:
        assert torch.isfinite(logits).all()
    assert out.aux_metrics is not None
    assert out.aux_metrics["many_class_empty_nodes"] >= 1.0


def test_classifier_manyclass_without_digit_position_embedding() -> None:
    model = TabFoundryClassifier(use_digit_position_embed=False)
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


def test_classifier_feature_mask_handles_non_finite_inputs() -> None:
    model = TabFoundryClassifier(missingness_mode="feature_mask")
    batch = TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [2.0, 4.0, float("inf")],
                [3.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [float("nan"), 7.0, 8.0],
                [9.0, 10.0, float("inf")],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata={},
        num_classes=2,
    )

    out = model(batch)

    assert out.logits is not None
    assert torch.isfinite(out.logits).all()


def test_regressor_feature_mask_handles_non_finite_inputs() -> None:
    model = TabFoundryRegressor(missingness_mode="feature_mask")
    batch = TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, 3.0],
                [float("inf"), 4.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        x_test=torch.tensor([[float("nan"), 5.0]], dtype=torch.float32),
        y_test=torch.tensor([0.4], dtype=torch.float32),
        metadata={},
        num_classes=None,
    )

    out = model(batch)

    assert torch.isfinite(out.quantiles).all()


def test_tabfoundry_rejects_explicit_token_missingness_mode() -> None:
    with pytest.raises(ValueError, match="explicit_token"):
        _ = TabFoundryClassifier(missingness_mode="explicit_token")


def test_classifier_respects_configured_many_class_base_for_logits_width() -> None:
    model = TabFoundryClassifier(many_class_base=12)
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
    model = TabFoundryClassifier(feature_group_size=32)
    x = torch.randn(5, 70)
    grouped, token_padding_mask = model._group_features(x)
    assert grouped.shape[:2] == (5, 3)
    assert token_padding_mask.shape == (3,)
    assert torch.equal(token_padding_mask, torch.zeros_like(token_padding_mask))


def test_feature_grouping_size_one_matches_feature_count() -> None:
    model = TabFoundryClassifier(feature_group_size=1)
    x = torch.randn(4, 13)
    grouped, token_padding_mask = model._group_features(x)
    assert grouped.shape[:2] == (4, 13)
    assert token_padding_mask.shape == (13,)
    assert torch.equal(token_padding_mask, torch.zeros_like(token_padding_mask))


def test_manyclass_mixed_radix_digit_limit() -> None:
    model = TabFoundryClassifier(max_mixed_radix_digits=1)
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


def test_icl_encode_allows_test_tokens_to_attend_to_themselves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TabFoundryClassifier()
    captured: dict[str, torch.Tensor | int | None] = {"allowed_mask": None, "n_context": None}

    def _capture_forward(
        x: torch.Tensor,
        *,
        allowed_mask: torch.Tensor | None = None,
        n_context: int | None = None,
        force_qass: bool | None = None,
    ) -> torch.Tensor:
        _ = force_qass
        captured["allowed_mask"] = allowed_mask.clone() if allowed_mask is not None else None
        captured["n_context"] = n_context
        return x

    monkeypatch.setattr(model.tficl, "forward", _capture_forward)

    row_embeddings = torch.randn(5, model.d_icl)
    train_target_embed = torch.randn(3, model.d_icl)
    encoded = model._icl_encode(row_embeddings, train_target_embed, n_train=3)

    assert encoded.shape == row_embeddings.shape
    assert captured["n_context"] == 3
    allowed_mask = captured["allowed_mask"]
    assert isinstance(allowed_mask, torch.Tensor)
    assert allowed_mask.shape == (1, 1, 5, 5)
    assert bool(allowed_mask[0, 0, 3, 3])
    assert bool(allowed_mask[0, 0, 4, 4])
    assert not bool(allowed_mask[0, 0, 3, 4])
    assert not bool(allowed_mask[0, 0, 4, 3])
    assert bool(allowed_mask[0, 0, 3, 0])
    assert bool(allowed_mask[0, 0, 4, 2])


def test_manyclass_forward_runs_per_digit_icl_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TabFoundryClassifier()
    model.eval()
    batch = TaskBatch(
        x_train=torch.randn(32, 12),
        y_train=torch.arange(32, dtype=torch.int64),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 32, (8,)),
        metadata={},
        num_classes=32,
    )
    digits = encode_mixed_radix(
        batch.y_train,
        bases=balanced_bases(num_classes=32, max_base=model.many_class_base),
    )
    captured_calls: list[torch.Tensor] = []

    def _capture_icl(
        row_embeddings: torch.Tensor,
        train_target_embed: torch.Tensor,
        *,
        n_train: int,
    ) -> torch.Tensor:
        assert n_train == int(batch.x_train.shape[0])
        captured_calls.append(train_target_embed.detach().clone())
        return row_embeddings

    def _dummy_probs(
        row_embeddings: torch.Tensor,
        y_train: torch.Tensor,
        tree: object,
        *,
        n_train: int,
        num_classes: int,
    ) -> tuple[torch.Tensor, int, int]:
        _ = row_embeddings, y_train, tree, n_train
        n_test = int(batch.x_test.shape[0])
        return torch.full((n_test, num_classes), 1.0 / float(num_classes)), 0, 0

    monkeypatch.setattr(model, "_icl_encode", _capture_icl)
    monkeypatch.setattr(model, "_hierarchical_probs", _dummy_probs)

    out = model(batch)

    assert out.class_probs is not None
    assert len(captured_calls) == int(digits.shape[0])
    assert any(
        not torch.allclose(captured_calls[index], captured_calls[0])
        for index in range(1, len(captured_calls))
    )


def test_regressor_forward_shapes() -> None:
    model = TabFoundryRegressor()
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
