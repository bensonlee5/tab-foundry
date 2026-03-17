"""Tests for dropout threading, pre-encoder clip, and stability ladder smoke checks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tab_foundry.model.architectures.tabfoundry_staged.model import TabFoundryStagedClassifier
from tab_foundry.model.architectures.tabfoundry_staged.subsystems import PreNormCellBlock
from tab_foundry.model.components.blocks import TFRowEncoder
from tab_foundry.model.spec import ModelBuildSpec
from tab_foundry.types import TaskBatch


def _batch(*, num_classes: int = 2) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [0.5, 1.5, 2.5],
                [3.5, 4.5, 5.5],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata={},
        num_classes=num_classes,
    )


def _staged(stage: str = "prenorm_block", **overrides: object) -> TabFoundryStagedClassifier:
    kwargs: dict[str, object] = {
        "stage": stage,
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
    }
    kwargs.update(overrides)
    return TabFoundryStagedClassifier(**kwargs)


# --- Dropout threading tests ---


class TestDropoutThreading:
    def test_prenorm_cell_block_has_dropout_modules(self) -> None:
        block = PreNormCellBlock(
            embedding_size=64,
            nhead=4,
            mlp_hidden_size=128,
            allow_test_self_attention=False,
            norm_type="layernorm",
            dropout=0.1,
        )
        assert isinstance(block.attn_dropout, nn.Dropout)
        assert block.attn_dropout.p == 0.1
        assert isinstance(block.ff_dropout, nn.Dropout)
        assert block.ff_dropout.p == 0.1

    def test_prenorm_cell_block_zero_dropout(self) -> None:
        block = PreNormCellBlock(
            embedding_size=64,
            nhead=4,
            mlp_hidden_size=128,
            allow_test_self_attention=False,
            norm_type="layernorm",
            dropout=0.0,
        )
        assert block.attn_dropout.p == 0.0
        assert block.ff_dropout.p == 0.0

    def test_staged_model_dropout_propagates(self) -> None:
        model = _staged(
            stage="prenorm_block",
            staged_dropout=0.1,
            module_overrides={"table_block_style": "prenorm"},
        )
        # Check that transformer blocks have dropout
        for block in model.transformer_blocks:
            if isinstance(block, PreNormCellBlock):
                assert block.attn_dropout.p == 0.1
                assert block.ff_dropout.p == 0.1

    def test_staged_model_zero_dropout_no_implicit_nonzero(self) -> None:
        """Regression test: TFRowEncoder should not have implicit 0.1 dropout."""
        model = _staged(
            stage="row_cls_pool",
            staged_dropout=0.0,
            module_overrides={
                "table_block_style": "prenorm",
                "row_pool": "row_cls",
            },
        )
        # Walk the model and check that no nn.Dropout has p > 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.0, (
                    f"Module {name} has dropout p={module.p}, expected 0.0"
                )

    def test_tfrow_encoder_explicit_dropout(self) -> None:
        """TFRowEncoder should use explicit dropout param, not implicit 0.1."""
        encoder = TFRowEncoder(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.0, (
                    f"TFRowEncoder module {name} has dropout p={module.p}, expected 0.0"
                )


# --- Pre-encoder clip tests ---


class TestPreEncoderClip:
    def test_clip_bounds_input(self) -> None:
        model = _staged(
            stage="prenorm_block",
            pre_encoder_clip=5.0,
            module_overrides={"table_block_style": "prenorm"},
        )
        batch = _batch()
        # Inject extreme values
        batch = TaskBatch(
            x_train=torch.tensor([[100.0, -200.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            y_train=batch.y_train,
            x_test=torch.tensor([[50.0, -50.0, 2.5], [3.5, 4.5, 5.5]]),
            y_test=batch.y_test,
            metadata={},
            num_classes=2,
        )
        model.eval()
        with torch.no_grad():
            output = model(batch)
        # Output should be finite (no NaN/Inf)
        assert output.logits is not None
        assert torch.isfinite(output.logits).all()

    def test_no_clip_when_none(self) -> None:
        model = _staged(
            stage="prenorm_block",
            pre_encoder_clip=None,
            module_overrides={"table_block_style": "prenorm"},
        )
        assert model.pre_encoder_clip is None

    def test_clip_value_stored(self) -> None:
        model = _staged(
            stage="prenorm_block",
            pre_encoder_clip=10.0,
            module_overrides={"table_block_style": "prenorm"},
        )
        assert model.pre_encoder_clip == 10.0

    def test_spec_validation_rejects_negative_clip(self) -> None:
        with pytest.raises(ValueError, match="pre_encoder_clip must be > 0"):
            ModelBuildSpec(task="classification", arch="tabfoundry_staged", pre_encoder_clip=-1.0)

    def test_spec_validation_rejects_zero_clip(self) -> None:
        with pytest.raises(ValueError, match="pre_encoder_clip must be > 0"):
            ModelBuildSpec(task="classification", arch="tabfoundry_staged", pre_encoder_clip=0.0)


# --- Spec validation tests ---


class TestSpecValidation:
    def test_staged_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="staged_dropout"):
            ModelBuildSpec(task="classification", arch="tabfoundry_staged", staged_dropout=0.6)

    def test_staged_dropout_negative(self) -> None:
        with pytest.raises(ValueError, match="staged_dropout"):
            ModelBuildSpec(task="classification", arch="tabfoundry_staged", staged_dropout=-0.1)

    def test_staged_dropout_valid(self) -> None:
        spec = ModelBuildSpec(task="classification", arch="tabfoundry_staged", staged_dropout=0.1)
        assert spec.staged_dropout == 0.1


# --- Stability ladder smoke tests ---

# Each ladder rung:
# A: prenorm_baseline
# B: A + warmup_ratio:0.05, grad_clip:1.0
# C: B + staged_dropout:0.1
# D: C + pre_encoder_clip:10.0

_LADDER_CONFIGS: list[dict[str, object]] = [
    {
        "stage": "prenorm_block",
        "module_overrides": {"table_block_style": "prenorm"},
    },
    {
        "stage": "prenorm_block",
        "module_overrides": {"table_block_style": "prenorm"},
    },
    {
        "stage": "prenorm_block",
        "module_overrides": {"table_block_style": "prenorm"},
        "staged_dropout": 0.1,
    },
    {
        "stage": "prenorm_block",
        "module_overrides": {"table_block_style": "prenorm"},
        "staged_dropout": 0.1,
        "pre_encoder_clip": 10.0,
    },
]


class TestStabilityLadderSmoke:
    @pytest.mark.parametrize("rung_idx", range(len(_LADDER_CONFIGS)))
    def test_forward_produces_finite(self, rung_idx: int) -> None:
        config = dict(_LADDER_CONFIGS[rung_idx])
        model = _staged(**config)
        model.eval()
        batch = _batch()
        with torch.no_grad():
            output = model(batch)
        assert output.logits is not None
        assert torch.isfinite(output.logits).all(), f"Rung {rung_idx} produced non-finite logits"

    @pytest.mark.parametrize("rung_idx", range(len(_LADDER_CONFIGS)))
    def test_backward_produces_finite_gradients(self, rung_idx: int) -> None:
        config = dict(_LADDER_CONFIGS[rung_idx])
        model = _staged(**config)
        model.train()
        batch = _batch()
        output = model(batch)
        assert output.logits is not None
        loss = output.logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Rung {rung_idx}: non-finite gradient in {name}"
                )
