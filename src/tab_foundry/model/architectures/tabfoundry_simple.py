"""Exact nanoTabPFN-style binary debug architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import Linear, MultiheadAttention

from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES, build_norm
from tab_foundry.types import TaskBatch

from .tabfoundry import ClassificationOutput, DEFAULT_HEAD_HIDDEN_DIM


_REQUIRED_INPUT_NORMALIZATION = "train_zscore_clip"
_REQUIRED_MANY_CLASS_BASE = 2


class _FeatureEncoder(nn.Module):
    """Exact nanoTabPFN feature encoder."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdim=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdim=True) + 1.0e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100.0, max=100.0)
        return self.linear_layer(x)


class _TargetEncoder(nn.Module):
    """Exact nanoTabPFN target encoder."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        if y_train.ndim == 2:
            y_train = y_train.unsqueeze(-1)
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class _TransformerEncoderLayer(nn.Module):
    """Exact nanoTabPFN transformer block."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        *,
        layer_norm_eps: float = 1.0e-5,
        batch_first: bool = True,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
        )
        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)
        self.norm1 = build_norm(norm_type, embedding_size, eps=layer_norm_eps)
        self.norm2 = build_norm(norm_type, embedding_size, eps=layer_norm_eps)
        self.norm3 = build_norm(norm_type, embedding_size, eps=layer_norm_eps)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)

        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
        )[0]
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)

        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class _Decoder(nn.Module):
    """Exact nanoTabPFN decoder."""

    def __init__(
        self, embedding_size: int, mlp_hidden_size: int, num_outputs: int
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class TabFoundrySimpleClassifier(nn.Module):
    """Exact nanoTabPFN-style binary classifier for benchmark/debug runs."""

    def __init__(
        self,
        *,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = _REQUIRED_INPUT_NORMALIZATION,
        feature_group_size: int = 1,
        many_class_train_mode: str = "path_nll",
        max_mixed_radix_digits: int = 64,
        norm_type: str = "layernorm",
        tfcol_n_heads: int = 8,
        tfcol_n_layers: int = 3,
        tfcol_n_inducing: int = 128,
        tfrow_n_heads: int = 8,
        tfrow_n_layers: int = 3,
        tfrow_cls_tokens: int = 4,
        tfrow_norm: str = "layernorm",
        tficl_n_heads: int = 8,
        tficl_n_layers: int = 12,
        tficl_ff_expansion: int = 2,
        many_class_base: int = _REQUIRED_MANY_CLASS_BASE,
        head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        use_digit_position_embed: bool = True,
    ) -> None:
        super().__init__()
        self._require_default("d_col", int(d_col), 128)
        self._require_default("feature_group_size", int(feature_group_size), 1)
        self._require_default(
            "many_class_train_mode", str(many_class_train_mode), "path_nll"
        )
        self._require_default("max_mixed_radix_digits", int(max_mixed_radix_digits), 64)
        self._require_default("tfcol_n_heads", int(tfcol_n_heads), 8)
        self._require_default("tfcol_n_layers", int(tfcol_n_layers), 3)
        self._require_default("tfcol_n_inducing", int(tfcol_n_inducing), 128)
        self._require_default("tfrow_n_heads", int(tfrow_n_heads), 8)
        self._require_default("tfrow_n_layers", int(tfrow_n_layers), 3)
        self._require_default("tfrow_cls_tokens", int(tfrow_cls_tokens), 4)
        self._require_default(
            "tfrow_norm", str(tfrow_norm).strip().lower(), "layernorm"
        )
        self._require_default(
            "use_digit_position_embed", bool(use_digit_position_embed), True
        )

        self.d_icl = int(d_icl)
        if self.d_icl <= 0:
            raise ValueError(f"d_icl must be positive, got {self.d_icl}")
        self.input_normalization = str(input_normalization).strip().lower()
        if self.input_normalization != _REQUIRED_INPUT_NORMALIZATION:
            raise ValueError(
                "tabfoundry_simple requires "
                f"input_normalization={_REQUIRED_INPUT_NORMALIZATION!r}, got "
                f"{self.input_normalization!r}"
            )
        self.tficl_n_heads = int(tficl_n_heads)
        self.tficl_n_layers = int(tficl_n_layers)
        self.tficl_ff_expansion = int(tficl_ff_expansion)
        self.many_class_base = int(many_class_base)
        self.head_hidden_dim = int(head_hidden_dim)
        self.norm_type = str(norm_type).strip().lower()
        if self.norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {SUPPORTED_NORM_TYPES}, got {self.norm_type!r}"
            )
        for name, value in (
            ("tficl_n_heads", self.tficl_n_heads),
            ("tficl_n_layers", self.tficl_n_layers),
            ("tficl_ff_expansion", self.tficl_ff_expansion),
            ("head_hidden_dim", self.head_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.many_class_base != _REQUIRED_MANY_CLASS_BASE:
            raise ValueError(
                "tabfoundry_simple requires many_class_base=2 for exact binary parity, "
                f"got {self.many_class_base}"
            )

        self.feature_encoder = _FeatureEncoder(self.d_icl)
        self.target_encoder = _TargetEncoder(self.d_icl)
        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    self.d_icl,
                    self.tficl_n_heads,
                    self.head_hidden_dim,
                    norm_type=self.norm_type,
                )
                for _ in range(self.tficl_n_layers)
            ]
        )
        self.decoder = _Decoder(
            self.d_icl,
            self.head_hidden_dim,
            _REQUIRED_MANY_CLASS_BASE,
        )

    @staticmethod
    def _require_default(name: str, value: object, expected: object) -> None:
        if value != expected:
            raise ValueError(
                f"tabfoundry_simple only supports the default {name}={expected!r}, got {value!r}"
            )

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        if batch.num_classes is not None:
            return int(batch.num_classes)
        if batch.y_train.numel() == 0:
            raise RuntimeError("tabfoundry_simple requires at least one training label")
        return int(batch.y_train.max().item()) + 1

    @staticmethod
    def _prepare_task_inputs(
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        train_test_split_index = int(batch.x_train.shape[0])
        if train_test_split_index <= 0:
            raise RuntimeError("tabfoundry_simple requires at least one training row")
        x_all = (
            torch.cat([batch.x_train, batch.x_test], dim=0)
            .to(torch.float32)
            .unsqueeze(0)
        )
        y_train = batch.y_train.to(torch.float32).unsqueeze(0)
        return x_all, y_train, train_test_split_index

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        if x_all.ndim != 3:
            raise ValueError(
                f"x_all must have shape [B, R, C], got {tuple(x_all.shape)}"
            )
        if y_train.ndim not in {2, 3}:
            raise ValueError(
                f"y_train must have shape [B, R_train] or [B, R_train, 1], got {tuple(y_train.shape)}"
            )
        if int(x_all.shape[0]) != int(y_train.shape[0]):
            raise ValueError(
                "x_all and y_train must have matching batch dimensions, got "
                f"{x_all.shape[0]} and {y_train.shape[0]}"
            )
        if train_test_split_index <= 0 or train_test_split_index >= int(x_all.shape[1]):
            raise ValueError(
                "train_test_split_index must satisfy 0 < split < num_rows, got "
                f"split={train_test_split_index}, num_rows={x_all.shape[1]}"
            )
        if int(y_train.shape[1]) != train_test_split_index:
            raise ValueError(
                "y_train length must match train_test_split_index, got "
                f"len={y_train.shape[1]}, split={train_test_split_index}"
            )

    def _build_table_tokens_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        x_src = self.feature_encoder(x_all.to(torch.float32), train_test_split_index)
        y_src = self.target_encoder(y_train.to(torch.float32), int(x_src.shape[1]))
        return torch.cat([x_src, y_src], dim=2)

    def _encode_table_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        src = self._build_table_tokens_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        return src

    def _build_table_tokens(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        x_all, y_train, train_test_split_index = self._prepare_task_inputs(batch)
        table = self._build_table_tokens_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        return table.squeeze(0), train_test_split_index

    def _encode_table(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        x_all, y_train, train_test_split_index = self._prepare_task_inputs(batch)
        encoded = self._encode_table_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        return encoded.squeeze(0), train_test_split_index

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        encoded = self._encode_table_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        return self.decoder(encoded[:, train_test_split_index:, -1, :])

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        if num_classes != _REQUIRED_MANY_CLASS_BASE:
            raise RuntimeError(
                "tabfoundry_simple is binary-only and requires num_classes=2, "
                f"got {num_classes}"
            )
        x_all, y_train, train_test_split_index = self._prepare_task_inputs(batch)
        logits = self.forward_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
        ).squeeze(0)
        return ClassificationOutput(
            logits=logits,
            num_classes=_REQUIRED_MANY_CLASS_BASE,
            class_probs=None,
        )
