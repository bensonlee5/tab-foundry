"""Latent-bank row/column sandwich classifier."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from tab_foundry.input_normalization import InputNormalizationMode, normalize_train_test_tensors
from tab_foundry.model.components.attention import multihead_attention_sdpa
from tab_foundry.model.components.non_finite import clip_finite_values
from tab_foundry.model.components.normalization import build_norm
from tab_foundry.model.components.tabular_primitives import (
    DirectClassifierHead,
    LabelTokenTargetConditioner,
    ScalarPerFeatureMissingnessTokenizer,
    SharedLinearFeatureEncoder,
)
from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows
from tab_foundry.model.spec import (
    DEFAULT_MODEL_D_ICL,
    DEFAULT_MODEL_HEAD_HIDDEN_DIM,
    DEFAULT_MODEL_INPUT_NORMALIZATION,
    DEFAULT_MODEL_MANY_CLASS_BASE,
    DEFAULT_MODEL_NORM_TYPE,
    DEFAULT_MODEL_PRE_ENCODER_CLIP,
    DEFAULT_MODEL_SANDWICH_COL_LATENTS,
    DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    DEFAULT_MODEL_SANDWICH_HEADS,
    DEFAULT_MODEL_SANDWICH_LAYERS,
    DEFAULT_MODEL_SANDWICH_ROW_LATENTS,
    ModelBuildSpec,
)
from tab_foundry.types import TaskBatch

_UNBATCHED_TASK_RANK = 2
_BATCHED_TASK_RANK = 3
_MIN_CLASS_COUNT = 2


class _CrossAttentionBlock(nn.Module):
    """Pre-norm residual cross-attention plus FFN."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.query_norm = build_norm(norm_type, embedding_size)
        self.kv_norm = build_norm(norm_type, embedding_size)
        self.ff_norm = build_norm(norm_type, embedding_size)
        self.attn = nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)
        ff_hidden = embedding_size * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embedding_size),
        )

    def forward(self, query: torch.Tensor, *, key_value: torch.Tensor) -> torch.Tensor:
        q_norm = self.query_norm(query)
        kv_norm = self.kv_norm(key_value)
        query = query + multihead_attention_sdpa(
            self.attn,
            q_norm,
            kv_norm,
            kv_norm,
        )
        return query + self.ff(self.ff_norm(query))


class TabFoundrySandwichClassifier(nn.Module):
    """Small-class latent-bank sandwich classifier."""

    def __init__(
        self,
        *,
        d_icl: int = DEFAULT_MODEL_D_ICL,
        input_normalization: str = DEFAULT_MODEL_INPUT_NORMALIZATION,
        many_class_base: int = DEFAULT_MODEL_MANY_CLASS_BASE,
        norm_type: str = DEFAULT_MODEL_NORM_TYPE,
        head_hidden_dim: int = DEFAULT_MODEL_HEAD_HIDDEN_DIM,
        pre_encoder_clip: float | None = DEFAULT_MODEL_PRE_ENCODER_CLIP,
        sandwich_row_latents: int = DEFAULT_MODEL_SANDWICH_ROW_LATENTS,
        sandwich_col_latents: int = DEFAULT_MODEL_SANDWICH_COL_LATENTS,
        sandwich_layers: int = DEFAULT_MODEL_SANDWICH_LAYERS,
        sandwich_heads: int = DEFAULT_MODEL_SANDWICH_HEADS,
        sandwich_ff_expansion: int = DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_sandwich",
            d_icl=d_icl,
            input_normalization=input_normalization,
            many_class_base=many_class_base,
            norm_type=norm_type,
            head_hidden_dim=head_hidden_dim,
            pre_encoder_clip=pre_encoder_clip,
            sandwich_row_latents=sandwich_row_latents,
            sandwich_col_latents=sandwich_col_latents,
            sandwich_layers=sandwich_layers,
            sandwich_heads=sandwich_heads,
            sandwich_ff_expansion=sandwich_ff_expansion,
        )
        self.arch = "tabfoundry_sandwich"
        self.d_icl = int(self.model_spec.d_icl)
        self.input_normalization = str(self.model_spec.input_normalization).strip().lower()
        self.many_class_base = int(self.model_spec.many_class_base)
        self.norm_type = str(self.model_spec.norm_type).strip().lower()
        self.head_hidden_dim = int(self.model_spec.head_hidden_dim)
        self.pre_encoder_clip = self.model_spec.pre_encoder_clip
        self.sandwich_row_latents = int(self.model_spec.sandwich_row_latents)
        self.sandwich_col_latents = int(self.model_spec.sandwich_col_latents)
        self.sandwich_layers = int(self.model_spec.sandwich_layers)
        self.sandwich_heads = int(self.model_spec.sandwich_heads)
        self.sandwich_ff_expansion = int(self.model_spec.sandwich_ff_expansion)
        if self.norm_type != "layernorm":
            raise ValueError(
                "tabfoundry_sandwich currently requires norm_type='layernorm', "
                f"got {self.norm_type!r}"
            )
        self.tokenizer = ScalarPerFeatureMissingnessTokenizer()
        self.feature_encoder = SharedLinearFeatureEncoder(
            token_dim=int(self.tokenizer.token_dim),
            embedding_size=self.d_icl,
        )
        self.row_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.row_latent_seed = nn.Parameter(
            torch.randn(1, self.sandwich_row_latents, self.d_icl) * 0.02
        )
        self.col_latent_seed = nn.Parameter(
            torch.randn(1, self.sandwich_col_latents, self.d_icl) * 0.02
        )
        self.row_write = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            norm_type=self.norm_type,
        )
        self.col_write = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            norm_type=self.norm_type,
        )
        self.row_from_col_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    norm_type=self.norm_type,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.col_from_row_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    norm_type=self.norm_type,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.test_readout = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            norm_type=self.norm_type,
        )
        self.direct_head = DirectClassifierHead(
            self.d_icl,
            self.head_hidden_dim,
            self.many_class_base,
        )
        self._activation_checkpointing_enabled = False
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_checkpointing(self) -> None:
        self._activation_checkpointing_enabled = True

    def disable_activation_checkpointing(self) -> None:
        self._activation_checkpointing_enabled = False

    def _apply_activation_checkpoint(
        self,
        function,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        if not self._activation_checkpointing_enabled or not self.training:
            return function(*args)
        if not any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args):
            return function(*args)
        return activation_checkpoint(function, *args, use_reentrant=False)

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        trace_tensor = tensor.detach().to(torch.float32)
        trace_sum_sq = float(trace_tensor.square().sum().item())
        trace_count = int(trace_tensor.numel())
        total_sum_sq, total_count = self._activation_trace.get(name, (0.0, 0))
        self._activation_trace[name] = (
            total_sum_sq + trace_sum_sq,
            total_count + trace_count,
        )

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        if self._activation_trace is None:
            return None
        snapshot = {
            name: (float(total_sum_sq), int(total_count))
            for name, (total_sum_sq, total_count) in self._activation_trace.items()
            if total_count > 0
        }
        self._activation_trace = {}
        return snapshot

    def flush_activation_trace(self) -> dict[str, float] | None:
        snapshot = self.flush_activation_trace_stats()
        if snapshot is None:
            return None
        return {
            name: float(math.sqrt(total_sum_sq / float(total_count)))
            for name, (total_sum_sq, total_count) in snapshot.items()
            if total_count > 0
        }

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        if batch.num_classes is not None:
            return int(batch.num_classes)
        if batch.y_train.numel() == 0:
            raise RuntimeError("tabfoundry_sandwich requires at least one training label")
        return int(batch.y_train.max().item()) + 1

    @staticmethod
    def _prepare_task_inputs(batch: TaskBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if batch.x_train.ndim == _UNBATCHED_TASK_RANK:
            train_test_split_index = int(batch.x_train.shape[0])
            if train_test_split_index <= 0:
                raise RuntimeError("tabfoundry_sandwich requires at least one training row")
            x_all = torch.cat([batch.x_train, batch.x_test], dim=0).to(torch.float32).unsqueeze(0)
            y_train = batch.y_train.to(torch.int64).unsqueeze(0)
            y_test = batch.y_test.to(torch.int64).unsqueeze(0)
            return x_all, y_train, y_test, train_test_split_index
        if batch.x_train.ndim != _BATCHED_TASK_RANK or batch.x_test.ndim != _BATCHED_TASK_RANK:
            raise RuntimeError(
                "tabfoundry_sandwich task batching requires x_train/x_test rank 2 or 3, "
                f"got x_train={tuple(int(dim) for dim in batch.x_train.shape)}, "
                f"x_test={tuple(int(dim) for dim in batch.x_test.shape)}"
            )
        if batch.y_train.ndim != _UNBATCHED_TASK_RANK or batch.y_test.ndim != _UNBATCHED_TASK_RANK:
            raise RuntimeError(
                "tabfoundry_sandwich task batching requires y_train/y_test rank 2 when batching, "
                f"got y_train={tuple(int(dim) for dim in batch.y_train.shape)}, "
                f"y_test={tuple(int(dim) for dim in batch.y_test.shape)}"
            )
        if int(batch.x_train.shape[0]) != int(batch.x_test.shape[0]):
            raise RuntimeError("tabfoundry_sandwich batched train/test tensors must share a batch dimension")
        train_test_split_index = int(batch.x_train.shape[1])
        if train_test_split_index <= 0:
            raise RuntimeError("tabfoundry_sandwich requires at least one training row")
        x_all = torch.cat([batch.x_train, batch.x_test], dim=1).to(torch.float32)
        y_train = batch.y_train.to(torch.int64)
        y_test = batch.y_test.to(torch.int64)
        return x_all, y_train, y_test, train_test_split_index

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        if x_all.ndim != _BATCHED_TASK_RANK:
            raise ValueError(f"x_all must have shape [B, R, C], got {tuple(x_all.shape)}")
        if y_train.ndim != _UNBATCHED_TASK_RANK:
            raise ValueError(f"y_train must have shape [B, R_train], got {tuple(y_train.shape)}")
        if int(x_all.shape[0]) != int(y_train.shape[0]):
            raise ValueError("x_all and y_train must have matching batch dimensions")
        if train_test_split_index <= 0 or train_test_split_index >= int(x_all.shape[1]):
            raise ValueError(
                "train_test_split_index must satisfy 0 < split < num_rows, got "
                f"split={train_test_split_index}, num_rows={x_all.shape[1]}"
            )
        if int(y_train.shape[1]) != train_test_split_index:
            raise ValueError("y_train length must match train_test_split_index")

    def _normalize_x_all(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        x_train = x_all[:, :train_test_split_index, :]
        x_test = x_all[:, train_test_split_index:, :]
        train_norm, test_norm = normalize_train_test_tensors(
            x_train,
            x_test,
            mode=cast(InputNormalizationMode, self.input_normalization),
            preserve_non_finite=True,
        )
        return torch.cat([train_norm, test_norm], dim=1)

    def _feature_cells(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        normalized = self._normalize_x_all(x_all, train_test_split_index=train_test_split_index)
        if self.pre_encoder_clip is not None:
            normalized = clip_finite_values(
                normalized,
                clip_value=float(self.pre_encoder_clip),
            )
        tokenized_x, _ = self.tokenizer(normalized)
        feature_cells = self.feature_encoder(tokenized_x)
        self.trace_activation("post_feature_encoder", feature_cells)
        return feature_cells

    def _row_summaries(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self.row_conditioner(
            y_train,
            num_rows=int(feature_cells.shape[1]),
        ).squeeze(2)
        self.trace_activation("post_target_conditioner", conditioned)
        summaries = feature_cells.mean(dim=2) + conditioned
        self.trace_activation("post_row_summary", summaries)
        return summaries

    def _column_summaries(
        self,
        feature_cells: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        summaries = feature_cells[:, :train_test_split_index, :, :].mean(dim=1)
        self.trace_activation("post_column_summary", summaries)
        return summaries

    def _cross_block(
        self,
        block: _CrossAttentionBlock,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor, current_kv: torch.Tensor) -> torch.Tensor:
            return block(current_query, key_value=current_kv)

        return self._apply_activation_checkpoint(_apply, query, key_value)

    def _validate_num_classes(self, num_classes: int) -> None:
        if num_classes < _MIN_CLASS_COUNT:
            raise RuntimeError(
                f"tabfoundry_sandwich requires at least {_MIN_CLASS_COUNT} classes, got {num_classes}"
            )
        if num_classes > self.many_class_base:
            raise RuntimeError(
                "tabfoundry_sandwich is small-class only and requires "
                f"num_classes <= many_class_base={self.many_class_base}, got {num_classes}"
            )

    def _forward_logits_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        feature_cells = self._feature_cells(x_all, train_test_split_index=train_test_split_index)
        row_summaries = self._row_summaries(feature_cells, y_train=y_train)
        column_summaries = self._column_summaries(
            feature_cells,
            train_test_split_index=train_test_split_index,
        )
        train_rows = row_summaries[:, :train_test_split_index, :]
        test_rows = row_summaries[:, train_test_split_index:, :]
        row_latents = self.row_latent_seed.expand(int(x_all.shape[0]), -1, -1)
        col_latents = self.col_latent_seed.expand(int(x_all.shape[0]), -1, -1)
        row_latents = self._cross_block(self.row_write, row_latents, train_rows)
        self.trace_activation("post_row_write", row_latents)
        col_latents = self._cross_block(self.col_write, col_latents, column_summaries)
        self.trace_activation("post_col_write", col_latents)
        for index, (row_block, col_block) in enumerate(
            zip(self.row_from_col_blocks, self.col_from_row_blocks, strict=True)
        ):
            row_latents = self._cross_block(
                cast(_CrossAttentionBlock, row_block),
                row_latents,
                col_latents,
            )
            self.trace_activation(f"post_row_latent_block_{index}", row_latents)
            col_latents = self._cross_block(
                cast(_CrossAttentionBlock, col_block),
                col_latents,
                row_latents,
            )
            self.trace_activation(f"post_col_latent_block_{index}", col_latents)
        fused_latents = torch.cat([row_latents, col_latents], dim=1)
        self.trace_activation("post_fused_latents", fused_latents)
        test_rows = self._cross_block(self.test_readout, test_rows, fused_latents)
        self.trace_activation("post_test_readout", test_rows)
        return self.direct_head(test_rows)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        return self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        logits = self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
        )
        return ClassificationOutput(
            logits=flatten_classification_output_rows(logits),
            num_classes=num_classes,
            class_probs=None,
        )
