"""Fixed-latent repeated-input Perceiver-style sandwich classifier."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from tab_foundry.feature_types import (
    DEFAULT_FEATURE_TYPE,
    FEATURE_TYPE_VOCAB,
    resolve_feature_types,
)
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
    DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    DEFAULT_MODEL_SANDWICH_HEADS,
    DEFAULT_MODEL_SANDWICH_LATENTS,
    DEFAULT_MODEL_SANDWICH_LAYERS,
    ModelBuildSpec,
)
from tab_foundry.types import TaskBatch

_UNBATCHED_TASK_RANK = 2
_BATCHED_TASK_RANK = 3
_MIN_CLASS_COUNT = 2
_ROW_SUMMARY_TOKEN_ID = 0
_COLUMN_SUMMARY_TOKEN_ID = 1
_TRAIN_ROLE_ID = 0
_TEST_ROLE_ID = 1
_FEATURE_TYPE_TO_ID = {name: index for index, name in enumerate(FEATURE_TYPE_VOCAB)}


def _init_truncated_normal_(
    tensor: torch.Tensor,
    *,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initialize a tensor from a truncated normal distribution."""

    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


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


class _SelfAttentionBlock(nn.Module):
    """Pre-norm residual self-attention plus FFN."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.attn_norm = build_norm(norm_type, embedding_size)
        self.ff_norm = build_norm(norm_type, embedding_size)
        self.attn = nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)
        ff_hidden = embedding_size * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embedding_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_norm = self.attn_norm(hidden)
        hidden = hidden + multihead_attention_sdpa(
            self.attn,
            hidden_norm,
            hidden_norm,
            hidden_norm,
        )
        return hidden + self.ff(self.ff_norm(hidden))


class _PerceiverStage(nn.Module):
    """One unshared Perceiver stage: input read, then latent self-attention."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.input_read = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )
        self.latent_block = _SelfAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )


class TabFoundrySandwichClassifier(nn.Module):
    """Small-class fixed-latent repeated-input Perceiver-style classifier."""

    def __init__(
        self,
        *,
        d_icl: int = DEFAULT_MODEL_D_ICL,
        input_normalization: str = DEFAULT_MODEL_INPUT_NORMALIZATION,
        many_class_base: int = DEFAULT_MODEL_MANY_CLASS_BASE,
        norm_type: str = DEFAULT_MODEL_NORM_TYPE,
        head_hidden_dim: int = DEFAULT_MODEL_HEAD_HIDDEN_DIM,
        pre_encoder_clip: float | None = DEFAULT_MODEL_PRE_ENCODER_CLIP,
        sandwich_latents: int = DEFAULT_MODEL_SANDWICH_LATENTS,
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
            sandwich_latents=sandwich_latents,
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
        self.sandwich_latents = int(self.model_spec.sandwich_latents)
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
        self.feature_type_embedding = nn.Embedding(len(FEATURE_TYPE_VOCAB), self.d_icl)
        self.row_summary_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.column_summary_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.row_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            norm_type=self.norm_type,
        )
        self.column_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            norm_type=self.norm_type,
        )
        self.byte_token_type = nn.Embedding(2, self.d_icl)
        self.y_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.y_role_embedding = nn.Embedding(2, self.d_icl)
        self.latent_seed = nn.Parameter(torch.empty(1, self.sandwich_latents, self.d_icl))
        _init_truncated_normal_(self.latent_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.perceiver_stages = nn.ModuleList(
            [
                _PerceiverStage(
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

    @staticmethod
    def _fourier_positions(
        *,
        num_positions: int,
        embedding_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(num_positions, device=device, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, embedding_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / float(embedding_size))
        )
        encoding = torch.zeros((num_positions, embedding_size), device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        odd_width = encoding[:, 1::2].shape[1]
        if odd_width > 0:
            encoding[:, 1::2] = torch.cos(positions * div_terms[:odd_width])
        return encoding.to(dtype=dtype).unsqueeze(0)

    @staticmethod
    def _default_feature_type_ids(
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        default_id = int(_FEATURE_TYPE_TO_ID[DEFAULT_FEATURE_TYPE])
        return torch.full(
            (int(batch_size), int(num_features)),
            default_id,
            device=device,
            dtype=torch.int64,
        )

    def _feature_type_ids_from_metadata(
        self,
        metadata: dict[str, Any],
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        members = metadata.get("task_members")
        resolved_types_by_task: list[list[str]]
        if isinstance(members, list) and members:
            if len(members) != int(batch_size):
                raise RuntimeError(
                    "tabfoundry_sandwich task-batched metadata must align with the tensor batch size: "
                    f"expected={int(batch_size)}, got={len(members)}"
                )
            resolved_types_by_task = []
            for index, member in enumerate(members):
                if not isinstance(member, dict):
                    raise RuntimeError(
                        "tabfoundry_sandwich task-batched metadata members must be objects, "
                        f"got task_members[{index}]={type(member).__name__}"
                    )
                resolved_types_by_task.append(
                    resolve_feature_types(
                        member.get("feature_types"),
                        expected_count=int(num_features),
                        context=f"batch.metadata.task_members[{index}].feature_types",
                    )
                )
        else:
            feature_types = resolve_feature_types(
                metadata.get("feature_types"),
                expected_count=int(num_features),
                context="batch.metadata.feature_types",
            )
            resolved_types_by_task = [list(feature_types) for _ in range(int(batch_size))]
        feature_type_ids = [
            [int(_FEATURE_TYPE_TO_ID[value]) for value in feature_types]
            for feature_types in resolved_types_by_task
        ]
        return torch.tensor(feature_type_ids, device=device, dtype=torch.int64)

    def _feature_cells(
        self,
        x_all: torch.Tensor,
        *,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self._normalize_x_all(x_all, train_test_split_index=train_test_split_index)
        if self.pre_encoder_clip is not None:
            normalized = clip_finite_values(
                normalized,
                clip_value=float(self.pre_encoder_clip),
            )
        tokenized_x, _ = self.tokenizer(normalized)
        feature_cells = self.feature_encoder(tokenized_x)
        self.trace_activation("post_feature_encoder", feature_cells)
        row_pos = self._fourier_positions(
            num_positions=int(feature_cells.shape[1]),
            embedding_size=int(feature_cells.shape[3]),
            device=feature_cells.device,
            dtype=feature_cells.dtype,
        ).unsqueeze(2)
        col_pos = self._fourier_positions(
            num_positions=int(feature_cells.shape[2]),
            embedding_size=int(feature_cells.shape[3]),
            device=feature_cells.device,
            dtype=feature_cells.dtype,
        ).unsqueeze(1)
        feature_type_embed = self.feature_type_embedding(feature_type_ids).unsqueeze(1)
        feature_type_embed = feature_type_embed.to(dtype=feature_cells.dtype)
        feature_cells = feature_cells + row_pos + col_pos + feature_type_embed
        self.trace_activation("post_cell_encoding", feature_cells)
        return feature_cells

    def _cross_block(
        self,
        block: _CrossAttentionBlock,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor, current_kv: torch.Tensor) -> torch.Tensor:
            return block(current_query, key_value=current_kv)

        return self._apply_activation_checkpoint(_apply, query, key_value)

    def _self_block(
        self,
        block: _SelfAttentionBlock,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_hidden: torch.Tensor) -> torch.Tensor:
            return block(current_hidden)

        return self._apply_activation_checkpoint(_apply, hidden)

    def _summary_query_attention(
        self,
        block: _CrossAttentionBlock,
        *,
        query: torch.Tensor,
        key_value: torch.Tensor,
        outer_count: int,
    ) -> torch.Tensor:
        batch_size, _, inner_count, embedding_size = (
            int(key_value.shape[0]),
            int(key_value.shape[1]),
            int(key_value.shape[2]),
            int(key_value.shape[3]),
        )
        flat_kv = key_value.reshape(batch_size * outer_count, inner_count, embedding_size)
        flat_query = query.expand(batch_size * outer_count, -1, -1).to(
            device=key_value.device,
            dtype=key_value.dtype,
        )
        summaries = self._cross_block(block, flat_query, flat_kv)
        return summaries.squeeze(1).reshape(batch_size, outer_count, embedding_size)

    def _row_summary_bytes(self, feature_cells: torch.Tensor) -> torch.Tensor:
        summaries = self._summary_query_attention(
            self.row_summary_builder,
            query=self.row_summary_query,
            key_value=feature_cells,
            outer_count=int(feature_cells.shape[1]),
        )
        self.trace_activation("post_row_summary", summaries)
        return summaries

    def _column_summary_bytes(self, feature_cells: torch.Tensor) -> torch.Tensor:
        column_major = feature_cells.transpose(1, 2).contiguous()
        summaries = self._summary_query_attention(
            self.column_summary_builder,
            query=self.column_summary_query,
            key_value=column_major,
            outer_count=int(column_major.shape[1]),
        )
        self.trace_activation("post_column_summary", summaries)
        return summaries

    def _x_byte_array_tokens(self, feature_cells: torch.Tensor) -> torch.Tensor:
        row_summaries = self._row_summary_bytes(feature_cells)
        column_summaries = self._column_summary_bytes(feature_cells)
        row_type_ids = torch.full(
            row_summaries.shape[:2],
            _ROW_SUMMARY_TOKEN_ID,
            device=row_summaries.device,
            dtype=torch.int64,
        )
        col_type_ids = torch.full(
            column_summaries.shape[:2],
            _COLUMN_SUMMARY_TOKEN_ID,
            device=column_summaries.device,
            dtype=torch.int64,
        )
        row_tokens = row_summaries + self.byte_token_type(row_type_ids).to(dtype=row_summaries.dtype)
        column_tokens = column_summaries + self.byte_token_type(col_type_ids).to(
            dtype=column_summaries.dtype
        )
        byte_tokens = torch.cat([row_tokens, column_tokens], dim=1)
        self.trace_activation("post_x_byte_array", byte_tokens)
        return byte_tokens

    def _row_summary_tokens(
        self,
        *,
        feature_cells: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        row_summaries = self._row_summary_bytes(feature_cells)
        num_rows = int(feature_cells.shape[1])
        conditioned = self.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
            dtype=row_summaries.dtype
        )
        row_pos = self._fourier_positions(
            num_positions=num_rows,
            embedding_size=int(row_summaries.shape[2]),
            device=row_summaries.device,
            dtype=row_summaries.dtype,
        )
        role_ids = torch.full(
            row_summaries.shape[:2],
            _TRAIN_ROLE_ID,
            device=row_summaries.device,
            dtype=torch.int64,
        )
        role_ids[:, int(y_train.shape[1]) :] = _TEST_ROLE_ID
        role_embed = self.y_role_embedding(role_ids).to(dtype=row_summaries.dtype)
        token_type_ids = torch.full(
            row_summaries.shape[:2],
            _ROW_SUMMARY_TOKEN_ID,
            device=row_summaries.device,
            dtype=torch.int64,
        )
        token_type = self.byte_token_type(token_type_ids).to(dtype=row_summaries.dtype)
        tokens = row_summaries + conditioned + row_pos + role_embed + token_type
        self.trace_activation("post_row_summary_tokens", tokens)
        return tokens

    def _column_summary_tokens(self, feature_cells: torch.Tensor) -> torch.Tensor:
        column_summaries = self._column_summary_bytes(feature_cells)
        token_type_ids = torch.full(
            column_summaries.shape[:2],
            _COLUMN_SUMMARY_TOKEN_ID,
            device=column_summaries.device,
            dtype=torch.int64,
        )
        token_type = self.byte_token_type(token_type_ids).to(dtype=column_summaries.dtype)
        tokens = column_summaries + token_type
        self.trace_activation("post_column_summary_tokens", tokens)
        return tokens

    def _perceiver_input_tokens(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_tokens = self._row_summary_tokens(feature_cells=feature_cells, y_train=y_train)
        column_tokens = self._column_summary_tokens(feature_cells)
        input_tokens = torch.cat([row_tokens, column_tokens], dim=1)
        self.trace_activation("post_perceiver_input", input_tokens)
        return input_tokens, row_tokens

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
        feature_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        feature_cells = self._feature_cells(
            x_all,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
        )
        perceiver_input, row_tokens = self._perceiver_input_tokens(feature_cells, y_train=y_train)
        latents = self.latent_seed.expand(int(x_all.shape[0]), -1, -1)
        for index, stage in enumerate(self.perceiver_stages):
            stage = cast(_PerceiverStage, stage)
            latents = self._cross_block(stage.input_read, latents, perceiver_input)
            self.trace_activation(f"post_stage_{index}_cross", latents)
            latents = self._self_block(stage.latent_block, latents)
            self.trace_activation(f"post_stage_{index}_self", latents)
        test_queries = row_tokens[:, train_test_split_index:, :]
        test_rows = self._cross_block(self.test_readout, test_queries, latents)
        self.trace_activation("post_test_readout", test_rows)
        return self.direct_head(test_rows)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        feature_type_ids = self._default_feature_type_ids(
            batch_size=int(x_all.shape[0]),
            num_features=int(x_all.shape[2]),
            device=x_all.device,
        )
        return self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        feature_type_ids = self._feature_type_ids_from_metadata(
            batch.metadata,
            batch_size=int(x_all.shape[0]),
            num_features=int(x_all.shape[2]),
            device=x_all.device,
        )
        logits = self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
        )
        return ClassificationOutput(
            logits=flatten_classification_output_rows(logits),
            num_classes=num_classes,
            class_probs=None,
        )
