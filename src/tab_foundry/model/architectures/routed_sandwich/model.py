"""Routed latent-memory sandwich classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from tab_foundry.feature_types import (
    FEATURE_TYPE_VOCAB,
    feature_type_ids_from_resolved,
    feature_type_ids_from_task_metadata,
    normalize_feature_types,
)
from tab_foundry.model.components.attention import multihead_attention_sdpa
from tab_foundry.model.components.tabular_primitives import (
    DirectMulticlassHead,
    FeatureTypeFiLM,
    LabelTokenTargetConditioner,
    ScalarPerFeatureMissingnessTokenizer,
    SharedLinearFeatureEncoder,
)
from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows
from tab_foundry.model.spec import ROUTED_SANDWICH_DEFAULTS as _D, ModelBuildSpec
from tab_foundry.types import TaskBatch

from .. import shared_forward as _shared_forward
from .. import shared_hooks as _shared_hooks
from ..tabfoundry_sandwich import feature_flow as _feature_flow
from ..tabfoundry_sandwich.blocks import (
    _CrossAttentionBlock,
    _InducedSetAttentionBlock,
    _NativePackedCrossAttention,
    _NativePackedSelfAttention,
    _SelfAttentionBlock,
    _build_sandwich_activation,
    _build_sandwich_block_norm,
    _init_truncated_normal_,
)
from ..tabfoundry_sandwich.states import SandwichFeatureState, SandwichRawInputState


_CLASSIFICATION_LOSS_SURFACE = "classification"
_MIN_CLASS_COUNT = 2
_ROW_SUMMARY_TOKEN_ID = 0
_COLUMN_SUMMARY_TOKEN_ID = 1
_EVIDENCE_TOKEN_ID = 2
_CELL_TOKEN_ID = 3


@dataclass(slots=True)
class _RoutedClassificationState:
    feature_state: SandwichFeatureState
    full_cell_stream: torch.Tensor
    context_bank: torch.Tensor
    stage0_input: torch.Tensor
    row_tokens: torch.Tensor


class _DynamicHyperConnection(nn.Module):
    """Input-dependent width/depth mixing over routed residual streams."""

    def __init__(self, *, embedding_size: int, num_streams: int) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.num_streams = int(num_streams)
        self.width_norm = nn.LayerNorm(self.embedding_size)
        self.depth_norm = nn.LayerNorm(self.embedding_size)
        self.width_proj = nn.Linear(self.embedding_size, self.num_streams)
        self.depth_proj = nn.Linear(self.embedding_size, self.num_streams)
        self.width_static = nn.Parameter(torch.zeros(self.num_streams))
        self.depth_static = nn.Parameter(torch.zeros(self.num_streams))
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

    def width_mix(self, streams: torch.Tensor) -> torch.Tensor:
        pooled = streams.mean(dim=1).mean(dim=1)
        dynamic = torch.tanh(self.width_proj(self.width_norm(pooled))) * self.dynamic_scale
        weights = torch.softmax(dynamic + self.width_static.unsqueeze(0), dim=-1)
        return torch.einsum("bs,bnsd->bnd", weights, streams)

    def depth_mix(self, streams: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        pooled = primary.mean(dim=1)
        dynamic = torch.tanh(self.depth_proj(self.depth_norm(pooled))) * self.dynamic_scale
        weights = torch.tanh(dynamic + self.depth_static.unsqueeze(0))
        weights = weights / float(self.num_streams)
        return streams + primary.unsqueeze(2) * weights.unsqueeze(1).unsqueeze(-1)


class _RoutedCrossAttentionBlock(nn.Module):
    """Width-mix routed streams, run cross-attention, then depth-mix back."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_streams: int,
        residual_multiplier: float,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.router = _DynamicHyperConnection(
            embedding_size=embedding_size,
            num_streams=num_streams,
        )
        self.query_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.kv_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.packed_attention = bool(packed_attention)
        self.residual_multiplier = float(residual_multiplier)
        self.attn = (
            _NativePackedCrossAttention(embedding_size=embedding_size, n_heads=n_heads)
            if self.packed_attention
            else nn.MultiheadAttention(
                embedding_size,
                n_heads,
                batch_first=True,
            )
        )
        ff_hidden = embedding_size * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            _build_sandwich_activation(activation),
            nn.Linear(ff_hidden, embedding_size),
        )

    def forward(self, query_streams: torch.Tensor, *, key_value: torch.Tensor) -> torch.Tensor:
        primary = self.router.width_mix(query_streams)
        q_norm = self.query_norm(primary)
        kv_norm = self.kv_norm(key_value)
        if self.packed_attention:
            if not isinstance(self.attn, _NativePackedCrossAttention):
                raise RuntimeError("packed routed cross-attention is missing native attention")
            primary = primary + (self.residual_multiplier * self.attn(q_norm, key_value=kv_norm))
        else:
            primary = primary + (
                self.residual_multiplier
                * multihead_attention_sdpa(
                    cast(nn.MultiheadAttention, self.attn),
                    q_norm,
                    kv_norm,
                    kv_norm,
                )
            )
        primary = primary + (self.residual_multiplier * self.ff(self.ff_norm(primary)))
        return self.router.depth_mix(query_streams, primary)


class _RoutedSelfAttentionBlock(nn.Module):
    """Width-mix routed streams, run self-attention, then depth-mix back."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_streams: int,
        residual_multiplier: float,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.router = _DynamicHyperConnection(
            embedding_size=embedding_size,
            num_streams=num_streams,
        )
        self.attn_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.packed_attention = bool(packed_attention)
        self.residual_multiplier = float(residual_multiplier)
        self.attn = (
            _NativePackedSelfAttention(embedding_size=embedding_size, n_heads=n_heads)
            if self.packed_attention
            else nn.MultiheadAttention(
                embedding_size,
                n_heads,
                batch_first=True,
            )
        )
        ff_hidden = embedding_size * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            _build_sandwich_activation(activation),
            nn.Linear(ff_hidden, embedding_size),
        )

    def forward(
        self,
        query_streams: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        primary = self.router.width_mix(query_streams)
        hidden_norm = self.attn_norm(primary)
        if self.packed_attention:
            if not isinstance(self.attn, _NativePackedSelfAttention):
                raise RuntimeError("packed routed self-attention is missing native attention")
            primary = primary + (
                self.residual_multiplier * self.attn(hidden_norm, attn_bias=attn_bias)
            )
        else:
            primary = primary + (
                self.residual_multiplier
                * multihead_attention_sdpa(
                    cast(nn.MultiheadAttention, self.attn),
                    hidden_norm,
                    hidden_norm,
                    hidden_norm,
                    attn_bias=attn_bias,
                )
            )
        primary = primary + (self.residual_multiplier * self.ff(self.ff_norm(primary)))
        return self.router.depth_mix(query_streams, primary)


class _RoutedPerceiverStage(nn.Module):
    """One routed context read followed by routed latent self-attention."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_streams: int,
        residual_multiplier: float,
        self_attention_per_cross: int,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.input_read = _RoutedCrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            num_streams=num_streams,
            residual_multiplier=residual_multiplier,
            packed_attention=packed_attention,
        )
        self.self_blocks = nn.ModuleList(
            [
                _RoutedSelfAttentionBlock(
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    activation=activation,
                    block_norm=block_norm,
                    num_streams=num_streams,
                    residual_multiplier=residual_multiplier,
                    packed_attention=packed_attention,
                )
                for _ in range(self_attention_per_cross)
            ]
        )


def _scale_routed_block_outputs(module: nn.Module, *, scale: float) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("ff.2.weight"):
                parameter.mul_(float(scale))


class RoutedSandwichClassifier(nn.Module):
    """Classification-only sandwich variant with routed latent/query residuals."""

    def __init__(
        self,
        *,
        d_icl: int = _D["d_icl"],
        input_normalization: str = _D["input_normalization"],
        many_class_base: int = _D["many_class_base"],
        norm_type: str = _D["norm_type"],
        head_hidden_dim: int = _D["head_hidden_dim"],
        pre_encoder_clip: float | None = _D["pre_encoder_clip"],
        sandwich_latents: int = _D["sandwich_latents"],
        sandwich_layers: int = _D["sandwich_layers"],
        sandwich_heads: int = _D["sandwich_heads"],
        sandwich_ff_expansion: int = _D["sandwich_ff_expansion"],
        sandwich_activation: str = _D["sandwich_activation"],
        sandwich_block_norm: str = _D["sandwich_block_norm"],
        sandwich_self_attention_per_cross: int = _D["sandwich_self_attention_per_cross"],
        sandwich_pre_row_attention_layers: int = _D["sandwich_pre_row_attention_layers"],
        sandwich_pre_column_attention_layers: int = _D["sandwich_pre_column_attention_layers"],
        sandwich_pre_column_inducing_tokens: int = _D["sandwich_pre_column_inducing_tokens"],
        sandwich_packed_attention: bool = _D["sandwich_packed_attention"],
        feature_type_conditioning: str = _D["feature_type_conditioning"],
        routed_residual_mode: str = _D["routed_residual_mode"],
        routed_residual_streams: int = _D["routed_residual_streams"],
        routed_residual_scale: str = _D["routed_residual_scale"],
        routed_row_summary_tokens: int = _D["routed_row_summary_tokens"],
        routed_column_summary_tokens: int = _D["routed_column_summary_tokens"],
        routed_evidence_tokens: int = _D["routed_evidence_tokens"],
        routed_direct_cell_bypass: bool = _D["routed_direct_cell_bypass"],
        floating_likelihood: str = _D["floating_likelihood"],
        integer_likelihood: str = _D["integer_likelihood"],
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
            arch="routed_sandwich",
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
            sandwich_activation=sandwich_activation,
            sandwich_block_norm=sandwich_block_norm,
            sandwich_self_attention_per_cross=sandwich_self_attention_per_cross,
            sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
            sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
            sandwich_pre_column_inducing_tokens=sandwich_pre_column_inducing_tokens,
            sandwich_packed_attention=sandwich_packed_attention,
            feature_type_conditioning=feature_type_conditioning,
            routed_residual_mode=routed_residual_mode,
            routed_residual_streams=routed_residual_streams,
            routed_residual_scale=routed_residual_scale,
            routed_row_summary_tokens=routed_row_summary_tokens,
            routed_column_summary_tokens=routed_column_summary_tokens,
            routed_evidence_tokens=routed_evidence_tokens,
            routed_direct_cell_bypass=routed_direct_cell_bypass,
            floating_likelihood=floating_likelihood,
            integer_likelihood=integer_likelihood,
        )
        self.arch = "routed_sandwich"
        self.loss_surface = _CLASSIFICATION_LOSS_SURFACE
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
        self.sandwich_activation = str(self.model_spec.sandwich_activation).strip().lower()
        self.sandwich_block_norm = str(self.model_spec.sandwich_block_norm).strip().lower()
        self.self_attention_per_cross = int(self.model_spec.sandwich_self_attention_per_cross)
        self.pre_row_attention_layers = int(self.model_spec.sandwich_pre_row_attention_layers)
        self.pre_column_attention_layers = int(
            self.model_spec.sandwich_pre_column_attention_layers
        )
        self.pre_column_inducing_tokens = int(self.model_spec.sandwich_pre_column_inducing_tokens)
        self.sandwich_packed_attention = bool(self.model_spec.sandwich_packed_attention)
        self.feature_type_conditioning = (
            str(self.model_spec.feature_type_conditioning).strip().lower()
        )
        self.routed_residual_mode = str(self.model_spec.routed_residual_mode).strip().lower()
        self.routed_residual_streams = int(self.model_spec.routed_residual_streams)
        self.routed_residual_scale = str(self.model_spec.routed_residual_scale).strip().lower()
        self.routed_row_summary_tokens = int(self.model_spec.routed_row_summary_tokens)
        self.routed_column_summary_tokens = int(self.model_spec.routed_column_summary_tokens)
        self.routed_evidence_tokens = int(self.model_spec.routed_evidence_tokens)
        self.routed_direct_cell_bypass = bool(self.model_spec.routed_direct_cell_bypass)
        if self.norm_type != "layernorm":
            raise ValueError(
                "routed_sandwich currently requires norm_type='layernorm', "
                f"got {self.norm_type!r}"
            )
        if self.routed_residual_mode != "dynamic_hyper":
            raise ValueError(
                "routed_sandwich currently requires routed_residual_mode='dynamic_hyper', "
                f"got {self.routed_residual_mode!r}"
            )
        if self.routed_residual_scale != "deepnorm":
            raise ValueError(
                "routed_sandwich currently requires routed_residual_scale='deepnorm', "
                f"got {self.routed_residual_scale!r}"
            )

        residual_depth = (
            self.sandwich_layers * (self.self_attention_per_cross + 1)
            + 1
            + int(self.routed_direct_cell_bypass)
        )
        self.deepnorm_residual_depth = max(1, int(residual_depth))
        self.deepnorm_alpha = float((2.0 * self.deepnorm_residual_depth) ** 0.25)
        self.deepnorm_beta = float((8.0 * self.deepnorm_residual_depth) ** (-0.25))

        self.tokenizer = ScalarPerFeatureMissingnessTokenizer()
        self.feature_encoder = SharedLinearFeatureEncoder(
            token_dim=int(self.tokenizer.token_dim),
            embedding_size=self.d_icl,
        )
        self.feature_type_film: FeatureTypeFiLM | None
        self.feature_type_embedding: nn.Embedding | None
        if self.feature_type_conditioning == "film":
            self.feature_type_film = FeatureTypeFiLM(len(FEATURE_TYPE_VOCAB), self.d_icl)
            self.feature_type_embedding = None
        else:
            self.feature_type_film = None
            self.feature_type_embedding = nn.Embedding(len(FEATURE_TYPE_VOCAB), self.d_icl)

        self.row_summary_query = nn.Parameter(
            torch.randn(1, self.routed_row_summary_tokens, self.d_icl) * 0.02
        )
        self.column_summary_query = nn.Parameter(
            torch.randn(1, self.routed_column_summary_tokens, self.d_icl) * 0.02
        )
        self.evidence_query = nn.Parameter(
            torch.randn(1, self.routed_evidence_tokens, self.d_icl) * 0.02
        )
        self.test_query_seed = nn.Parameter(
            torch.randn(1, 1, self.routed_residual_streams, self.d_icl) * 0.02
        )
        self.test_row_pool_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)

        self.row_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.column_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.evidence_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.pre_row_attention_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.pre_row_attention_layers)
            ]
        )
        self.pre_column_attention_blocks = nn.ModuleList(
            [
                _InducedSetAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    num_inducing=self.pre_column_inducing_tokens,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.pre_column_attention_layers)
            ]
        )
        self.token_type_embedding = nn.Embedding(4, self.d_icl)
        self.y_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.y_role_embedding = nn.Embedding(2, self.d_icl)
        self.latent_seed = nn.Parameter(
            torch.empty(1, self.sandwich_latents, self.routed_residual_streams, self.d_icl)
        )
        _init_truncated_normal_(self.latent_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.latent_memory_router = _DynamicHyperConnection(
            embedding_size=self.d_icl,
            num_streams=self.routed_residual_streams,
        )
        self.perceiver_stages = nn.ModuleList(
            [
                _RoutedPerceiverStage(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    num_streams=self.routed_residual_streams,
                    residual_multiplier=self.deepnorm_alpha,
                    self_attention_per_cross=self.self_attention_per_cross,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.latent_readout = _RoutedCrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            num_streams=self.routed_residual_streams,
            residual_multiplier=self.deepnorm_alpha,
            packed_attention=self.sandwich_packed_attention,
        )
        self.cell_readout = _RoutedCrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            num_streams=self.routed_residual_streams,
            residual_multiplier=self.deepnorm_alpha,
            packed_attention=self.sandwich_packed_attention,
        )
        self.test_row_pool = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.direct_head = DirectMulticlassHead(
            self.d_icl,
            self.head_hidden_dim,
            self.many_class_base,
        )

        for stage in self.perceiver_stages:
            _scale_routed_block_outputs(stage, scale=self.deepnorm_beta)
        _scale_routed_block_outputs(self.latent_readout, scale=self.deepnorm_beta)
        _scale_routed_block_outputs(self.cell_readout, scale=self.deepnorm_beta)

        self._activation_checkpointing_enabled = False
        self._activation_trace: dict[str, tuple[float, int]] | None = None
        self._fourier_position_cache: dict[
            tuple[int, int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def enable_activation_checkpointing(self) -> None:
        _shared_hooks.enable_activation_checkpointing(self)

    def disable_activation_checkpointing(self) -> None:
        _shared_hooks.disable_activation_checkpointing(self)

    def set_loss_surface(self, loss_surface: str) -> None:
        normalized = str(loss_surface).strip().lower()
        if normalized != _CLASSIFICATION_LOSS_SURFACE:
            raise ValueError(
                "routed_sandwich only supports loss_surface='classification', "
                f"got {loss_surface!r}"
            )
        self.loss_surface = normalized

    def _apply_activation_checkpoint(
        self,
        function,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        return _shared_hooks.apply_activation_checkpoint(self, function, *args)

    def enable_activation_trace(self) -> None:
        _shared_hooks.enable_activation_trace(self)

    def disable_activation_trace(self) -> None:
        _shared_hooks.disable_activation_trace(self)

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        _shared_hooks.trace_activation(self, name, tensor)

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        return _shared_hooks.flush_activation_trace_stats(self)

    def flush_activation_trace(self) -> dict[str, float] | None:
        return _shared_hooks.flush_activation_trace(self)

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        return _shared_forward.task_num_classes(batch, arch_name="routed_sandwich")

    @staticmethod
    def _prepare_task_inputs(
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return _shared_forward.prepare_task_inputs(batch, arch_name="routed_sandwich")

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        _shared_forward.validate_batched_inputs(x_all, y_train, train_test_split_index)

    def _normalize_x_all(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        return _shared_forward.normalize_x_all(
            x_all,
            train_test_split_index=train_test_split_index,
            input_normalization=self.input_normalization,
            preserve_non_finite=True,
        )

    def _build_raw_input_state(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None,
        train_test_split_index: int,
        num_classes: int,
        feature_type_ids: torch.Tensor,
    ) -> SandwichRawInputState:
        return _feature_flow.build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=num_classes,
            feature_type_ids=feature_type_ids,
        )

    def _build_feature_state(
        self,
        raw_state: SandwichRawInputState,
        *,
        apply_input_normalization: bool = True,
    ) -> SandwichFeatureState:
        return _feature_flow.build_feature_state(
            self,
            raw_state,
            apply_input_normalization=apply_input_normalization,
        )

    def _fourier_positions(
        self,
        *,
        num_positions: int,
        embedding_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(num_positions), int(embedding_size), torch.device(device), dtype)
        cached = self._fourier_position_cache.get(key)
        if cached is not None:
            return cached
        positions = _feature_flow.fourier_positions(
            num_positions=num_positions,
            embedding_size=embedding_size,
            device=device,
            dtype=dtype,
        )
        self._fourier_position_cache[key] = positions
        return positions

    @staticmethod
    def _feature_type_ids_from_resolved(
        resolved_types_by_task: list[list[str]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return feature_type_ids_from_resolved(
            resolved_types_by_task,
            device=device,
        )

    @staticmethod
    def _normalize_required_feature_types(
        feature_types: Any,
        *,
        expected_count: int,
        context: str,
    ) -> list[str]:
        if feature_types is None:
            raise ValueError(f"{context} is required for routed_sandwich")
        return normalize_feature_types(
            feature_types,
            expected_count=expected_count,
            context=context,
        )

    def _feature_type_ids_from_forward_batched(
        self,
        feature_types: list[str] | list[list[str]] | None,
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        if feature_types is None:
            raise ValueError("routed_sandwich forward_batched() requires explicit feature_types")
        if not feature_types or isinstance(feature_types[0], str):
            if batch_size != 1:
                raise ValueError(
                    "routed_sandwich forward_batched() requires one feature_types list per task "
                    f"when batch_size={batch_size}"
                )
            resolved = [
                self._normalize_required_feature_types(
                    feature_types,
                    expected_count=num_features,
                    context="forward_batched.feature_types",
                )
            ]
            return self._feature_type_ids_from_resolved(resolved, device=device)
        if not isinstance(feature_types, list) or len(feature_types) != batch_size:
            raise ValueError(
                "routed_sandwich forward_batched() requires one feature_types list per task "
                f"when batch_size={batch_size}, got {type(feature_types).__name__}"
            )
        resolved_types_by_task = [
            self._normalize_required_feature_types(
                value,
                expected_count=num_features,
                context=f"forward_batched.feature_types[{index}]",
            )
            for index, value in enumerate(feature_types)
        ]
        return self._feature_type_ids_from_resolved(
            resolved_types_by_task,
            device=device,
        )

    def _feature_type_ids_from_metadata(
        self,
        metadata: dict[str, Any],
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        return feature_type_ids_from_task_metadata(
            metadata,
            batch_size=batch_size,
            num_features=num_features,
            device=device,
        )

    def _feature_cells(
        self,
        x_all: torch.Tensor,
        *,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
        apply_input_normalization: bool = True,
    ) -> torch.Tensor:
        return _feature_flow.feature_cells(
            self,
            x_all,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
            apply_input_normalization=apply_input_normalization,
        )

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
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def _apply(current_hidden: torch.Tensor) -> torch.Tensor:
            return block(current_hidden, attn_bias=attn_bias)

        return self._apply_activation_checkpoint(_apply, hidden)

    def _routed_cross_block(
        self,
        block: _RoutedCrossAttentionBlock,
        query_streams: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor, current_kv: torch.Tensor) -> torch.Tensor:
            return block(current_query, key_value=current_kv)

        return self._apply_activation_checkpoint(_apply, query_streams, key_value)

    def _routed_self_block(
        self,
        block: _RoutedSelfAttentionBlock,
        query_streams: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor) -> torch.Tensor:
            return block(current_query)

        return self._apply_activation_checkpoint(_apply, query_streams)

    def _row_feature_self_attention(
        self,
        block: _SelfAttentionBlock,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        return _feature_flow.row_feature_self_attention(self, block, feature_cells)

    def _column_row_isab(
        self,
        block: _InducedSetAttentionBlock,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        return _feature_flow.column_row_isab(self, block, feature_cells)

    def _pre_perceiver_cell_mixer(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _feature_flow.pre_perceiver_cell_mixer(self, feature_cells)

    def _validate_num_classes(self, num_classes: int) -> None:
        if num_classes < _MIN_CLASS_COUNT:
            raise RuntimeError(f"routed_sandwich requires at least 2 classes, got {num_classes}")
        if num_classes > self.many_class_base:
            raise RuntimeError(
                "routed_sandwich uses a direct multiclass head and requires "
                f"num_classes <= many_class_base={self.many_class_base}, got {num_classes}"
            )

    def _summary_query_attention(
        self,
        block: _CrossAttentionBlock,
        *,
        query: torch.Tensor,
        key_value: torch.Tensor,
        outer_count: int,
    ) -> torch.Tensor:
        batch_size, _outer, inner_count, embedding_size = (
            int(key_value.shape[0]),
            int(key_value.shape[1]),
            int(key_value.shape[2]),
            int(key_value.shape[3]),
        )
        query_count = int(query.shape[1])
        flat_kv = key_value.reshape(batch_size * outer_count, inner_count, embedding_size)
        flat_query = query.expand(batch_size * outer_count, -1, -1).to(
            device=key_value.device,
            dtype=key_value.dtype,
        )
        summaries = self._cross_block(block, flat_query, flat_kv)
        return summaries.reshape(batch_size, outer_count, query_count, embedding_size)

    def _full_cell_tokens(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_rows, num_features, _embedding_size = (
            int(feature_cells.shape[0]),
            int(feature_cells.shape[1]),
            int(feature_cells.shape[2]),
            int(feature_cells.shape[3]),
        )
        conditioned = self.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
            dtype=feature_cells.dtype
        )
        role_embed = _feature_flow.role_ids(
            batch_size=batch_size,
            num_rows=num_rows,
            num_train_rows=int(y_train.shape[1]),
            device=feature_cells.device,
        )
        role_embed = self.y_role_embedding(role_embed).to(dtype=feature_cells.dtype)
        token_type = self.token_type_embedding.weight[_CELL_TOKEN_ID].to(dtype=feature_cells.dtype)
        full_cell_token_grid = (
            feature_cells
            + conditioned.unsqueeze(2)
            + role_embed.unsqueeze(2)
            + token_type.view(1, 1, 1, -1)
        )
        self.trace_activation("post_full_cell_tokens", full_cell_token_grid)
        full_cell_stream = full_cell_token_grid.reshape(batch_size, num_rows * num_features, self.d_icl)
        self.trace_activation("post_full_cell_stream", full_cell_stream)
        return full_cell_stream

    def _row_summary_tokens(
        self,
        *,
        feature_cells: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        row_summaries = self._summary_query_attention(
            self.row_summary_builder,
            query=self.row_summary_query,
            key_value=feature_cells,
            outer_count=int(feature_cells.shape[1]),
        )
        self.trace_activation("post_row_summary", row_summaries)
        num_rows = int(feature_cells.shape[1])
        conditioned = self.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
            dtype=row_summaries.dtype
        )
        row_pos = self._fourier_positions(
            num_positions=num_rows,
            embedding_size=int(row_summaries.shape[3]),
            device=row_summaries.device,
            dtype=row_summaries.dtype,
        )
        role_embed = _feature_flow.role_ids(
            batch_size=int(row_summaries.shape[0]),
            num_rows=num_rows,
            num_train_rows=int(y_train.shape[1]),
            device=row_summaries.device,
        )
        role_embed = self.y_role_embedding(role_embed).to(dtype=row_summaries.dtype)
        token_type = self.token_type_embedding.weight[_ROW_SUMMARY_TOKEN_ID].to(
            dtype=row_summaries.dtype
        )
        tokens = (
            row_summaries
            + conditioned.unsqueeze(2)
            + row_pos.unsqueeze(2)
            + role_embed.unsqueeze(2)
            + token_type.view(1, 1, 1, -1)
        )
        flattened_tokens = tokens.reshape(
            int(tokens.shape[0]),
            num_rows * self.routed_row_summary_tokens,
            int(tokens.shape[3]),
        )
        self.trace_activation("post_row_summary_tokens", flattened_tokens)
        return flattened_tokens

    def _column_summary_tokens(self, feature_cells: torch.Tensor) -> torch.Tensor:
        column_major = feature_cells.transpose(1, 2).contiguous()
        column_summaries = self._summary_query_attention(
            self.column_summary_builder,
            query=self.column_summary_query,
            key_value=column_major,
            outer_count=int(column_major.shape[1]),
        )
        self.trace_activation("post_column_summary", column_summaries)
        token_type = self.token_type_embedding.weight[_COLUMN_SUMMARY_TOKEN_ID].to(
            dtype=column_summaries.dtype
        )
        tokens = column_summaries + token_type.view(1, 1, 1, -1)
        flattened_tokens = tokens.reshape(
            int(tokens.shape[0]),
            int(tokens.shape[1]) * self.routed_column_summary_tokens,
            int(tokens.shape[3]),
        )
        self.trace_activation("post_column_summary_tokens", flattened_tokens)
        return flattened_tokens

    def _evidence_tokens(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_cell_stream = self._full_cell_tokens(feature_cells, y_train=y_train)
        evidence_query = self.evidence_query.expand(int(full_cell_stream.shape[0]), -1, -1).to(
            device=full_cell_stream.device,
            dtype=full_cell_stream.dtype,
        )
        evidence_tokens = self._cross_block(self.evidence_builder, evidence_query, full_cell_stream)
        token_type = self.token_type_embedding.weight[_EVIDENCE_TOKEN_ID].to(
            dtype=evidence_tokens.dtype
        )
        evidence_tokens = evidence_tokens + token_type.view(1, 1, -1)
        self.trace_activation("post_evidence_tokens", evidence_tokens)
        return evidence_tokens, full_cell_stream

    def _build_classification_state(
        self,
        feature_state: SandwichFeatureState,
    ) -> _RoutedClassificationState:
        feature_cells = feature_state.feature_cells
        y_train = feature_state.raw_state.y_train
        row_tokens = self._row_summary_tokens(feature_cells=feature_cells, y_train=y_train)
        column_tokens = self._column_summary_tokens(feature_cells)
        evidence_tokens, full_cell_stream = self._evidence_tokens(feature_cells, y_train=y_train)
        context_bank = torch.cat([row_tokens, column_tokens, evidence_tokens], dim=1)
        if self.routed_direct_cell_bypass:
            stage0_input = torch.cat([full_cell_stream, context_bank], dim=1)
        else:
            stage0_input = context_bank
        self.trace_activation("post_context_bank", context_bank)
        self.trace_activation("post_stage0_input", stage0_input)
        return _RoutedClassificationState(
            feature_state=feature_state,
            full_cell_stream=full_cell_stream,
            context_bank=context_bank,
            stage0_input=stage0_input,
            row_tokens=row_tokens,
        )

    def _build_routed_query_streams(self, primary_tokens: torch.Tensor) -> torch.Tensor:
        return primary_tokens.unsqueeze(2) + self.test_query_seed.to(
            device=primary_tokens.device,
            dtype=primary_tokens.dtype,
        )

    def _pool_test_rows(
        self,
        *,
        latent_query_streams: torch.Tensor,
        value_streams: torch.Tensor,
        num_test_rows: int,
    ) -> torch.Tensor:
        query_primary = self.latent_memory_router.width_mix(latent_query_streams)
        value_primary = self.latent_memory_router.width_mix(value_streams)
        query_tokens = query_primary.reshape(
            int(query_primary.shape[0]),
            num_test_rows,
            self.routed_row_summary_tokens,
            self.d_icl,
        )
        value_tokens = value_primary.reshape(
            int(value_primary.shape[0]),
            num_test_rows,
            self.routed_row_summary_tokens,
            self.d_icl,
        )
        pool_query = query_tokens.mean(dim=2, keepdim=True)
        pool_query = pool_query + self.test_row_pool_query.view(1, 1, 1, self.d_icl).to(
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        flat_query = pool_query.reshape(int(query_tokens.shape[0]) * num_test_rows, 1, self.d_icl)
        flat_kv = value_tokens.reshape(
            int(value_tokens.shape[0]) * num_test_rows,
            self.routed_row_summary_tokens,
            self.d_icl,
        )
        pooled = self._cross_block(self.test_row_pool, flat_query, flat_kv)
        pooled = pooled.reshape(int(query_tokens.shape[0]), num_test_rows, self.d_icl)
        self.trace_activation("post_test_row_pool", pooled)
        return pooled

    def _forward_logits_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        resolved_num_classes = int(num_classes) if num_classes is not None else 2
        if num_classes is None:
            resolved_num_classes = max(2, int(y_train.max().item()) + 1)
        self._validate_num_classes(resolved_num_classes)
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=None,
            train_test_split_index=train_test_split_index,
            num_classes=resolved_num_classes,
            feature_type_ids=feature_type_ids,
        )
        feature_state = self._build_feature_state(raw_state)
        classification_state = self._build_classification_state(feature_state)
        latents = self.latent_seed.expand(int(x_all.shape[0]), -1, -1, -1)
        for index, stage in enumerate(self.perceiver_stages):
            stage = cast(_RoutedPerceiverStage, stage)
            key_value = (
                classification_state.stage0_input
                if index == 0
                else classification_state.context_bank
            )
            latents = self._routed_cross_block(stage.input_read, latents, key_value)
            self.trace_activation(f"post_stage_{index}_cross", latents)
            for self_index, self_block in enumerate(stage.self_blocks):
                self_block = cast(_RoutedSelfAttentionBlock, self_block)
                latents = self._routed_self_block(self_block, latents)
                self.trace_activation(f"post_stage_{index}_self_{self_index}", latents)
            self.trace_activation(f"post_stage_{index}_self", latents)
        batch_size = int(x_all.shape[0])
        num_rows = int(feature_state.feature_cells.shape[1])
        num_test_rows = num_rows - train_test_split_index
        row_token_grid = classification_state.row_tokens.reshape(
            batch_size,
            num_rows,
            self.routed_row_summary_tokens,
            self.d_icl,
        )
        test_queries = row_token_grid[:, train_test_split_index:, :, :].reshape(
            batch_size,
            num_test_rows * self.routed_row_summary_tokens,
            self.d_icl,
        )
        query_streams = self._build_routed_query_streams(test_queries)
        latent_memory = self.latent_memory_router.width_mix(latents)
        latent_readout_streams = self._routed_cross_block(
            self.latent_readout,
            query_streams,
            latent_memory,
        )
        self.trace_activation("post_latent_readout", latent_readout_streams)
        value_streams = latent_readout_streams
        if self.routed_direct_cell_bypass:
            value_streams = self._routed_cross_block(
                self.cell_readout,
                latent_readout_streams,
                classification_state.full_cell_stream,
            )
            self.trace_activation("post_cell_readout", value_streams)
        pooled_test_rows = self._pool_test_rows(
            latent_query_streams=latent_readout_streams,
            value_streams=value_streams,
            num_test_rows=num_test_rows,
        )
        return self.direct_head(pooled_test_rows)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
        feature_types: list[str] | list[list[str]],
    ) -> torch.Tensor:
        feature_type_ids = self._feature_type_ids_from_forward_batched(
            feature_types,
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

    def forward_classification(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        feature_type_ids = batch.feature_type_ids
        if feature_type_ids is None:
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
            num_classes=num_classes,
        )
        return ClassificationOutput(
            logits=flatten_classification_output_rows(logits),
            class_probs=None,
            num_classes=num_classes,
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return self.forward_classification(batch)
