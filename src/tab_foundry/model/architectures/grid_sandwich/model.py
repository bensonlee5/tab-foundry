"""Grid-preserving sandwich classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from tab_foundry.feature_types import (
    FEATURE_TYPE_VOCAB,
    feature_type_ids_from_resolved,
    feature_type_ids_from_task_metadata,
    normalize_feature_types,
)
from tab_foundry.model.components.attention import _reshape_heads, multihead_attention_sdpa
from tab_foundry.model.components.tabular_primitives import (
    DirectMulticlassHead,
    FeatureTypeFiLM,
    LabelTokenTargetConditioner,
    ScalarPerFeatureMissingnessTokenizer,
    SharedLinearFeatureEncoder,
)
from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows
from tab_foundry.model.spec import GRID_SANDWICH_DEFAULTS as _D, ModelBuildSpec
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
_GRID_RESIDUAL_PRENORM = "prenorm"
_GRID_RESIDUAL_HYPER_CONNECTION_LITE = "hyper_connection_lite"
_GRID_ATTENTION_STANDARD = "standard"
_GRID_ATTENTION_DIFFERENTIAL = "differential"
_GRID_FFN_SWIGLU = "swiglu"
_GRID_HYPER_STREAMS = 2
_DIFFERENTIAL_ATTENTION_LAMBDA_INIT = 0.1
_SWIGLU_HIDDEN_MULTIPLE = 8
_GRID_INTERVENTION_NONE = "none"
_GRID_INTERVENTION_ABLATE_CHUNK = "ablate_chunk"
_GRID_INTERVENTION_REPEAT_CHUNK = "repeat_chunk"


@dataclass(frozen=True, slots=True)
class GridCoreIntervention:
    """Eval-only grid-core layer intervention for checkpoint diagnostics."""

    mode: str = _GRID_INTERVENTION_NONE
    start_layer: int | None = None
    end_layer: int | None = None
    repeat_count: int = 2


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def _swiglu_hidden_size(*, embedding_size: int, ff_expansion: int) -> int:
    raw_hidden = math.ceil((2.0 / 3.0) * float(ff_expansion) * float(embedding_size))
    return _round_up_to_multiple(int(raw_hidden), _SWIGLU_HIDDEN_MULTIPLE)


class _SwiGLUFFN(nn.Module):
    """Parameter-matched gated FFN for opt-in grid-core experiments."""

    def __init__(self, *, embedding_size: int, ff_expansion: int) -> None:
        super().__init__()
        hidden_size = _swiglu_hidden_size(
            embedding_size=embedding_size,
            ff_expansion=ff_expansion,
        )
        self.value = nn.Linear(embedding_size, hidden_size)
        self.gate = nn.Linear(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, embedding_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.value(hidden) * F.silu(self.gate(hidden)))


def _build_grid_ffn(
    *,
    embedding_size: int,
    ff_expansion: int,
    activation: str,
    ffn_mode: str,
) -> nn.Module:
    if str(ffn_mode).strip().lower() == _GRID_FFN_SWIGLU:
        return _SwiGLUFFN(embedding_size=embedding_size, ff_expansion=ff_expansion)
    ff_hidden = embedding_size * ff_expansion
    return nn.Sequential(
        nn.Linear(embedding_size, ff_hidden),
        _build_sandwich_activation(activation),
        nn.Linear(ff_hidden, embedding_size),
    )


class _GridAttentionCore(nn.Module):
    """Standard or differential multi-head attention for opt-in grid blocks."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        attention_mode: str,
        is_cross_attention: bool,
        packed_attention: bool,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.n_heads = int(n_heads)
        self.attention_mode = str(attention_mode).strip().lower()
        self.is_cross_attention = bool(is_cross_attention)
        self.packed_attention = bool(packed_attention)
        if self.embedding_size % self.n_heads != 0:
            raise ValueError(
                "grid attention requires embedding_size divisible by n_heads, "
                f"got embedding_size={self.embedding_size}, n_heads={self.n_heads}"
            )
        if self.attention_mode == _GRID_ATTENTION_DIFFERENTIAL:
            self.q1 = nn.Linear(self.embedding_size, self.embedding_size)
            self.k1 = nn.Linear(self.embedding_size, self.embedding_size)
            self.q2 = nn.Linear(self.embedding_size, self.embedding_size)
            self.k2 = nn.Linear(self.embedding_size, self.embedding_size)
            self.v = nn.Linear(self.embedding_size, self.embedding_size)
            self.out_proj = nn.Linear(self.embedding_size, self.embedding_size)
            self.lambda_scale = nn.Parameter(
                torch.tensor(float(_DIFFERENTIAL_ATTENTION_LAMBDA_INIT))
            )
        elif self.attention_mode == _GRID_ATTENTION_STANDARD:
            self.attn = (
                (
                    _NativePackedCrossAttention(
                        embedding_size=self.embedding_size,
                        n_heads=self.n_heads,
                    )
                    if self.is_cross_attention
                    else _NativePackedSelfAttention(
                        embedding_size=self.embedding_size,
                        n_heads=self.n_heads,
                    )
                )
                if self.packed_attention
                else nn.MultiheadAttention(
                    self.embedding_size,
                    self.n_heads,
                    batch_first=True,
                )
            )
        else:
            raise ValueError(
                "grid_attention_mode must be 'standard' or 'differential', "
                f"got {attention_mode!r}"
            )

    def forward(
        self,
        query: torch.Tensor,
        *,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention_mode == _GRID_ATTENTION_STANDARD:
            if self.packed_attention:
                if self.is_cross_attention:
                    return cast(_NativePackedCrossAttention, self.attn)(
                        query,
                        key_value=key_value,
                    )
                return cast(_NativePackedSelfAttention, self.attn)(
                    query,
                    attn_bias=attn_bias,
                )
            return multihead_attention_sdpa(
                cast(nn.MultiheadAttention, self.attn),
                query,
                key_value,
                key_value,
                attn_bias=attn_bias,
            )
        return self._differential_attention(
            query,
            key_value=key_value,
            attn_bias=attn_bias,
        )

    def _attention_weights(
        self,
        query_heads: torch.Tensor,
        key_heads: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        head_dim = int(query_heads.shape[-1])
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / math.sqrt(head_dim)
        if attn_bias is not None:
            scores = scores + attn_bias.to(device=scores.device, dtype=scores.dtype)
        return torch.softmax(scores, dim=-1)

    def _differential_attention(
        self,
        query: torch.Tensor,
        *,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        q1 = _reshape_heads(self.q1(query), num_heads=self.n_heads)
        k1 = _reshape_heads(self.k1(key_value), num_heads=self.n_heads)
        q2 = _reshape_heads(self.q2(query), num_heads=self.n_heads)
        k2 = _reshape_heads(self.k2(key_value), num_heads=self.n_heads)
        v = _reshape_heads(self.v(key_value), num_heads=self.n_heads)
        attn1 = self._attention_weights(q1, k1, attn_bias=attn_bias)
        attn2 = self._attention_weights(q2, k2, attn_bias=attn_bias)
        attended = torch.matmul(attn1 - (self.lambda_scale * attn2), v)
        batch_size, _num_heads, target_len, head_dim = attended.shape
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, target_len, self.n_heads * head_dim)
        )
        return self.out_proj(merged)


class _GridCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with opt-in grid attention and FFN modes."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        attention_mode: str,
        ffn_mode: str,
        packed_attention: bool,
    ) -> None:
        super().__init__()
        self.query_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.kv_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.attn = _GridAttentionCore(
            embedding_size=embedding_size,
            n_heads=n_heads,
            attention_mode=attention_mode,
            is_cross_attention=True,
            packed_attention=packed_attention,
        )
        self.ff = _build_grid_ffn(
            embedding_size=embedding_size,
            ff_expansion=ff_expansion,
            activation=activation,
            ffn_mode=ffn_mode,
        )

    def forward(self, query: torch.Tensor, *, key_value: torch.Tensor) -> torch.Tensor:
        q_norm = self.query_norm(query)
        kv_norm = self.kv_norm(key_value)
        query = query + self.attn(q_norm, key_value=kv_norm)
        return query + self.ff(self.ff_norm(query))


class _GridSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with opt-in grid attention and FFN modes."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        attention_mode: str,
        ffn_mode: str,
        packed_attention: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.attn = _GridAttentionCore(
            embedding_size=embedding_size,
            n_heads=n_heads,
            attention_mode=attention_mode,
            is_cross_attention=False,
            packed_attention=packed_attention,
        )
        self.ff = _build_grid_ffn(
            embedding_size=embedding_size,
            ff_expansion=ff_expansion,
            activation=activation,
            ffn_mode=ffn_mode,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_norm = self.attn_norm(hidden)
        hidden = hidden + self.attn(hidden_norm, key_value=hidden_norm, attn_bias=attn_bias)
        return hidden + self.ff(self.ff_norm(hidden))


class _GridInducedSetAttentionBlock(nn.Module):
    """ISAB mixer using the opt-in grid self/cross-attention blocks."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_inducing: int,
        attention_mode: str,
        ffn_mode: str,
        packed_attention: bool,
    ) -> None:
        super().__init__()
        self.inducing_seed = nn.Parameter(torch.empty(1, num_inducing, embedding_size))
        _init_truncated_normal_(self.inducing_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.rows_to_inducing = _GridCrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            attention_mode=attention_mode,
            ffn_mode=ffn_mode,
            packed_attention=packed_attention,
        )
        self.inducing_self = _GridSelfAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            attention_mode=attention_mode,
            ffn_mode=ffn_mode,
            packed_attention=packed_attention,
        )
        self.rows_from_inducing = _GridCrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            attention_mode=attention_mode,
            ffn_mode=ffn_mode,
            packed_attention=packed_attention,
        )


class _GridHyperConnection(nn.Module):
    """Input-dependent width/depth mixing over two grid residual streams."""

    def __init__(self, *, embedding_size: int, num_streams: int = _GRID_HYPER_STREAMS) -> None:
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


def _use_experimental_grid_blocks(*, attention_mode: str, ffn_mode: str) -> bool:
    return (
        str(attention_mode).strip().lower() != _GRID_ATTENTION_STANDARD
        or str(ffn_mode).strip().lower() == _GRID_FFN_SWIGLU
    )


class _GridMixerLayer(nn.Module):
    """Alternate row-wise and column-wise mixing while preserving the cell grid."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_inducing: int,
        residual_mode: str,
        attention_mode: str,
        ffn_mode: str,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.residual_mode = str(residual_mode).strip().lower()
        self.attention_mode = str(attention_mode).strip().lower()
        self.ffn_mode = str(ffn_mode).strip().lower()
        self.row_mixer: nn.Module
        self.column_mixer: nn.Module
        if _use_experimental_grid_blocks(
            attention_mode=self.attention_mode,
            ffn_mode=self.ffn_mode,
        ):
            self.row_mixer = _GridSelfAttentionBlock(
                embedding_size=embedding_size,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                activation=activation,
                block_norm=block_norm,
                attention_mode=self.attention_mode,
                ffn_mode=self.ffn_mode,
                packed_attention=packed_attention,
            )
            self.column_mixer = _GridInducedSetAttentionBlock(
                embedding_size=embedding_size,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                activation=activation,
                block_norm=block_norm,
                num_inducing=num_inducing,
                attention_mode=self.attention_mode,
                ffn_mode=self.ffn_mode,
                packed_attention=packed_attention,
            )
        else:
            self.row_mixer = _SelfAttentionBlock(
                embedding_size=embedding_size,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                activation=activation,
                block_norm=block_norm,
                packed_attention=packed_attention,
            )
            self.column_mixer = _InducedSetAttentionBlock(
                embedding_size=embedding_size,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                activation=activation,
                block_norm=block_norm,
                num_inducing=num_inducing,
                packed_attention=packed_attention,
            )
        self.row_router = (
            _GridHyperConnection(embedding_size=embedding_size)
            if self.residual_mode == _GRID_RESIDUAL_HYPER_CONNECTION_LITE
            else None
        )
        self.column_router = (
            _GridHyperConnection(embedding_size=embedding_size)
            if self.residual_mode == _GRID_RESIDUAL_HYPER_CONNECTION_LITE
            else None
        )


class GridSandwichClassifier(nn.Module):
    """Classification-only grid-preserving sandwich classifier."""

    def __init__(
        self,
        *,
        d_icl: int = _D["d_icl"],
        input_normalization: str = _D["input_normalization"],
        many_class_base: int = _D["many_class_base"],
        norm_type: str = _D["norm_type"],
        head_hidden_dim: int = _D["head_hidden_dim"],
        pre_encoder_clip: float | None = _D["pre_encoder_clip"],
        sandwich_layers: int = _D["sandwich_layers"],
        sandwich_heads: int = _D["sandwich_heads"],
        sandwich_ff_expansion: int = _D["sandwich_ff_expansion"],
        sandwich_activation: str = _D["sandwich_activation"],
        sandwich_block_norm: str = _D["sandwich_block_norm"],
        sandwich_pre_row_attention_layers: int = _D["sandwich_pre_row_attention_layers"],
        sandwich_pre_column_attention_layers: int = _D["sandwich_pre_column_attention_layers"],
        sandwich_pre_column_inducing_tokens: int = _D["sandwich_pre_column_inducing_tokens"],
        sandwich_packed_attention: bool = _D["sandwich_packed_attention"],
        feature_type_conditioning: str = _D["feature_type_conditioning"],
        grid_residual_mode: str = _D["grid_residual_mode"],
        grid_attention_mode: str = _D["grid_attention_mode"],
        grid_ffn_mode: str = _D["grid_ffn_mode"],
        grid_recurrence_steps: int | None = _D["grid_recurrence_steps"],
        grid_recurrence_unique_layers: int | None = _D["grid_recurrence_unique_layers"],
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
            arch="grid_sandwich",
            d_icl=d_icl,
            input_normalization=input_normalization,
            many_class_base=many_class_base,
            norm_type=norm_type,
            head_hidden_dim=head_hidden_dim,
            pre_encoder_clip=pre_encoder_clip,
            sandwich_layers=sandwich_layers,
            sandwich_heads=sandwich_heads,
            sandwich_ff_expansion=sandwich_ff_expansion,
            sandwich_activation=sandwich_activation,
            sandwich_block_norm=sandwich_block_norm,
            sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
            sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
            sandwich_pre_column_inducing_tokens=sandwich_pre_column_inducing_tokens,
            sandwich_packed_attention=sandwich_packed_attention,
            feature_type_conditioning=feature_type_conditioning,
            grid_residual_mode=grid_residual_mode,
            grid_attention_mode=grid_attention_mode,
            grid_ffn_mode=grid_ffn_mode,
            grid_recurrence_steps=grid_recurrence_steps,
            grid_recurrence_unique_layers=grid_recurrence_unique_layers,
        )
        self.arch = "grid_sandwich"
        self.loss_surface = _CLASSIFICATION_LOSS_SURFACE
        self.d_icl = int(self.model_spec.d_icl)
        self.input_normalization = str(self.model_spec.input_normalization).strip().lower()
        self.many_class_base = int(self.model_spec.many_class_base)
        self.norm_type = str(self.model_spec.norm_type).strip().lower()
        self.head_hidden_dim = int(self.model_spec.head_hidden_dim)
        self.pre_encoder_clip = self.model_spec.pre_encoder_clip
        self.sandwich_layers = int(self.model_spec.sandwich_layers)
        self.sandwich_heads = int(self.model_spec.sandwich_heads)
        self.sandwich_ff_expansion = int(self.model_spec.sandwich_ff_expansion)
        self.sandwich_activation = str(self.model_spec.sandwich_activation).strip().lower()
        self.sandwich_block_norm = str(self.model_spec.sandwich_block_norm).strip().lower()
        self.pre_row_attention_layers = int(self.model_spec.sandwich_pre_row_attention_layers)
        self.pre_column_attention_layers = int(
            self.model_spec.sandwich_pre_column_attention_layers
        )
        self.pre_column_inducing_tokens = int(self.model_spec.sandwich_pre_column_inducing_tokens)
        self.sandwich_packed_attention = bool(self.model_spec.sandwich_packed_attention)
        self.feature_type_conditioning = (
            str(self.model_spec.feature_type_conditioning).strip().lower()
        )
        self.grid_residual_mode = str(self.model_spec.grid_residual_mode).strip().lower()
        self.grid_attention_mode = str(self.model_spec.grid_attention_mode).strip().lower()
        self.grid_ffn_mode = str(self.model_spec.grid_ffn_mode).strip().lower()
        self.grid_recurrence_steps = (
            None
            if self.model_spec.grid_recurrence_steps is None
            else int(self.model_spec.grid_recurrence_steps)
        )
        self.grid_recurrence_unique_layers = (
            None
            if self.model_spec.grid_recurrence_unique_layers is None
            else int(self.model_spec.grid_recurrence_unique_layers)
        )
        self.grid_core_iterations = int(self.grid_recurrence_steps or self.sandwich_layers)
        self.grid_core_unique_layers = int(
            self.grid_recurrence_unique_layers
            or (1 if self.grid_recurrence_steps is not None else self.sandwich_layers)
        )
        if self.norm_type != "layernorm":
            raise ValueError(
                "grid_sandwich currently requires norm_type='layernorm', "
                f"got {self.norm_type!r}"
            )

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
        grid_layer_count = int(self.grid_core_unique_layers)
        self.grid_layers = nn.ModuleList(
            [
                _GridMixerLayer(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    num_inducing=self.pre_column_inducing_tokens,
                    residual_mode=self.grid_residual_mode,
                    attention_mode=self.grid_attention_mode,
                    ffn_mode=self.grid_ffn_mode,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(grid_layer_count)
            ]
        )
        self.y_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.y_role_embedding = nn.Embedding(2, self.d_icl)
        self.row_pool_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.row_pool = _CrossAttentionBlock(
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

        self._activation_checkpointing_enabled = False
        self._activation_trace: dict[str, tuple[float, int]] | None = None
        self._grid_core_intervention = GridCoreIntervention()
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
                "grid_sandwich only supports loss_surface='classification', "
                f"got {loss_surface!r}"
            )
        self.loss_surface = normalized

    def clear_grid_core_intervention(self) -> None:
        """Clear eval-only grid-core perturbations."""

        self._grid_core_intervention = GridCoreIntervention()

    def set_grid_core_intervention(
        self,
        *,
        mode: str,
        start_layer: int | None = None,
        end_layer: int | None = None,
        repeat_count: int = 2,
    ) -> None:
        """Set an eval-only contiguous grid-layer perturbation."""

        normalized_mode = str(mode).strip().lower()
        if normalized_mode == _GRID_INTERVENTION_NONE:
            self.clear_grid_core_intervention()
            return
        if normalized_mode not in {
            _GRID_INTERVENTION_ABLATE_CHUNK,
            _GRID_INTERVENTION_REPEAT_CHUNK,
        }:
            raise ValueError(
                "grid core intervention mode must be one of "
                f"{[_GRID_INTERVENTION_NONE, _GRID_INTERVENTION_ABLATE_CHUNK, _GRID_INTERVENTION_REPEAT_CHUNK]}, "
                f"got {mode!r}"
            )
        if self.grid_recurrence_steps is not None:
            raise ValueError(
                "grid core interventions require a non-recurrent grid_sandwich checkpoint "
                "with distinct grid mixer layers"
            )
        if start_layer is None or end_layer is None:
            raise ValueError("grid core interventions require start_layer and end_layer")
        start = int(start_layer)
        end = int(end_layer)
        layer_count = len(self.grid_layers)
        if start < 0 or end < start or end >= layer_count:
            raise ValueError(
                "grid core intervention layer range must satisfy "
                f"0 <= start_layer <= end_layer < {layer_count}, "
                f"got start_layer={start}, end_layer={end}"
            )
        repeats = int(repeat_count)
        if normalized_mode == _GRID_INTERVENTION_REPEAT_CHUNK and repeats <= 0:
            raise ValueError("repeat_chunk grid core interventions require repeat_count > 0")
        self._grid_core_intervention = GridCoreIntervention(
            mode=normalized_mode,
            start_layer=start,
            end_layer=end,
            repeat_count=repeats,
        )

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
        return _shared_forward.task_num_classes(batch, arch_name="grid_sandwich")

    @staticmethod
    def _prepare_task_inputs(
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return _shared_forward.prepare_task_inputs(batch, arch_name="grid_sandwich")

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
            raise ValueError(f"{context} is required for grid_sandwich")
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
            raise ValueError("grid_sandwich forward_batched() requires explicit feature_types")
        if not feature_types or isinstance(feature_types[0], str):
            if batch_size != 1:
                raise ValueError(
                    "grid_sandwich forward_batched() requires one feature_types list per task "
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
                "grid_sandwich forward_batched() requires one feature_types list per task "
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
        block: nn.Module,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor, current_kv: torch.Tensor) -> torch.Tensor:
            return block(current_query, key_value=current_kv)

        return self._apply_activation_checkpoint(_apply, query, key_value)

    def _self_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def _apply(current_hidden: torch.Tensor) -> torch.Tensor:
            return block(current_hidden, attn_bias=attn_bias)

        return self._apply_activation_checkpoint(_apply, hidden)

    def _row_feature_self_attention(
        self,
        block: nn.Module,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_rows, num_features, embedding_size = (
            int(feature_cells.shape[0]),
            int(feature_cells.shape[1]),
            int(feature_cells.shape[2]),
            int(feature_cells.shape[3]),
        )
        row_major = feature_cells.reshape(batch_size * num_rows, num_features, embedding_size)
        mixed = self._self_block(block, row_major)
        return mixed.reshape(batch_size, num_rows, num_features, embedding_size)

    def _induced_set_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        inducing_seed = cast(torch.Tensor, getattr(block, "inducing_seed"))
        rows_to_inducing = cast(nn.Module, getattr(block, "rows_to_inducing"))
        inducing_self = cast(nn.Module, getattr(block, "inducing_self"))
        rows_from_inducing = cast(nn.Module, getattr(block, "rows_from_inducing"))
        inducing = inducing_seed.expand(int(hidden.shape[0]), -1, -1).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        inducing = self._cross_block(rows_to_inducing, inducing, hidden)
        inducing = self._self_block(inducing_self, inducing)
        return self._cross_block(rows_from_inducing, hidden, inducing)

    def _column_row_isab(
        self,
        block: nn.Module,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_rows, num_features, embedding_size = (
            int(feature_cells.shape[0]),
            int(feature_cells.shape[1]),
            int(feature_cells.shape[2]),
            int(feature_cells.shape[3]),
        )
        column_major = feature_cells.transpose(1, 2).contiguous()
        column_major = column_major.reshape(batch_size * num_features, num_rows, embedding_size)
        mixed = self._induced_set_block(block, column_major)
        mixed = mixed.reshape(batch_size, num_features, num_rows, embedding_size)
        return mixed.transpose(1, 2).contiguous()

    def _initialize_grid_streams(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return feature_cells.unsqueeze(3).expand(-1, -1, -1, _GRID_HYPER_STREAMS, -1).contiguous()

    def _row_feature_self_attention_streams(
        self,
        layer: _GridMixerLayer,
        feature_streams: torch.Tensor,
    ) -> torch.Tensor:
        if layer.row_router is None:
            raise RuntimeError("grid hyper-connection row router is missing")
        batch_size, num_rows, num_features, num_streams, embedding_size = (
            int(feature_streams.shape[0]),
            int(feature_streams.shape[1]),
            int(feature_streams.shape[2]),
            int(feature_streams.shape[3]),
            int(feature_streams.shape[4]),
        )
        row_major = feature_streams.reshape(
            batch_size * num_rows,
            num_features,
            num_streams,
            embedding_size,
        )
        primary = layer.row_router.width_mix(row_major)
        mixed = self._self_block(layer.row_mixer, primary)
        mixed_streams = layer.row_router.depth_mix(row_major, mixed)
        return mixed_streams.reshape(
            batch_size,
            num_rows,
            num_features,
            num_streams,
            embedding_size,
        )

    def _column_row_isab_streams(
        self,
        layer: _GridMixerLayer,
        feature_streams: torch.Tensor,
    ) -> torch.Tensor:
        if layer.column_router is None:
            raise RuntimeError("grid hyper-connection column router is missing")
        batch_size, num_rows, num_features, num_streams, embedding_size = (
            int(feature_streams.shape[0]),
            int(feature_streams.shape[1]),
            int(feature_streams.shape[2]),
            int(feature_streams.shape[3]),
            int(feature_streams.shape[4]),
        )
        column_major = feature_streams.transpose(1, 2).contiguous()
        column_major = column_major.reshape(
            batch_size * num_features,
            num_rows,
            num_streams,
            embedding_size,
        )
        primary = layer.column_router.width_mix(column_major)
        mixed = self._induced_set_block(layer.column_mixer, primary)
        mixed_streams = layer.column_router.depth_mix(column_major, mixed)
        mixed_streams = mixed_streams.reshape(
            batch_size,
            num_features,
            num_rows,
            num_streams,
            embedding_size,
        )
        return mixed_streams.transpose(1, 2).contiguous()

    def _grid_core_layer_indices(self) -> tuple[int, ...]:
        intervention = self._grid_core_intervention
        if intervention.mode == _GRID_INTERVENTION_NONE:
            return tuple(range(self.grid_core_iterations))
        if self.grid_recurrence_steps is not None:
            raise RuntimeError("grid core interventions are not supported for recurrent checkpoints")
        start = int(cast(int, intervention.start_layer))
        end = int(cast(int, intervention.end_layer))
        if intervention.mode == _GRID_INTERVENTION_ABLATE_CHUNK:
            return tuple(
                index
                for index in range(len(self.grid_layers))
                if index < start or index > end
            )
        if intervention.mode == _GRID_INTERVENTION_REPEAT_CHUNK:
            before = tuple(range(0, start))
            chunk = tuple(range(start, end + 1))
            after = tuple(range(end + 1, len(self.grid_layers)))
            repeated = tuple(
                layer_index
                for _repeat_index in range(int(intervention.repeat_count))
                for layer_index in chunk
            )
            return before + repeated + after
        raise RuntimeError(f"unsupported grid core intervention mode: {intervention.mode!r}")

    def _grid_core_layer(self, logical_index: int) -> _GridMixerLayer:
        return cast(
            _GridMixerLayer,
            self.grid_layers[int(logical_index) % len(self.grid_layers)]
            if self.grid_recurrence_steps is not None
            else self.grid_layers[int(logical_index)],
        )

    def _pre_perceiver_cell_mixer(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _feature_flow.pre_perceiver_cell_mixer(self, feature_cells)

    def _validate_num_classes(self, num_classes: int) -> None:
        if num_classes < _MIN_CLASS_COUNT:
            raise RuntimeError(f"grid_sandwich requires at least 2 classes, got {num_classes}")
        if num_classes > self.many_class_base:
            raise RuntimeError(
                "grid_sandwich uses a direct multiclass head and requires "
                f"num_classes <= many_class_base={self.many_class_base}, got {num_classes}"
            )

    def _label_conditioned_cells(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = int(feature_cells.shape[1])
        conditioned = self.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
            dtype=feature_cells.dtype
        )
        conditioned[:, int(y_train.shape[1]) :, :] = 0.0
        role_ids = _feature_flow.role_ids(
            batch_size=int(feature_cells.shape[0]),
            num_rows=num_rows,
            num_train_rows=int(y_train.shape[1]),
            device=feature_cells.device,
        )
        role_embed = self.y_role_embedding(role_ids).to(dtype=feature_cells.dtype)
        conditioned_cells = feature_cells + conditioned.unsqueeze(2) + role_embed.unsqueeze(2)
        self.trace_activation("post_label_conditioned_cells", conditioned_cells)
        return conditioned_cells

    def _pool_test_rows(self, feature_cells: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        test_rows = feature_cells[:, train_test_split_index:, :, :]
        batch_size, num_test_rows, num_features, embedding_size = (
            int(test_rows.shape[0]),
            int(test_rows.shape[1]),
            int(test_rows.shape[2]),
            int(test_rows.shape[3]),
        )
        flat_rows = test_rows.reshape(batch_size * num_test_rows, num_features, embedding_size)
        row_query = self.row_pool_query.expand(batch_size * num_test_rows, -1, -1).to(
            device=test_rows.device,
            dtype=test_rows.dtype,
        )
        pooled = self._cross_block(self.row_pool, row_query, flat_rows)
        pooled = pooled.reshape(batch_size, num_test_rows, embedding_size)
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
        hidden = self._label_conditioned_cells(feature_state.feature_cells, y_train=y_train)
        if self.grid_residual_mode == _GRID_RESIDUAL_HYPER_CONNECTION_LITE:
            streams = self._initialize_grid_streams(hidden)
            for index in self._grid_core_layer_indices():
                layer = self._grid_core_layer(index)
                streams = self._row_feature_self_attention_streams(layer, streams)
                self.trace_activation(f"post_grid_row_mixer_{index}", streams.mean(dim=3))
                streams = self._column_row_isab_streams(layer, streams)
                self.trace_activation(f"post_grid_column_mixer_{index}", streams.mean(dim=3))
            hidden = streams.mean(dim=3)
            self.trace_activation("post_grid_hyper_connection_collapse", hidden)
        else:
            for index in self._grid_core_layer_indices():
                layer = self._grid_core_layer(index)
                hidden = self._row_feature_self_attention(layer.row_mixer, hidden)
                self.trace_activation(f"post_grid_row_mixer_{index}", hidden)
                hidden = self._column_row_isab(layer.column_mixer, hidden)
                self.trace_activation(f"post_grid_column_mixer_{index}", hidden)
        pooled_test_rows = self._pool_test_rows(
            hidden,
            train_test_split_index=train_test_split_index,
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
