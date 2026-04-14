"""Building blocks for the sandwich architecture family."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing import cast

from tab_foundry.model.components.attention import (
    _reshape_heads,
    multihead_attention_sdpa,
    scaled_dot_product_attention,
)
from tab_foundry.model.components.normalization import build_norm
from tab_foundry.model.components.rational import RationalActivation


SUPPORTED_SANDWICH_ACTIVATIONS = ("gelu", "rational")
SUPPORTED_SANDWICH_BLOCK_NORMS = ("layernorm", "none")


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


class _NativePackedSelfAttention(nn.Module):
    """Packed self-attention with explicit QKV projections."""

    def __init__(self, *, embedding_size: int, n_heads: int) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.embed_dim = int(embedding_size)
        self.n_heads = int(n_heads)
        self.num_heads = int(n_heads)
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.dropout = 0.0
        self.in_proj_weight = nn.Parameter(
            torch.empty(self.embedding_size * 3, self.embedding_size)
        )
        self.in_proj_bias = nn.Parameter(torch.empty(self.embedding_size * 3))
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size)
        reference = nn.MultiheadAttention(self.embedding_size, self.n_heads, batch_first=True)
        self.copy_from_multihead_attention(reference)

    def copy_from_multihead_attention(self, module: nn.MultiheadAttention) -> None:
        with torch.no_grad():
            self.in_proj_weight.copy_(module.in_proj_weight)
            if module.in_proj_bias is None:
                self.in_proj_bias.zero_()
            else:
                self.in_proj_bias.copy_(module.in_proj_bias)
            self.out_proj.weight.copy_(module.out_proj.weight)
            self.out_proj.bias.copy_(module.out_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_proj, k_proj, v_proj = F.linear(
            hidden,
            self.in_proj_weight,
            self.in_proj_bias,
        ).split(self.embedding_size, dim=-1)
        q_heads = _reshape_heads(q_proj, num_heads=self.n_heads)
        k_heads = _reshape_heads(k_proj, num_heads=self.n_heads)
        v_heads = _reshape_heads(v_proj, num_heads=self.n_heads)
        attn_out = scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_bias=attn_bias,
            dropout_p=self.dropout,
            training=self.training,
        )
        batch_size, _num_heads, target_len, head_dim = attn_out.shape
        merged = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, target_len, self.n_heads * head_dim)
        )
        return self.out_proj(merged)


class _NativePackedCrossAttention(nn.Module):
    """Packed cross-attention with explicit Q and fused KV projections."""

    def __init__(self, *, embedding_size: int, n_heads: int) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.embed_dim = int(embedding_size)
        self.n_heads = int(n_heads)
        self.num_heads = int(n_heads)
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.dropout = 0.0
        self.in_proj_weight = nn.Parameter(
            torch.empty(self.embedding_size * 3, self.embedding_size)
        )
        self.in_proj_bias = nn.Parameter(torch.empty(self.embedding_size * 3))
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size)
        reference = nn.MultiheadAttention(self.embedding_size, self.n_heads, batch_first=True)
        self.copy_from_multihead_attention(reference)

    def copy_from_multihead_attention(self, module: nn.MultiheadAttention) -> None:
        with torch.no_grad():
            self.in_proj_weight.copy_(module.in_proj_weight)
            self.in_proj_bias.copy_(module.in_proj_bias)
            self.out_proj.weight.copy_(module.out_proj.weight)
            self.out_proj.bias.copy_(module.out_proj.bias)

    def forward(self, query: torch.Tensor, *, key_value: torch.Tensor) -> torch.Tensor:
        q_proj = F.linear(
            query,
            self.in_proj_weight[: self.embedding_size],
            self.in_proj_bias[: self.embedding_size],
        )
        k_proj, v_proj = F.linear(
            key_value,
            self.in_proj_weight[self.embedding_size :],
            self.in_proj_bias[self.embedding_size :],
        ).split(self.embedding_size, dim=-1)
        q_heads = _reshape_heads(q_proj, num_heads=self.n_heads)
        k_heads = _reshape_heads(k_proj, num_heads=self.n_heads)
        v_heads = _reshape_heads(v_proj, num_heads=self.n_heads)
        attn_out = scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            dropout_p=self.dropout,
            training=self.training,
        )
        batch_size, _num_heads, target_len, head_dim = attn_out.shape
        merged = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, target_len, self.n_heads * head_dim)
        )
        return self.out_proj(merged)


class _CrossAttentionBlock(nn.Module):
    """Pre-norm residual cross-attention plus FFN."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.query_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.kv_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.packed_attention = bool(packed_attention)
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

    def forward(self, query: torch.Tensor, *, key_value: torch.Tensor) -> torch.Tensor:
        q_norm = self.query_norm(query)
        kv_norm = self.kv_norm(key_value)
        if self.packed_attention:
            if not isinstance(self.attn, _NativePackedCrossAttention):
                raise RuntimeError("packed cross-attention block is missing native attention")
            query = query + self.attn(q_norm, key_value=kv_norm)
        else:
            query = query + multihead_attention_sdpa(
                cast(nn.MultiheadAttention, self.attn),
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
        activation: str,
        block_norm: str,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.ff_norm = _build_sandwich_block_norm(block_norm, embedding_size)
        self.packed_attention = bool(packed_attention)
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
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_norm = self.attn_norm(hidden)
        if self.packed_attention:
            if not isinstance(self.attn, _NativePackedSelfAttention):
                raise RuntimeError("packed self-attention block is missing native attention")
            hidden = hidden + self.attn(hidden_norm, attn_bias=attn_bias)
        else:
            hidden = hidden + multihead_attention_sdpa(
                cast(nn.MultiheadAttention, self.attn),
                hidden_norm,
                hidden_norm,
                hidden_norm,
                attn_bias=attn_bias,
            )
        return hidden + self.ff(self.ff_norm(hidden))


class _PerceiverStage(nn.Module):
    """One unshared Perceiver stage: input read, then repeated latent self-attention."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        self_attention_per_cross: int,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.input_read = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            packed_attention=packed_attention,
        )
        self.self_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    activation=activation,
                    block_norm=block_norm,
                    packed_attention=packed_attention,
                )
                for _ in range(self_attention_per_cross)
            ]
        )


class _InducedSetAttentionBlock(nn.Module):
    """ISAB-style induced set mixer with learned inducing points."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_inducing: int,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.inducing_seed = nn.Parameter(torch.empty(1, num_inducing, embedding_size))
        _init_truncated_normal_(self.inducing_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.rows_to_inducing = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            packed_attention=packed_attention,
        )
        self.inducing_self = _SelfAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            packed_attention=packed_attention,
        )
        self.rows_from_inducing = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            packed_attention=packed_attention,
        )


def _build_sandwich_activation(activation: str) -> nn.Module:
    normalized = str(activation).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "rational":
        return RationalActivation(version="A", degrees=(5, 4), approx_func="gelu")
    raise ValueError(
        "sandwich_activation must be one of "
        f"{SUPPORTED_SANDWICH_ACTIVATIONS}, got {activation!r}"
    )


def _build_sandwich_block_norm(block_norm: str, embedding_size: int) -> nn.Module:
    normalized = str(block_norm).strip().lower()
    if normalized == "none":
        return nn.Identity()
    if normalized == "layernorm":
        return build_norm("layernorm", embedding_size)
    raise ValueError(
        "sandwich_block_norm must be one of "
        f"{SUPPORTED_SANDWICH_BLOCK_NORMS}, got {block_norm!r}"
    )
