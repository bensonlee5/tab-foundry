"""Building blocks for the sandwich architecture family."""

from __future__ import annotations

import torch
from torch import nn

from tab_foundry.model.components.attention import multihead_attention_sdpa
from tab_foundry.model.components.normalization import build_norm


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

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_norm = self.attn_norm(hidden)
        hidden = hidden + multihead_attention_sdpa(
            self.attn,
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
        norm_type: str,
        self_attention_per_cross: int,
    ) -> None:
        super().__init__()
        self.input_read = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )
        self.self_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    norm_type=norm_type,
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
        norm_type: str,
        num_inducing: int,
    ) -> None:
        super().__init__()
        self.inducing_seed = nn.Parameter(torch.empty(1, num_inducing, embedding_size))
        _init_truncated_normal_(self.inducing_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.rows_to_inducing = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )
        self.inducing_self = _SelfAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )
        self.rows_from_inducing = _CrossAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            norm_type=norm_type,
        )
