"""Query-aware scalable softmax attention blocks."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from .normalization import build_norm


class QASSScaler(nn.Module):
    """Implements QASS scaling q * base(log n) * (1 + tanh(gate(q)))."""

    def __init__(self, n_heads: int, d_head: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.mlp_base = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, n_heads * d_head, bias=False),
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(d_head, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_head, bias=False),
        )
        final_layer = cast(nn.Linear, self.mlp_gate[-1])
        nn.init.zeros_(final_layer.weight)

    def forward(self, q: torch.Tensor, n_context: int) -> torch.Tensor:
        """Scale query tensor of shape [B, H, N, D]."""

        if n_context <= 1:
            return q
        log_n = math.log(float(n_context))
        base_in = torch.full((1, 1), log_n, device=q.device, dtype=q.dtype)
        base = self.mlp_base(base_in).view(1, self.n_heads, 1, self.d_head)
        gate = 1.0 + torch.tanh(self.mlp_gate(q))
        return q * base * gate


class QASSMultiheadAttention(nn.Module):
    """Multihead attention with optional QASS query scaling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        use_qass: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_qass = use_qass

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaler = QASSScaler(n_heads=n_heads, d_head=self.d_head)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        allowed_mask: torch.Tensor | None = None,
        n_context: int | None = None,
        force_qass: bool | None = None,
    ) -> torch.Tensor:
        q = self._reshape(self.q_proj(query))
        k = self._reshape(self.k_proj(key))
        v = self._reshape(self.v_proj(value))

        apply_qass = self.use_qass if force_qass is None else force_qass
        if apply_qass and n_context is not None:
            q = self.scaler(q, n_context=n_context)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if allowed_mask is not None:
            # allowed_mask: [B, 1, Nq, Nk] bool
            scores = scores.masked_fill(~allowed_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.d_model)
        )
        return self.out_proj(out)


class QASSTransformerLayer(nn.Module):
    """Pre-norm transformer layer with QASS-enabled MHA."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        ff_expansion: int = 2,
        dropout: float = 0.0,
        use_qass: bool = True,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.attn = QASSMultiheadAttention(
            d_model, n_heads, dropout=dropout, use_qass=use_qass
        )
        ff_hidden = d_model * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        allowed_mask: torch.Tensor | None = None,
        n_context: int | None = None,
        force_qass: bool | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(
            h,
            h,
            h,
            allowed_mask=allowed_mask,
            n_context=n_context,
            force_qass=force_qass,
        )
        x = x + self.ff(self.norm2(x))
        return x


class QASSTransformerEncoder(nn.Module):
    """Stacked QASS transformer encoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        ff_expansion: int = 2,
        dropout: float = 0.0,
        use_qass: bool = True,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QASSTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    dropout=dropout,
                    use_qass=use_qass,
                    norm_type=norm_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        allowed_mask: torch.Tensor | None = None,
        n_context: int | None = None,
        force_qass: bool | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, allowed_mask=allowed_mask, n_context=n_context, force_qass=force_qass
            )
        return self.final_norm(x)
