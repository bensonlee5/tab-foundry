"""Reusable model blocks for the tabfoundry family."""

from __future__ import annotations

import torch
from torch import nn

from .normalization import build_norm
from .qass import QASSMultiheadAttention


def _reset_norm_module(module: nn.Module) -> None:
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()
        return
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    with torch.no_grad():
        if isinstance(weight, torch.Tensor):
            weight.fill_(1.0)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def _reinitialize_transformer_encoder_layer(layer: nn.TransformerEncoderLayer) -> None:
    # nn.TransformerEncoder clones one initialized layer N times, so we must
    # reinitialize each clone to avoid value-identical deep stacks.
    layer.self_attn._reset_parameters()
    layer.self_attn.out_proj.reset_parameters()
    layer.linear1.reset_parameters()
    layer.linear2.reset_parameters()
    _reset_norm_module(layer.norm1)
    _reset_norm_module(layer.norm2)


def _reinitialize_transformer_encoder(encoder: nn.TransformerEncoder) -> None:
    for layer in encoder.layers:
        _reinitialize_transformer_encoder_layer(layer)
    if encoder.norm is not None:
        _reset_norm_module(encoder.norm)


class ISABBlock(nn.Module):
    """Induced set attention block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        *,
        ff_expansion: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)

        self.norm_in = build_norm(norm_type, d_model)
        self.norm_mid = build_norm(norm_type, d_model)
        self.norm_out = build_norm(norm_type, d_model)

        self.attn_in_to_inducing = QASSMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_qass=True,
        )
        self.attn_inducing_to_in = QASSMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_qass=False,
        )
        ff_hidden = d_model * ff_expansion
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, n_context: int) -> torch.Tensor:
        # x: [B, N, D]
        inducing = self.inducing.expand(x.shape[0], -1, -1)
        h = self.norm_in(x)
        inducing = inducing + self.attn_in_to_inducing(
            inducing,
            h,
            h,
            n_context=n_context,
            force_qass=True,
        )
        inducing_mid = self.norm_mid(inducing)
        x = x + self.attn_inducing_to_in(
            h,
            inducing_mid,
            inducing_mid,
            n_context=n_context,
            force_qass=False,
        )
        x = x + self.ff(self.norm_out(x))
        return x


class TFColEncoder(nn.Module):
    """Column-wise set transformer stack."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        n_inducing: int = 128,
        norm_type: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ISABBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_inducing=n_inducing,
                    norm_type=norm_type,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B=m_columns, N=rows, D]
        h = x
        n_total = int(x.shape[1])
        for block in self.blocks:
            h = block(h, n_context=n_total)
        return h


class TFRowEncoder(nn.Module):
    """Row-wise transformer that aggregates feature tokens with CLS tokens."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        cls_tokens: int = 4,
        d_out: int = 512,
        ff_expansion: int = 2,
        norm_type: str = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cls_tokens = cls_tokens
        self.norm_type = str(norm_type).strip().lower()
        self._disable_transformer_fastpath = self.norm_type != "layernorm"
        self.cls = nn.Parameter(torch.randn(1, cls_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_expansion,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        setattr(encoder_layer, "norm1", build_norm(norm_type, d_model))
        setattr(encoder_layer, "norm2", build_norm(norm_type, d_model))
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=build_norm(norm_type, d_model),
            enable_nested_tensor=False,
        )
        if n_layers > 1:
            _reinitialize_transformer_encoder(self.encoder)
        self.out = nn.Linear(cls_tokens * d_model, d_out)

    def forward(
        self,
        per_row_features: torch.Tensor,
        *,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # per_row_features: [N, M, D]
        n_rows = per_row_features.shape[0]
        cls = self.cls.expand(n_rows, -1, -1)
        tokens = torch.cat([cls, per_row_features], dim=1)
        src_key_padding_mask: torch.Tensor | None = None
        if token_padding_mask is not None:
            if token_padding_mask.ndim != 1:
                raise ValueError("token_padding_mask must be 1D with shape [M]")
            if int(token_padding_mask.shape[0]) != int(per_row_features.shape[1]):
                raise ValueError(
                    "token_padding_mask shape mismatch with per_row_features"
                )
            cls_mask = torch.zeros(
                (n_rows, self.cls_tokens), dtype=torch.bool, device=tokens.device
            )
            feat_mask = token_padding_mask.to(
                device=tokens.device, dtype=torch.bool
            ).expand(n_rows, -1)
            src_key_padding_mask = torch.cat([cls_mask, feat_mask], dim=1)
        if self._disable_transformer_fastpath:
            previous_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                h = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
            finally:
                torch.backends.mha.set_fastpath_enabled(previous_fastpath)
        else:
            h = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        cls_out = h[:, : self.cls_tokens, :].reshape(n_rows, -1)
        return self.out(cls_out)
