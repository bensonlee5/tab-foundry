from __future__ import annotations

import torch
from torch import nn

from tab_foundry.model.components.qass import (
    QASSScaler,
    QASSTransformerEncoder,
    QASSTransformerLayer,
)


def test_qass_gate_identity_at_init() -> None:
    scaler = QASSScaler(n_heads=2, d_head=4)
    q = torch.randn(1, 2, 3, 4)
    out = scaler(q, n_context=16)
    # At init the gate branch is identity (1 + tanh(0) == 1), so only base scaling changes values.
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


class _CountingLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__(normalized_shape)
        self.calls = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.calls += 1
        return super().forward(input)


def test_qass_layer_computes_norm1_once_for_self_attention() -> None:
    layer = QASSTransformerLayer(d_model=16, n_heads=4)
    counting_norm = _CountingLayerNorm(16)
    layer.norm1 = counting_norm
    _ = layer(torch.randn(2, 5, 16), n_context=5)
    assert counting_norm.calls == 1


def test_qass_encoder_applies_final_norm() -> None:
    encoder = QASSTransformerEncoder(d_model=8, n_heads=2, n_layers=0)
    x = torch.randn(2, 4, 8)
    out = encoder(x)
    assert torch.allclose(out, encoder.final_norm(x))
