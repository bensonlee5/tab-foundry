from __future__ import annotations

import torch
from torch import nn

from tab_foundry.model.components.blocks import ISABBlock, TFColEncoder, TFRowEncoder


def test_isab_block_keeps_raw_residual_stream() -> None:
    block = ISABBlock(d_model=8, n_heads=2, n_inducing=4)
    for module in [block.attn_in_to_inducing, block.attn_inducing_to_in, block.ff]:
        for param in module.parameters():
            nn.init.zeros_(param)

    x = torch.randn(3, 6, 8)
    out = block(x, n_context=6)
    assert torch.allclose(out, x, atol=1e-6)


def test_tfcol_uses_total_rows_as_context_length() -> None:
    encoder = TFColEncoder(d_model=8, n_heads=2, n_layers=1, n_inducing=4)

    class _CaptureBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_context: int | None = None

        def forward(self, x: torch.Tensor, *, n_context: int) -> torch.Tensor:
            self.last_context = n_context
            return x

    capture = _CaptureBlock()
    encoder.blocks = nn.ModuleList([capture])
    _ = encoder(torch.randn(5, 11, 8))
    assert capture.last_context == 11


def test_tfrow_encoder_configures_final_norm() -> None:
    encoder = TFRowEncoder(d_model=8, n_heads=2, n_layers=1, cls_tokens=2, d_out=16)
    assert encoder.encoder.norm is not None


def test_tfrow_encoder_accepts_token_padding_mask() -> None:
    encoder = TFRowEncoder(d_model=8, n_heads=2, n_layers=1, cls_tokens=2, d_out=16)
    per_row = torch.randn(3, 5, 8)
    mask = torch.tensor([False, False, True, False, True], dtype=torch.bool)
    out = encoder(per_row, token_padding_mask=mask)
    assert out.shape == (3, 16)


def test_tfrow_encoder_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(0)
    encoder = TFRowEncoder(d_model=16, n_heads=4, n_layers=3, cls_tokens=2, d_out=16)
    layers = list(encoder.encoder.layers)
    assert len(layers) == 3

    for param_name in (
        "self_attn.in_proj_weight",
        "self_attn.out_proj.weight",
        "linear1.weight",
        "linear2.weight",
    ):
        layer0_param = dict(layers[0].named_parameters())[param_name]
        layer1_param = dict(layers[1].named_parameters())[param_name]
        layer2_param = dict(layers[2].named_parameters())[param_name]

        assert layer0_param.data_ptr() != layer1_param.data_ptr()
        assert layer1_param.data_ptr() != layer2_param.data_ptr()
        assert not torch.equal(layer0_param, layer1_param)
        assert not torch.equal(layer1_param, layer2_param)
