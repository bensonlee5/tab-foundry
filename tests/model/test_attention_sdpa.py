from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tab_foundry.model.architectures.tabfoundry_staged.subsystems import PreNormCellBlock
from tab_foundry.model.components.attention import (
    attention_bias_from_allowed_mask,
    multihead_attention_sdpa,
    packed_projection_multihead_attention_sdpa,
)
from tab_foundry.model.components.qass import QASSMultiheadAttention


def _allowed_mask(*, batch_size: int, target_len: int, source_len: int) -> torch.Tensor:
    allowed = torch.ones((batch_size, 1, target_len, source_len), dtype=torch.bool)
    allowed[:, :, -1, -1] = False
    return allowed


def _row_mask_reference(
    *,
    n_total: int,
    n_train: int,
    allow_test_self_attention: bool,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((n_total, n_total), device=device, dtype=torch.float32)
    if n_train >= n_total:
        return mask
    test_slice = slice(n_train, n_total)
    mask[test_slice, test_slice] = float("-inf")
    if allow_test_self_attention:
        diag = torch.arange(n_train, n_total, device=device)
        mask[diag, diag] = 0.0
    return mask


def _qass_reference(
    module: QASSMultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None,
    n_context: int | None,
    force_qass: bool | None,
) -> torch.Tensor:
    q = module._reshape(module.q_proj(query))
    k = module._reshape(module.k_proj(key))
    v = module._reshape(module.v_proj(value))

    apply_qass = module.use_qass if force_qass is None else force_qass
    if apply_qass and n_context is not None:
        q = module.scaler(q, n_context=n_context)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(module.d_head)
    if allowed_mask is not None:
        scores = scores.masked_fill(~allowed_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = module.dropout(attn)
    out = torch.matmul(attn, v)
    out = (
        out.permute(0, 2, 1, 3)
        .contiguous()
        .view(query.shape[0], query.shape[1], module.d_model)
    )
    return module.out_proj(out)


def test_multihead_attention_sdpa_matches_torch_module_without_mask() -> None:
    torch.manual_seed(1)
    module = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0)
    query = torch.randn(2, 5, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)

    expected = module(query, key, value, need_weights=False)[0]
    observed = multihead_attention_sdpa(module, query, key, value)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_multihead_attention_sdpa_matches_torch_module_with_additive_bias() -> None:
    torch.manual_seed(2)
    module = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0)
    query = torch.randn(2, 5, 16)
    allowed_mask = torch.ones((5, 5), dtype=torch.bool)
    allowed_mask[-1, -1] = False
    attn_bias = attention_bias_from_allowed_mask(allowed_mask, dtype=query.dtype)

    expected = module(
        query,
        query,
        query,
        attn_mask=attn_bias,
        need_weights=False,
    )[0]
    observed = multihead_attention_sdpa(
        module,
        query,
        query,
        query,
        attn_bias=attn_bias,
    )

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_packed_projection_multihead_attention_sdpa_matches_self_attention_path() -> None:
    torch.manual_seed(7)
    module = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0)
    query = torch.randn(2, 5, 16)
    allowed_mask = torch.ones((5, 5), dtype=torch.bool)
    allowed_mask[-1, -1] = False
    attn_bias = attention_bias_from_allowed_mask(allowed_mask, dtype=query.dtype)

    expected = multihead_attention_sdpa(
        module,
        query,
        query,
        query,
        attn_bias=attn_bias,
    )
    observed = packed_projection_multihead_attention_sdpa(
        module,
        query,
        query,
        query,
        attn_bias=attn_bias,
    )

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_packed_projection_multihead_attention_sdpa_matches_cross_attention_path() -> None:
    torch.manual_seed(8)
    module = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0)
    query = torch.randn(2, 3, 16)
    key_value = torch.randn(2, 5, 16)

    expected = multihead_attention_sdpa(module, query, key_value, key_value)
    observed = packed_projection_multihead_attention_sdpa(module, query, key_value, key_value)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("allow_test_self_attention", [False, True])
def test_prenorm_cell_block_row_attention_matches_reference(
    allow_test_self_attention: bool,
) -> None:
    torch.manual_seed(3)
    block = PreNormCellBlock(
        embedding_size=16,
        nhead=4,
        mlp_hidden_size=32,
        allow_test_self_attention=allow_test_self_attention,
        norm_type="layernorm",
        dropout=0.0,
    )
    with torch.no_grad():
        block.self_attention_between_features.in_proj_weight.zero_()
        assert block.self_attention_between_features.in_proj_bias is not None
        block.self_attention_between_features.in_proj_bias.zero_()
        block.self_attention_between_features.out_proj.weight.zero_()
        block.self_attention_between_features.out_proj.bias.zero_()
        block.linear1.weight.zero_()
        block.linear1.bias.zero_()
        block.linear2.weight.zero_()
        block.linear2.bias.zero_()

    cells = torch.randn(1, 5, 4, 16)
    observed = block(cells.clone(), train_test_split_index=3)

    row_in = cells.transpose(1, 2).reshape(4, 5, 16)
    row_norm = block.row_norm(row_in)
    row_mask = _row_mask_reference(
        n_total=5,
        n_train=3,
        allow_test_self_attention=allow_test_self_attention,
        device=cells.device,
    )
    row_out = block.self_attention_between_datapoints(
        row_norm,
        row_norm,
        row_norm,
        attn_mask=row_mask,
        need_weights=False,
    )[0]
    expected = (row_in + row_out).reshape(1, 4, 5, 16).transpose(2, 1)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_qass_multihead_attention_matches_dense_reference() -> None:
    torch.manual_seed(4)
    module = QASSMultiheadAttention(d_model=16, n_heads=4, dropout=0.0, use_qass=True)
    query = torch.randn(2, 5, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    allowed_mask = _allowed_mask(batch_size=2, target_len=5, source_len=5)

    expected = _qass_reference(
        module,
        query,
        key,
        value,
        allowed_mask=allowed_mask,
        n_context=4,
        force_qass=None,
    )
    observed = module(
        query,
        key,
        value,
        allowed_mask=allowed_mask,
        n_context=4,
    )

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_qass_multihead_attention_backward_stays_finite() -> None:
    torch.manual_seed(5)
    module = QASSMultiheadAttention(d_model=16, n_heads=4, dropout=0.0, use_qass=True)
    query = torch.randn(2, 5, 16, requires_grad=True)
    allowed_mask = _allowed_mask(batch_size=2, target_len=5, source_len=5)

    output = module(
        query,
        query,
        query,
        allowed_mask=allowed_mask,
        n_context=4,
    )
    loss = output.square().mean()
    loss.backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    for param in module.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multihead_attention_sdpa_cuda_long_row_backward_smoke() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 CUDA support required")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < 70 * 1024**3:
        pytest.skip("long-row smoke expects a large-memory CUDA device")

    torch.manual_seed(6)
    torch.cuda.empty_cache()
    module = nn.MultiheadAttention(
        128,
        8,
        batch_first=True,
        dropout=0.0,
        dtype=torch.bfloat16,
        device="cuda",
    ).train()
    query = torch.randn(
        147,
        8192,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    attn_bias = torch.zeros((8192, 8192), device="cuda", dtype=torch.bfloat16)
    attn_bias[6144:, 6144:] = float("-inf")
    diag = torch.arange(6144, 8192, device="cuda")
    attn_bias[diag, diag] = 0.0

    output = multihead_attention_sdpa(
        module,
        query,
        query,
        query,
        attn_bias=attn_bias,
    )
    loss = output.square().mean()
    loss.backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
