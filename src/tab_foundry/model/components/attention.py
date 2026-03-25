"""Shared scaled-dot-product attention helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention


def _reshape_heads(x: Tensor, *, num_heads: int) -> Tensor:
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def attention_bias_from_allowed_mask(allowed_mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Convert an allow-mask into an additive SDPA bias tensor."""

    if allowed_mask.dtype is not torch.bool:
        raise TypeError(
            "allowed_mask must use bool dtype, "
            f"got {allowed_mask.dtype}"
        )
    bias = torch.zeros(allowed_mask.shape, device=allowed_mask.device, dtype=dtype)
    return bias.masked_fill(~allowed_mask, float("-inf"))


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_bias: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool,
) -> Tensor:
    """Apply scaled dot-product attention with training-aware dropout."""

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=(float(dropout_p) if training else 0.0),
    )


def multihead_attention_sdpa(
    module: MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_bias: Tensor | None = None,
) -> Tensor:
    """Run a MultiheadAttention module through SDPA using its existing weights."""

    if not module.batch_first:
        raise ValueError("multihead_attention_sdpa requires batch_first=True")
    if not module._qkv_same_embed_dim:
        raise ValueError("multihead_attention_sdpa requires qkv_same_embed_dim=True")
    if module.in_proj_weight is None:
        raise ValueError("multihead_attention_sdpa requires packed in_proj_weight")
    if module.bias_k is not None or module.bias_v is not None:
        raise ValueError("multihead_attention_sdpa does not support bias_k/bias_v")
    if module.add_zero_attn:
        raise ValueError("multihead_attention_sdpa does not support add_zero_attn")

    embed_dim = int(module.embed_dim)
    in_proj_bias = module.in_proj_bias

    q_proj = F.linear(
        query,
        module.in_proj_weight[:embed_dim],
        None if in_proj_bias is None else in_proj_bias[:embed_dim],
    )
    k_proj = F.linear(
        key,
        module.in_proj_weight[embed_dim : 2 * embed_dim],
        None if in_proj_bias is None else in_proj_bias[embed_dim : 2 * embed_dim],
    )
    v_proj = F.linear(
        value,
        module.in_proj_weight[2 * embed_dim :],
        None if in_proj_bias is None else in_proj_bias[2 * embed_dim :],
    )

    q_heads = _reshape_heads(q_proj, num_heads=module.num_heads)
    k_heads = _reshape_heads(k_proj, num_heads=module.num_heads)
    v_heads = _reshape_heads(v_proj, num_heads=module.num_heads)

    attn_out = scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        attn_bias=attn_bias,
        dropout_p=module.dropout,
        training=module.training,
    )
    batch_size, _num_heads, target_len, head_dim = attn_out.shape
    merged = (
        attn_out.transpose(1, 2)
        .contiguous()
        .view(batch_size, target_len, module.num_heads * head_dim)
    )
    return module.out_proj(merged)
