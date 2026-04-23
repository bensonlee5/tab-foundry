"""Shared scaled-dot-product attention helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention


def _reshape_heads(x: Tensor, *, num_heads: int) -> Tensor:
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def qk_norm_scale_init(*, embed_dim: int, num_heads: int) -> float:
    if int(embed_dim) % int(num_heads) != 0:
        raise ValueError(
            "QK-norm requires embed_dim divisible by num_heads, "
            f"got embed_dim={embed_dim}, num_heads={num_heads}"
        )
    return math.sqrt(int(embed_dim) // int(num_heads))


def apply_qk_norm(
    query_heads: Tensor,
    key_heads: Tensor,
    qk_norm_scale: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if qk_norm_scale is None:
        return query_heads, key_heads
    query_norm = F.normalize(query_heads, p=2.0, dim=-1)
    key_norm = F.normalize(key_heads, p=2.0, dim=-1)
    scale = qk_norm_scale.to(device=query_norm.device, dtype=query_norm.dtype).view(1, -1, 1, 1)
    return query_norm * scale, key_norm


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


def _validate_packed_multihead_attention(module: MultiheadAttention) -> None:
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


def multihead_attention_sdpa(
    module: MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_bias: Tensor | None = None,
    qk_norm_scale: Tensor | None = None,
) -> Tensor:
    """Run a MultiheadAttention module through SDPA using its existing weights."""

    _validate_packed_multihead_attention(module)

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
    q_heads, k_heads = apply_qk_norm(q_heads, k_heads, qk_norm_scale)

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


def packed_projection_multihead_attention_sdpa(
    module: MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_bias: Tensor | None = None,
    qk_norm_scale: Tensor | None = None,
) -> Tensor:
    """Run MultiheadAttention through SDPA with fused QKV/KV projections when possible."""

    _validate_packed_multihead_attention(module)

    embed_dim = int(module.embed_dim)
    in_proj_bias = module.in_proj_bias
    if query is key and key is value:
        q_proj, k_proj, v_proj = F.linear(
            query,
            module.in_proj_weight,
            in_proj_bias,
        ).split(embed_dim, dim=-1)
    elif key is value:
        q_proj = F.linear(
            query,
            module.in_proj_weight[:embed_dim],
            None if in_proj_bias is None else in_proj_bias[:embed_dim],
        )
        k_proj, v_proj = F.linear(
            key,
            module.in_proj_weight[embed_dim:],
            None if in_proj_bias is None else in_proj_bias[embed_dim:],
        ).split(embed_dim, dim=-1)
    else:
        return multihead_attention_sdpa(
            module,
            query,
            key,
            value,
            attn_bias=attn_bias,
            qk_norm_scale=qk_norm_scale,
        )

    q_heads = _reshape_heads(q_proj, num_heads=module.num_heads)
    k_heads = _reshape_heads(k_proj, num_heads=module.num_heads)
    v_heads = _reshape_heads(v_proj, num_heads=module.num_heads)
    q_heads, k_heads = apply_qk_norm(q_heads, k_heads, qk_norm_scale)

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
