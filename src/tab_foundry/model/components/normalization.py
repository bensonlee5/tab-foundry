"""Shared normalization helpers for tabfoundry model components."""

from __future__ import annotations

import torch
from torch import nn


SUPPORTED_NORM_TYPES = ("layernorm", "rmsnorm")


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean centering."""

    def __init__(self, normalized_shape: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        if int(normalized_shape) <= 0:
            raise ValueError(f"normalized_shape must be positive, got {normalized_shape}")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        # Match the LayerNorm attribute contract that PyTorch's transformer
        # internals expect when they access norm.weight and norm.bias directly.
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_norm(norm_type: str, normalized_shape: int, *, eps: float = 1.0e-5) -> nn.Module:
    """Construct one supported normalization layer."""

    normalized = str(norm_type).strip().lower()
    if normalized == "layernorm":
        return nn.LayerNorm(normalized_shape, eps=eps)
    if normalized == "rmsnorm":
        return RMSNorm(normalized_shape, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type!r}; supported={SUPPORTED_NORM_TYPES}")
