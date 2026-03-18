"""Helpers for preserving and encoding non-finite numeric inputs."""

from __future__ import annotations

import torch


def clip_finite_values(values: torch.Tensor, *, clip_value: float) -> torch.Tensor:
    """Clamp only finite values, preserving non-finite sentinels for downstream logic."""

    finite_mask = torch.isfinite(values)
    if not torch.any(finite_mask):
        return values
    clipped = values.clone()
    clipped[finite_mask] = clipped[finite_mask].clamp(min=-clip_value, max=clip_value)
    return clipped


def encode_non_finite_token_features(values: torch.Tensor) -> torch.Tensor:
    """Encode scalar values plus exhaustive non-finite-type flags before embedding."""

    filled = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.stack(
        [
            filled,
            torch.isnan(values).to(dtype=filled.dtype),
            torch.isposinf(values).to(dtype=filled.dtype),
            torch.isneginf(values).to(dtype=filled.dtype),
        ],
        dim=-1,
    )
