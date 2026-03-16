"""Shared train-split normalization helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


InputNormalizationMode = Literal["none", "train_zscore", "train_zscore_clip"]
SUPPORTED_INPUT_NORMALIZATION_MODES: tuple[InputNormalizationMode, ...] = (
    "none",
    "train_zscore",
    "train_zscore_clip",
)

_EPS = 1.0e-8
_CLIP_VALUE = 100.0


def _tensor_stats_dtype(device: torch.device) -> torch.dtype:
    """Choose a reduction dtype that is supported on the active device."""

    return torch.float32 if device.type == "mps" else torch.float64


def normalize_train_test_arrays(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    mode: InputNormalizationMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize numpy train/test arrays using train-only statistics."""

    normalized_mode = str(mode).strip().lower()
    train = np.asarray(x_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    train_stats = np.asarray(x_train, dtype=np.float64)
    test_stats = np.asarray(x_test, dtype=np.float64)
    if normalized_mode == "none":
        return train, test

    mean = train_stats.mean(axis=-2, dtype=np.float64, keepdims=True)
    std = train_stats.std(axis=-2, dtype=np.float64, keepdims=True)
    std = np.where(std < _EPS, 1.0, std)
    train_norm = (train_stats - mean) / std
    test_norm = (test_stats - mean) / std
    if normalized_mode == "train_zscore":
        return train_norm.astype(np.float32, copy=False), test_norm.astype(np.float32, copy=False)
    if normalized_mode == "train_zscore_clip":
        return (
            np.clip(train_norm, -_CLIP_VALUE, _CLIP_VALUE).astype(np.float32, copy=False),
            np.clip(test_norm, -_CLIP_VALUE, _CLIP_VALUE).astype(np.float32, copy=False),
        )
    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")


def normalize_train_test_tensors(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize torch train/test tensors using train-only statistics."""

    normalized_mode = str(mode).strip().lower()
    train = x_train.to(torch.float32)
    test = x_test.to(torch.float32)
    stats_dtype = _tensor_stats_dtype(x_train.device)
    train_stats = x_train.to(stats_dtype)
    test_stats = x_test.to(stats_dtype)
    if normalized_mode == "none":
        return train, test

    mean = train_stats.mean(dim=-2, keepdim=True)
    std = train_stats.std(dim=-2, keepdim=True, unbiased=False)
    std = torch.where(std < _EPS, torch.ones_like(std), std)
    train_norm = ((train_stats - mean) / std).to(torch.float32)
    test_norm = ((test_stats - mean) / std).to(torch.float32)
    if normalized_mode == "train_zscore":
        return train_norm, test_norm
    if normalized_mode == "train_zscore_clip":
        return train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE), test_norm.clamp(
            min=-_CLIP_VALUE,
            max=_CLIP_VALUE,
        )
    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")


def prepare_train_test_tensors_with_missing(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize tensors while preserving a separate non-finite mask."""

    train_missing = ~torch.isfinite(x_train)
    test_missing = ~torch.isfinite(x_test)
    if not bool(train_missing.any() or test_missing.any()):
        train_norm, test_norm = normalize_train_test_tensors(x_train, x_test, mode=mode)
        return (
            train_norm,
            test_norm,
            train_missing.to(torch.bool),
            test_missing.to(torch.bool),
        )

    train = x_train.to(torch.float32)
    test = x_test.to(torch.float32)
    train_observed = (~train_missing).to(torch.float32)
    train_safe = torch.where(train_missing, torch.zeros_like(train), train)
    observed_count = train_observed.sum(dim=-2, keepdim=False)
    observed_sum = train_safe.sum(dim=-2, keepdim=False)
    fill_values = torch.where(
        observed_count > 0,
        observed_sum / observed_count.clamp_min(1.0),
        torch.zeros_like(observed_sum),
    )
    train_filled = torch.where(train_missing, fill_values.unsqueeze(-2), train)
    test_filled = torch.where(test_missing, fill_values.unsqueeze(-2), test)
    train_norm, test_norm = normalize_train_test_tensors(train_filled, test_filled, mode=mode)
    return train_norm, test_norm, train_missing.to(torch.bool), test_missing.to(torch.bool)
