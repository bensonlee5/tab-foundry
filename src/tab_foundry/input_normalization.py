"""Shared train-split normalization helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


InputNormalizationMode = Literal["none", "train_zscore", "train_zscore_clip"]

_EPS = 1.0e-8
_CLIP_VALUE = 100.0


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
    if normalized_mode == "none":
        return train, test

    mean = train.mean(axis=0, dtype=np.float32)
    std = train.std(axis=0, dtype=np.float32)
    std = np.where(std < _EPS, 1.0, std).astype(np.float32, copy=False)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
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
    if normalized_mode == "none":
        return train, test

    mean = train.mean(dim=0, keepdim=False)
    std = train.std(dim=0, keepdim=False, unbiased=False)
    std = torch.where(std < _EPS, torch.ones_like(std), std)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    if normalized_mode == "train_zscore":
        return train_norm, test_norm
    if normalized_mode == "train_zscore_clip":
        return train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE), test_norm.clamp(
            min=-_CLIP_VALUE,
            max=_CLIP_VALUE,
        )
    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")
