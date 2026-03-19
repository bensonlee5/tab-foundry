"""Shared constants and helpers for train-split normalization."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch


InputNormalizationMode = Literal[
    "none",
    "train_zscore",
    "train_zscore_clip",
    "train_rankgauss",
    "train_robust",
    "train_winsorize_zscore",
    "train_zscore_tanh",
    "train_robust_tanh",
]
SUPPORTED_INPUT_NORMALIZATION_MODES: tuple[InputNormalizationMode, ...] = (
    "none",
    "train_zscore",
    "train_zscore_clip",
    "train_rankgauss",
    "train_robust",
    "train_winsorize_zscore",
    "train_zscore_tanh",
    "train_robust_tanh",
)

EPS = 1.0e-8
CLIP_VALUE = 100.0
WINSORIZE_LO = 1.0
WINSORIZE_HI = 99.0
SMOOTH_TAIL_LIMIT = 3.0
SQRT2 = math.sqrt(2.0)
ZSCORE_BASED_MODES = (
    "train_zscore",
    "train_zscore_clip",
    "train_winsorize_zscore",
    "train_zscore_tanh",
)
ROBUST_MODES = ("train_robust", "train_robust_tanh")


def normalize_mode(mode: InputNormalizationMode) -> str:
    """Canonicalize the configured normalization mode for comparisons."""

    return str(mode).strip().lower()


def tensor_stats_dtype(device: torch.device) -> torch.dtype:
    """Choose a reduction dtype that is supported on the active device."""

    return torch.float32 if device.type == "mps" else torch.float64


def smooth_tanh_np(values: np.ndarray) -> np.ndarray:
    """Apply a bounded smooth-tail transform that stays near-linear around zero."""

    return SMOOTH_TAIL_LIMIT * np.tanh(values / SMOOTH_TAIL_LIMIT)


def smooth_tanh_torch(values: torch.Tensor) -> torch.Tensor:
    """Apply a bounded smooth-tail transform that stays near-linear around zero."""

    limit = values.new_tensor(SMOOTH_TAIL_LIMIT)
    return limit * torch.tanh(values / limit)
