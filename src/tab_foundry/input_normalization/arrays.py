"""Numpy train-split normalization paths."""

from __future__ import annotations

import numpy as np

from .rankgauss import rankgauss_test_np, rankgauss_train_np
from .shared import (
    CLIP_VALUE,
    EPS,
    ROBUST_MODES,
    WINSORIZE_HI,
    WINSORIZE_LO,
    ZSCORE_BASED_MODES,
    InputNormalizationMode,
    normalize_mode,
    smooth_tanh_np,
)


def normalize_train_test_arrays(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    mode: InputNormalizationMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize numpy train/test arrays using train-only statistics."""

    normalized_mode = normalize_mode(mode)
    train = np.asarray(x_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    train_stats = np.asarray(x_train, dtype=np.float64)
    test_stats = np.asarray(x_test, dtype=np.float64)
    if normalized_mode == "none":
        return train, test

    if normalized_mode in ZSCORE_BASED_MODES:
        if normalized_mode == "train_winsorize_zscore":
            lo = np.percentile(train_stats, WINSORIZE_LO, axis=0)
            hi = np.percentile(train_stats, WINSORIZE_HI, axis=0)
            train_stats = np.clip(train_stats, lo, hi)
            test_stats = np.clip(test_stats, lo, hi)

        mean = train_stats.mean(axis=0, dtype=np.float64)
        std = train_stats.std(axis=0, dtype=np.float64)
        std = np.where(std < EPS, 1.0, std)
        train_norm = (train_stats - mean) / std
        test_norm = (test_stats - mean) / std

        if normalized_mode == "train_zscore_clip":
            return (
                np.clip(train_norm, -CLIP_VALUE, CLIP_VALUE).astype(np.float32, copy=False),
                np.clip(test_norm, -CLIP_VALUE, CLIP_VALUE).astype(np.float32, copy=False),
            )
        if normalized_mode == "train_zscore_tanh":
            return (
                smooth_tanh_np(train_norm).astype(np.float32, copy=False),
                smooth_tanh_np(test_norm).astype(np.float32, copy=False),
            )
        return train_norm.astype(np.float32, copy=False), test_norm.astype(np.float32, copy=False)

    if normalized_mode == "train_rankgauss":
        n_cols = train_stats.shape[1]
        train_out = np.empty_like(train_stats)
        test_out = np.empty_like(test_stats)
        for c in range(n_cols):
            col_std = train_stats[:, c].std()
            if col_std < EPS:
                train_out[:, c] = 0.0
                test_out[:, c] = 0.0
                continue
            train_out[:, c] = rankgauss_train_np(train_stats[:, c])
            sorted_idx = np.argsort(train_stats[:, c])
            train_sorted = train_stats[sorted_idx, c]
            test_out[:, c] = rankgauss_test_np(train_sorted, len(train_stats), test_stats[:, c])
        return train_out.astype(np.float32, copy=False), test_out.astype(np.float32, copy=False)

    if normalized_mode in ROBUST_MODES:
        median = np.median(train_stats, axis=0)
        q25 = np.percentile(train_stats, 25.0, axis=0)
        q75 = np.percentile(train_stats, 75.0, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < EPS, 1.0, iqr)
        train_norm = (train_stats - median) / iqr
        test_norm = (test_stats - median) / iqr
        if normalized_mode == "train_robust_tanh":
            return (
                smooth_tanh_np(train_norm).astype(np.float32, copy=False),
                smooth_tanh_np(test_norm).astype(np.float32, copy=False),
            )
        return train_norm.astype(np.float32, copy=False), test_norm.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")
