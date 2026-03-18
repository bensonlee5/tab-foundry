"""Shared train-split normalization helpers."""

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

_EPS = 1.0e-8
_CLIP_VALUE = 100.0
_WINSORIZE_LO = 1.0
_WINSORIZE_HI = 99.0
_SMOOTH_TAIL_LIMIT = 3.0
_SQRT2 = math.sqrt(2.0)
_ZSCORE_BASED_MODES = (
    "train_zscore",
    "train_zscore_clip",
    "train_winsorize_zscore",
    "train_zscore_tanh",
)
_ROBUST_MODES = ("train_robust", "train_robust_tanh")


def _tensor_stats_dtype(device: torch.device) -> torch.dtype:
    """Choose a reduction dtype that is supported on the active device."""

    return torch.float32 if device.type == "mps" else torch.float64


def _smooth_tanh_np(values: np.ndarray) -> np.ndarray:
    """Apply a bounded smooth-tail transform that stays near-linear around zero."""

    return _SMOOTH_TAIL_LIMIT * np.tanh(values / _SMOOTH_TAIL_LIMIT)


def _smooth_tanh_torch(values: torch.Tensor) -> torch.Tensor:
    """Apply a bounded smooth-tail transform that stays near-linear around zero."""

    limit = values.new_tensor(_SMOOTH_TAIL_LIMIT)
    return limit * torch.tanh(values / limit)


# ---------------------------------------------------------------------------
# Rank-Gauss helpers
# ---------------------------------------------------------------------------

def _rankgauss_train_np(col: np.ndarray) -> np.ndarray:
    """Rank-Gauss transform for a single train column (numpy)."""
    n = len(col)
    perm = np.argsort(col, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[perm] = np.arange(n, dtype=np.int64)
    quantiles = (ranks + 0.5) / n  # open (0, 1)
    q_t = torch.as_tensor(quantiles, dtype=torch.float64)
    gauss = (_SQRT2 * torch.erfinv(2.0 * q_t - 1.0)).numpy()
    return gauss


def _rankgauss_test_np(
    train_sorted: np.ndarray,
    train_gauss_sorted: np.ndarray,
    n_train: int,
    col: np.ndarray,
) -> np.ndarray:
    """Map test column through the train rank-gauss mapping (numpy)."""
    # fractional position in train
    insert_idx = np.searchsorted(train_sorted, col, side="right")  # 0..n_train
    quantiles = (insert_idx + 0.5) / (n_train + 1)  # keep in (0,1)
    quantiles = np.clip(quantiles, 1e-7, 1.0 - 1e-7)
    q_t = torch.as_tensor(quantiles, dtype=torch.float64)
    gauss = (_SQRT2 * torch.erfinv(2.0 * q_t - 1.0)).numpy()
    return gauss


def _rankgauss_train_torch(col: torch.Tensor) -> torch.Tensor:
    """Rank-Gauss transform for a single train column (torch)."""
    n = int(col.shape[0])
    perm = torch.argsort(col, stable=True)
    ranks = torch.empty_like(perm)
    ranks[perm] = torch.arange(n, device=col.device, dtype=perm.dtype)
    quantiles = (ranks.to(col.dtype) + 0.5) / n
    return _SQRT2 * torch.erfinv(2.0 * quantiles - 1.0)


def _rankgauss_test_torch(
    train_sorted: torch.Tensor,
    n_train: int,
    col: torch.Tensor,
) -> torch.Tensor:
    """Map test column through the train rank-gauss mapping (torch)."""
    train_sorted = train_sorted.contiguous()
    col = col.contiguous()
    insert_idx = torch.searchsorted(train_sorted, col, right=True)
    quantiles = (insert_idx.to(col.dtype) + 0.5) / (n_train + 1)
    quantiles = quantiles.clamp(1e-7, 1.0 - 1e-7)
    return _SQRT2 * torch.erfinv(2.0 * quantiles - 1.0)


# ---------------------------------------------------------------------------
# Public API — numpy
# ---------------------------------------------------------------------------

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

    # --- z-score based modes ---
    if normalized_mode in (
        "train_zscore",
        "train_zscore_clip",
        "train_winsorize_zscore",
        "train_zscore_tanh",
    ):
        if normalized_mode == "train_winsorize_zscore":
            lo = np.percentile(train_stats, _WINSORIZE_LO, axis=0)
            hi = np.percentile(train_stats, _WINSORIZE_HI, axis=0)
            train_stats = np.clip(train_stats, lo, hi)
            test_stats = np.clip(test_stats, lo, hi)

        mean = train_stats.mean(axis=0, dtype=np.float64)
        std = train_stats.std(axis=0, dtype=np.float64)
        std = np.where(std < _EPS, 1.0, std)
        train_norm = (train_stats - mean) / std
        test_norm = (test_stats - mean) / std

        if normalized_mode == "train_zscore_clip":
            return (
                np.clip(train_norm, -_CLIP_VALUE, _CLIP_VALUE).astype(np.float32, copy=False),
                np.clip(test_norm, -_CLIP_VALUE, _CLIP_VALUE).astype(np.float32, copy=False),
            )
        if normalized_mode == "train_zscore_tanh":
            return (
                _smooth_tanh_np(train_norm).astype(np.float32, copy=False),
                _smooth_tanh_np(test_norm).astype(np.float32, copy=False),
            )
        return train_norm.astype(np.float32, copy=False), test_norm.astype(np.float32, copy=False)

    # --- rank-gauss ---
    if normalized_mode == "train_rankgauss":
        n_cols = train_stats.shape[1]
        train_out = np.empty_like(train_stats)
        test_out = np.empty_like(test_stats)
        for c in range(n_cols):
            col_std = train_stats[:, c].std()
            if col_std < _EPS:
                train_out[:, c] = 0.0
                test_out[:, c] = 0.0
                continue
            train_out[:, c] = _rankgauss_train_np(train_stats[:, c])
            sorted_idx = np.argsort(train_stats[:, c])
            train_sorted = train_stats[sorted_idx, c]
            train_gauss_sorted = train_out[sorted_idx, c]
            test_out[:, c] = _rankgauss_test_np(
                train_sorted, train_gauss_sorted, len(train_stats), test_stats[:, c],
            )
        return train_out.astype(np.float32, copy=False), test_out.astype(np.float32, copy=False)

    # --- robust (median / IQR) ---
    if normalized_mode in ("train_robust", "train_robust_tanh"):
        median = np.median(train_stats, axis=0)
        q25 = np.percentile(train_stats, 25.0, axis=0)
        q75 = np.percentile(train_stats, 75.0, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < _EPS, 1.0, iqr)
        train_norm = (train_stats - median) / iqr
        test_norm = (test_stats - median) / iqr
        if normalized_mode == "train_robust_tanh":
            return (
                _smooth_tanh_np(train_norm).astype(np.float32, copy=False),
                _smooth_tanh_np(test_norm).astype(np.float32, copy=False),
            )
        return train_norm.astype(np.float32, copy=False), test_norm.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")


# ---------------------------------------------------------------------------
# Public API — torch
# ---------------------------------------------------------------------------

def _normalize_train_test_tensors_2d(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
    preserve_non_finite: bool = False,
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
    if preserve_non_finite:
        train_out = train.clone()
        test_out = test.clone()
        n_cols = int(train_stats.shape[1])
        for c in range(n_cols):
            train_col = train_stats[:, c]
            test_col = test_stats[:, c]
            train_mask = torch.isfinite(train_col)
            test_mask = torch.isfinite(test_col)
            finite_train = train_col[train_mask]
            if int(finite_train.numel()) <= 0:
                if torch.any(test_mask):
                    test_out[test_mask, c] = 0.0
                continue

            if normalized_mode in _ZSCORE_BASED_MODES:
                if normalized_mode == "train_winsorize_zscore":
                    lo = torch.quantile(finite_train, _WINSORIZE_LO / 100.0)
                    hi = torch.quantile(finite_train, _WINSORIZE_HI / 100.0)
                    finite_train = finite_train.clamp(min=lo, max=hi)
                    finite_test = (
                        test_col[test_mask].clamp(min=lo, max=hi)
                        if torch.any(test_mask)
                        else None
                    )
                else:
                    finite_test = test_col[test_mask] if torch.any(test_mask) else None

                mean = finite_train.mean()
                std = finite_train.std(unbiased=False)
                if float(std) < _EPS:
                    std = torch.ones_like(std)
                train_norm = ((finite_train - mean) / std).to(torch.float32)
                test_norm = (
                    ((finite_test - mean) / std).to(torch.float32)
                    if finite_test is not None
                    else None
                )
                if normalized_mode == "train_zscore_clip":
                    train_norm = train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE)
                    if test_norm is not None:
                        test_norm = test_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE)
                elif normalized_mode == "train_zscore_tanh":
                    train_norm = _smooth_tanh_torch(train_norm)
                    if test_norm is not None:
                        test_norm = _smooth_tanh_torch(test_norm)
                train_out[train_mask, c] = train_norm
                if test_norm is not None:
                    test_out[test_mask, c] = test_norm
                continue

            if normalized_mode == "train_rankgauss":
                col_std = finite_train.std(unbiased=False)
                if float(col_std) < _EPS:
                    train_out[train_mask, c] = 0.0
                    if torch.any(test_mask):
                        test_out[test_mask, c] = 0.0
                    continue
                train_out[train_mask, c] = _rankgauss_train_torch(finite_train).to(torch.float32)
                if torch.any(test_mask):
                    train_sorted, _ = finite_train.sort()
                    test_out[test_mask, c] = _rankgauss_test_torch(
                        train_sorted,
                        int(finite_train.shape[0]),
                        test_col[test_mask],
                    ).to(torch.float32)
                continue

            if normalized_mode in ("train_robust", "train_robust_tanh"):
                median = torch.quantile(finite_train, 0.5)
                q25 = torch.quantile(finite_train, 0.25)
                q75 = torch.quantile(finite_train, 0.75)
                iqr = q75 - q25
                if float(iqr) < _EPS:
                    iqr = torch.ones_like(iqr)
                train_norm = ((finite_train - median) / iqr).to(torch.float32)
                if normalized_mode == "train_robust_tanh":
                    train_norm = _smooth_tanh_torch(train_norm)
                train_out[train_mask, c] = train_norm
                if torch.any(test_mask):
                    test_norm = ((test_col[test_mask] - median) / iqr).to(torch.float32)
                    if normalized_mode == "train_robust_tanh":
                        test_norm = _smooth_tanh_torch(test_norm)
                    test_out[test_mask, c] = test_norm
                continue

            raise ValueError(f"Unsupported input_normalization mode: {mode!r}")
        return train_out, test_out

    # --- z-score based modes ---
    if normalized_mode in _ZSCORE_BASED_MODES:
        if normalized_mode == "train_winsorize_zscore":
            lo = torch.quantile(train_stats, _WINSORIZE_LO / 100.0, dim=0)
            hi = torch.quantile(train_stats, _WINSORIZE_HI / 100.0, dim=0)
            train_stats = train_stats.clamp(min=lo, max=hi)
            test_stats = test_stats.clamp(min=lo, max=hi)

        mean = train_stats.mean(dim=0, keepdim=False)
        std = train_stats.std(dim=0, keepdim=False, unbiased=False)
        std = torch.where(std < _EPS, torch.ones_like(std), std)
        train_norm = ((train_stats - mean) / std).to(torch.float32)
        test_norm = ((test_stats - mean) / std).to(torch.float32)

        if normalized_mode == "train_zscore_clip":
            return train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE), test_norm.clamp(
                min=-_CLIP_VALUE,
                max=_CLIP_VALUE,
            )
        if normalized_mode == "train_zscore_tanh":
            return _smooth_tanh_torch(train_norm), _smooth_tanh_torch(test_norm)
        return train_norm, test_norm

    # --- rank-gauss ---
    if normalized_mode == "train_rankgauss":
        n_cols = train_stats.shape[1]
        train_out = torch.empty_like(train_stats)
        test_out = torch.empty_like(test_stats)
        for c in range(n_cols):
            col_std = train_stats[:, c].std(unbiased=False)
            if col_std < _EPS:
                train_out[:, c] = 0.0
                test_out[:, c] = 0.0
                continue
            train_out[:, c] = _rankgauss_train_torch(train_stats[:, c])
            train_sorted, _ = train_stats[:, c].sort()
            test_out[:, c] = _rankgauss_test_torch(
                train_sorted, train_stats.shape[0], test_stats[:, c],
            )
        return train_out.to(torch.float32), test_out.to(torch.float32)

    # --- robust (median / IQR) ---
    if normalized_mode in _ROBUST_MODES:
        median = torch.quantile(train_stats, 0.5, dim=0)
        q25 = torch.quantile(train_stats, 0.25, dim=0)
        q75 = torch.quantile(train_stats, 0.75, dim=0)
        iqr = q75 - q25
        iqr = torch.where(iqr < _EPS, torch.ones_like(iqr), iqr)
        train_norm = ((train_stats - median) / iqr).to(torch.float32)
        test_norm = ((test_stats - median) / iqr).to(torch.float32)
        if normalized_mode == "train_robust_tanh":
            return _smooth_tanh_torch(train_norm), _smooth_tanh_torch(test_norm)
        return train_norm, test_norm

    raise ValueError(f"Unsupported input_normalization mode: {mode!r}")


def _normalize_train_test_tensors_3d_fallback(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
    preserve_non_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for batch_idx in range(int(x_train.shape[0])):
        train_norm, test_norm = _normalize_train_test_tensors_2d(
            x_train[batch_idx],
            x_test[batch_idx],
            mode=mode,
            preserve_non_finite=preserve_non_finite,
        )
        train_parts.append(train_norm)
        test_parts.append(test_norm)
    return torch.stack(train_parts, dim=0), torch.stack(test_parts, dim=0)


def _normalize_train_test_tensors_3d_zscore(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
    preserve_non_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_mode = str(mode).strip().lower()
    train = x_train.to(torch.float32)
    test = x_test.to(torch.float32)
    stats_dtype = _tensor_stats_dtype(x_train.device)
    train_stats = x_train.to(stats_dtype)
    test_stats = x_test.to(stats_dtype)

    if preserve_non_finite and normalized_mode == "train_winsorize_zscore":
        return _normalize_train_test_tensors_3d_fallback(
            x_train,
            x_test,
            mode=mode,
            preserve_non_finite=preserve_non_finite,
        )

    if preserve_non_finite:
        train_mask = torch.isfinite(train_stats)
        test_mask = torch.isfinite(test_stats)
        has_finite = torch.any(train_mask, dim=1)

        if normalized_mode == "train_winsorize_zscore":
            lo = torch.quantile(train_stats, _WINSORIZE_LO / 100.0, dim=1)
            hi = torch.quantile(train_stats, _WINSORIZE_HI / 100.0, dim=1)
            train_stats = train_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))
            test_stats = test_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))

        safe_train = torch.where(train_mask, train_stats, torch.zeros_like(train_stats))
        finite_count = train_mask.sum(dim=1).to(dtype=train_stats.dtype).clamp_min(1.0)
        mean = safe_train.sum(dim=1) / finite_count
        centered_sq = torch.where(
            train_mask,
            (train_stats - mean.unsqueeze(1)).square(),
            torch.zeros_like(train_stats),
        )
        std = torch.sqrt(centered_sq.sum(dim=1) / finite_count)
        std = torch.where(std < _EPS, torch.ones_like(std), std)

        train_norm = ((train_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
        test_norm = ((test_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)

        if normalized_mode == "train_zscore_clip":
            train_norm = train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE)
            test_norm = test_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE)
        elif normalized_mode == "train_zscore_tanh":
            train_norm = _smooth_tanh_torch(train_norm)
            test_norm = _smooth_tanh_torch(test_norm)

        train_out = train.clone()
        test_out = test.clone()
        train_out[train_mask] = train_norm[train_mask]

        valid_test_mask = test_mask & has_finite.unsqueeze(1)
        test_out[valid_test_mask] = test_norm[valid_test_mask]

        zero_test_mask = test_mask & (~has_finite).unsqueeze(1)
        test_out[zero_test_mask] = 0.0
        return train_out, test_out

    if normalized_mode == "train_winsorize_zscore":
        lo = torch.quantile(train_stats, _WINSORIZE_LO / 100.0, dim=1)
        hi = torch.quantile(train_stats, _WINSORIZE_HI / 100.0, dim=1)
        train_stats = train_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))
        test_stats = test_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))

    mean = train_stats.mean(dim=1)
    std = train_stats.std(dim=1, unbiased=False)
    std = torch.where(std < _EPS, torch.ones_like(std), std)
    train_norm = ((train_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
    test_norm = ((test_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)

    if normalized_mode == "train_zscore_clip":
        return train_norm.clamp(min=-_CLIP_VALUE, max=_CLIP_VALUE), test_norm.clamp(
            min=-_CLIP_VALUE,
            max=_CLIP_VALUE,
        )
    if normalized_mode == "train_zscore_tanh":
        return _smooth_tanh_torch(train_norm), _smooth_tanh_torch(test_norm)
    return train_norm, test_norm


def normalize_train_test_tensors(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
    preserve_non_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize torch train/test tensors using train-only statistics."""

    if x_train.ndim != x_test.ndim:
        raise ValueError(
            "x_train and x_test must have matching ranks, got "
            f"{tuple(x_train.shape)} and {tuple(x_test.shape)}"
        )
    if x_train.ndim == 2:
        return _normalize_train_test_tensors_2d(
            x_train,
            x_test,
            mode=mode,
            preserve_non_finite=preserve_non_finite,
        )
    if x_train.ndim != 3:
        raise ValueError(
            "normalize_train_test_tensors expects 2D or 3D inputs, got "
            f"{tuple(x_train.shape)} and {tuple(x_test.shape)}"
        )
    if int(x_train.shape[0]) != int(x_test.shape[0]):
        raise ValueError("3D x_train and x_test must have matching batch dimensions")
    if int(x_train.shape[2]) != int(x_test.shape[2]):
        raise ValueError("x_train and x_test must have matching feature dimensions")

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "none":
        return x_train.to(torch.float32), x_test.to(torch.float32)
    if normalized_mode in _ZSCORE_BASED_MODES:
        return _normalize_train_test_tensors_3d_zscore(
            x_train,
            x_test,
            mode=mode,
            preserve_non_finite=preserve_non_finite,
        )
    return _normalize_train_test_tensors_3d_fallback(
        x_train,
        x_test,
        mode=mode,
        preserve_non_finite=preserve_non_finite,
    )
