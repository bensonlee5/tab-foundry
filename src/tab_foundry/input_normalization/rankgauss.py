"""Rank-Gauss helpers for numpy and torch normalization paths."""

from __future__ import annotations

import numpy as np
import torch

from .shared import RANKGAUSS_CLAMP_EPS, SQRT2


_RANKGAUSS_CENTER_OFFSET = 0.5


def rankgauss_train_np(col: np.ndarray) -> np.ndarray:
    """Rank-Gauss transform for a single train column."""

    n = len(col)
    perm = np.argsort(col, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[perm] = np.arange(n, dtype=np.int64)
    quantiles = (ranks + _RANKGAUSS_CENTER_OFFSET) / n
    q_t = torch.as_tensor(quantiles, dtype=torch.float64)
    return (SQRT2 * torch.erfinv(2.0 * q_t - 1.0)).numpy()


def rankgauss_test_np(
    train_sorted: np.ndarray,
    n_train: int,
    col: np.ndarray,
) -> np.ndarray:
    """Map a test column through the train rank-gauss mapping."""

    insert_idx = np.searchsorted(train_sorted, col, side="right")
    quantiles = (insert_idx + _RANKGAUSS_CENTER_OFFSET) / (n_train + 1)
    quantiles = np.clip(quantiles, RANKGAUSS_CLAMP_EPS, 1.0 - RANKGAUSS_CLAMP_EPS)
    q_t = torch.as_tensor(quantiles, dtype=torch.float64)
    return (SQRT2 * torch.erfinv(2.0 * q_t - 1.0)).numpy()


def rankgauss_train_torch(col: torch.Tensor) -> torch.Tensor:
    """Rank-Gauss transform for a single train column."""

    n = int(col.shape[0])
    perm = torch.argsort(col, stable=True)
    ranks = torch.empty_like(perm)
    ranks[perm] = torch.arange(n, device=col.device, dtype=perm.dtype)
    quantiles = (ranks.to(col.dtype) + _RANKGAUSS_CENTER_OFFSET) / n
    return SQRT2 * torch.erfinv(2.0 * quantiles - 1.0)


def rankgauss_test_torch(
    train_sorted: torch.Tensor,
    n_train: int,
    col: torch.Tensor,
) -> torch.Tensor:
    """Map a test column through the train rank-gauss mapping."""

    insert_idx = torch.searchsorted(train_sorted.contiguous(), col.contiguous(), right=True)
    quantiles = (insert_idx.to(col.dtype) + _RANKGAUSS_CENTER_OFFSET) / (n_train + 1)
    quantiles = quantiles.clamp(RANKGAUSS_CLAMP_EPS, 1.0 - RANKGAUSS_CLAMP_EPS)
    return SQRT2 * torch.erfinv(2.0 * quantiles - 1.0)
