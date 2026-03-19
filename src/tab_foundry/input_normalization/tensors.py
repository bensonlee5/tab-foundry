"""Torch train-split normalization paths."""

from __future__ import annotations

import torch

from .rankgauss import rankgauss_test_torch, rankgauss_train_torch
from .shared import (
    CLIP_VALUE,
    EPS,
    ROBUST_MODES,
    WINSORIZE_HI,
    WINSORIZE_LO,
    ZSCORE_BASED_MODES,
    InputNormalizationMode,
    normalize_mode,
    smooth_tanh_torch,
    tensor_stats_dtype,
)


def _postprocess_zscore_torch(
    train_norm: torch.Tensor,
    test_norm: torch.Tensor | None,
    *,
    normalized_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if normalized_mode == "train_zscore_clip":
        train_norm = train_norm.clamp(min=-CLIP_VALUE, max=CLIP_VALUE)
        if test_norm is not None:
            test_norm = test_norm.clamp(min=-CLIP_VALUE, max=CLIP_VALUE)
    elif normalized_mode == "train_zscore_tanh":
        train_norm = smooth_tanh_torch(train_norm)
        if test_norm is not None:
            test_norm = smooth_tanh_torch(test_norm)
    return train_norm, test_norm


def _normalize_train_test_tensors_2d(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: InputNormalizationMode,
    preserve_non_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize torch train/test tensors using train-only statistics."""

    normalized_mode = normalize_mode(mode)
    train = x_train.to(torch.float32)
    test = x_test.to(torch.float32)
    stats_dtype = tensor_stats_dtype(x_train.device)
    train_stats = x_train.to(stats_dtype)
    test_stats = x_test.to(stats_dtype)
    if normalized_mode == "none":
        return train, test
    if preserve_non_finite:
        train_out = train.clone()
        test_out = test.clone()
        for c in range(int(train_stats.shape[1])):
            train_col = train_stats[:, c]
            test_col = test_stats[:, c]
            train_mask = torch.isfinite(train_col)
            test_mask = torch.isfinite(test_col)
            finite_train = train_col[train_mask]
            if int(finite_train.numel()) <= 0:
                if torch.any(test_mask):
                    test_out[test_mask, c] = 0.0
                continue

            if normalized_mode in ZSCORE_BASED_MODES:
                if normalized_mode == "train_winsorize_zscore":
                    lo = torch.quantile(finite_train, WINSORIZE_LO / 100.0)
                    hi = torch.quantile(finite_train, WINSORIZE_HI / 100.0)
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
                if float(std) < EPS:
                    std = torch.ones_like(std)
                train_norm = ((finite_train - mean) / std).to(torch.float32)
                test_norm = (
                    ((finite_test - mean) / std).to(torch.float32)
                    if finite_test is not None
                    else None
                )
                train_norm, test_norm = _postprocess_zscore_torch(
                    train_norm,
                    test_norm,
                    normalized_mode=normalized_mode,
                )
                train_out[train_mask, c] = train_norm
                if test_norm is not None:
                    test_out[test_mask, c] = test_norm
                continue

            if normalized_mode == "train_rankgauss":
                col_std = finite_train.std(unbiased=False)
                if float(col_std) < EPS:
                    train_out[train_mask, c] = 0.0
                    if torch.any(test_mask):
                        test_out[test_mask, c] = 0.0
                    continue
                train_out[train_mask, c] = rankgauss_train_torch(finite_train).to(torch.float32)
                if torch.any(test_mask):
                    train_sorted, _ = finite_train.sort()
                    test_out[test_mask, c] = rankgauss_test_torch(
                        train_sorted,
                        int(finite_train.shape[0]),
                        test_col[test_mask],
                    ).to(torch.float32)
                continue

            if normalized_mode in ROBUST_MODES:
                median = torch.quantile(finite_train, 0.5)
                q25 = torch.quantile(finite_train, 0.25)
                q75 = torch.quantile(finite_train, 0.75)
                iqr = q75 - q25
                if float(iqr) < EPS:
                    iqr = torch.ones_like(iqr)
                train_norm = ((finite_train - median) / iqr).to(torch.float32)
                if normalized_mode == "train_robust_tanh":
                    train_norm = smooth_tanh_torch(train_norm)
                train_out[train_mask, c] = train_norm
                if torch.any(test_mask):
                    test_norm = ((test_col[test_mask] - median) / iqr).to(torch.float32)
                    if normalized_mode == "train_robust_tanh":
                        test_norm = smooth_tanh_torch(test_norm)
                    test_out[test_mask, c] = test_norm
                continue

            raise ValueError(f"Unsupported input_normalization mode: {mode!r}")
        return train_out, test_out

    if normalized_mode in ZSCORE_BASED_MODES:
        if normalized_mode == "train_winsorize_zscore":
            lo = torch.quantile(train_stats, WINSORIZE_LO / 100.0, dim=0)
            hi = torch.quantile(train_stats, WINSORIZE_HI / 100.0, dim=0)
            train_stats = train_stats.clamp(min=lo, max=hi)
            test_stats = test_stats.clamp(min=lo, max=hi)

        mean = train_stats.mean(dim=0, keepdim=False)
        std = train_stats.std(dim=0, keepdim=False, unbiased=False)
        std = torch.where(std < EPS, torch.ones_like(std), std)
        train_norm = ((train_stats - mean) / std).to(torch.float32)
        test_norm = ((test_stats - mean) / std).to(torch.float32)
        processed_train_norm, processed_test_norm = _postprocess_zscore_torch(
            train_norm,
            test_norm,
            normalized_mode=normalized_mode,
        )
        assert processed_test_norm is not None
        return processed_train_norm, processed_test_norm

    if normalized_mode == "train_rankgauss":
        train_out = torch.empty_like(train_stats)
        test_out = torch.empty_like(test_stats)
        for c in range(int(train_stats.shape[1])):
            col_std = train_stats[:, c].std(unbiased=False)
            if col_std < EPS:
                train_out[:, c] = 0.0
                test_out[:, c] = 0.0
                continue
            train_out[:, c] = rankgauss_train_torch(train_stats[:, c])
            train_sorted, _ = train_stats[:, c].sort()
            test_out[:, c] = rankgauss_test_torch(train_sorted, train_stats.shape[0], test_stats[:, c])
        return train_out.to(torch.float32), test_out.to(torch.float32)

    if normalized_mode in ROBUST_MODES:
        median = torch.quantile(train_stats, 0.5, dim=0)
        q25 = torch.quantile(train_stats, 0.25, dim=0)
        q75 = torch.quantile(train_stats, 0.75, dim=0)
        iqr = q75 - q25
        iqr = torch.where(iqr < EPS, torch.ones_like(iqr), iqr)
        train_norm = ((train_stats - median) / iqr).to(torch.float32)
        test_norm = ((test_stats - median) / iqr).to(torch.float32)
        if normalized_mode == "train_robust_tanh":
            return smooth_tanh_torch(train_norm), smooth_tanh_torch(test_norm)
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
    normalized_mode = normalize_mode(mode)
    train = x_train.to(torch.float32)
    test = x_test.to(torch.float32)
    stats_dtype = tensor_stats_dtype(x_train.device)
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
            lo = torch.quantile(train_stats, WINSORIZE_LO / 100.0, dim=1)
            hi = torch.quantile(train_stats, WINSORIZE_HI / 100.0, dim=1)
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
        std = torch.where(std < EPS, torch.ones_like(std), std)

        train_norm = ((train_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
        test_norm = ((test_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
        processed_train_norm, processed_test_norm = _postprocess_zscore_torch(
            train_norm,
            test_norm,
            normalized_mode=normalized_mode,
        )
        assert processed_test_norm is not None

        train_out = train.clone()
        test_out = test.clone()
        train_out[train_mask] = processed_train_norm[train_mask]
        valid_test_mask = test_mask & has_finite.unsqueeze(1)
        test_out[valid_test_mask] = processed_test_norm[valid_test_mask]
        zero_test_mask = test_mask & (~has_finite).unsqueeze(1)
        test_out[zero_test_mask] = 0.0
        return train_out, test_out

    if normalized_mode == "train_winsorize_zscore":
        lo = torch.quantile(train_stats, WINSORIZE_LO / 100.0, dim=1)
        hi = torch.quantile(train_stats, WINSORIZE_HI / 100.0, dim=1)
        train_stats = train_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))
        test_stats = test_stats.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))

    mean = train_stats.mean(dim=1)
    std = train_stats.std(dim=1, unbiased=False)
    std = torch.where(std < EPS, torch.ones_like(std), std)
    train_norm = ((train_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
    test_norm = ((test_stats - mean.unsqueeze(1)) / std.unsqueeze(1)).to(torch.float32)
    processed_train_norm, processed_test_norm = _postprocess_zscore_torch(
        train_norm,
        test_norm,
        normalized_mode=normalized_mode,
    )
    assert processed_test_norm is not None
    return processed_train_norm, processed_test_norm


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

    normalized_mode = normalize_mode(mode)
    if normalized_mode == "none":
        return x_train.to(torch.float32), x_test.to(torch.float32)
    if normalized_mode in ZSCORE_BASED_MODES:
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
