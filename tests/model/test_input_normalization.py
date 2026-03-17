from __future__ import annotations

import numpy as np
import pytest
import torch

from tab_foundry.input_normalization import (
    _tensor_stats_dtype,
    normalize_train_test_arrays,
    normalize_train_test_tensors,
)


def test_train_zscore_clip_normalizes_from_train_only_and_clips() -> None:
    x_train = np.asarray([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    x_test = np.asarray([[1000.0, 1.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_zscore_clip",
    )

    assert np.allclose(train_norm[:, 0], np.asarray([-1.0, 1.0], dtype=np.float32))
    assert np.allclose(train_norm[:, 1], np.asarray([0.0, 0.0], dtype=np.float32))
    assert test_norm[0, 0] == np.float32(100.0)
    assert test_norm[0, 1] == np.float32(0.0)


def test_tensor_stats_dtype_uses_float32_on_mps() -> None:
    assert _tensor_stats_dtype(torch.device("cpu")) == torch.float64
    assert _tensor_stats_dtype(torch.device("mps")) == torch.float32


def test_train_rankgauss_produces_roughly_normal_output() -> None:
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((200, 3)).astype(np.float32)
    x_test = rng.standard_normal((50, 3)).astype(np.float32)

    train_norm, _ = normalize_train_test_arrays(x_train, x_test, mode="train_rankgauss")

    for c in range(3):
        col = train_norm[:, c]
        assert abs(float(np.mean(col))) < 0.15, f"col {c} mean too far from 0"
        assert abs(float(np.std(col)) - 1.0) < 0.3, f"col {c} std too far from 1"


def test_train_robust_uses_median_and_iqr() -> None:
    # Hand-computed: col0 = [1, 2, 3, 4, 5], median=3, Q25=2, Q75=4, IQR=2
    x_train = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    x_test = np.asarray([[7.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(x_train, x_test, mode="train_robust")

    expected_train = np.asarray([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32)
    np.testing.assert_allclose(train_norm, expected_train, atol=1e-5)
    # (7 - 3) / 2 = 2.0
    np.testing.assert_allclose(test_norm, np.asarray([[2.0]], dtype=np.float32), atol=1e-5)


def test_train_winsorize_zscore_clips_at_percentiles() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((1000, 2)).astype(np.float32)
    # Inject extreme outliers
    x_train[0, 0] = 100.0
    x_train[1, 0] = -100.0
    x_test = np.asarray([[200.0, 0.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(
        x_train, x_test, mode="train_winsorize_zscore",
    )

    # After winsorization the effective range is much smaller than raw,
    # so the z-scored values should not be extreme
    lo = np.percentile(x_train.astype(np.float64), 1.0, axis=0)
    hi = np.percentile(x_train.astype(np.float64), 99.0, axis=0)
    # Test values should be clipped to [lo, hi] before z-scoring,
    # so the result is bounded
    clipped_test_col0 = np.clip(200.0, lo[0], hi[0])
    clipped_train = np.clip(x_train.astype(np.float64), lo, hi)
    mean_c0 = clipped_train[:, 0].mean()
    std_c0 = clipped_train[:, 0].std()
    expected = (clipped_test_col0 - mean_c0) / std_c0
    np.testing.assert_allclose(float(test_norm[0, 0]), expected, atol=1e-4)


def test_preserve_non_finite_normalizes_only_finite_values() -> None:
    x_train = torch.tensor([[1.0, float("nan")], [3.0, 5.0]], dtype=torch.float32)
    x_test = torch.tensor([[5.0, float("nan")], [float("nan"), 7.0]], dtype=torch.float32)

    train_norm, test_norm = normalize_train_test_tensors(
        x_train,
        x_test,
        mode="train_zscore_clip",
        preserve_non_finite=True,
    )

    assert torch.isnan(train_norm[0, 1])
    assert torch.isnan(test_norm[0, 1])
    assert torch.isnan(test_norm[1, 0])
    assert train_norm[0, 0].item() == pytest.approx(-1.0)
    assert train_norm[1, 0].item() == pytest.approx(1.0)
    assert test_norm[0, 0].item() == pytest.approx(3.0)
    assert train_norm[1, 1].item() == pytest.approx(0.0)
    assert test_norm[1, 1].item() == pytest.approx(2.0)
