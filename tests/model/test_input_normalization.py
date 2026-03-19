from __future__ import annotations

import numpy as np
import pytest
import torch

from tab_foundry.input_normalization import (
    _SMOOTH_TAIL_LIMIT,
    _tensor_stats_dtype,
    normalize_train_test_arrays,
    normalize_train_test_tensors,
)


def _stack_2d_normalization(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    mode: str,
    preserve_non_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for batch_idx in range(int(x_train.shape[0])):
        train_norm, test_norm = normalize_train_test_tensors(
            x_train[batch_idx],
            x_test[batch_idx],
            mode=mode,
            preserve_non_finite=preserve_non_finite,
        )
        train_parts.append(train_norm)
        test_parts.append(test_norm)
    return torch.stack(train_parts, dim=0), torch.stack(test_parts, dim=0)


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


def test_train_zscore_tanh_uses_train_only_stats_and_bounded_smooth_tail() -> None:
    x_train = np.asarray([[0.0], [2.0]], dtype=np.float32)
    x_test = np.asarray([[1000.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_zscore_tanh",
    )

    expected_train = np.asarray(
        [
            [-_SMOOTH_TAIL_LIMIT * np.tanh(1.0 / _SMOOTH_TAIL_LIMIT)],
            [_SMOOTH_TAIL_LIMIT * np.tanh(1.0 / _SMOOTH_TAIL_LIMIT)],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(train_norm, expected_train, atol=1e-6)
    assert float(test_norm[0, 0]) == pytest.approx(_SMOOTH_TAIL_LIMIT, abs=1.0e-4)


def test_train_robust_tanh_uses_median_iqr_and_bounded_smooth_tail() -> None:
    x_train = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    x_test = np.asarray([[25.0]], dtype=np.float32)

    train_norm, test_norm = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_robust_tanh",
    )

    expected_raw_train = np.asarray([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32)
    expected_train = (_SMOOTH_TAIL_LIMIT * np.tanh(expected_raw_train / _SMOOTH_TAIL_LIMIT)).astype(
        np.float32,
    )
    np.testing.assert_allclose(train_norm, expected_train, atol=1e-6)
    assert float(test_norm[0, 0]) == pytest.approx(2.9960823, abs=1.0e-4)


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


def test_preserve_non_finite_keeps_signed_infinities_distinct() -> None:
    x_train = torch.tensor(
        [
            [float("inf"), float("-inf")],
            [2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    x_test = torch.tensor(
        [
            [float("inf"), float("-inf")],
            [6.0, 8.0],
        ],
        dtype=torch.float32,
    )

    train_norm, test_norm = normalize_train_test_tensors(
        x_train,
        x_test,
        mode="train_zscore_clip",
        preserve_non_finite=True,
    )

    assert torch.isposinf(train_norm[0, 0])
    assert torch.isneginf(train_norm[0, 1])
    assert torch.isposinf(test_norm[0, 0])
    assert torch.isneginf(test_norm[0, 1])
    assert train_norm[1, 0].item() == pytest.approx(0.0)
    assert train_norm[1, 1].item() == pytest.approx(0.0)
    assert test_norm[1, 0].item() == pytest.approx(4.0)
    assert test_norm[1, 1].item() == pytest.approx(4.0)


@pytest.mark.parametrize(
    "mode",
    [
        "train_zscore",
        "train_zscore_clip",
        "train_winsorize_zscore",
        "train_zscore_tanh",
    ],
)
def test_batched_zscore_modes_match_stacked_2d_behavior(mode: str) -> None:
    rng = np.random.default_rng(7)
    x_train = torch.tensor(rng.standard_normal((3, 9, 4)).astype(np.float32))
    x_test = torch.tensor(rng.standard_normal((3, 5, 4)).astype(np.float32))

    observed_train, observed_test = normalize_train_test_tensors(
        x_train,
        x_test,
        mode=mode,
    )
    expected_train, expected_test = _stack_2d_normalization(
        x_train,
        x_test,
        mode=mode,
    )

    torch.testing.assert_close(observed_train, expected_train, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(observed_test, expected_test, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize(
    "mode",
    [
        "train_rankgauss",
        "train_robust",
        "train_robust_tanh",
    ],
)
def test_batched_fallback_modes_match_stacked_2d_behavior(mode: str) -> None:
    rng = np.random.default_rng(17)
    x_train = torch.tensor(rng.standard_normal((2, 11, 3)).astype(np.float32))
    x_test = torch.tensor(rng.standard_normal((2, 4, 3)).astype(np.float32))

    observed_train, observed_test = normalize_train_test_tensors(
        x_train,
        x_test,
        mode=mode,
    )
    expected_train, expected_test = _stack_2d_normalization(
        x_train,
        x_test,
        mode=mode,
    )

    torch.testing.assert_close(observed_train, expected_train, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(observed_test, expected_test, atol=1.0e-6, rtol=1.0e-6)


def test_batched_preserve_non_finite_matches_stacked_2d_behavior() -> None:
    x_train = torch.tensor(
        [
            [
                [1.0, float("nan"), float("inf"), float("-inf")],
                [3.0, 5.0, 7.0, 9.0],
                [5.0, 7.0, 11.0, 13.0],
            ],
            [
                [2.0, float("nan"), float("inf"), float("-inf")],
                [6.0, 8.0, 10.0, 12.0],
                [8.0, 10.0, 14.0, 16.0],
            ],
        ],
        dtype=torch.float32,
    )
    x_test = torch.tensor(
        [
            [
                [7.0, float("nan"), float("inf"), float("-inf")],
                [9.0, 11.0, 15.0, 17.0],
            ],
            [
                [10.0, float("nan"), float("inf"), float("-inf")],
                [12.0, 14.0, 18.0, 20.0],
            ],
        ],
        dtype=torch.float32,
    )

    observed_train, observed_test = normalize_train_test_tensors(
        x_train,
        x_test,
        mode="train_zscore_clip",
        preserve_non_finite=True,
    )
    expected_train, expected_test = _stack_2d_normalization(
        x_train,
        x_test,
        mode="train_zscore_clip",
        preserve_non_finite=True,
    )

    torch.testing.assert_close(
        observed_train,
        expected_train,
        equal_nan=True,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        observed_test,
        expected_test,
        equal_nan=True,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
