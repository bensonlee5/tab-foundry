from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tab_foundry.input_normalization import (
    _CLIP_VALUE,
    _SMOOTH_TAIL_LIMIT,
    normalize_train_test_arrays,
    normalize_train_test_tensors,
    SUPPORTED_INPUT_NORMALIZATION_MODES,
)


_FLOAT32S = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

_MODES_WITH_TRAIN_STATS = (
    "train_zscore",
    "train_zscore_clip",
    "train_rankgauss",
    "train_robust",
    "train_winsorize_zscore",
    "train_zscore_tanh",
    "train_robust_tanh",
)


@st.composite
def _train_test_arrays(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    train_rows = draw(st.integers(min_value=1, max_value=6))
    test_rows = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=6))
    x_train = draw(hnp.arrays(np.float32, (train_rows, cols), elements=_FLOAT32S))
    x_test = draw(hnp.arrays(np.float32, (test_rows, cols), elements=_FLOAT32S))
    return x_train, x_test


@st.composite
def _constant_column_case(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int]:
    x_train, x_test = draw(_train_test_arrays())
    col_idx = draw(st.integers(min_value=0, max_value=int(x_train.shape[1]) - 1))
    constant_value = np.float32(draw(_FLOAT32S))
    x_train = x_train.copy()
    x_train[:, col_idx] = constant_value
    return x_train, x_test, col_idx


@st.composite
def _train_and_two_tests(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_rows = draw(st.integers(min_value=1, max_value=6))
    test_rows_a = draw(st.integers(min_value=1, max_value=6))
    test_rows_b = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=6))
    x_train = draw(hnp.arrays(np.float32, (train_rows, cols), elements=_FLOAT32S))
    x_test_a = draw(hnp.arrays(np.float32, (test_rows_a, cols), elements=_FLOAT32S))
    x_test_b = draw(hnp.arrays(np.float32, (test_rows_b, cols), elements=_FLOAT32S))
    return x_train, x_test_a, x_test_b


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays(), mode=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES))
def test_normalizers_preserve_shape_and_float32_dtype(
    data: tuple[np.ndarray, np.ndarray],
    mode: str,
) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode=mode)
    train_t, test_t = normalize_train_test_tensors(
        torch.from_numpy(x_train),
        torch.from_numpy(x_test),
        mode=mode,
    )

    assert train_np.shape == x_train.shape
    assert test_np.shape == x_test.shape
    assert train_np.dtype == np.float32
    assert test_np.dtype == np.float32
    assert train_t.shape == x_train.shape
    assert test_t.shape == x_test.shape
    assert train_t.dtype == torch.float32
    assert test_t.dtype == torch.float32


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays())
def test_none_mode_matches_float32_cast_only(data: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode="none")
    train_t, test_t = normalize_train_test_tensors(
        torch.from_numpy(x_train),
        torch.from_numpy(x_test),
        mode="none",
    )

    np.testing.assert_array_equal(train_np, np.asarray(x_train, dtype=np.float32))
    np.testing.assert_array_equal(test_np, np.asarray(x_test, dtype=np.float32))
    assert torch.equal(train_t, torch.from_numpy(np.asarray(x_train, dtype=np.float32)))
    assert torch.equal(test_t, torch.from_numpy(np.asarray(x_test, dtype=np.float32)))


@settings(deadline=None, max_examples=40)
@given(
    case=_train_and_two_tests(),
    mode=st.sampled_from(_MODES_WITH_TRAIN_STATS),
)
def test_train_normalization_depends_only_on_train_split(
    case: tuple[np.ndarray, np.ndarray, np.ndarray],
    mode: str,
) -> None:
    x_train, x_test_a, x_test_b = case

    first_train, _ = normalize_train_test_arrays(x_train, x_test_a, mode=mode)
    second_train, _ = normalize_train_test_arrays(x_train, x_test_b, mode=mode)

    np.testing.assert_allclose(first_train, second_train, atol=1.0e-6, rtol=1.0e-6)


@settings(deadline=None, max_examples=40)
@given(case=_constant_column_case(), mode=st.sampled_from(_MODES_WITH_TRAIN_STATS))
def test_constant_train_columns_normalize_to_zero(
    case: tuple[np.ndarray, np.ndarray, int],
    mode: str,
) -> None:
    x_train, x_test, col_idx = case

    train_np, _ = normalize_train_test_arrays(x_train, x_test, mode=mode)

    np.testing.assert_allclose(train_np[:, col_idx], np.zeros(x_train.shape[0], dtype=np.float32))


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays())
def test_clipped_mode_stays_within_clip_bounds(data: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode="train_zscore_clip")

    assert float(np.max(np.abs(train_np))) <= _CLIP_VALUE + 1.0e-6
    assert float(np.max(np.abs(test_np))) <= _CLIP_VALUE + 1.0e-6


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays(), mode=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES))
def test_numpy_and_torch_normalizers_agree(
    data: tuple[np.ndarray, np.ndarray],
    mode: str,
) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode=mode)
    train_t, test_t = normalize_train_test_tensors(
        torch.from_numpy(x_train),
        torch.from_numpy(x_test),
        mode=mode,
    )

    np.testing.assert_allclose(train_np, train_t.numpy(), atol=1.0e-5, rtol=1.0e-5)
    np.testing.assert_allclose(test_np, test_t.numpy(), atol=1.0e-5, rtol=1.0e-5)


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays())
def test_rankgauss_output_bounded(data: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode="train_rankgauss")

    assert np.all(np.isfinite(train_np))
    assert np.all(np.isfinite(test_np))
    # erfinv maps (0,1) quantiles to roughly [-5,5] for practical sample sizes
    assert float(np.max(np.abs(train_np))) < 10.0
    assert float(np.max(np.abs(test_np))) < 10.0


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays())
def test_winsorize_clips_within_train_percentiles(data: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = data

    train_norm, test_norm = normalize_train_test_arrays(
        x_train, x_test, mode="train_winsorize_zscore",
    )

    # Output should be finite (no NaN/Inf from division)
    assert np.all(np.isfinite(train_norm))
    assert np.all(np.isfinite(test_norm))


@settings(deadline=None, max_examples=40)
@given(data=_train_test_arrays(), mode=st.sampled_from(("train_zscore_tanh", "train_robust_tanh")))
def test_smooth_tail_modes_are_bounded(data: tuple[np.ndarray, np.ndarray], mode: str) -> None:
    x_train, x_test = data

    train_np, test_np = normalize_train_test_arrays(x_train, x_test, mode=mode)

    assert np.all(np.isfinite(train_np))
    assert np.all(np.isfinite(test_np))
    assert float(np.max(np.abs(train_np))) <= _SMOOTH_TAIL_LIMIT + 1.0e-6
    assert float(np.max(np.abs(test_np))) <= _SMOOTH_TAIL_LIMIT + 1.0e-6
