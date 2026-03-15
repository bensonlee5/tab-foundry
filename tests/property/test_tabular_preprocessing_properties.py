from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tab_foundry.preprocessing import apply_fitted_preprocessor, fit_fitted_preprocessor


_FLOAT32S = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_INT_LABELS = st.integers(min_value=-50, max_value=50)
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


@st.composite
def _masked_matrix(
    draw: st.DrawFn,
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> np.ndarray:
    resolved_rows = rows if rows is not None else draw(st.integers(min_value=1, max_value=6))
    resolved_cols = cols if cols is not None else draw(st.integers(min_value=1, max_value=5))
    base = draw(hnp.arrays(np.float32, (resolved_rows, resolved_cols), elements=_FLOAT32S))
    nan_mask = draw(hnp.arrays(np.bool_, (resolved_rows, resolved_cols), elements=st.booleans()))
    matrix = base.copy()
    matrix[nan_mask] = np.nan
    return matrix


@st.composite
def _regression_case(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    train_rows = draw(st.integers(min_value=1, max_value=6))
    test_rows = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=5))
    x_train = draw(_masked_matrix(rows=train_rows, cols=cols))
    x_test = draw(_masked_matrix(rows=test_rows, cols=cols))
    y_train = draw(hnp.arrays(np.float32, train_rows, elements=_FLOAT32S))
    y_test = draw(hnp.arrays(np.float32, test_rows, elements=_FLOAT32S))
    all_nan_fill = float(draw(_FLOAT32S))
    return x_train, y_train, x_test, y_test, all_nan_fill


@st.composite
def _classification_case(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_rows = draw(st.integers(min_value=1, max_value=6))
    test_rows = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=5))
    x_train = draw(_masked_matrix(rows=train_rows, cols=cols))
    x_test = draw(_masked_matrix(rows=test_rows, cols=cols))
    label_pool = draw(st.lists(_INT_LABELS, min_size=1, max_size=min(5, train_rows + 2), unique=True))
    y_train = np.asarray(
        draw(st.lists(st.sampled_from(label_pool), min_size=train_rows, max_size=train_rows)),
        dtype=np.int64,
    )
    unseen_label = draw(st.one_of(st.none(), _INT_LABELS.filter(lambda value: value not in label_pool)))
    test_pool = label_pool if unseen_label is None else [*label_pool, unseen_label]
    y_test = np.asarray(
        draw(st.lists(st.sampled_from(test_pool), min_size=test_rows, max_size=test_rows)),
        dtype=np.int64,
    )
    return x_train, y_train, x_test, y_test


@st.composite
def _all_nan_column_case(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    x_train, y_train, x_test, y_test, all_nan_fill = draw(_regression_case())
    col_idx = draw(st.integers(min_value=0, max_value=int(x_train.shape[1]) - 1))
    x_train = x_train.copy()
    x_train[:, col_idx] = np.nan
    return x_train, y_train, x_test, y_test, col_idx, all_nan_fill


@settings(deadline=None, max_examples=35)
@given(case=_regression_case())
def test_regression_fit_apply_imputes_with_fitted_fill_values(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    x_train, y_train, x_test, y_test, all_nan_fill = case

    state = fit_fitted_preprocessor(
        task="regression",
        x_train=x_train,
        y_train=y_train,
        all_nan_fill=all_nan_fill,
    )
    processed = apply_fitted_preprocessor(
        task="regression",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        impute_missing=True,
    )

    fill_values = np.asarray(state.missing_value_policy.fill_values, dtype=np.float32)
    expected_train = np.where(np.isnan(x_train), fill_values[np.newaxis, :], x_train).astype(np.float32)
    expected_test = np.where(np.isnan(x_test), fill_values[np.newaxis, :], x_test).astype(np.float32)

    assert processed.x_train.shape == x_train.shape
    assert processed.x_test.shape == x_test.shape
    assert processed.y_train.shape == y_train.shape
    assert processed.y_test is not None
    assert processed.y_test.shape == y_test.shape
    assert processed.num_classes is None
    assert processed.valid_test_mask is None
    assert not np.isnan(processed.x_train).any()
    assert not np.isnan(processed.x_test).any()
    np.testing.assert_allclose(processed.x_train, expected_train, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(processed.x_test, expected_test, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(processed.y_train, y_train.astype(np.float32), atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(processed.y_test, y_test.astype(np.float32), atol=1.0e-6, rtol=1.0e-6)


@settings(deadline=None, max_examples=35)
@given(case=_all_nan_column_case())
def test_all_nan_train_columns_use_all_nan_fill(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float],
) -> None:
    x_train, y_train, x_test, y_test, col_idx, all_nan_fill = case

    state = fit_fitted_preprocessor(
        task="regression",
        x_train=x_train,
        y_train=y_train,
        all_nan_fill=all_nan_fill,
    )
    processed = apply_fitted_preprocessor(
        task="regression",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        impute_missing=True,
    )

    expected_fill = np.float32(all_nan_fill)
    assert state.missing_value_policy.fill_values[col_idx] == pytest.approx(float(expected_fill))
    np.testing.assert_allclose(
        processed.x_train[:, col_idx],
        np.full(x_train.shape[0], expected_fill, dtype=np.float32),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    expected_test_col = np.where(np.isnan(x_test[:, col_idx]), expected_fill, x_test[:, col_idx]).astype(
        np.float32
    )
    np.testing.assert_allclose(
        processed.x_test[:, col_idx],
        expected_test_col,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


@settings(deadline=None, max_examples=35)
@given(case=_classification_case())
def test_classification_fit_apply_remaps_labels_and_filters_unseen_test_targets(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x_train, y_train, x_test, y_test = case

    state = fit_fitted_preprocessor(task="classification", x_train=x_train, y_train=y_train)
    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        impute_missing=True,
    )

    expected_label_values = np.unique(y_train)
    expected_mask = np.isin(y_test, expected_label_values)

    assert processed.num_classes == int(expected_label_values.shape[0])
    assert processed.valid_test_mask is not None
    assert processed.valid_test_mask.tolist() == expected_mask.tolist()
    assert processed.y_train.shape == y_train.shape
    assert processed.x_train.shape == x_train.shape
    assert processed.x_test.shape[1] == x_test.shape[1]
    assert processed.x_test.shape[0] == int(expected_mask.sum())
    assert not np.isnan(processed.x_train).any()
    assert not np.isnan(processed.x_test).any()
    assert sorted(np.unique(processed.y_train).tolist()) == list(range(processed.num_classes))
    assert processed.y_test is not None
    assert processed.y_test.shape == (int(expected_mask.sum()),)
    if processed.y_test.size > 0:
        assert int(processed.y_test.min()) >= 0
        assert int(processed.y_test.max()) < processed.num_classes


@settings(deadline=None, max_examples=25)
@given(
    task=st.sampled_from(("classification", "regression")),
    train_rows=st.integers(min_value=1, max_value=6),
    cols=st.integers(min_value=1, max_value=5),
)
def test_fit_preprocessor_rejects_mismatched_train_rows(
    task: str,
    train_rows: int,
    cols: int,
) -> None:
    x_train = np.zeros((train_rows, cols), dtype=np.float32)
    if task == "classification":
        y_train = np.zeros(train_rows + 1, dtype=np.int64)
    else:
        y_train = np.zeros(train_rows + 1, dtype=np.float32)

    with pytest.raises(RuntimeError, match="x_train row count must match y_train"):
        _ = fit_fitted_preprocessor(task=task, x_train=x_train, y_train=y_train)


@settings(deadline=None, max_examples=25)
@given(
    task=st.sampled_from(("classification", "regression")),
    train_rows=st.integers(min_value=1, max_value=6),
    test_rows=st.integers(min_value=1, max_value=6),
    cols=st.integers(min_value=1, max_value=5),
)
def test_apply_preprocessor_rejects_mismatched_test_rows(
    task: str,
    train_rows: int,
    test_rows: int,
    cols: int,
) -> None:
    x_train = np.zeros((train_rows, cols), dtype=np.float32)
    x_test = np.zeros((test_rows, cols), dtype=np.float32)
    if task == "classification":
        y_train = np.arange(train_rows, dtype=np.int64)
        y_test = np.arange(test_rows + 1, dtype=np.int64)
    else:
        y_train = np.zeros(train_rows, dtype=np.float32)
        y_test = np.zeros(test_rows + 1, dtype=np.float32)

    state = fit_fitted_preprocessor(task=task, x_train=x_train, y_train=y_train)

    with pytest.raises(RuntimeError, match="x_test row count must match y_test"):
        _ = apply_fitted_preprocessor(
            task=task,
            state=state,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
