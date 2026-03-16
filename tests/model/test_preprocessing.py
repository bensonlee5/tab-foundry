from __future__ import annotations

import numpy as np
import pytest

from tab_foundry.preprocessing import apply_fitted_preprocessor, fit_fitted_preprocessor


def test_classification_preprocessing_persists_fill_values_and_filters_unseen_test_labels() -> None:
    x_train = np.asarray(
        [
            [1.0, np.nan, 3.0],
            [3.0, 5.0, np.nan],
            [5.0, 7.0, 9.0],
        ],
        dtype=np.float32,
    )
    y_train = np.asarray([7, 3, 7], dtype=np.int64)
    x_test = np.asarray(
        [
            [np.nan, 10.0, 11.0],
            [2.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    y_test = np.asarray([7, 999], dtype=np.int64)

    state = fit_fitted_preprocessor(
        task="classification",
        x_train=x_train,
        y_train=y_train,
    )

    assert state.feature_ids == [0, 1, 2]
    assert state.missing_value_policy.fill_values == [3.0, 6.0, 6.0]
    assert state.classification_label_policy is not None
    assert state.classification_label_policy.label_values == [3, 7]

    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    assert processed.num_classes == 2
    assert processed.valid_test_mask is not None
    assert processed.valid_test_mask.tolist() == [True, False]
    assert processed.y_train.tolist() == [1, 0, 1]
    assert processed.y_test is not None
    assert processed.y_test.tolist() == [1]
    assert processed.x_train.tolist() == [
        [1.0, 6.0, 3.0],
        [3.0, 5.0, 6.0],
        [5.0, 7.0, 9.0],
    ]
    assert processed.x_test.tolist() == [[3.0, 10.0, 11.0]]


def test_regression_preprocessing_round_trips_fill_values_without_label_policy() -> None:
    x_train = np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    y_train = np.asarray([0.25, 0.75], dtype=np.float32)
    x_test = np.asarray([[np.nan, 5.0]], dtype=np.float32)

    state = fit_fitted_preprocessor(
        task="regression",
        x_train=x_train,
        y_train=y_train,
    )
    assert state.classification_label_policy is None
    assert state.missing_value_policy.fill_values == [2.0, 4.0]

    processed = apply_fitted_preprocessor(
        task="regression",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=np.asarray([1.5], dtype=np.float32),
    )

    assert processed.num_classes is None
    assert processed.valid_test_mask is None
    assert processed.x_test.tolist() == [[2.0, 5.0]]
    assert processed.y_test is not None
    assert processed.y_test.tolist() == [1.5]


def test_classification_preprocessing_can_skip_imputation_and_still_remap_labels() -> None:
    x_train = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    y_train = np.asarray([10, 20], dtype=np.int64)
    x_test = np.asarray([[np.nan, 4.0], [5.0, np.nan]], dtype=np.float32)
    y_test = np.asarray([20, 999], dtype=np.int64)

    state = fit_fitted_preprocessor(
        task="classification",
        x_train=x_train,
        y_train=y_train,
    )
    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        impute_missing=False,
    )

    assert np.isnan(processed.x_train[0, 1])
    assert np.isnan(processed.x_test[0, 0])
    assert processed.y_train.tolist() == [0, 1]
    assert processed.y_test is not None
    assert processed.y_test.tolist() == [1]
    assert processed.num_classes == 2


def test_classification_preprocessing_supports_categorical_feature_state() -> None:
    x_train = np.asarray(
        [
            [1.0, 10.0, np.nan],
            [3.0, 10.0, 1.0],
            [np.nan, 20.0, 2.0],
        ],
        dtype=np.float32,
    )
    y_train = np.asarray([10, 20, 10], dtype=np.int64)
    x_test = np.asarray(
        [
            [7.0, 20.0, 2.0],
            [9.0, 30.0, np.nan],
        ],
        dtype=np.float32,
    )
    y_test = np.asarray([20, 10], dtype=np.int64)

    state = fit_fitted_preprocessor(
        task="classification",
        x_train=x_train,
        y_train=y_train,
        feature_types=["num", "categorical", "cat"],
    )

    assert state.missing_value_policy.fill_values == [2.0, 0.0, 0.0]
    assert state.categorical_feature_policy.feature_types == ["num", "cat", "cat"]
    assert state.categorical_feature_policy.cardinalities == [0, 2, 2]
    assert state.categorical_feature_policy.train_value_vocab == [[], [10.0, 20.0], [1.0, 2.0]]

    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_types=["num", "cat", "categorical"],
    )

    assert processed.x_train.tolist() == [
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
    ]
    assert processed.x_test.tolist() == [
        [7.0, 1.0, 1.0],
        [9.0, 2.0, 2.0],
    ]
    assert processed.feature_state is not None
    assert processed.feature_state.categorical_mask.tolist() == [False, True, True]
    assert processed.feature_state.categorical_cardinalities.tolist() == [0, 2, 2]
    assert processed.feature_state.x_train_categorical_ids.tolist() == [
        [0, 0, 2],
        [0, 0, 0],
        [0, 1, 1],
    ]
    assert processed.feature_state.x_test_categorical_ids.tolist() == [
        [0, 1, 1],
        [0, 2, 2],
    ]


def test_classification_preprocessing_preserves_large_integer_categorical_ids() -> None:
    x_train = np.asarray([[16777216], [16777217]], dtype=np.int64)
    y_train = np.asarray([0, 1], dtype=np.int64)
    x_test = np.asarray([[16777217], [16777218]], dtype=np.int64)
    y_test = np.asarray([1, 0], dtype=np.int64)

    state = fit_fitted_preprocessor(
        task="classification",
        x_train=x_train,
        y_train=y_train,
        feature_types=["cat"],
    )

    assert state.categorical_feature_policy.cardinalities == [2]
    assert state.categorical_feature_policy.train_value_vocab == [[16777216, 16777217]]

    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_types=["categorical"],
    )

    assert processed.x_train.tolist() == [[0.0], [1.0]]
    assert processed.x_test.tolist() == [[1.0], [2.0]]
    assert processed.feature_state is not None
    assert processed.feature_state.categorical_mask.tolist() == [True]
    assert processed.feature_state.categorical_cardinalities.tolist() == [2]
    assert processed.feature_state.x_train_categorical_ids.tolist() == [[0], [1]]
    assert processed.feature_state.x_test_categorical_ids.tolist() == [[1], [2]]


def test_apply_fitted_preprocessor_rejects_feature_type_mismatch() -> None:
    state = fit_fitted_preprocessor(
        task="classification",
        x_train=np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        y_train=np.asarray([0, 1], dtype=np.int64),
        feature_types=["num", "cat"],
    )

    with pytest.raises(RuntimeError, match="feature_types do not match fitted preprocessing state"):
        _ = apply_fitted_preprocessor(
            task="classification",
            state=state,
            x_train=np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
            y_train=np.asarray([0, 1], dtype=np.int64),
            x_test=np.asarray([[3.0, 10.0]], dtype=np.float32),
            y_test=np.asarray([0], dtype=np.int64),
            feature_types=["num", "num"],
        )


def test_fit_fitted_preprocessor_rejects_invalid_feature_type_inputs() -> None:
    with pytest.raises(RuntimeError, match="feature_types length must match feature count"):
        _ = fit_fitted_preprocessor(
            task="classification",
            x_train=np.asarray([[1.0, 2.0]], dtype=np.float32),
            y_train=np.asarray([0], dtype=np.int64),
            feature_types=["num"],
        )

    with pytest.raises(RuntimeError, match="feature_types entries must be one of"):
        _ = fit_fitted_preprocessor(
            task="classification",
            x_train=np.asarray([[1.0, 2.0]], dtype=np.float32),
            y_train=np.asarray([0], dtype=np.int64),
            feature_types=["num", "bogus"],
        )


def test_classification_preprocessing_rejects_misaligned_train_rows() -> None:
    with pytest.raises(RuntimeError, match="x_train row count must match y_train"):
        _ = fit_fitted_preprocessor(
            task="classification",
            x_train=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            y_train=np.asarray([10, 20], dtype=np.int64),
        )


def test_classification_preprocessing_rejects_misaligned_test_rows() -> None:
    state = fit_fitted_preprocessor(
        task="classification",
        x_train=np.asarray([[1.0], [2.0]], dtype=np.float32),
        y_train=np.asarray([10, 20], dtype=np.int64),
    )

    with pytest.raises(RuntimeError, match="x_test row count must match y_test"):
        _ = apply_fitted_preprocessor(
            task="classification",
            state=state,
            x_train=np.asarray([[1.0], [2.0]], dtype=np.float32),
            y_train=np.asarray([10, 20], dtype=np.int64),
            x_test=np.asarray([[3.0], [4.0]], dtype=np.float32),
            y_test=np.asarray([10], dtype=np.int64),
        )


def test_regression_preprocessing_rejects_misaligned_train_rows() -> None:
    with pytest.raises(RuntimeError, match="x_train row count must match y_train"):
        _ = fit_fitted_preprocessor(
            task="regression",
            x_train=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            y_train=np.asarray([0.25, 0.75], dtype=np.float32),
        )
