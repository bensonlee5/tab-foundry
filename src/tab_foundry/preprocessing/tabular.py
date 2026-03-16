"""Shared task-level preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tab_foundry.feature_state import PreprocessedFeatureState

from .state import (
    CATEGORICAL_FEATURE_TYPE,
    CATEGORICAL_VALUE_POLICY_OOV_BUCKET,
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    NUMERIC_FEATURE_TYPE,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
    CategoricalFeaturePolicyState,
    ClassificationLabelPolicyState,
    FittedPreprocessorState,
    MissingValuePolicyState,
    normalize_feature_types,
)
from .surface import (
    SUPPORTED_LABEL_MAPPINGS,
    SUPPORTED_UNSEEN_TEST_LABEL_POLICIES,
)


SUPPORTED_TASKS = ("classification", "regression")


@dataclass(slots=True)
class PreprocessedTaskArrays:
    """Task arrays after shared preprocessing."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray | None
    num_classes: int | None
    valid_test_mask: np.ndarray | None
    feature_state: PreprocessedFeatureState | None


def _require_task(task: str) -> str:
    normalized = str(task).strip().lower()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported preprocessing task: {task!r}")
    return normalized


def _require_matrix(x: Any, *, context: str) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim != 2:
        raise RuntimeError(f"{context} must be a rank-2 matrix, got shape={array.shape!r}")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise RuntimeError(f"{context} must contain numeric values, got dtype={array.dtype!r}")
    return array


def _require_vector(y: Any, *, dtype: Any, context: str) -> np.ndarray:
    array = np.asarray(y, dtype=dtype)
    if array.ndim != 1:
        raise RuntimeError(f"{context} must be rank-1, got shape={array.shape!r}")
    return array


def _require_matching_rows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_context: str,
    y_context: str,
) -> None:
    if int(x.shape[0]) != int(y.shape[0]):
        raise RuntimeError(
            f"{x_context} row count must match {y_context}: "
            f"{x_context}_rows={x.shape[0]}, {y_context}_rows={y.shape[0]}"
        )


def _fit_fill_values(
    x_train: np.ndarray,
    *,
    feature_types: Sequence[str],
    all_nan_fill: float,
) -> list[float]:
    fill_values = np.full((int(x_train.shape[1]),), float(all_nan_fill), dtype=np.float32)
    numeric_mask = np.asarray(
        [feature_type == NUMERIC_FEATURE_TYPE for feature_type in feature_types],
        dtype=bool,
    )
    if np.any(numeric_mask):
        means = np.nanmean(x_train[:, numeric_mask], axis=0)
        means = np.where(np.isnan(means), float(all_nan_fill), means)
        fill_values[numeric_mask] = means.astype(np.float32, copy=False)
    return [float(value) for value in fill_values.tolist()]


def _fit_categorical_feature_policy(
    x_train: np.ndarray,
    *,
    feature_types: Sequence[str],
) -> CategoricalFeaturePolicyState:
    train_value_vocab: list[list[int | float]] = []
    value_dtypes: list[str | None] = []
    cardinalities: list[int] = []
    for col_idx, feature_type in enumerate(feature_types):
        if feature_type != CATEGORICAL_FEATURE_TYPE:
            train_value_vocab.append([])
            value_dtypes.append(None)
            cardinalities.append(0)
            continue
        column = np.asarray(x_train[:, col_idx])
        finite_values = column[np.isfinite(column)]
        vocab = np.unique(finite_values)
        train_value_vocab.append(list(vocab.tolist()))
        value_dtypes.append(np.dtype(column.dtype).name)
        cardinalities.append(int(vocab.shape[0]))
    return CategoricalFeaturePolicyState(
        feature_types=list(feature_types),
        train_value_vocab=train_value_vocab,
        value_dtypes=value_dtypes,
        cardinalities=cardinalities,
        value_policy=CATEGORICAL_VALUE_POLICY_OOV_BUCKET,
    )


def _encode_categorical_column(
    values: np.ndarray,
    *,
    vocab: np.ndarray,
    lookup_dtype: str | None,
    context: str,
    allow_oov: bool,
) -> np.ndarray:
    if lookup_dtype is not None:
        vocab = np.asarray(vocab, dtype=np.dtype(lookup_dtype))
    oov_id = int(vocab.shape[0])
    encoded = np.full(values.shape, oov_id, dtype=np.int64)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return encoded
    if vocab.size <= 0:
        if not allow_oov:
            raise RuntimeError(f"{context} contains finite values absent from categorical vocabulary")
        return encoded

    finite_idx = np.where(finite_mask)[0]
    finite_values = values[finite_mask]
    if lookup_dtype is not None and np.issubdtype(vocab.dtype, np.floating):
        finite_values = finite_values.astype(vocab.dtype, copy=False)
    positions = np.searchsorted(vocab, finite_values)
    clipped = np.clip(positions, 0, vocab.shape[0] - 1)
    valid = vocab[clipped] == finite_values
    encoded[finite_idx[valid]] = positions[valid].astype(np.int64, copy=False)
    if not allow_oov and not bool(np.all(valid)):
        raise RuntimeError(f"{context} contains values absent from fitted categorical vocabulary")
    return encoded


def fit_fitted_preprocessor(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    feature_types: Sequence[str] | None = None,
    all_nan_fill: float = 0.0,
    label_mapping: str = CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    unseen_test_label_policy: str = UNSEEN_TEST_LABEL_POLICY_FILTER,
) -> FittedPreprocessorState:
    """Fit the shared preprocessing state from one train split."""

    normalized_task = _require_task(task)
    normalized_label_mapping = str(label_mapping).strip()
    if normalized_label_mapping not in SUPPORTED_LABEL_MAPPINGS:
        raise ValueError(
            f"label_mapping must be one of {SUPPORTED_LABEL_MAPPINGS}, got {label_mapping!r}"
        )
    normalized_unseen_test_label_policy = str(unseen_test_label_policy).strip()
    if normalized_unseen_test_label_policy not in SUPPORTED_UNSEEN_TEST_LABEL_POLICIES:
        raise ValueError(
            "unseen_test_label_policy must be one of "
            f"{SUPPORTED_UNSEEN_TEST_LABEL_POLICIES}, got {unseen_test_label_policy!r}"
        )
    x_train_matrix = _require_matrix(x_train, context="x_train")
    feature_ids = list(range(int(x_train_matrix.shape[1])))
    normalized_feature_types = normalize_feature_types(
        feature_types,
        width=int(x_train_matrix.shape[1]),
    )
    missing_value_policy = MissingValuePolicyState(
        strategy=MISSING_VALUE_STRATEGY_TRAIN_MEAN,
        all_nan_fill=float(all_nan_fill),
        fill_values=_fit_fill_values(
            x_train_matrix,
            feature_types=normalized_feature_types,
            all_nan_fill=float(all_nan_fill),
        ),
    )
    categorical_feature_policy = _fit_categorical_feature_policy(
        x_train_matrix,
        feature_types=normalized_feature_types,
    )

    classification_label_policy: ClassificationLabelPolicyState | None = None
    if normalized_task == "classification":
        labels = _require_vector(y_train, dtype=np.int64, context="y_train")
        _require_matching_rows(x_train_matrix, labels, x_context="x_train", y_context="y_train")
        label_values = np.unique(labels)
        if label_values.size <= 0:
            raise RuntimeError("classification train split has no labels")
        classification_label_policy = ClassificationLabelPolicyState(
            mapping=normalized_label_mapping,
            unseen_test_label=normalized_unseen_test_label_policy,
            label_values=[int(value) for value in label_values.tolist()],
        )
    else:
        targets = _require_vector(y_train, dtype=np.float32, context="y_train")
        _require_matching_rows(x_train_matrix, targets, x_context="x_train", y_context="y_train")

    return FittedPreprocessorState(
        feature_order_policy=FEATURE_ORDER_POLICY_POSITIONAL,
        feature_ids=feature_ids,
        missing_value_policy=missing_value_policy,
        categorical_feature_policy=categorical_feature_policy,
        classification_label_policy=classification_label_policy,
        dtype_policy=dict(DTYPE_POLICY),
    )


def _apply_feature_preprocessing(
    state: FittedPreprocessorState,
    *,
    x_train: Any,
    x_test: Any,
    impute_missing: bool,
) -> tuple[np.ndarray, np.ndarray, PreprocessedFeatureState]:
    x_train_matrix = _require_matrix(x_train, context="x_train")
    x_test_matrix = _require_matrix(x_test, context="x_test")
    expected_width = len(state.feature_ids)
    if int(x_train_matrix.shape[1]) != expected_width or int(x_test_matrix.shape[1]) != expected_width:
        raise RuntimeError(
            "feature count does not match fitted preprocessing state: "
            f"expected={expected_width}, got_train={x_train_matrix.shape[1]}, got_test={x_test_matrix.shape[1]}"
        )

    fill_values = np.asarray(state.missing_value_policy.fill_values, dtype=np.float32)
    train = x_train_matrix.astype(np.float32, copy=True)
    test = x_test_matrix.astype(np.float32, copy=True)
    categorical_policy = state.categorical_feature_policy
    categorical_mask = np.asarray(
        [feature_type == CATEGORICAL_FEATURE_TYPE for feature_type in categorical_policy.feature_types],
        dtype=bool,
    )
    categorical_cardinalities = np.asarray(categorical_policy.cardinalities, dtype=np.int64)
    train_categorical_ids = np.zeros(train.shape, dtype=np.int64)
    test_categorical_ids = np.zeros(test.shape, dtype=np.int64)

    for col_idx, is_categorical in enumerate(categorical_mask.tolist()):
        if is_categorical:
            vocab = np.asarray(categorical_policy.train_value_vocab[col_idx])
            vocab_dtype = categorical_policy.value_dtypes[col_idx]
            train_ids = _encode_categorical_column(
                x_train_matrix[:, col_idx],
                vocab=vocab,
                lookup_dtype=vocab_dtype,
                context=f"x_train column {col_idx}",
                allow_oov=False,
            )
            test_ids = _encode_categorical_column(
                x_test_matrix[:, col_idx],
                vocab=vocab,
                lookup_dtype=vocab_dtype,
                context=f"x_test column {col_idx}",
                allow_oov=True,
            )
            train_categorical_ids[:, col_idx] = train_ids
            test_categorical_ids[:, col_idx] = test_ids
            train[:, col_idx] = train_ids.astype(np.float32, copy=False)
            test[:, col_idx] = test_ids.astype(np.float32, copy=False)
            continue
        if not impute_missing:
            continue
        train_nan = np.isnan(train[:, col_idx])
        if np.any(train_nan):
            train[train_nan, col_idx] = float(fill_values[col_idx])
        test_nan = np.isnan(test[:, col_idx])
        if np.any(test_nan):
            test[test_nan, col_idx] = float(fill_values[col_idx])

    return train, test, PreprocessedFeatureState(
        categorical_mask=categorical_mask,
        categorical_cardinalities=categorical_cardinalities,
        x_train_categorical_ids=train_categorical_ids,
        x_test_categorical_ids=test_categorical_ids,
    )


def _map_classification_targets(
    state: FittedPreprocessorState,
    *,
    y_train: Any,
    y_test: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, int, np.ndarray | None]:
    label_policy = state.classification_label_policy
    if label_policy is None:
        raise RuntimeError("classification preprocessing state is missing label policy")

    label_values = np.asarray(label_policy.label_values, dtype=np.int64)
    train_raw = _require_vector(y_train, dtype=np.int64, context="y_train")

    train_pos = np.searchsorted(label_values, train_raw)
    train_in_bounds = train_pos < label_values.shape[0]
    train_valid = train_in_bounds & (label_values[np.clip(train_pos, 0, label_values.shape[0] - 1)] == train_raw)
    if not bool(np.all(train_valid)):
        raise RuntimeError("classification y_train contains labels absent from fitted preprocessing state")
    remapped_train = train_pos.astype(np.int64, copy=False)

    remapped_test: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    if y_test is not None:
        test_raw = _require_vector(y_test, dtype=np.int64, context="y_test")
        test_pos = np.searchsorted(label_values, test_raw)
        test_in_bounds = test_pos < label_values.shape[0]
        test_valid = test_in_bounds & (label_values[np.clip(test_pos, 0, label_values.shape[0] - 1)] == test_raw)
        if label_policy.unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
            raise RuntimeError(
                "unsupported classification unseen_test_label policy: "
                f"{label_policy.unseen_test_label!r}"
            )
        valid_mask = test_valid.astype(bool, copy=False)
        remapped_test = test_pos[valid_mask].astype(np.int64, copy=False)

    return remapped_train, remapped_test, int(label_values.shape[0]), valid_mask


def apply_fitted_preprocessor(
    *,
    task: str,
    state: FittedPreprocessorState,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any | None = None,
    feature_types: Sequence[str] | None = None,
    impute_missing: bool = True,
) -> PreprocessedTaskArrays:
    """Apply one fitted preprocessing state to raw task arrays."""

    normalized_task = _require_task(task)
    if feature_types is not None:
        expected_feature_types = list(state.categorical_feature_policy.feature_types)
        provided_feature_types = normalize_feature_types(
            feature_types,
            width=len(expected_feature_types),
        )
        if provided_feature_types != expected_feature_types:
            raise RuntimeError(
                "feature_types do not match fitted preprocessing state: "
                f"expected={expected_feature_types}, got={provided_feature_types}"
            )
    train_x, test_x, feature_state = _apply_feature_preprocessing(
        state,
        x_train=x_train,
        x_test=x_test,
        impute_missing=bool(impute_missing),
    )
    if normalized_task == "classification":
        train_targets = _require_vector(y_train, dtype=np.int64, context="y_train")
        _require_matching_rows(train_x, train_targets, x_context="x_train", y_context="y_train")
        if y_test is not None:
            test_targets = _require_vector(y_test, dtype=np.int64, context="y_test")
            _require_matching_rows(test_x, test_targets, x_context="x_test", y_context="y_test")
        train_y, test_y, num_classes, valid_mask = _map_classification_targets(
            state,
            y_train=train_targets,
            y_test=y_test if y_test is None else test_targets,
        )
        if valid_mask is not None:
            test_x = test_x[valid_mask]
            feature_state = feature_state.filter_test_rows(valid_mask)
        return PreprocessedTaskArrays(
            x_train=train_x,
            y_train=train_y.astype(np.int64, copy=False),
            x_test=test_x,
            y_test=None if test_y is None else test_y.astype(np.int64, copy=False),
            num_classes=num_classes,
            valid_test_mask=valid_mask,
            feature_state=feature_state,
        )

    train_y = _require_vector(y_train, dtype=np.float32, context="y_train")
    _require_matching_rows(train_x, train_y, x_context="x_train", y_context="y_train")
    test_y_array: np.ndarray | None = None
    if y_test is not None:
        test_y_array = _require_vector(y_test, dtype=np.float32, context="y_test")
        _require_matching_rows(test_x, test_y_array, x_context="x_test", y_context="y_test")
    return PreprocessedTaskArrays(
        x_train=train_x,
        y_train=train_y.astype(np.float32, copy=False),
        x_test=test_x,
        y_test=None if test_y_array is None else test_y_array.astype(np.float32, copy=False),
        num_classes=None,
        valid_test_mask=None,
        feature_state=feature_state,
    )


def preprocess_runtime_task_arrays(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any | None = None,
    feature_types: Sequence[str] | None = None,
    impute_missing: bool = True,
    all_nan_fill: float = 0.0,
    label_mapping: str = CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    unseen_test_label_policy: str = UNSEEN_TEST_LABEL_POLICY_FILTER,
) -> PreprocessedTaskArrays:
    """Fit and apply preprocessing from the runtime support set."""

    state = fit_fitted_preprocessor(
        task=task,
        x_train=x_train,
        y_train=y_train,
        feature_types=feature_types,
        all_nan_fill=all_nan_fill,
        label_mapping=label_mapping,
        unseen_test_label_policy=unseen_test_label_policy,
    )
    return apply_fitted_preprocessor(
        task=task,
        state=state,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_types=feature_types,
        impute_missing=impute_missing,
    )
