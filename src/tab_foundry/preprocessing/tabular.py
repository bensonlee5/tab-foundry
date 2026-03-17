"""Shared task-level preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .state import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
    ClassificationLabelPolicyState,
    FittedPreprocessorState,
    MissingValuePolicyState,
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


def _require_task(task: str) -> str:
    normalized = str(task).strip().lower()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported preprocessing task: {task!r}")
    return normalized


def _require_matrix(x: Any, *, context: str) -> np.ndarray:
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 2:
        raise RuntimeError(f"{context} must be a rank-2 matrix, got shape={array.shape!r}")
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
    all_nan_fill: float,
) -> list[float]:
    finite_mask = np.isfinite(x_train)
    finite_sums = np.where(finite_mask, x_train, 0.0).sum(axis=0, dtype=np.float64)
    finite_counts = finite_mask.sum(axis=0)
    means = np.full((x_train.shape[1],), float(all_nan_fill), dtype=np.float64)
    np.divide(finite_sums, finite_counts, out=means, where=finite_counts > 0)
    return [float(value) for value in means.astype(np.float32, copy=False).tolist()]


def fit_fitted_preprocessor(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
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
    missing_value_policy = MissingValuePolicyState(
        strategy=MISSING_VALUE_STRATEGY_TRAIN_MEAN,
        all_nan_fill=float(all_nan_fill),
        fill_values=_fit_fill_values(x_train_matrix, all_nan_fill=float(all_nan_fill)),
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
        classification_label_policy=classification_label_policy,
        dtype_policy=dict(DTYPE_POLICY),
    )


def _apply_feature_preprocessing(
    state: FittedPreprocessorState,
    *,
    x_train: Any,
    x_test: Any,
    impute_missing: bool,
) -> tuple[np.ndarray, np.ndarray]:
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
    if not impute_missing:
        return train, test

    train_missing = ~np.isfinite(train)
    if np.any(train_missing):
        train[train_missing] = np.take(fill_values, np.where(train_missing)[1])

    test_missing = ~np.isfinite(test)
    if np.any(test_missing):
        test[test_missing] = np.take(fill_values, np.where(test_missing)[1])

    return train, test


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
    impute_missing: bool = True,
) -> PreprocessedTaskArrays:
    """Apply one fitted preprocessing state to raw task arrays."""

    normalized_task = _require_task(task)
    train_x, test_x = _apply_feature_preprocessing(
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
        return PreprocessedTaskArrays(
            x_train=train_x,
            y_train=train_y.astype(np.int64, copy=False),
            x_test=test_x,
            y_test=None if test_y is None else test_y.astype(np.int64, copy=False),
            num_classes=num_classes,
            valid_test_mask=valid_mask,
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
    )


def preprocess_runtime_task_arrays(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any | None = None,
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
        impute_missing=impute_missing,
    )
