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


def _fit_fill_values(
    x_train: np.ndarray,
    *,
    all_nan_fill: float,
) -> list[float]:
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isnan(means), float(all_nan_fill), means)
    return [float(value) for value in means.astype(np.float32, copy=False).tolist()]


def fit_fitted_preprocessor(
    *,
    task: str,
    x_train: Any,
    y_train: Any,
    all_nan_fill: float = 0.0,
) -> FittedPreprocessorState:
    """Fit the shared preprocessing state from one train split."""

    normalized_task = _require_task(task)
    x_train_matrix = _require_matrix(x_train, context="x_train")
    feature_ids = list(range(int(x_train_matrix.shape[1])))
    missing_value_policy = MissingValuePolicyState(
        strategy=MISSING_VALUE_STRATEGY_TRAIN_MEAN,
        all_nan_fill=float(all_nan_fill),
        fill_values=_fit_fill_values(x_train_matrix, all_nan_fill=float(all_nan_fill)),
    )

    classification_label_policy: ClassificationLabelPolicyState | None = None
    if normalized_task == "classification":
        labels = np.asarray(y_train, dtype=np.int64)
        if labels.ndim != 1:
            raise RuntimeError(f"y_train must be rank-1 for classification, got shape={labels.shape!r}")
        label_values = np.unique(labels)
        if label_values.size <= 0:
            raise RuntimeError("classification train split has no labels")
        classification_label_policy = ClassificationLabelPolicyState(
            mapping=CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
            unseen_test_label=UNSEEN_TEST_LABEL_POLICY_FILTER,
            label_values=[int(value) for value in label_values.tolist()],
        )

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

    train_nan = np.isnan(train)
    if np.any(train_nan):
        train[train_nan] = np.take(fill_values, np.where(train_nan)[1])

    test_nan = np.isnan(test)
    if np.any(test_nan):
        test[test_nan] = np.take(fill_values, np.where(test_nan)[1])

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
    train_raw = np.asarray(y_train, dtype=np.int64)
    if train_raw.ndim != 1:
        raise RuntimeError(f"y_train must be rank-1 for classification, got shape={train_raw.shape!r}")

    train_pos = np.searchsorted(label_values, train_raw)
    train_in_bounds = train_pos < label_values.shape[0]
    train_valid = train_in_bounds & (label_values[np.clip(train_pos, 0, label_values.shape[0] - 1)] == train_raw)
    if not bool(np.all(train_valid)):
        raise RuntimeError("classification y_train contains labels absent from fitted preprocessing state")
    remapped_train = train_pos.astype(np.int64, copy=False)

    remapped_test: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    if y_test is not None:
        test_raw = np.asarray(y_test, dtype=np.int64)
        if test_raw.ndim != 1:
            raise RuntimeError(f"y_test must be rank-1 for classification, got shape={test_raw.shape!r}")
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
        train_y, test_y, num_classes, valid_mask = _map_classification_targets(
            state,
            y_train=y_train,
            y_test=y_test,
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

    train_y = np.asarray(y_train, dtype=np.float32)
    if train_y.ndim != 1:
        raise RuntimeError(f"y_train must be rank-1 for regression, got shape={train_y.shape!r}")
    test_y_array: np.ndarray | None = None
    if y_test is not None:
        test_y_array = np.asarray(y_test, dtype=np.float32)
        if test_y_array.ndim != 1:
            raise RuntimeError(f"y_test must be rank-1 for regression, got shape={test_y_array.shape!r}")
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
) -> PreprocessedTaskArrays:
    """Fit and apply preprocessing from the runtime support set."""

    state = fit_fitted_preprocessor(
        task=task,
        x_train=x_train,
        y_train=y_train,
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
