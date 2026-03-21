"""Benchmark metric helpers for classification and regression."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .dataset_common import BenchmarkDatasetEvaluationError, _assert_finite_benchmark_datasets


_LOG_LOSS_EPS = 1.0e-15
_PICP_CENTRAL_COVERAGE = 0.90
_PICP_LOWER_QUANTILE = (1.0 - _PICP_CENTRAL_COVERAGE) / 2.0
_PICP_UPPER_QUANTILE = 1.0 - _PICP_LOWER_QUANTILE

_CLASSIFICATION_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
_REGRESSION_KF = KFold(n_splits=5, shuffle=True, random_state=0)


def evaluate_classifier(
    classifier: Any,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    allow_missing_values: bool = False,
) -> dict[str, float]:
    """Evaluate a sklearn-style classifier on the cached benchmark suite."""

    if not allow_missing_values:
        _assert_finite_benchmark_datasets(datasets, context="benchmark evaluation inputs")
    metrics: dict[str, float] = {}
    for dataset_name, (x, y) in datasets.items():
        try:
            targets: list[np.ndarray] = []
            probability_matrices: list[np.ndarray] = []
            all_labels = np.asarray(sorted(int(label) for label in np.unique(y)), dtype=np.int64)
            for train_idx, test_idx in _CLASSIFICATION_SKF.split(x, y):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                targets.append(y_test)
                classifier.fit(x_train, y_train)
                probability_matrices.append(
                    _aligned_classification_probabilities(
                        classifier,
                        classifier.predict_proba(x_test),
                        labels=all_labels,
                    )
                )

            target_array = np.concatenate(targets, axis=0)
            probability_matrix = np.concatenate(probability_matrices, axis=0)
            _assert_finite_benchmark_datasets(
                {"probabilities": (probability_matrix, target_array)},
                context=f"benchmark classifier outputs dataset={dataset_name!r}",
            )
            roc_auc_probabilities: np.ndarray = (
                probability_matrix[:, 1]
                if probability_matrix.shape[1] == 2
                else probability_matrix
            )
            metrics[f"{dataset_name}/ROC AUC"] = float(
                roc_auc_score(target_array, roc_auc_probabilities, multi_class="ovr")
            )
            metrics[f"{dataset_name}/Log Loss"] = float(
                log_loss(
                    target_array,
                    probability_matrix,
                    labels=all_labels.tolist(),
                )
            )
            metrics[f"{dataset_name}/Brier Score"] = float(
                _classification_brier_score(target_array, probability_matrix)
            )
        except Exception as exc:
            raise BenchmarkDatasetEvaluationError(str(dataset_name), exc) from exc

    roc_auc_values = [value for key, value in metrics.items() if key.endswith("/ROC AUC")]
    log_loss_values = [value for key, value in metrics.items() if key.endswith("/Log Loss")]
    brier_score_values = [value for key, value in metrics.items() if key.endswith("/Brier Score")]
    metrics["ROC AUC"] = float(np.mean(roc_auc_values))
    metrics["Log Loss"] = float(np.mean(log_loss_values))
    metrics["Brier Score"] = float(np.mean(brier_score_values))
    return metrics


def evaluate_regressor(
    regressor: Any,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    allow_missing_values: bool = False,
) -> dict[str, float]:
    """Evaluate a sklearn-style regressor on the cached benchmark suite."""

    if not allow_missing_values:
        _assert_finite_benchmark_datasets(datasets, context="benchmark evaluation inputs")
    metrics: dict[str, float] = {}
    for dataset_name, (x, y) in datasets.items():
        try:
            targets: list[np.ndarray] = []
            quantile_predictions: list[np.ndarray] = []
            quantile_levels: np.ndarray | None = None
            for train_idx, test_idx in _REGRESSION_KF.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train = np.asarray(y[train_idx], dtype=np.float32)
                y_test = np.asarray(y[test_idx], dtype=np.float32)
                targets.append(y_test)
                regressor.fit(x_train, y_train)
                raw_quantiles, raw_levels = regressor.predict_quantiles(x_test)
                fold_quantiles, fold_levels = _normalize_quantile_predictions(
                    raw_quantiles,
                    raw_levels,
                )
                if quantile_levels is None:
                    quantile_levels = fold_levels
                elif not np.allclose(quantile_levels, fold_levels, atol=1.0e-8, rtol=1.0e-8):
                    raise RuntimeError("regressor quantile levels changed between folds")
                quantile_predictions.append(fold_quantiles)

            if quantile_levels is None:
                raise RuntimeError("regression benchmark produced no quantile levels")
            target_array = np.concatenate(targets, axis=0)
            quantile_array = np.concatenate(quantile_predictions, axis=0)
            _assert_finite_benchmark_datasets(
                {"quantiles": (quantile_array, target_array)},
                context=f"benchmark regressor outputs dataset={dataset_name!r}",
            )
            metrics[f"{dataset_name}/CRPS"] = float(
                _crps_from_quantiles(target_array, quantile_array, quantile_levels)
            )
            metrics[f"{dataset_name}/Average Pinball Loss"] = float(
                _average_pinball_loss(target_array, quantile_array, quantile_levels)
            )
            metrics[f"{dataset_name}/PICP 90"] = float(
                _prediction_interval_coverage_probability(
                    target_array,
                    quantile_array,
                    quantile_levels,
                    lower_quantile=_PICP_LOWER_QUANTILE,
                    upper_quantile=_PICP_UPPER_QUANTILE,
                )
            )
        except Exception as exc:
            raise BenchmarkDatasetEvaluationError(str(dataset_name), exc) from exc

    crps_values = [value for key, value in metrics.items() if key.endswith("/CRPS")]
    avg_pinball_values = [
        value for key, value in metrics.items() if key.endswith("/Average Pinball Loss")
    ]
    picp_values = [value for key, value in metrics.items() if key.endswith("/PICP 90")]
    metrics["CRPS"] = float(np.mean(crps_values))
    metrics["Average Pinball Loss"] = float(np.mean(avg_pinball_values))
    metrics["PICP 90"] = float(np.mean(picp_values))
    return metrics


def _normalize_classification_probabilities(probabilities: np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float64)
    if raw.ndim == 1:
        positive = np.clip(raw, _LOG_LOSS_EPS, 1.0 - _LOG_LOSS_EPS)
        return np.stack([1.0 - positive, positive], axis=1)
    if raw.ndim != 2 or raw.shape[1] <= 0:
        raise RuntimeError(
            "predict_proba must return a 1D probability vector or a 2D probability matrix"
        )
    if raw.shape[1] == 1:
        positive = np.clip(raw[:, 0], _LOG_LOSS_EPS, 1.0 - _LOG_LOSS_EPS)
        return np.stack([1.0 - positive, positive], axis=1)
    clipped = np.clip(raw, _LOG_LOSS_EPS, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("predict_proba returned a non-positive probability row")
    normalized = clipped / row_sums
    normalized = np.clip(normalized, _LOG_LOSS_EPS, 1.0)
    return normalized / normalized.sum(axis=1, keepdims=True)


def _classifier_classes(classifier: Any) -> np.ndarray | None:
    raw_classes = getattr(classifier, "classes_", None)
    if raw_classes is None:
        raw_classes = getattr(classifier, "_classes", None)
    if raw_classes is None:
        return None
    return np.asarray(raw_classes, dtype=np.int64)


def _aligned_classification_probabilities(
    classifier: Any,
    probabilities: np.ndarray,
    *,
    labels: np.ndarray,
) -> np.ndarray:
    normalized = _normalize_classification_probabilities(probabilities)
    classifier_classes = _classifier_classes(classifier)
    if classifier_classes is None:
        if normalized.shape[1] == labels.size:
            return normalized
        raise RuntimeError(
            "predict_proba output could not be aligned to benchmark labels; "
            "expose classes_/_classes or return the full probability matrix"
        )
    if normalized.shape[1] != classifier_classes.size:
        raise RuntimeError(
            "predict_proba output width does not match classifier classes metadata"
        )
    if normalized.shape[1] == labels.size and np.array_equal(classifier_classes, labels):
        return normalized
    label_to_index = {int(label): index for index, label in enumerate(labels.tolist())}
    aligned = np.zeros((normalized.shape[0], labels.size), dtype=np.float64)
    for source_index, raw_label in enumerate(classifier_classes.tolist()):
        label = int(raw_label)
        if label not in label_to_index:
            raise RuntimeError(
                f"predict_proba exposed unexpected class label {label!r} for benchmark labels {labels.tolist()!r}"
            )
        aligned[:, label_to_index[label]] = normalized[:, source_index]
    return _normalize_classification_probabilities(aligned)


def _classification_brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.zeros_like(probability_array, dtype=np.float64)
    one_hot[np.arange(target_array.shape[0]), target_array] = 1.0
    squared_error = np.square(probability_array - one_hot)
    return float(np.mean(np.sum(squared_error, axis=1)))


def _normalize_quantile_predictions(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    if normalized_quantiles.ndim != 2:
        raise RuntimeError("regressor quantiles must be a 2D matrix")
    if levels.ndim != 1 or levels.size != normalized_quantiles.shape[1]:
        raise RuntimeError("regressor quantile levels must be a 1D vector aligned with quantiles")
    if levels.size <= 0:
        raise RuntimeError("regressor quantile levels must be non-empty")
    if np.any(levels <= 0.0) or np.any(levels >= 1.0):
        raise RuntimeError("regressor quantile levels must lie strictly between 0 and 1")
    order = np.argsort(levels)
    sorted_levels = levels[order]
    if np.any(np.diff(sorted_levels) <= 0.0):
        raise RuntimeError("regressor quantile levels must be strictly increasing")
    return normalized_quantiles[:, order], sorted_levels


def _pinball_loss_matrix(
    targets: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> np.ndarray:
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    quantile_array = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(1, -1)
    errors = target_array - quantile_array
    return np.maximum(levels * errors, (levels - 1.0) * errors)


def _average_pinball_loss(
    targets: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    return float(np.mean(_pinball_loss_matrix(targets, quantiles, quantile_levels)))


def _crps_from_quantiles(
    targets: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    pinball = _pinball_loss_matrix(targets, quantiles, quantile_levels)
    return float(2.0 * np.mean(np.trapezoid(pinball, x=quantile_levels, axis=1)))


def _prediction_interval_coverage_probability(
    targets: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> float:
    lower_index = int(np.argmin(np.abs(np.asarray(quantile_levels, dtype=np.float64) - float(lower_quantile))))
    upper_index = int(np.argmin(np.abs(np.asarray(quantile_levels, dtype=np.float64) - float(upper_quantile))))
    if lower_index >= upper_index:
        raise RuntimeError("prediction interval quantile indices must be strictly ordered")
    target_array = np.asarray(targets, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)
    lower = quantile_array[:, lower_index]
    upper = quantile_array[:, upper_index]
    return float(np.mean((target_array >= lower) & (target_array <= upper)))


def _dataset_metric_summary(metrics: Mapping[str, float], *, metric_name: str) -> dict[str, float]:
    suffix = f"/{metric_name}"
    return {
        str(key[: -len(suffix)]): float(value)
        for key, value in metrics.items()
        if key.endswith(suffix) and key != metric_name
    }


def dataset_roc_auc_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="ROC AUC")


def dataset_log_loss_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="Log Loss")


def dataset_brier_score_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="Brier Score")


def dataset_crps_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="CRPS")


def dataset_avg_pinball_loss_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="Average Pinball Loss")


def dataset_picp_90_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return _dataset_metric_summary(metrics, metric_name="PICP 90")
