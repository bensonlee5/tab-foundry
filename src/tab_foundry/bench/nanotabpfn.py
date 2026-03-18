"""Notebook-style nanoTabPFN comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import openml
from openml.tasks import TaskType
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from tab_foundry.data.validation import assert_no_non_finite_values
from tab_foundry.bench.artifacts import checkpoint_snapshots_from_history


BENCHMARK_BUNDLE_FILENAME = "nanotabpfn_openml_binary_medium_v1.json"
_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_REGRESSION_TASK_TYPE = "supervised_regression"
_ALLOWED_BUNDLE_SELECTION_TASK_TYPES = {
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
}
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED = 0
_LOG_LOSS_EPS = 1.0e-15
_PICP_CENTRAL_COVERAGE = 0.90
_PICP_LOWER_QUANTILE = (1.0 - _PICP_CENTRAL_COVERAGE) / 2.0
_PICP_UPPER_QUANTILE = 1.0 - _PICP_LOWER_QUANTILE

_CLASSIFICATION_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
_REGRESSION_KF = KFold(n_splits=5, shuffle=True, random_state=0)


@dataclass(slots=True)
class PreparedOpenMLBenchmarkTask:
    """Materialized OpenML benchmark task after notebook-style preprocessing."""

    task_id: int
    dataset_name: str
    x: np.ndarray
    y: np.ndarray
    observed_task: dict[str, Any]
    qualities: dict[str, float]
    task_type: str = _CLASSIFICATION_TASK_TYPE


class BenchmarkDatasetEvaluationError(RuntimeError):
    """One benchmark dataset failed within a checkpoint evaluation."""

    def __init__(self, dataset_name: str, cause: Exception) -> None:
        self.dataset_name = str(dataset_name)
        self.error_type = type(cause).__name__
        super().__init__(
            f"benchmark evaluation failed for dataset {self.dataset_name!r}: {cause}"
        )


def get_feature_preprocessor(x: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """Replicate the nanoTabPFN notebook preprocessing logic."""

    frame = pd.DataFrame(x)
    num_mask: list[bool] = []
    cat_mask: list[bool] = []
    for column in frame:
        unique_non_nan_entries = frame[column].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = frame[column].notna().sum()
        numeric_entries = pd.to_numeric(frame[column], errors="coerce").notna().sum()
        num_mask.append(bool(non_nan_entries == numeric_entries))
        cat_mask.append(bool(non_nan_entries != numeric_entries))

    num_transformer = Pipeline(
        [
            (
                "to_pandas",
                FunctionTransformer(
                    lambda value: pd.DataFrame(value) if not isinstance(value, pd.DataFrame) else value
                ),
            ),
            (
                "to_numeric",
                FunctionTransformer(lambda value: value.apply(pd.to_numeric, errors="coerce").to_numpy()),
            ),
        ]
    )
    cat_transformer = Pipeline(
        [("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, np.asarray(num_mask)),
            ("cat", cat_transformer, np.asarray(cat_mask)),
        ]
    )


def default_benchmark_bundle_path() -> Path:
    """Return the repo-tracked canonical benchmark bundle path."""

    return Path(__file__).resolve().with_name(BENCHMARK_BUNDLE_FILENAME)


def _normalize_selection(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle selection metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle selection must be an object")
    task_type = payload.get("task_type", _CLASSIFICATION_TASK_TYPE)
    if task_type not in _ALLOWED_BUNDLE_SELECTION_TASK_TYPES:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be one of "
            f"{sorted(_ALLOWED_BUNDLE_SELECTION_TASK_TYPES)!r}"
        )
    expected_keys = (
        {
            "new_instances",
            "max_features",
            "max_classes",
            "max_missing_pct",
            "min_minority_class_pct",
        }
        | ({"task_type"} if "task_type" in payload else set())
        if task_type == _CLASSIFICATION_TASK_TYPE
        else {
            "new_instances",
            "task_type",
            "max_features",
            "max_missing_pct",
        }
    )
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle selection keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    new_instances = payload["new_instances"]
    max_features = payload["max_features"]
    max_missing_pct = payload["max_missing_pct"]

    if not isinstance(new_instances, int) or isinstance(new_instances, bool) or new_instances <= 0:
        raise RuntimeError("benchmark bundle selection.new_instances must be a positive int")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0:
        raise RuntimeError("benchmark bundle selection.max_features must be a positive int")
    if not isinstance(max_missing_pct, (int, float)) or not 0 <= float(max_missing_pct) <= 100:
        raise RuntimeError("benchmark bundle selection.max_missing_pct must be a percentage between 0 and 100")
    normalized = {
        "new_instances": int(new_instances),
        "task_type": str(task_type),
        "max_features": int(max_features),
        "max_missing_pct": float(max_missing_pct),
    }
    if task_type == _CLASSIFICATION_TASK_TYPE:
        max_classes = payload["max_classes"]
        min_minority_class_pct = payload["min_minority_class_pct"]
        if not isinstance(max_classes, int) or isinstance(max_classes, bool) or max_classes <= 0:
            raise RuntimeError("benchmark bundle selection.max_classes must be a positive int")
        if not isinstance(min_minority_class_pct, (int, float)) or not 0 <= float(min_minority_class_pct) <= 100:
            raise RuntimeError(
                "benchmark bundle selection.min_minority_class_pct must be a percentage between 0 and 100"
            )
        normalized["max_classes"] = int(max_classes)
        normalized["min_minority_class_pct"] = float(min_minority_class_pct)
    return normalized


def read_required_openml_quality(raw_qualities: Any, *, task_id: int, quality_name: str) -> float:
    """Read a numeric OpenML quality and raise a drift error if it is missing."""

    if not isinstance(raw_qualities, dict):
        raise RuntimeError(f"benchmark bundle drift: task {task_id} dataset qualities are missing")
    value = raw_qualities.get(quality_name)
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"benchmark bundle drift: task {task_id} missing numeric quality {quality_name!r}"
        )
    return float(value)


def normalize_benchmark_bundle(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle must be a JSON object")
    expected_keys = {"name", "version", "selection", "task_ids", "tasks"}
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    name = payload["name"]
    version = payload["version"]
    selection = payload["selection"]
    task_ids = payload["task_ids"]
    tasks = payload["tasks"]
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("benchmark bundle name must be a non-empty string")
    if not isinstance(version, int) or version <= 0:
        raise RuntimeError("benchmark bundle version must be a positive int")
    if not isinstance(task_ids, list) or not task_ids:
        raise RuntimeError("benchmark bundle task_ids must be a non-empty list")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("benchmark bundle tasks must be a non-empty list")

    normalized_selection = _normalize_selection(selection)
    selection_task_type = str(normalized_selection["task_type"])
    normalized_task_ids = [int(task_id) for task_id in task_ids]
    normalized_tasks: list[dict[str, Any]] = []
    for index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            raise RuntimeError(f"benchmark bundle task {index} must be an object")
        task_keys = (
            {"task_id", "dataset_name", "n_rows", "n_features", "n_classes"}
            if selection_task_type == _CLASSIFICATION_TASK_TYPE
            else {"task_id", "dataset_name", "n_rows", "n_features"}
        )
        actual_task_keys = set(task_payload.keys())
        if actual_task_keys != task_keys:
            raise RuntimeError(
                f"benchmark bundle task keys mismatch at index {index}: "
                f"expected={sorted(task_keys)}, actual={sorted(actual_task_keys)}"
            )
        dataset_name = task_payload["dataset_name"]
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise RuntimeError(f"benchmark bundle task dataset_name must be non-empty at index {index}")
        normalized_task = {
            "task_id": int(task_payload["task_id"]),
            "dataset_name": str(dataset_name),
            "n_rows": int(task_payload["n_rows"]),
            "n_features": int(task_payload["n_features"]),
        }
        if selection_task_type == _CLASSIFICATION_TASK_TYPE:
            normalized_task["n_classes"] = int(task_payload["n_classes"])
        normalized_tasks.append(normalized_task)

    if normalized_task_ids != [int(task["task_id"]) for task in normalized_tasks]:
        raise RuntimeError("benchmark bundle task_ids must match tasks[].task_id order exactly")

    normalized_bundle: dict[str, Any] = {
        "name": str(name),
        "version": int(version),
        "selection": normalized_selection,
        "task_ids": normalized_task_ids,
        "tasks": normalized_tasks,
    }
    return normalized_bundle


def _validate_bundle_missing_value_policy(
    bundle: Mapping[str, Any],
    *,
    allow_missing_values: bool,
    source_path: Path,
) -> None:
    if allow_missing_values:
        return
    selection = cast(dict[str, Any], bundle["selection"])
    max_missing_pct = float(selection["max_missing_pct"])
    if max_missing_pct > 0.0:
        raise RuntimeError(
            "benchmark bundle permits missing-valued inputs while allow_missing_values=False: "
            f"path={source_path}, max_missing_pct={max_missing_pct}"
        )


def _assert_finite_benchmark_datasets(
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    context: str,
) -> None:
    for dataset_name, (x, y) in datasets.items():
        assert_no_non_finite_values(
            {"x": x, "y": y},
            context=f"{context} dataset={dataset_name!r}",
        )


def benchmark_bundle_allows_missing_values(bundle: Mapping[str, Any]) -> bool:
    """Return whether the bundle contract permits missing-valued inputs."""

    selection = cast(dict[str, Any], bundle["selection"])
    raw_max_missing_pct = selection.get("max_missing_pct")
    if not isinstance(raw_max_missing_pct, (int, float)):
        return False
    return bool(float(raw_max_missing_pct) > 0.0)


def benchmark_bundle_task_type(bundle: Mapping[str, Any]) -> str:
    """Return the bundle task type."""

    selection = cast(dict[str, Any], bundle["selection"])
    task_type = selection.get("task_type", _CLASSIFICATION_TASK_TYPE)
    if task_type not in _ALLOWED_BUNDLE_SELECTION_TASK_TYPES:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be one of "
            f"{sorted(_ALLOWED_BUNDLE_SELECTION_TASK_TYPES)!r}"
        )
    return str(task_type)


def load_benchmark_bundle(
    path: Path | None = None,
    *,
    allow_missing_values: bool = False,
) -> dict[str, Any]:
    """Load and validate the canonical benchmark bundle metadata."""

    bundle_path = (path or default_benchmark_bundle_path()).expanduser().resolve()
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        bundle = normalize_benchmark_bundle(payload)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}: {bundle_path}") from exc
    _validate_bundle_missing_value_policy(
        bundle,
        allow_missing_values=bool(allow_missing_values),
        source_path=bundle_path,
    )
    return bundle


def load_benchmark_bundle_for_execution(path: Path | None = None) -> tuple[dict[str, Any], bool]:
    """Load a bundle and resolve whether execution should allow missing values."""

    bundle = load_benchmark_bundle(path, allow_missing_values=True)
    return bundle, benchmark_bundle_allows_missing_values(bundle)


def benchmark_bundle_summary(
    bundle: Mapping[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Build compact bundle metadata for run summaries."""

    task_ids = [int(task_id) for task_id in cast(list[Any], bundle["task_ids"])]
    selection_raw = bundle.get("selection")
    selection = (
        cast(dict[str, Any], json.loads(json.dumps(selection_raw, sort_keys=True)))
        if isinstance(selection_raw, Mapping)
        else None
    )
    allow_missing_values = (
        None
        if not isinstance(selection_raw, Mapping)
        else benchmark_bundle_allows_missing_values(bundle)
    )
    return {
        "name": str(bundle["name"]),
        "version": int(bundle["version"]),
        "source_path": str(source_path.expanduser().resolve()),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
        "selection": selection,
        "allow_missing_values": allow_missing_values,
        "all_tasks_no_missing": None if allow_missing_values is None else (not allow_missing_values),
    }


def _task_type_value(task_type: TaskType | int) -> int:
    return int(task_type.value) if isinstance(task_type, TaskType) else int(task_type)


def _openml_task_type_for_bundle_task_type(task_type: str) -> int:
    if task_type == _CLASSIFICATION_TASK_TYPE:
        return _task_type_value(TaskType.SUPERVISED_CLASSIFICATION)
    if task_type == _REGRESSION_TASK_TYPE:
        return _task_type_value(TaskType.SUPERVISED_REGRESSION)
    raise RuntimeError(f"unsupported benchmark bundle task_type: {task_type!r}")


def prepare_openml_benchmark_task(
    task_id: int,
    *,
    new_instances: int,
    task_type: str,
) -> PreparedOpenMLBenchmarkTask:
    """Load and preprocess one OpenML task using the nanoTabPFN notebook logic."""

    task = openml.tasks.get_task(task_id, download_splits=False)
    expected_task_type_id = _openml_task_type_for_bundle_task_type(task_type)
    observed_task_type_id = _task_type_value(cast(TaskType | int, task.task_type_id))
    if observed_task_type_id != expected_task_type_id:
        raise RuntimeError(
            "benchmark bundle drift: "
            f"task {task_id} is no longer {task_type}"
        )
    task_any: Any = task
    dataset = task_any.get_dataset(download_data=False)
    dataset_any: Any = dataset
    raw_qualities = dataset_any.qualities
    number_of_features = read_required_openml_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="NumberOfFeatures",
    )
    missing_pct = read_required_openml_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="PercentageOfInstancesWithMissingValues",
    )
    number_of_classes = None
    minority_class_pct = None
    if task_type == _CLASSIFICATION_TASK_TYPE:
        number_of_classes = read_required_openml_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="NumberOfClasses",
        )
        minority_class_pct = read_required_openml_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="MinorityClassPercentage",
        )

    x_frame, y_raw, _categorical_indicator, _attribute_names = dataset_any.get_data(
        target=str(task_any.target_name),
        dataset_format="dataframe",
    )
    if new_instances < int(len(cast(Any, y_raw))):
        train_test_split_kwargs: dict[str, Any] = {
            "test_size": new_instances,
            "random_state": 0,
        }
        if task_type == _CLASSIFICATION_TASK_TYPE:
            train_test_split_kwargs["stratify"] = y_raw
        _x_unused, x_sub, _y_unused, y_sub = train_test_split(x_frame, y_raw, **train_test_split_kwargs)
    else:
        x_sub = x_frame
        y_sub = y_raw

    preprocessor = get_feature_preprocessor(x_sub)
    x = np.asarray(preprocessor.fit_transform(x_sub), dtype=np.float32)
    observed_task = {
        "task_id": int(task_id),
        "dataset_name": str(dataset.name),
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }
    if task_type == _CLASSIFICATION_TASK_TYPE:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_sub.to_numpy(copy=True)).astype(np.int64, copy=False)
        observed_task["n_classes"] = int(np.unique(y).size)
    else:
        y = np.asarray(pd.to_numeric(y_sub, errors="raise"), dtype=np.float32)
    return PreparedOpenMLBenchmarkTask(
        task_id=int(task_id),
        task_type=str(task_type),
        dataset_name=str(dataset.name),
        x=x,
        y=y,
        observed_task=observed_task,
        qualities={
            "NumberOfFeatures": float(number_of_features),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            **(
                {}
                if number_of_classes is None
                else {"NumberOfClasses": float(number_of_classes)}
            ),
            **(
                {}
                if minority_class_pct is None
                else {"MinorityClassPercentage": float(minority_class_pct)}
            ),
        },
    )


def load_openml_benchmark_datasets(
    *,
    new_instances: int = 200,
    benchmark_bundle_path: Path | None = None,
    allow_missing_values: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Load the nanoTabPFN OpenML benchmark suite."""

    bundle = load_benchmark_bundle(
        benchmark_bundle_path,
        allow_missing_values=allow_missing_values,
    )
    selection = cast(dict[str, Any], bundle["selection"])
    selection_task_type = benchmark_bundle_task_type(bundle)
    expected_new_instances = int(selection["new_instances"])
    if new_instances != expected_new_instances:
        raise RuntimeError(
            "benchmark bundle selection mismatch: "
            f"expected new_instances={expected_new_instances}, got {new_instances}"
        )
    expected_tasks = cast(list[dict[str, Any]], bundle["tasks"])
    expected_by_task_id = {int(task["task_id"]): task for task in expected_tasks}
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    benchmark_tasks: list[dict[str, Any]] = []
    for task_id in cast(list[int], bundle["task_ids"]):
        prepared = prepare_openml_benchmark_task(
            int(task_id),
            new_instances=new_instances,
            task_type=selection_task_type,
        )
        number_of_features = prepared.qualities["NumberOfFeatures"]
        missing_pct = prepared.qualities["PercentageOfInstancesWithMissingValues"]
        if number_of_features > int(selection["max_features"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_features expected<={selection['max_features']}, actual={number_of_features}"
            )
        if missing_pct > float(selection["max_missing_pct"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_missing_pct expected<={selection['max_missing_pct']}, actual={missing_pct}"
            )
        if selection_task_type == _CLASSIFICATION_TASK_TYPE:
            number_of_classes = prepared.qualities["NumberOfClasses"]
            minority_class_pct = prepared.qualities["MinorityClassPercentage"]
            if number_of_classes > int(selection["max_classes"]):
                raise RuntimeError(
                    "benchmark bundle drift: "
                    f"task {task_id} exceeds max_classes expected<={selection['max_classes']}, actual={number_of_classes}"
                )
            if minority_class_pct < float(selection["min_minority_class_pct"]):
                raise RuntimeError(
                    "benchmark bundle drift: "
                    "task "
                    f"{task_id} violates min_minority_class_pct expected>={selection['min_minority_class_pct']}, "
                    f"actual={minority_class_pct}"
                )
        expected_task = expected_by_task_id[int(task_id)]
        if prepared.observed_task != expected_task:
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} metadata mismatch expected={expected_task}, actual={prepared.observed_task}"
            )

        datasets[prepared.dataset_name] = (prepared.x, prepared.y)
        benchmark_tasks.append(dict(prepared.observed_task))
    if not datasets:
        raise RuntimeError("OpenML benchmark produced no datasets after filtering")
    if len(benchmark_tasks) != len(expected_tasks):
        raise RuntimeError(
            "benchmark bundle drift: "
            f"task count mismatch expected={len(expected_tasks)}, actual={len(benchmark_tasks)}"
        )
    if not allow_missing_values:
        _assert_finite_benchmark_datasets(
            datasets,
            context=f"benchmark bundle {bundle['name']!r}",
        )
    return datasets, benchmark_tasks


def save_dataset_cache(path: Path, datasets: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Persist benchmark datasets for reuse across envs."""

    payload: dict[str, Any] = {"names": np.asarray(list(datasets.keys()), dtype=str)}
    for index, (name, (x, y)) in enumerate(datasets.items()):
        payload[f"x_{index:03d}"] = np.asarray(x, dtype=np.float32)
        payload[f"y_{index:03d}"] = np.asarray(y)
        payload[f"name_{index:03d}"] = np.asarray(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_dataset_cache(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a cached benchmark dataset bundle."""

    cache = np.load(path, allow_pickle=False)
    names = [str(name) for name in cache["names"].tolist()]
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, name in enumerate(names):
        datasets[name] = (
            np.asarray(cache[f"x_{index:03d}"], dtype=np.float32),
            np.asarray(cache[f"y_{index:03d}"]),
        )
    if not datasets:
        raise RuntimeError(f"dataset cache is empty: {path}")
    return datasets


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
            assert_no_non_finite_values(
                {
                    "probabilities": probability_matrix,
                },
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
            assert_no_non_finite_values(
                {
                    "targets": target_array,
                    "quantiles": quantile_array,
                    "quantile_levels": quantile_levels,
                },
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
    """Normalize classifier probabilities for stable benchmark evaluation."""

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
    return float(np.mean(np.square(probability_array - one_hot)))


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


def _ensure_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _dataset_metric_summary(metrics: Mapping[str, float], *, metric_name: str) -> dict[str, float]:
    """Extract per-dataset metrics for one benchmark metric family."""

    suffix = f"/{metric_name}"
    return {
        str(key[: -len(suffix)]): float(value)
        for key, value in metrics.items()
        if key.endswith(suffix) and key != metric_name
    }


def dataset_roc_auc_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset ROC AUC values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="ROC AUC")


def dataset_log_loss_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset log-loss values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="Log Loss")


def dataset_brier_score_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset Brier score values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="Brier Score")


def dataset_crps_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset CRPS values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="CRPS")


def dataset_avg_pinball_loss_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset average pinball loss values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="Average Pinball Loss")


def dataset_picp_90_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset 90% PICP values from a benchmark metrics dict."""

    return _dataset_metric_summary(metrics, metric_name="PICP 90")


def task_bootstrap_roc_auc_interval(
    dataset_roc_auc: Mapping[str, float],
    *,
    bootstrap_samples: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int] | None:
    """Estimate a task-bootstrap confidence interval for mean ROC AUC."""

    if bootstrap_samples <= 0:
        return None
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    values = np.asarray([float(value) for value in dataset_roc_auc.values()], dtype=np.float64)
    if values.size <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(bootstrap_samples), values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return {
        "samples": int(bootstrap_samples),
        "confidence": float(confidence),
        "lower": lower,
        "upper": upper,
    }


def annotate_curve_records_with_task_statistics(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    """Add per-checkpoint task-count and optional bootstrap diagnostics."""

    annotated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        raw_dataset_roc_auc = record.get("dataset_roc_auc")
        dataset_roc_auc = (
            {
                str(dataset_name): float(value)
                for dataset_name, value in raw_dataset_roc_auc.items()
            }
            if isinstance(raw_dataset_roc_auc, Mapping)
            else {}
        )
        enriched = dict(record)
        dataset_counts = [
            len(dataset_roc_auc),
            *[
                len(value)
                for key, value in record.items()
                if key.startswith("dataset_") and isinstance(value, Mapping)
            ],
        ]
        enriched["dataset_count"] = int(max(dataset_counts, default=0))
        bootstrap_interval = task_bootstrap_roc_auc_interval(
            dataset_roc_auc,
            bootstrap_samples=int(bootstrap_samples),
            confidence=float(bootstrap_confidence),
            seed=int(bootstrap_seed) + index,
        )
        if bootstrap_interval is not None:
            enriched["roc_auc_task_bootstrap_ci"] = bootstrap_interval
        annotated.append(enriched)
    return annotated


def _interval_overlap(interval_a: Mapping[str, Any], interval_b: Mapping[str, Any]) -> bool:
    return float(interval_a["lower"]) <= float(interval_b["upper"]) and float(
        interval_b["lower"]
    ) <= float(interval_a["upper"])


def _is_successful_curve_record(record: Mapping[str, Any]) -> bool:
    for key in ("roc_auc", "log_loss", "crps"):
        raw_value = record.get(key)
        if raw_value is None:
            continue
        try:
            if math.isfinite(float(raw_value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _curve_ranking_metric(records: list[dict[str, Any]]) -> tuple[str, str]:
    if any(record.get("roc_auc") is not None for record in records):
        return ("roc_auc", "max")
    if any(record.get("crps") is not None for record in records):
        return ("crps", "min")
    if any(record.get("log_loss") is not None for record in records):
        return ("log_loss", "min")
    raise RuntimeError("checkpoint curve records do not expose a ranking metric")


def curve_adjacent_ci_overlap_fraction(records: list[dict[str, Any]]) -> float | None:
    """Estimate how often adjacent checkpoint CIs overlap."""

    sorted_records = sorted(
        [dict(record) for record in records if _is_successful_curve_record(record)],
        key=lambda record: int(record["step"]),
    )
    overlaps = 0
    pairs = 0
    for previous, current in zip(sorted_records, sorted_records[1:], strict=False):
        prev_ci = previous.get("roc_auc_task_bootstrap_ci")
        curr_ci = current.get("roc_auc_task_bootstrap_ci")
        if not isinstance(prev_ci, Mapping) or not isinstance(curr_ci, Mapping):
            continue
        pairs += 1
        if _interval_overlap(prev_ci, curr_ci):
            overlaps += 1
    if pairs <= 0:
        return None
    return float(overlaps) / float(pairs)


def curve_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one checkpoint curve after task statistics are attached."""

    successful_records = sorted(
        [dict(record) for record in records if _is_successful_curve_record(record)],
        key=lambda record: int(record["step"]),
    )
    if not successful_records:
        return {
            "checkpoint_count": 0,
            "task_count": 0,
            "best_step": 0,
            "best_roc_auc": float("nan"),
            "best_crps": None,
            "final_step": 0,
            "final_roc_auc": float("nan"),
            "final_crps": None,
            "adjacent_ci_overlap_fraction": None,
        }
    ranking_key, ranking_direction = _curve_ranking_metric(successful_records)
    best_record = (
        max(successful_records, key=lambda record: float(record[ranking_key]))
        if ranking_direction == "max"
        else min(successful_records, key=lambda record: float(record[ranking_key]))
    )
    final_record = successful_records[-1]
    task_count = max(
        int(
            record.get(
                "dataset_count",
                max(
                    [
                        len(value)
                        for key, value in record.items()
                        if key.startswith("dataset_") and isinstance(value, Mapping)
                    ],
                    default=0,
                ),
            )
        )
        for record in successful_records
    )
    return {
        "checkpoint_count": int(len(successful_records)),
        "task_count": int(task_count),
        "best_step": int(best_record["step"]),
        "best_roc_auc": None
        if best_record.get("roc_auc") is None
        else float(best_record["roc_auc"]),
        "best_crps": None if best_record.get("crps") is None else float(best_record["crps"]),
        "final_step": int(final_record["step"]),
        "final_roc_auc": None
        if final_record.get("roc_auc") is None
        else float(final_record["roc_auc"]),
        "final_crps": None if final_record.get("crps") is None else float(final_record["crps"]),
        "adjacent_ci_overlap_fraction": curve_adjacent_ci_overlap_fraction(successful_records),
    }


def summarize_checkpoint_curve(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Build one checkpoint-trace summary with failure and CI metadata."""

    successful_records = [
        dict(record)
        for record in records
        if _is_successful_curve_record(record)
    ]
    failed_records = [
        dict(record)
        for record in records
        if not _is_successful_curve_record(record)
    ]
    annotated_successful_records = annotate_curve_records_with_task_statistics(
        successful_records,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_confidence=float(bootstrap_confidence),
        bootstrap_seed=int(bootstrap_seed),
    )
    sorted_successful_records = sorted(
        annotated_successful_records,
        key=lambda record: int(record["step"]),
    )
    trace_summary = curve_summary(sorted_successful_records)
    best_record = None
    if sorted_successful_records:
        ranking_key, ranking_direction = _curve_ranking_metric(sorted_successful_records)
        best_record = (
            max(sorted_successful_records, key=lambda record: float(record[ranking_key]))
            if ranking_direction == "max"
            else min(sorted_successful_records, key=lambda record: float(record[ranking_key]))
        )
    final_record = None if not sorted_successful_records else sorted_successful_records[-1]
    best_checkpoint_path = (
        None
        if best_record is None or best_record.get("checkpoint_path") is None
        else str(best_record["checkpoint_path"])
    )
    final_checkpoint_path = (
        None
        if final_record is None or final_record.get("checkpoint_path") is None
        else str(final_record["checkpoint_path"])
    )
    sorted_records = sorted(
        [dict(record) for record in sorted_successful_records + failed_records],
        key=lambda record: int(record["step"]),
    )
    for record in sorted_records:
        checkpoint_path = (
            None if record.get("checkpoint_path") is None else str(record["checkpoint_path"])
        )
        record["is_best_checkpoint"] = bool(
            checkpoint_path is not None and checkpoint_path == best_checkpoint_path
        )
        record["is_final_checkpoint"] = bool(
            checkpoint_path is not None and checkpoint_path == final_checkpoint_path
        )
    return {
        "records": sorted_records,
        "successful_records": sorted_successful_records,
        "failed_records": sorted(
            [dict(record) for record in failed_records],
            key=lambda record: int(record["step"]),
        ),
        "summary": trace_summary,
        "checkpoint_count": int(len(sorted_records)),
        "successful_checkpoint_count": int(len(sorted_successful_records)),
        "failed_checkpoint_count": int(len(failed_records)),
        "best_record": None if best_record is None else dict(best_record),
        "final_record": None if final_record is None else dict(final_record),
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "last_attempted_step": 0 if not sorted_records else int(sorted_records[-1]["step"]),
        "last_attempted_checkpoint_path": (
            None
            if not sorted_records or sorted_records[-1].get("checkpoint_path") is None
            else str(sorted_records[-1]["checkpoint_path"])
        ),
        "best_to_final_roc_auc_delta": (
            None
            if best_record is None
            or final_record is None
            or best_record.get("roc_auc") is None
            or final_record.get("roc_auc") is None
            else float(final_record["roc_auc"]) - float(best_record["roc_auc"])
        ),
        "best_to_final_crps_delta": (
            None
            if best_record is None
            or final_record is None
            or best_record.get("crps") is None
            or final_record.get("crps") is None
            else float(final_record["crps"]) - float(best_record["crps"])
        ),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "confidence": float(bootstrap_confidence),
            "seed": int(bootstrap_seed),
        },
    }


def resolve_device(device: str) -> str:
    """Resolve auto device selection to a concrete torch device string."""

    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_tab_foundry_run_artifact_paths(run_dir: Path) -> tuple[Path, Path]:
    """Resolve the training-history JSONL and checkpoint directory for a run."""

    resolved_run_dir = run_dir.expanduser().resolve()
    candidates = [
        (
            resolved_run_dir / "train_history.jsonl",
            resolved_run_dir / "checkpoints",
        ),
        (
            resolved_run_dir / "train_outputs" / "train_history.jsonl",
            resolved_run_dir / "train_outputs" / "checkpoints",
        ),
    ]
    for history_path, checkpoint_dir in candidates:
        if history_path.exists() and checkpoint_dir.exists():
            return history_path, checkpoint_dir
    expected = ", ".join(
        f"history={history_path}, checkpoints={checkpoint_dir}"
        for history_path, checkpoint_dir in candidates
    )
    raise RuntimeError(f"missing tab-foundry run artifacts under {resolved_run_dir}; checked {expected}")


def resolve_tab_foundry_best_checkpoint(run_dir: Path) -> Path:
    """Resolve the best checkpoint path for a plain or smoke tab-foundry run."""

    resolved_run_dir = run_dir.expanduser().resolve()
    candidates = [
        resolved_run_dir / "checkpoints" / "best.pt",
        resolved_run_dir / "train_outputs" / "checkpoints" / "best.pt",
    ]
    for checkpoint_path in candidates:
        if checkpoint_path.exists():
            return checkpoint_path.resolve()
    expected = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"missing best checkpoint under {resolved_run_dir}; checked {expected}")


def collect_checkpoint_snapshots(run_dir: Path) -> list[dict[str, Any]]:
    """Resolve step checkpoints and their elapsed training times."""

    resolved_run_dir = run_dir.expanduser().resolve()
    telemetry_path = resolved_run_dir / "telemetry.json"
    if telemetry_path.exists():
        payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
        snapshots = payload.get("checkpoint_snapshots")
        if isinstance(snapshots, list) and snapshots:
            return sorted(
                [
                    {
                        "step": int(snapshot["step"]),
                        "path": str(Path(str(snapshot["path"])).expanduser().resolve()),
                        "elapsed_seconds": float(
                            snapshot.get("train_elapsed_seconds", snapshot["elapsed_seconds"])
                        ),
                    }
                    for snapshot in snapshots
                ],
                key=lambda snapshot: int(snapshot["step"]),
            )

    history_path, checkpoint_dir = resolve_tab_foundry_run_artifact_paths(resolved_run_dir)
    snapshots = checkpoint_snapshots_from_history(history_path, checkpoint_dir)
    return [
        {
            "step": int(snapshot["step"]),
            "path": str(snapshot["path"]),
            "elapsed_seconds": float(snapshot["train_elapsed_seconds"]),
        }
        for snapshot in snapshots
    ]


def evaluate_tab_foundry_run(
    run_dir: Path,
    *,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    task_type: str,
    device: str,
    allow_checkpoint_failures: bool = False,
    allow_missing_values: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate smoke-run checkpoints on the notebook benchmark suite."""

    from tab_foundry.bench.checkpoint import TabFoundryClassifier, TabFoundryRegressor

    resolved_device = resolve_device(device)
    curve_records: list[dict[str, Any]] = []
    for snapshot in collect_checkpoint_snapshots(run_dir):
        checkpoint_path = Path(str(snapshot["path"]))
        try:
            predictor: Any
            if task_type == _CLASSIFICATION_TASK_TYPE:
                predictor = TabFoundryClassifier(checkpoint_path, device=resolved_device)
                metrics = evaluate_classifier(
                    predictor,
                    datasets,
                    allow_missing_values=allow_missing_values,
                )
            elif task_type == _REGRESSION_TASK_TYPE:
                predictor = TabFoundryRegressor(checkpoint_path, device=resolved_device)
                metrics = evaluate_regressor(
                    predictor,
                    datasets,
                    allow_missing_values=allow_missing_values,
                )
            else:
                raise RuntimeError(f"unsupported benchmark task_type: {task_type!r}")
        except Exception as exc:
            if not allow_checkpoint_failures:
                raise
            failed_dataset = None
            error_type = type(exc).__name__
            if isinstance(exc, BenchmarkDatasetEvaluationError):
                failed_dataset = exc.dataset_name
                error_type = str(exc.error_type)
            curve_records.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "step": int(snapshot["step"]),
                    "training_time": float(snapshot["elapsed_seconds"]),
                    "evaluation_error": str(exc),
                    "evaluation_error_type": error_type,
                    "failed_dataset": failed_dataset,
                }
            )
            continue
        model_arch = str(getattr(predictor.model_spec, "arch", "tabfoundry")).strip().lower()
        model_stage_raw = getattr(predictor.model_spec, "stage", None)
        model_stage = None if model_stage_raw is None else str(model_stage_raw).strip().lower()
        benchmark_profile_raw = getattr(predictor.model, "benchmark_profile", None)
        record: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "step": int(snapshot["step"]),
            "training_time": float(snapshot["elapsed_seconds"]),
            "model_arch": model_arch,
            "model_stage": model_stage,
            "benchmark_profile": None
            if benchmark_profile_raw is None
            else str(benchmark_profile_raw),
        }
        if "ROC AUC" in metrics:
            record["roc_auc"] = float(metrics["ROC AUC"])
            record["dataset_roc_auc"] = dataset_roc_auc_metrics(metrics)
        if "Log Loss" in metrics:
            record["log_loss"] = float(metrics["Log Loss"])
            record["dataset_log_loss"] = dataset_log_loss_metrics(metrics)
        if "Brier Score" in metrics:
            record["brier_score"] = float(metrics["Brier Score"])
            record["dataset_brier_score"] = dataset_brier_score_metrics(metrics)
        if "CRPS" in metrics:
            record["crps"] = float(metrics["CRPS"])
            record["dataset_crps"] = dataset_crps_metrics(metrics)
        if "Average Pinball Loss" in metrics:
            record["avg_pinball_loss"] = float(metrics["Average Pinball Loss"])
            record["dataset_avg_pinball_loss"] = dataset_avg_pinball_loss_metrics(metrics)
        if "PICP 90" in metrics:
            record["picp_90"] = float(metrics["PICP 90"])
            record["dataset_picp_90"] = dataset_picp_90_metrics(metrics)
        curve_records.append(record)
    return curve_records


def aggregate_curve(records: list[dict[str, Any]], *, value_key: str = "roc_auc") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one or more benchmark runs by interpolating on shared training times."""

    if not records:
        return np.asarray([]), np.asarray([]), np.asarray([])
    frame = pd.DataFrame(records)
    if value_key not in frame.columns:
        return np.asarray([]), np.asarray([]), np.asarray([])
    frame = frame[frame[value_key].notna()]
    if frame.empty:
        return np.asarray([]), np.asarray([]), np.asarray([])
    if "seed" not in frame.columns or frame["seed"].isna().all():
        ordered = frame.sort_values("training_time")
        return (
            ordered["training_time"].to_numpy(dtype=np.float64),
            ordered[value_key].to_numpy(dtype=np.float64),
            np.zeros((ordered.shape[0],), dtype=np.float64),
        )

    runs: list[pd.Series] = []
    all_times = sorted({float(value) for value in frame["training_time"].tolist()})
    for _seed, seed_frame in frame.groupby("seed", sort=True):
        run = seed_frame[[value_key, "training_time"]].sort_values("training_time")
        series = run.set_index("training_time")[value_key].reindex(all_times).interpolate()
        runs.append(series)
    if not runs:
        return np.asarray([]), np.asarray([]), np.asarray([])
    shared = pd.concat(runs, axis=1).dropna()
    return (
        shared.index.to_numpy(dtype=np.float64),
        shared.mean(axis=1).to_numpy(dtype=np.float64),
        shared.std(axis=1).fillna(0.0).to_numpy(dtype=np.float64),
    )


def plot_comparison_curve(
    *,
    tab_foundry_records: list[dict[str, Any]],
    nanotabpfn_records: list[dict[str, Any]],
    task_type: str,
    out_path: Path,
) -> Path:
    """Render the tab-foundry vs nanoTabPFN time-vs-quality curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_key = "log_loss" if task_type == _CLASSIFICATION_TASK_TYPE else "crps"
    ylabel = "mean log loss" if task_type == _CLASSIFICATION_TASK_TYPE else "mean CRPS"
    title = (
        "tab-foundry vs nanoTabPFN"
        if nanotabpfn_records
        else "tab-foundry benchmark"
    )
    tab_times, tab_mean, _tab_std = aggregate_curve(tab_foundry_records, value_key=metric_key)
    nano_times, nano_mean, nano_std = aggregate_curve(nanotabpfn_records, value_key=metric_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if tab_times.size > 0:
        ax.plot(tab_times, tab_mean, label="tab-foundry", color="#1f77b4", linewidth=2.0)
    if nano_times.size > 0:
        ax.plot(nano_times, nano_mean, label="nanoTabPFN", color="#d62728", linewidth=2.0)
        if np.any(nano_std > 0):
            ax.fill_between(nano_times, nano_mean - nano_std, nano_mean + nano_std, alpha=0.2, color="#d62728")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if tab_times.size > 0 or nano_times.size > 0:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    return out_path


def _metric_direction(metric_key: str) -> str:
    return "max" if metric_key in {"roc_auc", "picp_90"} else "min"


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        metric_key
        for metric_key in (
            "roc_auc",
            "log_loss",
            "brier_score",
            "crps",
            "avg_pinball_loss",
            "picp_90",
        )
        if metric_key in frame.columns
    ]


def build_comparison_summary(
    *,
    tab_foundry_records: list[dict[str, Any]],
    nanotabpfn_records: list[dict[str, Any]],
    benchmark_tasks: list[dict[str, Any]],
    benchmark_bundle: Mapping[str, Any],
    benchmark_bundle_path: Path,
    tab_foundry_run_dir: Path,
    task_type: str,
    nanotabpfn_root: Path | None = None,
    nanotabpfn_python: Path | None = None,
    control_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON summary for the benchmark comparison."""

    def _summary_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
        if not records:
            return {
                "best_step": 0.0,
                "best_training_time": 0.0,
                "final_step": 0.0,
                "final_training_time": 0.0,
            }
        frame = pd.DataFrame(records)
        metric_columns = _metric_columns(frame)
        if not metric_columns:
            return {
                "best_step": 0.0,
                "best_training_time": 0.0,
                "final_step": 0.0,
                "final_training_time": 0.0,
            }
        group_key = "step" if "step" in frame.columns else "training_time"
        aggregate_columns = ["training_time", *metric_columns]
        grouped = (
            frame.groupby(group_key, sort=True)[aggregate_columns]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(group_key)
        )
        if grouped.empty:
            return {
                "best_step": 0.0,
                "best_training_time": 0.0,
                "final_step": 0.0,
                "final_training_time": 0.0,
            }
        ranking_key, ranking_direction = _curve_ranking_metric(records)
        ranking_series = grouped[ranking_key].astype(float)
        best_index = int(
            ranking_series.idxmax() if ranking_direction == "max" else ranking_series.idxmin()
        )
        best_row = grouped.loc[best_index]
        final_row = grouped.iloc[-1]
        summary_metrics: dict[str, float | None] = {
            "best_step": 0.0 if group_key == "training_time" else float(best_row[group_key]),
            "best_training_time": float(best_row["training_time"]),
            "final_step": 0.0 if group_key == "training_time" else float(final_row[group_key]),
            "final_training_time": float(final_row["training_time"]),
        }
        for metric_key in metric_columns:
            metric_series = grouped[metric_key].astype(float)
            metric_direction = _metric_direction(metric_key)
            best_metric = (
                metric_series.max() if metric_direction == "max" else metric_series.min()
            )
            summary_metrics[f"best_{metric_key}"] = float(best_metric)
            summary_metrics[f"final_{metric_key}"] = float(final_row[metric_key])
        return summary_metrics

    def _dataset_summary(
        records: list[dict[str, Any]],
        *,
        step: float,
        record_key: str,
    ) -> dict[str, float]:
        per_dataset: dict[str, list[float]] = {}
        for record in records:
            if float(record.get("step", -1.0)) != float(step):
                continue
            raw_metrics = record.get(record_key)
            if not isinstance(raw_metrics, dict):
                continue
            for dataset_name, value in raw_metrics.items():
                per_dataset.setdefault(str(dataset_name), []).append(float(value))
        return {
            dataset_name: float(np.mean(values))
            for dataset_name, values in sorted(per_dataset.items())
            if values
        }

    def _dataset_delta_summary(
        records: list[dict[str, Any]],
        *,
        from_step: float,
        to_step: float,
        record_key: str,
    ) -> dict[str, float]:
        from_metrics = _dataset_summary(records, step=from_step, record_key=record_key)
        to_metrics = _dataset_summary(records, step=to_step, record_key=record_key)
        shared_dataset_names = sorted(set(from_metrics) & set(to_metrics))
        return {
            dataset_name: float(to_metrics[dataset_name]) - float(from_metrics[dataset_name])
            for dataset_name in shared_dataset_names
        }

    def _identity(records: list[dict[str, Any]]) -> dict[str, str | None]:
        if not records:
            return {
                "model_arch": None,
                "model_stage": None,
                "benchmark_profile": None,
            }
        first = records[0]
        return {
            "model_arch": None if first.get("model_arch") is None else str(first["model_arch"]),
            "model_stage": None if first.get("model_stage") is None else str(first["model_stage"]),
            "benchmark_profile": None
            if first.get("benchmark_profile") is None
            else str(first["benchmark_profile"]),
        }

    def _apply_metric_summaries(
        section: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        best_step: float,
        final_step: float,
        best_record: Mapping[str, Any] | None,
        final_record: Mapping[str, Any] | None,
    ) -> None:
        metric_record_keys = {
            "roc_auc": "dataset_roc_auc",
            "log_loss": "dataset_log_loss",
            "brier_score": "dataset_brier_score",
            "crps": "dataset_crps",
            "avg_pinball_loss": "dataset_avg_pinball_loss",
            "picp_90": "dataset_picp_90",
        }
        for metric_key, record_key in metric_record_keys.items():
            best_value = None if best_record is None else _ensure_optional_float(best_record.get(metric_key))
            final_value = None if final_record is None else _ensure_optional_float(final_record.get(metric_key))
            if best_value is not None:
                section[f"best_{metric_key}"] = best_value
            if final_value is not None:
                section[f"final_{metric_key}"] = final_value
            if best_value is not None and final_value is not None:
                section[f"best_to_final_{metric_key}_delta"] = float(final_value) - float(best_value)
            best_dataset = _dataset_summary(records, step=best_step, record_key=record_key)
            final_dataset = _dataset_summary(records, step=final_step, record_key=record_key)
            if best_dataset:
                section[f"best_{record_key}"] = best_dataset
            if final_dataset:
                section[f"final_{record_key}"] = final_dataset
            dataset_delta = _dataset_delta_summary(
                records,
                from_step=best_step,
                to_step=final_step,
                record_key=record_key,
            )
            if dataset_delta:
                section[f"best_to_final_{record_key}_delta"] = dataset_delta

    tab_foundry_curve = summarize_checkpoint_curve(
        tab_foundry_records,
        bootstrap_samples=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
        bootstrap_confidence=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
        bootstrap_seed=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
    )
    tab_foundry_successful_records = cast(
        list[dict[str, Any]],
        tab_foundry_curve["successful_records"],
    )
    tab_foundry_curve_summary = cast(dict[str, Any], tab_foundry_curve["summary"])
    summary = {
        "dataset_count": int(len(benchmark_tasks)),
        "benchmark_bundle": benchmark_bundle_summary(
            benchmark_bundle,
            source_path=benchmark_bundle_path,
        ),
        "tab_foundry": {
            "best_step": float(tab_foundry_curve_summary["best_step"]),
            "best_training_time": float(
                0.0
                if tab_foundry_curve["best_record"] is None
                else cast(dict[str, Any], tab_foundry_curve["best_record"])["training_time"]
            ),
            "final_step": float(tab_foundry_curve_summary["final_step"]),
            "final_training_time": float(
                0.0
                if tab_foundry_curve["final_record"] is None
                else cast(dict[str, Any], tab_foundry_curve["final_record"])["training_time"]
            ),
            "run_dir": str(tab_foundry_run_dir.expanduser().resolve()),
            **_identity(tab_foundry_successful_records),
        },
    }
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    _apply_metric_summaries(
        tab_foundry_summary,
        tab_foundry_successful_records,
        best_step=float(tab_foundry_summary["best_step"]),
        final_step=float(tab_foundry_summary["final_step"]),
        best_record=cast(Mapping[str, Any] | None, tab_foundry_curve["best_record"]),
        final_record=cast(Mapping[str, Any] | None, tab_foundry_curve["final_record"]),
    )
    if tab_foundry_curve_summary.get("best_roc_auc") is not None:
        tab_foundry_summary["best_roc_auc"] = float(tab_foundry_curve_summary["best_roc_auc"])
    if tab_foundry_curve_summary.get("final_roc_auc") is not None:
        tab_foundry_summary["final_roc_auc"] = float(tab_foundry_curve_summary["final_roc_auc"])
    if tab_foundry_curve_summary.get("best_crps") is not None:
        tab_foundry_summary["best_crps"] = float(tab_foundry_curve_summary["best_crps"])
    if tab_foundry_curve_summary.get("final_crps") is not None:
        tab_foundry_summary["final_crps"] = float(tab_foundry_curve_summary["final_crps"])
    if tab_foundry_curve.get("best_to_final_roc_auc_delta") is not None:
        tab_foundry_summary["best_to_final_roc_auc_delta"] = tab_foundry_curve["best_to_final_roc_auc_delta"]
    if tab_foundry_curve.get("best_to_final_crps_delta") is not None:
        tab_foundry_summary["best_to_final_crps_delta"] = tab_foundry_curve["best_to_final_crps_delta"]
    tab_foundry_summary["checkpoint_diagnostics"] = {
        "checkpoint_count": int(tab_foundry_curve["checkpoint_count"]),
        "successful_checkpoint_count": int(tab_foundry_curve["successful_checkpoint_count"]),
        "failed_checkpoint_count": int(tab_foundry_curve["failed_checkpoint_count"]),
        "task_count": int(tab_foundry_curve_summary["task_count"]),
        "adjacent_ci_overlap_fraction": tab_foundry_curve_summary[
            "adjacent_ci_overlap_fraction"
        ],
        "best_checkpoint_path": tab_foundry_curve["best_checkpoint_path"],
        "final_checkpoint_path": tab_foundry_curve["final_checkpoint_path"],
        "last_attempted_step": int(tab_foundry_curve["last_attempted_step"]),
        "last_attempted_checkpoint_path": tab_foundry_curve["last_attempted_checkpoint_path"],
        "bootstrap": dict(cast(dict[str, Any], tab_foundry_curve["bootstrap"])),
        "best_checkpoint": tab_foundry_curve["best_record"],
        "final_checkpoint": tab_foundry_curve["final_record"],
        "checkpoints": cast(list[dict[str, Any]], tab_foundry_curve["records"]),
        "failed_checkpoints": cast(list[dict[str, Any]], tab_foundry_curve["failed_records"]),
    }
    if nanotabpfn_records:
        nanotabpfn_summary = {
            **_summary_metrics(nanotabpfn_records),
            "root": None if nanotabpfn_root is None else str(nanotabpfn_root.expanduser().resolve()),
            "python": None
            if nanotabpfn_python is None
            else str(nanotabpfn_python.expanduser().resolve()),
            "num_seeds": int(len({int(record["seed"]) for record in nanotabpfn_records})),
        }
        best_step_raw = nanotabpfn_summary.get("best_step")
        final_step_raw = nanotabpfn_summary.get("final_step")
        best_step = 0.0 if best_step_raw is None else float(best_step_raw)
        final_step = 0.0 if final_step_raw is None else float(final_step_raw)
        best_record = None
        if best_step > 0.0:
            best_record = next(
                (record for record in nanotabpfn_records if float(record.get("step", -1.0)) == best_step),
                None,
            )
        final_record = None
        if final_step > 0.0:
            final_record = next(
                (record for record in nanotabpfn_records if float(record.get("step", -1.0)) == final_step),
                None,
            )
        _apply_metric_summaries(
            nanotabpfn_summary,
            nanotabpfn_records,
            best_step=best_step,
            final_step=final_step,
            best_record=best_record,
            final_record=final_record,
        )
        summary["nanotabpfn"] = nanotabpfn_summary
    if task_type == _CLASSIFICATION_TASK_TYPE:
        if tab_foundry_summary.get("final_log_loss") is None:
            raise RuntimeError("tab-foundry benchmark produced no log-loss values")
        if nanotabpfn_records and cast(dict[str, Any], summary["nanotabpfn"]).get("final_log_loss") is None:
            raise RuntimeError("nanoTabPFN benchmark produced no log-loss values")
    elif task_type == _REGRESSION_TASK_TYPE:
        if tab_foundry_summary.get("final_crps") is None:
            raise RuntimeError("tab-foundry benchmark produced no CRPS values")
    else:
        raise RuntimeError(f"unsupported benchmark task_type: {task_type!r}")
    if control_baseline is not None:
        summary["control_baseline"] = json.loads(json.dumps(control_baseline, sort_keys=True))
    return summary
