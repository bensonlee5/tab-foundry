"""OpenML benchmark dataset preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import openml
from openml.tasks import TaskType
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from tab_foundry.data.validation import assert_no_non_finite_values

from .bundle import (
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
    benchmark_bundle_task_type,
    load_benchmark_bundle,
)


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
