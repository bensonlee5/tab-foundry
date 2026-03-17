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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from tab_foundry.data.validation import assert_no_non_finite_values
from tab_foundry.bench.artifacts import checkpoint_snapshots_from_history


BENCHMARK_BUNDLE_FILENAME = "nanotabpfn_openml_binary_medium_v1.json"
_BUNDLE_SELECTION_TASK_TYPE = "supervised_classification"
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED = 0

_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


@dataclass(slots=True)
class PreparedOpenMLBenchmarkTask:
    """Materialized OpenML benchmark task after notebook-style preprocessing."""

    task_id: int
    dataset_name: str
    x: np.ndarray
    y: np.ndarray
    observed_task: dict[str, Any]
    qualities: dict[str, float]


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
    expected_keys = {
        "new_instances",
        "task_type",
        "max_features",
        "max_classes",
        "max_missing_pct",
        "min_minority_class_pct",
    }
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle selection keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    new_instances = payload["new_instances"]
    task_type = payload["task_type"]
    max_features = payload["max_features"]
    max_classes = payload["max_classes"]
    max_missing_pct = payload["max_missing_pct"]
    min_minority_class_pct = payload["min_minority_class_pct"]

    if not isinstance(new_instances, int) or isinstance(new_instances, bool) or new_instances <= 0:
        raise RuntimeError("benchmark bundle selection.new_instances must be a positive int")
    if task_type != _BUNDLE_SELECTION_TASK_TYPE:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be "
            f"{_BUNDLE_SELECTION_TASK_TYPE!r}"
        )
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0:
        raise RuntimeError("benchmark bundle selection.max_features must be a positive int")
    if not isinstance(max_classes, int) or isinstance(max_classes, bool) or max_classes <= 0:
        raise RuntimeError("benchmark bundle selection.max_classes must be a positive int")
    if not isinstance(max_missing_pct, (int, float)) or not 0 <= float(max_missing_pct) <= 100:
        raise RuntimeError("benchmark bundle selection.max_missing_pct must be a percentage between 0 and 100")
    if not isinstance(min_minority_class_pct, (int, float)) or not 0 <= float(min_minority_class_pct) <= 100:
        raise RuntimeError(
            "benchmark bundle selection.min_minority_class_pct must be a percentage between 0 and 100"
        )
    return {
        "new_instances": int(new_instances),
        "task_type": str(task_type),
        "max_features": int(max_features),
        "max_classes": int(max_classes),
        "max_missing_pct": float(max_missing_pct),
        "min_minority_class_pct": float(min_minority_class_pct),
    }


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

    normalized_task_ids = [int(task_id) for task_id in task_ids]
    normalized_tasks: list[dict[str, Any]] = []
    for index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            raise RuntimeError(f"benchmark bundle task {index} must be an object")
        task_keys = {"task_id", "dataset_name", "n_rows", "n_features", "n_classes"}
        actual_task_keys = set(task_payload.keys())
        if actual_task_keys != task_keys:
            raise RuntimeError(
                f"benchmark bundle task keys mismatch at index {index}: "
                f"expected={sorted(task_keys)}, actual={sorted(actual_task_keys)}"
            )
        dataset_name = task_payload["dataset_name"]
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise RuntimeError(f"benchmark bundle task dataset_name must be non-empty at index {index}")
        normalized_tasks.append(
            {
                "task_id": int(task_payload["task_id"]),
                "dataset_name": str(dataset_name),
                "n_rows": int(task_payload["n_rows"]),
                "n_features": int(task_payload["n_features"]),
                "n_classes": int(task_payload["n_classes"]),
            }
        )

    if normalized_task_ids != [int(task["task_id"]) for task in normalized_tasks]:
        raise RuntimeError("benchmark bundle task_ids must match tasks[].task_id order exactly")

    normalized_bundle: dict[str, Any] = {
        "name": str(name),
        "version": int(version),
        "selection": _normalize_selection(selection),
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


def prepare_openml_benchmark_task(
    task_id: int,
    *,
    new_instances: int,
) -> PreparedOpenMLBenchmarkTask:
    """Load and preprocess one OpenML task using the nanoTabPFN notebook logic."""

    task = openml.tasks.get_task(task_id, download_splits=False)
    if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
        raise RuntimeError(
            "benchmark bundle drift: "
            f"task {task_id} is no longer supervised classification"
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
    number_of_classes = read_required_openml_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="NumberOfClasses",
    )
    missing_pct = read_required_openml_quality(
        raw_qualities,
        task_id=int(task_id),
        quality_name="PercentageOfInstancesWithMissingValues",
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
        _x_unused, x_sub, _y_unused, y_sub = train_test_split(
            x_frame,
            y_raw,
            test_size=new_instances,
            stratify=y_raw,
            random_state=0,
        )
    else:
        x_sub = x_frame
        y_sub = y_raw

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_sub.to_numpy(copy=True))
    preprocessor = get_feature_preprocessor(x_sub)
    x = np.asarray(preprocessor.fit_transform(x_sub), dtype=np.float32)

    observed_task = {
        "task_id": int(task_id),
        "dataset_name": str(dataset.name),
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(np.unique(y).size),
    }
    return PreparedOpenMLBenchmarkTask(
        task_id=int(task_id),
        dataset_name=str(dataset.name),
        x=x,
        y=y.astype(np.int64, copy=False),
        observed_task=observed_task,
        qualities={
            "NumberOfFeatures": float(number_of_features),
            "NumberOfClasses": float(number_of_classes),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            "MinorityClassPercentage": float(minority_class_pct),
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
        prepared = prepare_openml_benchmark_task(int(task_id), new_instances=new_instances)
        number_of_features = prepared.qualities["NumberOfFeatures"]
        number_of_classes = prepared.qualities["NumberOfClasses"]
        missing_pct = prepared.qualities["PercentageOfInstancesWithMissingValues"]
        minority_class_pct = prepared.qualities["MinorityClassPercentage"]
        if number_of_features > int(selection["max_features"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_features expected<={selection['max_features']}, actual={number_of_features}"
            )
        if number_of_classes > int(selection["max_classes"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_classes expected<={selection['max_classes']}, actual={number_of_classes}"
            )
        if missing_pct > float(selection["max_missing_pct"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_missing_pct expected<={selection['max_missing_pct']}, actual={missing_pct}"
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
        payload[f"y_{index:03d}"] = np.asarray(y, dtype=np.int64)
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
            np.asarray(cache[f"y_{index:03d}"], dtype=np.int64),
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
            probabilities: list[np.ndarray] = []
            for train_idx, test_idx in _SKF.split(x, y):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                targets.append(y_test)
                classifier.fit(x_train, y_train)
                y_proba = classifier.predict_proba(x_test)
                if y_proba.shape[1] == 2:
                    probabilities.append(np.asarray(y_proba[:, 1], dtype=np.float64))
                else:
                    probabilities.append(np.asarray(y_proba, dtype=np.float64))

            target_array = np.concatenate(targets, axis=0)
            probability_array = np.concatenate(probabilities, axis=0)
            assert_no_non_finite_values(
                {"probabilities": probability_array},
                context=f"benchmark classifier outputs dataset={dataset_name!r}",
            )
            metrics[f"{dataset_name}/ROC AUC"] = float(
                roc_auc_score(target_array, probability_array, multi_class="ovr")
            )
        except Exception as exc:
            raise BenchmarkDatasetEvaluationError(str(dataset_name), exc) from exc

    roc_auc_values = [value for key, value in metrics.items() if key.endswith("/ROC AUC")]
    metrics["ROC AUC"] = float(np.mean(roc_auc_values))
    return metrics


def _evaluate_classifier_fast(
    model: Any,
    model_spec: Any,
    device: Any,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    allow_missing_values: bool = False,
) -> dict[str, float]:
    """Evaluate a tab-foundry model on the benchmark suite with reduced overhead.

    Mirrors the logic of ``evaluate_classifier`` + ``TabFoundryClassifier.predict_proba``
    but inlines it to avoid per-call overhead: enters torch.no_grad() once,
    pre-resolves normalization mode once, and uses torch.from_numpy for
    zero-copy tensor creation.  Keep in sync with those two implementations.
    """
    import torch
    import torch.nn.functional as F
    from tab_foundry.input_normalization import (
        InputNormalizationMode,
        normalize_train_test_arrays,
    )
    from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
        staged_surface_uses_internal_benchmark_normalization,
    )
    from tab_foundry.types import TaskBatch

    if not allow_missing_values:
        _assert_finite_benchmark_datasets(datasets, context="benchmark evaluation inputs")

    model_arch = str(getattr(model_spec, "arch", "tabfoundry")).strip().lower()
    normalization_mode: InputNormalizationMode = cast(
        InputNormalizationMode,
        str(getattr(model_spec, "input_normalization", "none")).strip().lower(),
    )
    internal_normalization = model_arch == "tabfoundry_simple"
    if model_arch == "tabfoundry_staged":
        internal_normalization = staged_surface_uses_internal_benchmark_normalization(
            model_spec,
        )
    skip_normalization = internal_normalization or normalization_mode == "none"

    metrics: dict[str, float] = {}
    with torch.no_grad():
        for dataset_name, (x, y) in datasets.items():
            try:
                targets: list[np.ndarray] = []
                probabilities: list[np.ndarray] = []
                for train_idx, test_idx in _SKF.split(x, y):
                    x_train, x_test = x[train_idx], x[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    targets.append(y_test)

                    classes, encoded = np.unique(y_train, return_inverse=True)
                    num_classes = int(classes.size)
                    x_train_f = np.ascontiguousarray(x_train, dtype=np.float32)
                    y_train_enc = encoded.astype(np.int64, copy=False)
                    x_test_f = np.ascontiguousarray(x_test, dtype=np.float32)

                    if skip_normalization:
                        x_train_norm, x_test_norm = x_train_f, x_test_f
                    else:
                        x_train_norm, x_test_norm = normalize_train_test_arrays(
                            x_train_f, x_test_f, mode=normalization_mode,
                        )

                    batch = TaskBatch(
                        x_train=torch.from_numpy(x_train_norm).to(device),
                        y_train=torch.from_numpy(y_train_enc).to(device),
                        x_test=torch.from_numpy(x_test_norm).to(device),
                        y_test=torch.zeros(x_test_norm.shape[0], dtype=torch.int64, device=device),
                        metadata={"dataset": "external_benchmark"},
                        num_classes=num_classes,
                    )

                    output = model(batch)
                    if output.logits is not None:
                        probs = F.softmax(output.logits[:, :num_classes], dim=-1)
                    elif output.class_probs is not None:
                        probs = output.class_probs[:, :num_classes]
                    else:
                        raise RuntimeError("checkpoint output does not expose logits or class probabilities")

                    y_proba = probs.cpu().numpy()
                    if y_proba.shape[1] == 2:
                        probabilities.append(np.asarray(y_proba[:, 1], dtype=np.float64))
                    else:
                        probabilities.append(np.asarray(y_proba, dtype=np.float64))

                target_array = np.concatenate(targets, axis=0)
                probability_array = np.concatenate(probabilities, axis=0)
                assert_no_non_finite_values(
                    {"probabilities": probability_array},
                    context=f"benchmark classifier outputs dataset={dataset_name!r}",
                )
                metrics[f"{dataset_name}/ROC AUC"] = float(
                    roc_auc_score(target_array, probability_array, multi_class="ovr")
                )
            except Exception as exc:
                raise BenchmarkDatasetEvaluationError(str(dataset_name), exc) from exc

    roc_auc_values = [value for key, value in metrics.items() if key.endswith("/ROC AUC")]
    metrics["ROC AUC"] = float(np.mean(roc_auc_values))
    return metrics


def dataset_roc_auc_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Extract per-dataset ROC AUC values from a benchmark metrics dict."""

    return {
        str(key[: -len("/ROC AUC")]): float(value)
        for key, value in metrics.items()
        if key.endswith("/ROC AUC") and key != "ROC AUC"
    }


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
        enriched["dataset_count"] = int(len(dataset_roc_auc))
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
    raw_roc_auc = record.get("roc_auc")
    if raw_roc_auc is None:
        return False
    try:
        return math.isfinite(float(raw_roc_auc))
    except (TypeError, ValueError):
        return False


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
            "final_step": 0,
            "final_roc_auc": float("nan"),
            "adjacent_ci_overlap_fraction": None,
        }
    best_record = max(successful_records, key=lambda record: float(record["roc_auc"]))
    final_record = successful_records[-1]
    task_count = max(
        int(
            record.get(
                "dataset_count",
                len(record["dataset_roc_auc"])
                if isinstance(record.get("dataset_roc_auc"), Mapping)
                else 0,
            )
        )
        for record in successful_records
    )
    return {
        "checkpoint_count": int(len(successful_records)),
        "task_count": int(task_count),
        "best_step": int(best_record["step"]),
        "best_roc_auc": float(best_record["roc_auc"]),
        "final_step": int(final_record["step"]),
        "final_roc_auc": float(final_record["roc_auc"]),
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
    best_record = (
        None
        if not sorted_successful_records
        else max(sorted_successful_records, key=lambda record: float(record["roc_auc"]))
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
            if best_record is None or final_record is None
            else float(final_record["roc_auc"]) - float(best_record["roc_auc"])
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
    device: str,
    allow_checkpoint_failures: bool = False,
    allow_missing_values: bool = False,
    compile_model: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate smoke-run checkpoints on the notebook benchmark suite."""

    import torch as _torch

    from tab_foundry.bench.checkpoint import TabFoundryClassifier

    resolved_device = resolve_device(device)
    curve_records: list[dict[str, Any]] = []
    classifier: TabFoundryClassifier | None = None
    use_fast_path = False
    compiled = False
    for snapshot in collect_checkpoint_snapshots(run_dir):
        checkpoint_path = Path(str(snapshot["path"]))
        try:
            if classifier is None:
                classifier = TabFoundryClassifier(checkpoint_path, device=resolved_device)
                use_fast_path = isinstance(getattr(classifier, "model", None), _torch.nn.Module)
                if compile_model and not compiled and use_fast_path:
                    try:
                        classifier.model = _torch.compile(classifier.model)  # type: ignore[assignment]
                        compiled = True
                    except Exception as compile_exc:
                        import sys

                        print(
                            f"[benchmark] torch.compile failed, falling back to eager: {compile_exc}",
                            file=sys.stderr,
                        )
            elif hasattr(classifier, "reload_weights"):
                classifier.reload_weights(checkpoint_path)
            else:
                classifier = TabFoundryClassifier(checkpoint_path, device=resolved_device)
            if use_fast_path:
                metrics = _evaluate_classifier_fast(
                    classifier.model,
                    classifier.model_spec,
                    classifier.device,
                    datasets,
                    allow_missing_values=allow_missing_values,
                )
            else:
                metrics = evaluate_classifier(
                    classifier,
                    datasets,
                    allow_missing_values=allow_missing_values,
                )
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
            classifier = None
            continue
        model_arch = str(getattr(classifier.model_spec, "arch", "tabfoundry")).strip().lower()
        model_stage_raw = getattr(classifier.model_spec, "stage", None)
        model_stage = None if model_stage_raw is None else str(model_stage_raw).strip().lower()
        benchmark_profile_raw = getattr(classifier.model, "benchmark_profile", None)
        curve_records.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "step": int(snapshot["step"]),
                "training_time": float(snapshot["elapsed_seconds"]),
                "roc_auc": float(metrics["ROC AUC"]),
                "dataset_roc_auc": dataset_roc_auc_metrics(metrics),
                "model_arch": model_arch,
                "model_stage": model_stage,
                "benchmark_profile": None
                if benchmark_profile_raw is None
                else str(benchmark_profile_raw),
            }
        )
    return curve_records


def aggregate_curve(records: list[dict[str, Any]], *, value_key: str = "roc_auc") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one or more benchmark runs by interpolating on shared training times."""

    if not records:
        return np.asarray([]), np.asarray([]), np.asarray([])
    frame = pd.DataFrame(records)
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
    out_path: Path,
) -> Path:
    """Render the tab-foundry vs nanoTabPFN time-vs-quality curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab_times, tab_mean, _tab_std = aggregate_curve(tab_foundry_records)
    nano_times, nano_mean, nano_std = aggregate_curve(nanotabpfn_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if tab_times.size > 0:
        ax.plot(tab_times, tab_mean, label="tab-foundry", color="#1f77b4", linewidth=2.0)
    if nano_times.size > 0:
        ax.plot(nano_times, nano_mean, label="nanoTabPFN", color="#d62728", linewidth=2.0)
        if np.any(nano_std > 0):
            ax.fill_between(nano_times, nano_mean - nano_std, nano_mean + nano_std, alpha=0.2, color="#d62728")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel("mean ROC AUC")
    ax.set_title("tab-foundry vs nanoTabPFN")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    return out_path


def build_comparison_summary(
    *,
    tab_foundry_records: list[dict[str, Any]],
    nanotabpfn_records: list[dict[str, Any]],
    benchmark_tasks: list[dict[str, Any]],
    benchmark_bundle: Mapping[str, Any],
    benchmark_bundle_path: Path,
    tab_foundry_run_dir: Path,
    nanotabpfn_root: Path,
    nanotabpfn_python: Path,
    control_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON summary for the benchmark comparison."""

    def _summary_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {
                "best_step": 0.0,
                "best_training_time": 0.0,
                "best_roc_auc": float("nan"),
                "final_step": 0.0,
                "final_training_time": 0.0,
                "final_roc_auc": float("nan"),
            }
        frame = pd.DataFrame(records)
        if "step" not in frame.columns:
            times, mean, _std = aggregate_curve(records)
            if times.size == 0 or mean.size == 0:
                return {
                    "best_step": 0.0,
                    "best_training_time": 0.0,
                    "best_roc_auc": float("nan"),
                    "final_step": 0.0,
                    "final_training_time": 0.0,
                    "final_roc_auc": float("nan"),
                }
            best_idx = int(np.nanargmax(mean))
            return {
                "best_step": 0.0,
                "best_training_time": float(times[best_idx]),
                "best_roc_auc": float(mean[best_idx]),
                "final_step": 0.0,
                "final_training_time": float(times[-1]),
                "final_roc_auc": float(mean[-1]),
            }

        grouped = (
            frame.groupby("step", sort=True)[["training_time", "roc_auc"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("step")
        )
        if grouped.empty:
            return {
                "best_step": 0.0,
                "best_training_time": 0.0,
                "best_roc_auc": float("nan"),
                "final_step": 0.0,
                "final_training_time": 0.0,
                "final_roc_auc": float("nan"),
            }
        best_index = int(grouped["roc_auc"].astype(float).idxmax())
        best_row = grouped.loc[best_index]
        final_row = grouped.iloc[-1]
        return {
            "best_step": float(best_row["step"]),
            "best_training_time": float(best_row["training_time"]),
            "best_roc_auc": float(best_row["roc_auc"]),
            "final_step": float(final_row["step"]),
            "final_training_time": float(final_row["training_time"]),
            "final_roc_auc": float(final_row["roc_auc"]),
        }

    def _dataset_summary(records: list[dict[str, Any]], *, step: float) -> dict[str, float]:
        per_dataset: dict[str, list[float]] = {}
        for record in records:
            if float(record.get("step", -1.0)) != float(step):
                continue
            raw_metrics = record.get("dataset_roc_auc")
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
    ) -> dict[str, float]:
        from_metrics = _dataset_summary(records, step=from_step)
        to_metrics = _dataset_summary(records, step=to_step)
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
            "best_roc_auc": float(tab_foundry_curve_summary["best_roc_auc"]),
            "final_step": float(tab_foundry_curve_summary["final_step"]),
            "final_training_time": float(
                0.0
                if tab_foundry_curve["final_record"] is None
                else cast(dict[str, Any], tab_foundry_curve["final_record"])["training_time"]
            ),
            "final_roc_auc": float(tab_foundry_curve_summary["final_roc_auc"]),
            "run_dir": str(tab_foundry_run_dir.expanduser().resolve()),
            **_identity(tab_foundry_successful_records),
        },
        "nanotabpfn": {
            **_summary_metrics(nanotabpfn_records),
            "root": str(nanotabpfn_root.expanduser().resolve()),
            "python": str(nanotabpfn_python.expanduser().resolve()),
            "num_seeds": int(len({int(record["seed"]) for record in nanotabpfn_records})),
        },
    }
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    nanotabpfn_summary = cast(dict[str, Any], summary["nanotabpfn"])
    tab_foundry_summary["best_dataset_roc_auc"] = _dataset_summary(
        tab_foundry_successful_records,
        step=float(tab_foundry_summary["best_step"]),
    )
    tab_foundry_summary["final_dataset_roc_auc"] = _dataset_summary(
        tab_foundry_successful_records,
        step=float(tab_foundry_summary["final_step"]),
    )
    tab_foundry_summary["best_to_final_dataset_roc_auc_delta"] = _dataset_delta_summary(
        tab_foundry_successful_records,
        from_step=float(tab_foundry_summary["best_step"]),
        to_step=float(tab_foundry_summary["final_step"]),
    )
    tab_foundry_summary["best_to_final_roc_auc_delta"] = tab_foundry_curve[
        "best_to_final_roc_auc_delta"
    ]
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
    nanotabpfn_summary["best_dataset_roc_auc"] = _dataset_summary(
        nanotabpfn_records,
        step=float(nanotabpfn_summary["best_step"]),
    )
    nanotabpfn_summary["final_dataset_roc_auc"] = _dataset_summary(
        nanotabpfn_records,
        step=float(nanotabpfn_summary["final_step"]),
    )
    nanotabpfn_summary["best_to_final_dataset_roc_auc_delta"] = _dataset_delta_summary(
        nanotabpfn_records,
        from_step=float(nanotabpfn_summary["best_step"]),
        to_step=float(nanotabpfn_summary["final_step"]),
    )
    nanotabpfn_summary["best_to_final_roc_auc_delta"] = float(
        nanotabpfn_summary["final_roc_auc"]
    ) - float(nanotabpfn_summary["best_roc_auc"])
    if math.isnan(float(tab_foundry_summary["final_roc_auc"])):
        raise RuntimeError("tab-foundry benchmark produced no ROC AUC values")
    if math.isnan(float(nanotabpfn_summary["final_roc_auc"])):
        raise RuntimeError("nanoTabPFN benchmark produced no ROC AUC values")
    if control_baseline is not None:
        summary["control_baseline"] = json.loads(json.dumps(control_baseline, sort_keys=True))
    return summary
