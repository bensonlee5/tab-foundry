"""Notebook-style nanoTabPFN comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
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

from tab_foundry.bench.artifacts import checkpoint_snapshots_from_history


BENCHMARK_BUNDLE_FILENAME = "nanotabpfn_openml_benchmark_v1.json"
_BUNDLE_SELECTION_TASK_TYPE = "supervised_classification"

_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


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


def _read_required_quality(raw_qualities: Any, *, task_id: int, quality_name: str) -> float:
    """Read a numeric OpenML quality and raise a drift error if it is missing."""

    if not isinstance(raw_qualities, dict):
        raise RuntimeError(f"benchmark bundle drift: task {task_id} dataset qualities are missing")
    value = raw_qualities.get(quality_name)
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"benchmark bundle drift: task {task_id} missing numeric quality {quality_name!r}"
        )
    return float(value)


def load_benchmark_bundle(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the canonical benchmark bundle metadata."""

    bundle_path = (path or default_benchmark_bundle_path()).expanduser().resolve()
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"benchmark bundle must be a JSON object: {bundle_path}")
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


def benchmark_bundle_summary(
    bundle: Mapping[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Build compact bundle metadata for run summaries."""

    task_ids = [int(task_id) for task_id in cast(list[Any], bundle["task_ids"])]
    return {
        "name": str(bundle["name"]),
        "version": int(bundle["version"]),
        "source_path": str(source_path.expanduser().resolve()),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
    }


def load_openml_benchmark_datasets(
    *,
    new_instances: int = 200,
    benchmark_bundle_path: Path | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Load the nanoTabPFN OpenML benchmark suite."""

    bundle = load_benchmark_bundle(benchmark_bundle_path)
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
        number_of_features = _read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="NumberOfFeatures",
        )
        number_of_classes = _read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="NumberOfClasses",
        )
        missing_pct = _read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="PercentageOfInstancesWithMissingValues",
        )
        minority_class_pct = _read_required_quality(
            raw_qualities,
            task_id=int(task_id),
            quality_name="MinorityClassPercentage",
        )
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
        expected_task = expected_by_task_id[int(task_id)]
        if observed_task != expected_task:
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} metadata mismatch expected={expected_task}, actual={observed_task}"
            )

        datasets[str(dataset.name)] = (x, y.astype(np.int64, copy=False))
        benchmark_tasks.append(observed_task)
    if not datasets:
        raise RuntimeError("OpenML benchmark produced no datasets after filtering")
    if len(benchmark_tasks) != len(expected_tasks):
        raise RuntimeError(
            "benchmark bundle drift: "
            f"task count mismatch expected={len(expected_tasks)}, actual={len(benchmark_tasks)}"
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
) -> dict[str, float]:
    """Evaluate a sklearn-style classifier on the cached benchmark suite."""

    metrics: dict[str, float] = {}
    for dataset_name, (x, y) in datasets.items():
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
        metrics[f"{dataset_name}/ROC AUC"] = float(
            roc_auc_score(target_array, probability_array, multi_class="ovr")
        )

    roc_auc_values = [value for key, value in metrics.items() if key.endswith("/ROC AUC")]
    metrics["ROC AUC"] = float(np.mean(roc_auc_values))
    return metrics


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

    history_path = resolved_run_dir / "train_outputs" / "train_history.jsonl"
    checkpoint_dir = resolved_run_dir / "train_outputs" / "checkpoints"
    if not history_path.exists():
        raise RuntimeError(f"missing smoke history file: {history_path}")
    if not checkpoint_dir.exists():
        raise RuntimeError(f"missing smoke checkpoint directory: {checkpoint_dir}")

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
) -> list[dict[str, Any]]:
    """Evaluate smoke-run checkpoints on the notebook benchmark suite."""

    from tab_foundry.bench.checkpoint import TabFoundryClassifier

    resolved_device = resolve_device(device)
    curve_records: list[dict[str, Any]] = []
    for snapshot in collect_checkpoint_snapshots(run_dir):
        checkpoint_path = Path(str(snapshot["path"]))
        classifier = TabFoundryClassifier(checkpoint_path, device=resolved_device)
        metrics = evaluate_classifier(classifier, datasets)
        curve_records.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "step": int(snapshot["step"]),
                "training_time": float(snapshot["elapsed_seconds"]),
                "roc_auc": float(metrics["ROC AUC"]),
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

    summary = {
        "dataset_count": int(len(benchmark_tasks)),
        "benchmark_bundle": benchmark_bundle_summary(
            benchmark_bundle,
            source_path=benchmark_bundle_path,
        ),
        "tab_foundry": {
            **_summary_metrics(tab_foundry_records),
            "run_dir": str(tab_foundry_run_dir.expanduser().resolve()),
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
    if math.isnan(float(tab_foundry_summary["final_roc_auc"])):
        raise RuntimeError("tab-foundry benchmark produced no ROC AUC values")
    if math.isnan(float(nanotabpfn_summary["final_roc_auc"])):
        raise RuntimeError("nanoTabPFN benchmark produced no ROC AUC values")
    return summary
