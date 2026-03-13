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

from tab_foundry.bench.artifacts import checkpoint_snapshots_from_history, write_json
from tab_foundry.export.checksums import sha256_file


NANOTABPFN_TASK_IDS: tuple[int, ...] = (
    363612,
    363613,
    363614,
    363615,
    363616,
    363618,
    363619,
    363620,
    363621,
    363623,
    363624,
    363625,
    363626,
    363627,
    363628,
    363629,
    363630,
    363631,
    363632,
    363671,
    363672,
    363673,
    363674,
    363675,
    363676,
    363677,
    363678,
    363679,
    363681,
    363682,
    363683,
    363684,
    363685,
    363686,
    363689,
    363691,
    363693,
    363694,
    363696,
    363697,
    363698,
    363699,
    363700,
    363702,
    363704,
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
)

BENCHMARK_INPUTS_FILENAME = "benchmark_inputs.json"
BENCHMARK_DATASET_CACHE_FILENAME = "benchmark_dataset_cache.npz"
DEFAULT_BENCHMARK_MAX_FEATURES = 10
DEFAULT_BENCHMARK_NEW_INSTANCES = 200
DEFAULT_BENCHMARK_TARGET_CLASSES_FILTER = 2

_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


@dataclass(frozen=True, slots=True)
class BenchmarkBundle:
    """Pinned benchmark bundle metadata and resolved dataset cache."""

    bundle_dir: Path
    benchmark_inputs_path: Path
    dataset_cache_path: Path
    datasets: dict[str, tuple[np.ndarray, np.ndarray]]
    benchmark_inputs: dict[str, Any]
    source: str


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


def _canonical_task_ids() -> list[int]:
    return [int(task_id) for task_id in NANOTABPFN_TASK_IDS]


def _benchmark_loader_config(
    *,
    max_features: int = DEFAULT_BENCHMARK_MAX_FEATURES,
    new_instances: int = DEFAULT_BENCHMARK_NEW_INSTANCES,
    target_classes_filter: int = DEFAULT_BENCHMARK_TARGET_CLASSES_FILTER,
) -> dict[str, int]:
    return {
        "max_features": int(max_features),
        "new_instances": int(new_instances),
        "target_classes_filter": int(target_classes_filter),
    }


def load_openml_benchmark_datasets(
    *,
    max_features: int = DEFAULT_BENCHMARK_MAX_FEATURES,
    new_instances: int = DEFAULT_BENCHMARK_NEW_INSTANCES,
    target_classes_filter: int = DEFAULT_BENCHMARK_TARGET_CLASSES_FILTER,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Load the nanoTabPFN OpenML benchmark suite."""

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    benchmark_tasks: list[dict[str, Any]] = []
    for task_id in NANOTABPFN_TASK_IDS:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        task_any: Any = task
        dataset = task_any.get_dataset(download_data=False)
        dataset_any: Any = dataset
        raw_qualities = dataset_any.qualities
        if not isinstance(raw_qualities, dict):
            continue
        number_of_features_raw = raw_qualities.get("NumberOfFeatures")
        number_of_classes_raw = raw_qualities.get("NumberOfClasses")
        missing_pct_raw = raw_qualities.get("PercentageOfInstancesWithMissingValues")
        minority_pct_raw = raw_qualities.get("MinorityClassPercentage")
        if not isinstance(number_of_features_raw, (int, float)):
            continue
        if not isinstance(number_of_classes_raw, (int, float)):
            continue
        if not isinstance(missing_pct_raw, (int, float)):
            continue
        if not isinstance(minority_pct_raw, (int, float)):
            continue
        number_of_features = float(number_of_features_raw)
        number_of_classes = float(number_of_classes_raw)
        missing_pct = float(missing_pct_raw)
        minority_pct = float(minority_pct_raw)
        if (
            number_of_features > max_features
            or number_of_classes > target_classes_filter
            or missing_pct > 0
            or minority_pct < 2.5
        ):
            continue

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

        datasets[str(dataset.name)] = (x, y.astype(np.int64, copy=False))
        benchmark_tasks.append(
            {
                "task_id": int(task_id),
                "dataset_name": str(dataset.name),
                "n_rows": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "n_classes": int(np.unique(y).size),
            }
        )
    if not datasets:
        raise RuntimeError("OpenML benchmark produced no datasets after filtering")
    return datasets, benchmark_tasks


def benchmark_bundle_paths(bundle_dir: Path) -> tuple[Path, Path]:
    """Return canonical metadata and cache paths for a benchmark bundle."""

    resolved_bundle_dir = bundle_dir.expanduser().resolve()
    return (
        resolved_bundle_dir / BENCHMARK_INPUTS_FILENAME,
        resolved_bundle_dir / BENCHMARK_DATASET_CACHE_FILENAME,
    )


def _load_benchmark_inputs(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"benchmark inputs payload must be a JSON object: {path}")
    return dict(payload)


def _validate_benchmark_inputs_payload(
    payload: Mapping[str, Any],
    *,
    benchmark_inputs_path: Path,
) -> dict[str, Any]:
    task_ids_raw = payload.get("task_ids")
    if not isinstance(task_ids_raw, list) or any(not isinstance(task_id, int) for task_id in task_ids_raw):
        raise RuntimeError(f"benchmark inputs missing canonical task ids: {benchmark_inputs_path}")
    task_ids = [int(task_id) for task_id in task_ids_raw]
    canonical_task_ids = _canonical_task_ids()
    if task_ids != canonical_task_ids:
        raise RuntimeError(f"benchmark task-id drift detected: {benchmark_inputs_path}")

    loader_raw = payload.get("loader")
    if not isinstance(loader_raw, dict):
        raise RuntimeError(f"benchmark inputs missing loader metadata: {benchmark_inputs_path}")
    expected_loader = _benchmark_loader_config()
    loader: dict[str, int] = {}
    for key, expected_value in expected_loader.items():
        value = loader_raw.get(key)
        if not isinstance(value, int):
            raise RuntimeError(f"benchmark inputs missing loader field {key!r}: {benchmark_inputs_path}")
        loader[key] = int(value)
    if loader != expected_loader:
        raise RuntimeError(f"benchmark loader drift detected: {benchmark_inputs_path}")

    benchmark_tasks_raw = payload.get("benchmark_tasks")
    if not isinstance(benchmark_tasks_raw, list) or any(not isinstance(task, dict) for task in benchmark_tasks_raw):
        raise RuntimeError(f"benchmark inputs missing task metadata: {benchmark_inputs_path}")
    benchmark_tasks = [dict(task) for task in benchmark_tasks_raw]

    benchmark_task_ids: list[int] = []
    dataset_names_from_tasks: list[str] = []
    for task in benchmark_tasks:
        task_id = task.get("task_id")
        dataset_name = task.get("dataset_name")
        if not isinstance(task_id, int) or not isinstance(dataset_name, str):
            raise RuntimeError(f"benchmark task metadata is invalid: {benchmark_inputs_path}")
        benchmark_task_ids.append(int(task_id))
        dataset_names_from_tasks.append(str(dataset_name))

    benchmark_task_id_set = set(benchmark_task_ids)
    expected_task_list = [task_id for task_id in canonical_task_ids if task_id in benchmark_task_id_set]
    if benchmark_task_ids != expected_task_list:
        raise RuntimeError(f"benchmark task-list drift detected: {benchmark_inputs_path}")

    task_count = payload.get("task_count")
    if not isinstance(task_count, int) or int(task_count) != len(benchmark_tasks):
        raise RuntimeError(f"benchmark task-count drift detected: {benchmark_inputs_path}")

    dataset_names_raw = payload.get("dataset_names")
    if not isinstance(dataset_names_raw, list) or any(not isinstance(name, str) for name in dataset_names_raw):
        raise RuntimeError(f"benchmark inputs missing dataset names: {benchmark_inputs_path}")
    dataset_names = [str(name) for name in dataset_names_raw]
    if dataset_names != dataset_names_from_tasks:
        raise RuntimeError(f"benchmark task-list drift detected: {benchmark_inputs_path}")

    dataset_cache_sha256 = payload.get("dataset_cache_sha256")
    if not isinstance(dataset_cache_sha256, str) or len(dataset_cache_sha256) != 64:
        raise RuntimeError(f"benchmark inputs missing dataset cache checksum: {benchmark_inputs_path}")

    return {
        "task_ids": task_ids,
        "loader": loader,
        "task_count": int(task_count),
        "benchmark_tasks": benchmark_tasks,
        "dataset_names": dataset_names,
        "dataset_cache_sha256": dataset_cache_sha256,
    }


def _build_benchmark_inputs_payload(
    *,
    benchmark_tasks: list[dict[str, Any]],
    dataset_names: list[str],
    dataset_cache_path: Path,
) -> dict[str, Any]:
    return {
        "task_ids": _canonical_task_ids(),
        "loader": _benchmark_loader_config(),
        "task_count": int(len(benchmark_tasks)),
        "benchmark_tasks": [dict(task) for task in benchmark_tasks],
        "dataset_names": [str(name) for name in dataset_names],
        "dataset_cache_sha256": sha256_file(dataset_cache_path),
    }


def _write_benchmark_bundle(
    *,
    bundle_dir: Path,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    benchmark_tasks: list[dict[str, Any]],
    source: str,
) -> BenchmarkBundle:
    resolved_bundle_dir = bundle_dir.expanduser().resolve()
    benchmark_inputs_path, dataset_cache_path = benchmark_bundle_paths(resolved_bundle_dir)
    save_dataset_cache(dataset_cache_path, datasets)
    benchmark_inputs = _validate_benchmark_inputs_payload(
        _build_benchmark_inputs_payload(
            benchmark_tasks=benchmark_tasks,
            dataset_names=list(datasets),
            dataset_cache_path=dataset_cache_path,
        ),
        benchmark_inputs_path=benchmark_inputs_path,
    )
    write_json(benchmark_inputs_path, benchmark_inputs)
    return BenchmarkBundle(
        bundle_dir=resolved_bundle_dir,
        benchmark_inputs_path=benchmark_inputs_path,
        dataset_cache_path=dataset_cache_path,
        datasets=datasets,
        benchmark_inputs=benchmark_inputs,
        source=source,
    )


def materialize_benchmark_bundle(bundle_dir: Path) -> BenchmarkBundle:
    """Always regenerate the benchmark bundle under the target directory."""

    datasets, benchmark_tasks = load_openml_benchmark_datasets(**_benchmark_loader_config())
    return _write_benchmark_bundle(
        bundle_dir=bundle_dir,
        datasets=datasets,
        benchmark_tasks=benchmark_tasks,
        source="created",
    )


def prepare_benchmark_bundle(bundle_dir: Path) -> BenchmarkBundle:
    """Create or reuse a pinned benchmark bundle with drift validation."""

    resolved_bundle_dir = bundle_dir.expanduser().resolve()
    benchmark_inputs_path, dataset_cache_path = benchmark_bundle_paths(resolved_bundle_dir)
    has_inputs = benchmark_inputs_path.exists()
    has_cache = dataset_cache_path.exists()
    if has_inputs != has_cache:
        raise RuntimeError(
            f"incomplete benchmark bundle at {resolved_bundle_dir}: expected both {benchmark_inputs_path.name} "
            f"and {dataset_cache_path.name}"
        )

    if has_inputs and has_cache:
        benchmark_inputs = _validate_benchmark_inputs_payload(
            _load_benchmark_inputs(benchmark_inputs_path),
            benchmark_inputs_path=benchmark_inputs_path,
        )
        actual_cache_sha256 = sha256_file(dataset_cache_path)
        if actual_cache_sha256 != benchmark_inputs["dataset_cache_sha256"]:
            raise RuntimeError(f"benchmark dataset cache checksum mismatch: {dataset_cache_path}")
        datasets = load_dataset_cache(dataset_cache_path)
        if list(datasets) != benchmark_inputs["dataset_names"]:
            raise RuntimeError(f"benchmark dataset cache metadata mismatch: {dataset_cache_path}")
        return BenchmarkBundle(
            bundle_dir=resolved_bundle_dir,
            benchmark_inputs_path=benchmark_inputs_path,
            dataset_cache_path=dataset_cache_path,
            datasets=datasets,
            benchmark_inputs=benchmark_inputs,
            source="reused",
        )

    return materialize_benchmark_bundle(resolved_bundle_dir)


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
    benchmark_inputs: Mapping[str, Any],
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
        "dataset_count": int(benchmark_inputs["task_count"]),
        "benchmark_inputs": {
            "task_ids": [int(task_id) for task_id in cast(list[int], benchmark_inputs["task_ids"])],
            "loader": dict(cast(dict[str, int], benchmark_inputs["loader"])),
            "task_count": int(benchmark_inputs["task_count"]),
            "benchmark_tasks": [dict(task) for task in cast(list[dict[str, Any]], benchmark_inputs["benchmark_tasks"])],
            "dataset_names": [str(name) for name in cast(list[str], benchmark_inputs["dataset_names"])],
            "bundle_dir": str(benchmark_inputs["bundle_dir"]),
            "cache_path": str(benchmark_inputs["cache_path"]),
            "dataset_cache_sha256": str(benchmark_inputs["dataset_cache_sha256"]),
            "source": str(benchmark_inputs["source"]),
        },
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
