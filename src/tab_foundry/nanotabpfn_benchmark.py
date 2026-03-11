"""Notebook-style nanoTabPFN comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import openml
from openml.tasks import TaskType
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder


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

_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


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


def load_openml_benchmark_datasets(
    *,
    max_features: int = 10,
    new_instances: int = 200,
    target_classes_filter: int = 2,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Load the nanoTabPFN OpenML benchmark suite."""

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    benchmark_tasks: list[dict[str, Any]] = []
    for task_id in NANOTABPFN_TASK_IDS:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        dataset = task.get_dataset(download_data=False)
        qualities = dataset.qualities
        if (
            qualities["NumberOfFeatures"] > max_features
            or qualities["NumberOfClasses"] > target_classes_filter
            or qualities["PercentageOfInstancesWithMissingValues"] > 0
            or qualities["MinorityClassPercentage"] < 2.5
        ):
            continue

        x_frame, y_raw, _categorical_indicator, _attribute_names = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )
        if new_instances < len(y_raw):
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


def _step_time_map_from_history(history_path: Path) -> dict[int, float]:
    step_times: dict[int, float] = {}
    for record in load_jsonl(history_path):
        raw_time = record.get("train_elapsed_seconds", record.get("elapsed_seconds", 0.0))
        step_times[int(record["step"])] = max(0.0, float(raw_time))
    return step_times


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

    step_times = _step_time_map_from_history(history_path)
    snapshots: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt")):
        try:
            step = int(checkpoint.stem.removeprefix("step_"))
        except ValueError as exc:
            raise RuntimeError(f"invalid step checkpoint name: {checkpoint.name}") from exc
        if step not in step_times:
            raise RuntimeError(f"missing elapsed time for checkpoint step={step}")
        snapshots.append(
            {
                "step": step,
                "path": str(checkpoint.resolve()),
                "elapsed_seconds": float(step_times[step]),
            }
        )
    if not snapshots:
        raise RuntimeError(f"no step checkpoints found under {checkpoint_dir}")
    return snapshots


def evaluate_tab_foundry_run(
    run_dir: Path,
    *,
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate smoke-run checkpoints on the notebook benchmark suite."""

    from tab_foundry.checkpoint_classifier import TabFoundryClassifier

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
    tab_foundry_run_dir: Path,
    nanotabpfn_root: Path,
    nanotabpfn_python: Path,
) -> dict[str, Any]:
    """Build a compact JSON summary for the benchmark comparison."""

    def _final_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        times, mean, _std = aggregate_curve(records)
        if times.size == 0 or mean.size == 0:
            return {"final_training_time": 0.0, "final_roc_auc": float("nan")}
        return {
            "final_training_time": float(times[-1]),
            "final_roc_auc": float(mean[-1]),
        }

    summary = {
        "dataset_count": int(len(benchmark_tasks)),
        "tab_foundry": {
            **_final_metrics(tab_foundry_records),
            "run_dir": str(tab_foundry_run_dir.expanduser().resolve()),
        },
        "nanotabpfn": {
            **_final_metrics(nanotabpfn_records),
            "root": str(nanotabpfn_root.expanduser().resolve()),
            "python": str(nanotabpfn_python.expanduser().resolve()),
            "num_seeds": int(len({int(record["seed"]) for record in nanotabpfn_records})),
        },
    }
    if math.isnan(float(summary["tab_foundry"]["final_roc_auc"])):
        raise RuntimeError("tab-foundry benchmark produced no ROC AUC values")
    if math.isnan(float(summary["nanotabpfn"]["final_roc_auc"])):
        raise RuntimeError("nanoTabPFN benchmark produced no ROC AUC values")
    return summary
