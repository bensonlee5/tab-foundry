"""Comparison-summary and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from .bundle import (
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
    benchmark_bundle_summary,
)
from .curves import (
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
    _curve_ranking_metric,
    summarize_checkpoint_curve,
)


def aggregate_curve(
    records: list[dict[str, Any]],
    *,
    value_key: str = "roc_auc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    tabiclv2_records: list[dict[str, Any]] | None = None,
    task_type: str,
    out_path: Path,
) -> Path:
    """Render the tab-foundry time-vs-quality curve with optional external baselines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_key = "log_loss" if task_type == _CLASSIFICATION_TASK_TYPE else "crps"
    ylabel = "mean log loss" if task_type == _CLASSIFICATION_TASK_TYPE else "mean CRPS"
    external_labels: list[str] = []
    if nanotabpfn_records:
        external_labels.append("nanoTabPFN")
    if tabiclv2_records:
        external_labels.append("TabICLv2")
    title = "tab-foundry benchmark" if not external_labels else "tab-foundry vs external baselines"
    tab_times, tab_mean, _tab_std = aggregate_curve(tab_foundry_records, value_key=metric_key)
    nano_times, nano_mean, nano_std = aggregate_curve(nanotabpfn_records, value_key=metric_key)
    tabicl_times, tabicl_mean, tabicl_std = aggregate_curve(
        [] if tabiclv2_records is None else tabiclv2_records,
        value_key=metric_key,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if tab_times.size > 0:
        ax.plot(tab_times, tab_mean, label="tab-foundry", color="#1f77b4", linewidth=2.0)
    if nano_times.size > 0:
        ax.plot(nano_times, nano_mean, label="nanoTabPFN", color="#d62728", linewidth=2.0)
        if np.any(nano_std > 0):
            ax.fill_between(nano_times, nano_mean - nano_std, nano_mean + nano_std, alpha=0.2, color="#d62728")
    if tabicl_times.size > 0:
        ax.plot(tabicl_times, tabicl_mean, label="TabICLv2", color="#2ca02c", linewidth=2.0)
        if np.any(tabicl_std > 0):
            ax.fill_between(
                tabicl_times,
                tabicl_mean - tabicl_std,
                tabicl_mean + tabicl_std,
                alpha=0.2,
                color="#2ca02c",
            )
    ax.set_xlabel("training time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if tab_times.size > 0 or nano_times.size > 0 or tabicl_times.size > 0:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    return out_path


def _ensure_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
    tabiclv2_records: list[dict[str, Any]] | None = None,
    benchmark_tasks: list[dict[str, Any]],
    benchmark_bundle: Mapping[str, Any],
    benchmark_bundle_path: Path,
    tab_foundry_run_dir: Path,
    task_type: str,
    nanotabpfn_root: Path | None = None,
    nanotabpfn_python: Path | None = None,
    tabiclv2_root: Path | None = None,
    tabiclv2_python: Path | None = None,
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
    def _external_baseline_summary(
        records: list[dict[str, Any]],
        *,
        root: Path | None,
        python: Path | None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        baseline_summary = {
            **_summary_metrics(records),
            "root": None if root is None else str(root.expanduser().resolve()),
            "python": None if python is None else str(python.expanduser().resolve()),
        }
        if extra_fields is not None:
            baseline_summary.update(dict(extra_fields))
        best_step_raw = baseline_summary.get("best_step")
        final_step_raw = baseline_summary.get("final_step")
        best_step = 0.0 if best_step_raw is None else float(best_step_raw)
        final_step = 0.0 if final_step_raw is None else float(final_step_raw)
        best_record = next(
            (record for record in records if float(record.get("step", -1.0)) == best_step),
            None,
        )
        final_record = next(
            (record for record in records if float(record.get("step", -1.0)) == final_step),
            None,
        )
        _apply_metric_summaries(
            baseline_summary,
            records,
            best_step=best_step,
            final_step=final_step,
            best_record=best_record,
            final_record=final_record,
        )
        return baseline_summary

    if nanotabpfn_records:
        summary["nanotabpfn"] = _external_baseline_summary(
            nanotabpfn_records,
            root=nanotabpfn_root,
            python=nanotabpfn_python,
            extra_fields={"num_seeds": int(len({int(record["seed"]) for record in nanotabpfn_records}))},
        )
    if tabiclv2_records:
        summary["tabiclv2"] = _external_baseline_summary(
            tabiclv2_records,
            root=tabiclv2_root,
            python=tabiclv2_python,
        )
    if task_type == _CLASSIFICATION_TASK_TYPE:
        if tab_foundry_summary.get("final_log_loss") is None:
            raise RuntimeError("tab-foundry benchmark produced no log-loss values")
        if nanotabpfn_records and cast(dict[str, Any], summary["nanotabpfn"]).get("final_log_loss") is None:
            raise RuntimeError("nanoTabPFN benchmark produced no log-loss values")
        if tabiclv2_records and cast(dict[str, Any], summary["tabiclv2"]).get("final_log_loss") is None:
            raise RuntimeError("TabICLv2 benchmark produced no log-loss values")
    elif task_type == _REGRESSION_TASK_TYPE:
        if tab_foundry_summary.get("final_crps") is None:
            raise RuntimeError("tab-foundry benchmark produced no CRPS values")
        if nanotabpfn_records and cast(dict[str, Any], summary["nanotabpfn"]).get("final_crps") is None:
            raise RuntimeError("nanoTabPFN benchmark produced no CRPS values")
        if tabiclv2_records and cast(dict[str, Any], summary["tabiclv2"]).get("final_crps") is None:
            raise RuntimeError("TabICLv2 benchmark produced no CRPS values")
    else:
        raise RuntimeError(f"unsupported benchmark task_type: {task_type!r}")
    if control_baseline is not None:
        summary["control_baseline"] = json.loads(json.dumps(control_baseline, sort_keys=True))
    return summary
