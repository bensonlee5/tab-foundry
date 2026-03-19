"""Signal extraction helpers for benchmark bounce diagnosis."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from tab_foundry.bench.nanotabpfn import curve_summary


def shared_bundle_analysis(
    primary_records: list[dict[str, Any]],
    confirmation_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    primary_by_step = {int(record["step"]): record for record in primary_records}
    confirmation_by_step = (
        {}
        if confirmation_records is None
        else {int(record["step"]): record for record in confirmation_records}
    )
    shared_steps = sorted(set(primary_by_step) & set(confirmation_by_step))
    primary_summary = curve_summary(primary_records)
    confirmation_summary = (
        None if confirmation_records is None else curve_summary(confirmation_records)
    )
    best_step_changed = bool(
        int(primary_summary["checkpoint_count"]) > 0
        and confirmation_summary is not None
        and int(confirmation_summary["checkpoint_count"]) > 0
        and int(primary_summary["best_step"]) != int(confirmation_summary["best_step"])
    )
    likely_benchmark_noise = bool(
        best_step_changed
        and primary_summary["adjacent_ci_overlap_fraction"] is not None
        and float(primary_summary["adjacent_ci_overlap_fraction"]) >= 0.5
        and confirmation_summary is not None
        and int(confirmation_summary["task_count"]) > int(primary_summary["task_count"])
    )
    return {
        "shared_step_count": int(len(shared_steps)),
        "shared_steps": shared_steps,
        "best_step_changed_between_bundles": bool(best_step_changed),
        "primary": primary_summary,
        "confirmation": confirmation_summary,
        "likely_benchmark_noise": likely_benchmark_noise,
    }


def curve_summary_compat(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper around the shared checkpoint-curve summary helper."""

    return curve_summary(records)


def history_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def training_signal(
    *,
    history: list[dict[str, Any]],
    curve_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grad_norms = [
        float(record["grad_norm"])
        for record in history
        if record.get("grad_norm") is not None and math.isfinite(float(record["grad_norm"]))
    ]
    train_losses = [
        float(record["train_loss"])
        for record in history
        if record.get("train_loss") is not None and math.isfinite(float(record["train_loss"]))
    ]
    median_grad_norm = float(np.median(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else 0.0
    max_grad_norm = float(max(grad_norms)) if grad_norms else 0.0
    grad_spike_ratio = (
        float(max_grad_norm / median_grad_norm)
        if grad_norms and median_grad_norm > 0.0
        else float("inf") if max_grad_norm > 0.0 else 0.0
    )
    sorted_records = sorted(curve_records, key=lambda record: int(record["step"]))
    worst_drop: dict[str, Any] | None = None
    for previous, current in zip(sorted_records, sorted_records[1:], strict=False):
        delta = float(current["roc_auc"]) - float(previous["roc_auc"])
        if worst_drop is None or delta < float(worst_drop["roc_auc_delta"]):
            window = [
                record
                for record in history
                if int(previous["step"]) < int(record["step"]) <= int(current["step"])
            ]
            window_grad_norms = [
                float(record["grad_norm"])
                for record in window
                if record.get("grad_norm") is not None and math.isfinite(float(record["grad_norm"]))
            ]
            window_losses = [
                float(record["train_loss"])
                for record in window
                if record.get("train_loss") is not None and math.isfinite(float(record["train_loss"]))
            ]
            worst_drop = {
                "from_step": int(previous["step"]),
                "to_step": int(current["step"]),
                "roc_auc_delta": float(delta),
                "window_max_grad_norm": None if not window_grad_norms else float(max(window_grad_norms)),
                "window_train_loss_var": history_variance(window_losses),
            }
    likely_optimization_instability = bool(
        max_grad_norm >= 50.0
        or (
            worst_drop is not None
            and worst_drop["window_max_grad_norm"] is not None
            and float(worst_drop["window_max_grad_norm"]) >= max(50.0, 10.0 * median_grad_norm)
            and float(worst_drop["roc_auc_delta"]) < -0.02
        )
    )
    return {
        "history_step_count": int(len(history)),
        "median_grad_norm": float(median_grad_norm),
        "max_grad_norm": float(max_grad_norm),
        "grad_spike_ratio": float(grad_spike_ratio),
        "train_loss_variance": history_variance(train_losses),
        "worst_checkpoint_drop": worst_drop,
        "likely_optimization_instability": likely_optimization_instability,
    }


def task_tradeoff_signal(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "positive_task_count": 0,
            "negative_task_count": 0,
            "top_quartile_abs_delta_share": 0.0,
            "likely_heterogeneous_task_tradeoff": False,
        }
    sorted_records = sorted(records, key=lambda record: int(record["step"]))
    best_record = max(sorted_records, key=lambda record: float(record["roc_auc"]))
    final_record = sorted_records[-1]
    best_dataset = cast(dict[str, float], best_record.get("dataset_roc_auc", {}))
    final_dataset = cast(dict[str, float], final_record.get("dataset_roc_auc", {}))
    shared_dataset_names = sorted(set(best_dataset) & set(final_dataset))
    deltas = {
        dataset_name: float(final_dataset[dataset_name]) - float(best_dataset[dataset_name])
        for dataset_name in shared_dataset_names
    }
    positive_count = sum(1 for value in deltas.values() if value > 0)
    negative_count = sum(1 for value in deltas.values() if value < 0)
    abs_deltas = sorted((abs(value) for value in deltas.values()), reverse=True)
    total_abs_delta = float(sum(abs_deltas))
    top_count = max(1, math.ceil(len(abs_deltas) / 4.0)) if abs_deltas else 0
    top_share = (
        float(sum(abs_deltas[:top_count]) / total_abs_delta)
        if total_abs_delta > 0.0 and top_count > 0
        else 0.0
    )
    likely_tradeoff = bool(positive_count > 0 and negative_count > 0 and top_share >= 0.5)
    return {
        "positive_task_count": int(positive_count),
        "negative_task_count": int(negative_count),
        "top_quartile_abs_delta_share": float(top_share),
        "likely_heterogeneous_task_tradeoff": likely_tradeoff,
    }


def checkpoint_aliasing_signal(
    *,
    coarse_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not dense_records:
        return {
            "available": False,
            "likely_checkpoint_aliasing": False,
        }
    coarse_summary = curve_summary(coarse_records)
    dense_summary = curve_summary(dense_records)
    coarse_steps = {int(record["step"]) for record in coarse_records}
    dense_best_step = int(dense_summary["best_step"])
    likely_aliasing = bool(
        dense_best_step not in coarse_steps
        and float(dense_summary["best_roc_auc"]) > float(coarse_summary["best_roc_auc"])
    )
    dense_intervals = [
        int(current["step"]) - int(previous["step"])
        for previous, current in zip(
            sorted(dense_records, key=lambda record: int(record["step"])),
            sorted(dense_records, key=lambda record: int(record["step"]))[1:],
            strict=False,
        )
    ]
    return {
        "available": True,
        "coarse": coarse_summary,
        "dense": dense_summary,
        "dense_checkpoint_interval": None if not dense_intervals else min(dense_intervals),
        "likely_checkpoint_aliasing": likely_aliasing,
    }


def classify_causes(
    *,
    bundle_analysis: dict[str, Any],
    training_signal: dict[str, Any],
    task_tradeoff_signal: dict[str, Any],
    checkpoint_aliasing_signal: dict[str, Any],
    evaluation_failures: dict[str, Any],
) -> dict[str, Any]:
    primary_causes: list[str] = []
    evidence: list[str] = []
    if int(evaluation_failures["failure_count"]) > 0:
        primary_causes.append("checkpoint_evaluation_failure")
        evidence.append(
            "One or more checkpoints could not be benchmarked cleanly on the selected bundle, "
            "which is itself diagnostic evidence rather than a plotting artifact."
        )
    if bool(bundle_analysis["likely_benchmark_noise"]):
        primary_causes.append("benchmark_noise")
        evidence.append(
            "The best checkpoint changes between the primary and confirmation bundles while adjacent "
            "primary-bundle confidence intervals overlap heavily."
        )
    if bool(checkpoint_aliasing_signal.get("likely_checkpoint_aliasing")):
        primary_causes.append("checkpoint_aliasing")
        evidence.append(
            "The denser checkpoint run finds a better checkpoint that is not present in the coarse "
            "25-step snapshot grid."
        )
    if bool(training_signal["likely_optimization_instability"]):
        primary_causes.append("optimization_instability")
        evidence.append(
            "Gradient norms spike sharply relative to the run median or the worst ROC AUC drop aligns "
            "with a high-gradient interval."
        )
    if bool(task_tradeoff_signal["likely_heterogeneous_task_tradeoff"]):
        primary_causes.append("heterogeneous_task_tradeoff")
        evidence.append(
            "A minority of datasets account for most of the best-to-final ROC AUC change while other "
            "datasets move in the opposite direction."
        )
    if not primary_causes:
        primary_causes.append("unclear")
        evidence.append(
            "The current traces do not isolate a single dominant cause; the next step is more repeated "
            "measurement rather than an optimizer change."
        )
    return {
        "primary_causes": primary_causes,
        "evidence": evidence,
    }
