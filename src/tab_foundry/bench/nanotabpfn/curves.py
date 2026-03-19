"""Checkpoint-curve summarization helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED = 0


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
    if any(record.get("log_loss") is not None for record in records):
        return ("log_loss", "min")
    if any(record.get("crps") is not None for record in records):
        return ("crps", "min")
    if any(record.get("roc_auc") is not None for record in records):
        return ("roc_auc", "max")
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
