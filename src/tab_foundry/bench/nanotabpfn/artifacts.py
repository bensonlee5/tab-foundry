"""Benchmark artifact and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tab_foundry.bench.artifacts import (
    checkpoint_snapshots_from_history,
    resolve_train_elapsed_seconds,
)

from .bundle import _CLASSIFICATION_TASK_TYPE
from .datasets import BenchmarkDatasetEvaluationError
from .metrics import (
    dataset_brier_score_metrics,
    dataset_log_loss_metrics,
    dataset_roc_auc_metrics,
    evaluate_classifier,
)


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
                        "elapsed_seconds": resolve_train_elapsed_seconds(
                            snapshot,
                            context=f"telemetry checkpoint step={snapshot['step']}",
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

    from tab_foundry.bench.checkpoint import TabFoundryClassifier

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
            else:
                raise RuntimeError(
                    "tab-foundry benchmark checkpoint evaluation is classification-only in this branch; "
                    f"got task_type={task_type!r}"
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
            continue
        model_arch = str(getattr(predictor.model_spec, "arch", "tabfoundry_staged")).strip().lower()
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
        curve_records.append(record)
    return curve_records
