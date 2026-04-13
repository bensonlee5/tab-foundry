"""Benchmark artifact and checkpoint helpers."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from tab_foundry.bench.artifacts import (
    checkpoint_snapshots_from_history,
    load_history,
    resolve_train_elapsed_seconds,
)
from tab_foundry.device import resolve_device
from tab_foundry.training.checkpoint_paths import resolve_latest_checkpoint_path

from .bundle import _CLASSIFICATION_TASK_TYPE
from .dataset_common import BenchmarkDataset, BenchmarkDatasetEvaluationError
from .metrics import (
    dataset_bpc_metrics,
    dataset_bpf_metrics,
    dataset_brier_score_metrics,
    dataset_log_loss_metrics,
    dataset_roc_auc_metrics,
    evaluate_classifier,
)


def benchmark_host_fingerprint() -> str:
    """Return a stable same-machine fingerprint for benchmark timing reuse."""

    parts = [
        platform.node().strip().lower(),
        platform.system().strip().lower(),
        platform.machine().strip().lower(),
    ]
    normalized = [part if part else "unknown" for part in parts]
    return "|".join(normalized)


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


def _resolve_telemetry_checkpoint_path(*, checkpoint_path: str, checkpoint_dir: Path) -> Path | None:
    recorded_path = Path(str(checkpoint_path)).expanduser()
    if recorded_path.exists():
        return recorded_path.resolve()
    candidate = checkpoint_dir / recorded_path.name
    if candidate.exists():
        return candidate.resolve()
    return None


def _history_step_elapsed_seconds(history_path: Path) -> dict[int, float]:
    return {
        int(record["step"]): resolve_train_elapsed_seconds(
            record,
            context=f"history step={record['step']}",
        )
        for record in load_history(history_path)
    }


def collect_checkpoint_snapshots(run_dir: Path) -> list[dict[str, Any]]:
    """Resolve step checkpoints and their elapsed training times."""

    resolved_run_dir = run_dir.expanduser().resolve()
    history_path, checkpoint_dir = resolve_tab_foundry_run_artifact_paths(resolved_run_dir)
    telemetry_path = resolved_run_dir / "telemetry.json"
    if telemetry_path.exists():
        payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
        snapshots = payload.get("checkpoint_snapshots")
        if isinstance(snapshots, list) and snapshots:
            resolved_snapshots: list[dict[str, Any]] = []
            elapsed_seconds_by_step: dict[int, float] = {}
            for snapshot in snapshots:
                step = int(snapshot["step"])
                elapsed_seconds = resolve_train_elapsed_seconds(
                    snapshot,
                    context=f"telemetry checkpoint step={snapshot['step']}",
                )
                elapsed_seconds_by_step[step] = elapsed_seconds
                resolved_checkpoint_path = _resolve_telemetry_checkpoint_path(
                    checkpoint_path=str(snapshot["path"]),
                    checkpoint_dir=checkpoint_dir,
                )
                if resolved_checkpoint_path is None:
                    continue
                resolved_snapshots.append(
                    {
                        "step": step,
                        "path": str(resolved_checkpoint_path),
                        "elapsed_seconds": elapsed_seconds,
                    }
                )

            latest_checkpoint_path = resolve_latest_checkpoint_path(resolved_run_dir)
            if latest_checkpoint_path is not None:
                resolved_latest_checkpoint_path = latest_checkpoint_path.expanduser().resolve()
                latest_checkpoint_step = _checkpoint_global_step(resolved_latest_checkpoint_path)
                highest_retained_step = max(
                    (int(snapshot["step"]) for snapshot in resolved_snapshots),
                    default=0,
                )
                if (
                    latest_checkpoint_step is not None
                    and str(resolved_latest_checkpoint_path)
                    not in {str(snapshot["path"]) for snapshot in resolved_snapshots}
                    and int(latest_checkpoint_step) > int(highest_retained_step)
                ):
                    latest_elapsed_seconds = elapsed_seconds_by_step.get(int(latest_checkpoint_step))
                    if latest_elapsed_seconds is None:
                        latest_elapsed_seconds = _history_step_elapsed_seconds(history_path).get(
                            int(latest_checkpoint_step)
                        )
                    if latest_elapsed_seconds is None:
                        raise RuntimeError(
                            "missing elapsed time for terminal latest checkpoint "
                            f"step={latest_checkpoint_step}"
                        )
                    resolved_snapshots.append(
                        {
                            "step": int(latest_checkpoint_step),
                            "path": str(resolved_latest_checkpoint_path),
                            "elapsed_seconds": float(latest_elapsed_seconds),
                        }
                    )

            if resolved_snapshots:
                return sorted(
                    resolved_snapshots,
                    key=lambda snapshot: int(snapshot["step"]),
                )

    snapshots = checkpoint_snapshots_from_history(history_path, checkpoint_dir)
    return [
        {
            "step": int(snapshot["step"]),
            "path": str(snapshot["path"]),
            "elapsed_seconds": float(snapshot["train_elapsed_seconds"]),
        }
        for snapshot in snapshots
    ]


def _resolve_checkpoint_selection(checkpoint_selection: str) -> str:
    normalized = str(checkpoint_selection).strip().lower()
    if normalized in {"", "all"}:
        return "all"
    if normalized == "best_and_final":
        return normalized
    raise RuntimeError(
        "checkpoint_selection must be one of ['all', 'best_and_final'], "
        f"got {checkpoint_selection!r}"
    )


def _checkpoint_global_step(checkpoint_path: Path) -> int | None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    raw_global_step = payload.get("global_step")
    if raw_global_step is None:
        return None
    return int(raw_global_step)


def selected_checkpoint_snapshots(
    run_dir: Path,
    *,
    checkpoint_selection: str = "all",
) -> list[dict[str, Any]]:
    snapshots = collect_checkpoint_snapshots(run_dir)
    selection = _resolve_checkpoint_selection(checkpoint_selection)
    if selection == "all":
        return snapshots

    final_snapshot = snapshots[-1]
    snapshots_by_step = {int(snapshot["step"]): snapshot for snapshot in snapshots}
    final_checkpoint_path = resolve_latest_checkpoint_path(run_dir)
    if final_checkpoint_path is None:
        final_checkpoint_path = Path(str(final_snapshot["path"]))
    try:
        best_checkpoint_path = resolve_tab_foundry_best_checkpoint(run_dir)
    except RuntimeError:
        best_checkpoint_path = final_checkpoint_path
    candidates = [
        best_checkpoint_path,
        final_checkpoint_path,
    ]
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    fallback_elapsed_seconds = float(final_snapshot["elapsed_seconds"])
    fallback_step = int(final_snapshot["step"])

    for checkpoint_path in candidates:
        resolved_checkpoint_path = checkpoint_path.expanduser().resolve()
        resolved_path_str = str(resolved_checkpoint_path)
        if resolved_path_str in seen_paths:
            continue
        checkpoint_step = _checkpoint_global_step(resolved_checkpoint_path)
        base_snapshot = (
            None
            if checkpoint_step is None
            else snapshots_by_step.get(int(checkpoint_step))
        )
        if base_snapshot is None and checkpoint_step is None:
            selected_step = fallback_step
        elif base_snapshot is not None:
            selected_step = int(base_snapshot["step"])
        else:
            if checkpoint_step is None:
                raise RuntimeError(
                    "checkpoint selection resolved without a fallback step or checkpoint step"
                )
            selected_step = int(checkpoint_step)
        selected.append(
            {
                "step": selected_step,
                "path": resolved_path_str,
                "elapsed_seconds": (
                    fallback_elapsed_seconds
                    if base_snapshot is None
                    else float(base_snapshot["elapsed_seconds"])
                ),
            }
        )
        seen_paths.add(resolved_path_str)
    return selected


def evaluate_tab_foundry_run(
    run_dir: Path,
    *,
    datasets: Mapping[str, BenchmarkDataset],
    task_type: str,
    device: str,
    allow_checkpoint_failures: bool = False,
    allow_missing_values: bool = False,
    checkpoint_selection: str = "all",
) -> list[dict[str, Any]]:
    """Evaluate smoke-run checkpoints on the notebook benchmark suite."""

    from tab_foundry.bench.checkpoint import TabFoundryClassifier

    resolved_device = resolve_device(device)
    curve_records: list[dict[str, Any]] = []
    for snapshot in selected_checkpoint_snapshots(
        run_dir,
        checkpoint_selection=checkpoint_selection,
    ):
        checkpoint_path = Path(str(snapshot["path"]))
        benchmark_started = time.perf_counter()
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
                    "benchmark_elapsed_seconds": float(time.perf_counter() - benchmark_started),
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
            "benchmark_elapsed_seconds": float(time.perf_counter() - benchmark_started),
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
        if "BPC" in metrics:
            record["bpc"] = float(metrics["BPC"])
            record["dataset_bpc"] = dataset_bpc_metrics(metrics)
        if "BPF" in metrics:
            record["bpf"] = float(metrics["BPF"])
            record["dataset_bpf"] = dataset_bpf_metrics(metrics)
        curve_records.append(record)
    return curve_records
