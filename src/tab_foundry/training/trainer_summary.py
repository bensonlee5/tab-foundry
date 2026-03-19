"""Training summary payload helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def _trainer_summary_payload(
    *,
    output_dir: Path,
    optimizer_requested_name: str,
    optimizer_resolved_name: str,
    optimizer_fallback_reason: str | None,
    global_step: int,
    best_checkpoint: Path | None,
    latest_checkpoint: Path | None,
    best_val: float,
    best_val_step: float,
    final_train_loss: float | None,
    final_train_loss_ema: float | None,
    last_train_metrics: Mapping[str, float] | None,
    last_val_metrics: dict[str, float] | None,
    final_grad_norm: float,
    grad_norm_sum: float,
    grad_norm_count: int,
    max_grad_norm: float,
    train_elapsed_seconds: float,
    wall_elapsed_seconds: float,
    nan_skip_count: int = 0,
    error: BaseException | None = None,
) -> dict[str, Any]:
    def _summary_float(value: float | None) -> float | None:
        if value is None:
            return None
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None

    metrics_payload: dict[str, float | None] = {
        "best_val_loss": float(best_val),
        "best_val_step": float(best_val_step) if best_val_step > 0 else None,
        "final_train_loss": _summary_float(final_train_loss),
        "final_train_loss_ema": _summary_float(final_train_loss_ema),
        "final_val_loss": None
        if last_val_metrics is None
        else _summary_float(last_val_metrics.get("val_loss")),
        "final_grad_norm": float(final_grad_norm),
        "mean_grad_norm": float(grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else 0.0,
        "max_grad_norm": float(max_grad_norm),
        "train_elapsed_seconds": float(train_elapsed_seconds),
        "wall_elapsed_seconds": float(wall_elapsed_seconds),
        "nan_skip_count": float(nan_skip_count),
    }
    if last_train_metrics is not None:
        final_train_acc = _summary_float(last_train_metrics.get("acc"))
        if final_train_acc is not None:
            metrics_payload["final_train_acc"] = final_train_acc
        final_train_rmse = _summary_float(last_train_metrics.get("rmse"))
        if final_train_rmse is not None:
            metrics_payload["final_train_rmse"] = final_train_rmse

    summary: dict[str, Any] = {
        "optimizer": {
            "requested_name": optimizer_requested_name,
            "resolved_name": optimizer_resolved_name,
            "fallback_reason": optimizer_fallback_reason,
        },
        "run": {
            "output_dir": str(output_dir),
            "global_step": int(global_step),
            "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint.resolve()),
            "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint.resolve()),
        },
        "metrics": metrics_payload,
    }
    if error is not None:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    return summary
