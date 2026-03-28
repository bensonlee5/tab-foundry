"""Finalization and telemetry helpers for trainer runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig

from tab_foundry.types import TrainResult

from .instability import (
    build_regime_budget_summary,
    build_runtime_summary,
    build_training_telemetry,
    grad_norm_summary_from_running_totals,
    peak_device_memory_summary,
    write_training_telemetry,
)
from .trainer_loop import TrainingLoopState
from .trainer_summary import _training_telemetry_summary_payload, _trainer_summary_payload
from .wandb import update_wandb_summary, wandb_identity_payload


def finalize_training_run(
    *,
    accelerator: Any,
    output_dir: Path,
    task: str,
    cfg: DictConfig,
    run: Any,
    training_surface_payload: dict[str, Any] | None,
    telemetry_output_path: Path,
    artifacts: Mapping[str, Any],
    optimizer_requested_name: str,
    optimizer_resolved_name: str,
    optimizer_fallback_reason: str | None,
    state: TrainingLoopState,
    train_start: float,
    success: bool,
    error: Exception | None = None,
) -> tuple[TrainResult | None, float, Mapping[str, Any] | None]:
    wall_elapsed_seconds = time.perf_counter() - train_start
    task_batching_summary: Mapping[str, Any] | None = None
    if accelerator.is_main_process:
        runtime_summary = build_runtime_summary(
            train_elapsed_seconds=state.train_elapsed_seconds,
            wall_elapsed_seconds=wall_elapsed_seconds,
            examples_seen=state.examples_seen,
            tokens_seen=state.tokens_seen,
            peak_memory_summary=peak_device_memory_summary(accelerator.device),
        )
        regime_budget = build_regime_budget_summary(
            task=task,
            training_surface_record=training_surface_payload,
            global_step=state.global_step,
            tokens_seen=state.tokens_seen,
        )
        telemetry_payload = build_training_telemetry(
            run_dir=output_dir,
            task=task,
            global_step=state.global_step,
            success=success,
            artifacts=dict(artifacts),
            checkpoint_snapshots=state.checkpoint_snapshots,
            history_records=state.history_records,
            gradient_records=state.gradient_records,
            runtime_summary=runtime_summary,
            regime_budget=regime_budget,
            training_surface_record=training_surface_payload,
            wandb=wandb_identity_payload(run, cfg=cfg),
            error=error,
        )
        raw_diagnostics = telemetry_payload.get("diagnostics")
        if isinstance(raw_diagnostics, Mapping):
            raw_task_batching = raw_diagnostics.get("task_batching")
            if isinstance(raw_task_batching, Mapping):
                task_batching_summary = raw_task_batching
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        update_wandb_summary(
            run,
            _training_telemetry_summary_payload(telemetry_payload=telemetry_payload),
        )
    update_wandb_summary(
        run,
        _trainer_summary_payload(
            output_dir=output_dir,
            optimizer_requested_name=optimizer_requested_name,
            optimizer_resolved_name=optimizer_resolved_name,
            optimizer_fallback_reason=optimizer_fallback_reason,
            global_step=state.global_step,
            best_checkpoint=state.best_checkpoint,
            latest_checkpoint=state.latest_checkpoint,
            best_val=state.best_val,
            best_val_step=state.best_val_step,
            final_train_loss=state.final_train_loss,
            final_train_loss_ema=state.final_train_loss_ema,
            last_train_metrics=state.last_train_metrics,
            last_val_metrics=state.last_val_metrics,
            final_grad_norm=state.final_grad_norm,
            grad_norm_sum=state.grad_norm_sum,
            grad_norm_count=state.grad_norm_count,
            max_grad_norm=state.max_grad_norm,
            train_elapsed_seconds=state.train_elapsed_seconds,
            wall_elapsed_seconds=wall_elapsed_seconds,
            nan_skip_count=state.nan_skip_count,
            task_batching=task_batching_summary,
            error=error,
        ),
    )
    if not success:
        return None, wall_elapsed_seconds, task_batching_summary
    return (
        TrainResult(
            output_dir=output_dir,
            best_checkpoint=state.best_checkpoint,
            latest_checkpoint=state.latest_checkpoint,
            global_step=state.global_step,
            metrics={
                "best_val_loss": float(state.best_val),
                "best_val_step": float(state.best_val_step),
                "final_val_loss": float(state.last_val_metrics["val_loss"])
                if state.last_val_metrics is not None
                else float(state.best_val),
                "train_elapsed_seconds": float(state.train_elapsed_seconds),
                "wall_elapsed_seconds": float(wall_elapsed_seconds),
                "nan_skip_count": float(state.nan_skip_count),
                **grad_norm_summary_from_running_totals(
                    grad_norm_sum=state.grad_norm_sum,
                    grad_norm_count=state.grad_norm_count,
                    max_grad_norm=state.max_grad_norm,
                    final_grad_norm=state.final_grad_norm,
                ),
            },
        ),
        wall_elapsed_seconds,
        task_batching_summary,
    )
