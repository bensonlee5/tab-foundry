"""Trainer step-loop execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, cast

import torch
from omegaconf import DictConfig

from tab_foundry.task_batching import move_batch, task_batch_diagnostics
from tab_foundry.types import TaskBatch

from .artifacts import (
    append_history_record,
    append_jsonl_record,
    gradient_history_record,
    history_record,
    save_eval_mode_checkpoint,
)
from .checkpoint_paths import canonical_latest_checkpoint_path, stage_latest_checkpoint_path
from .distributed import _reduction_float_dtype, _reduce_any_flag, _reduce_keyed_weighted_scalars
from .instability import (
    module_grad_norms,
    normalize_grad_norm_value,
    task_batch_examples_seen,
    task_batch_token_count,
    total_grad_norm,
    train_loss_delta,
    update_loss_ema,
)
from .schedule import stage_base_lr
from .trainer_guards import (
    GuardStepUpdate,
    _global_grad_norm_kind,
    _merge_activation_norms,
    _reduce_global_grad_norm_kind,
    handle_non_finite_grad,
    handle_non_finite_loss,
)
from .trainer_metrics import _compute_loss_and_metrics, _evaluate_loader, cycle_loader
from .trainer_optimizer import _set_optimizer_base_lr, _set_optimizer_training_mode
from .wandb import log_wandb_metrics


@dataclass(slots=True)
class TrainingLoopState:
    global_step: int = 0
    best_checkpoint: Path | None = None
    latest_checkpoint: Path | None = None
    best_val: float = float("inf")
    best_val_step: float = 0.0
    last_val_metrics: dict[str, float] | None = None
    train_elapsed_seconds: float = 0.0
    stop_requested: bool = False
    grad_norm_sum: float = 0.0
    grad_norm_count: int = 0
    max_grad_norm: float = 0.0
    final_grad_norm: float = 0.0
    nan_skip_count: int = 0
    previous_train_loss: float | None = None
    loss_ema: float | None = None
    final_train_loss: float | None = None
    final_train_loss_ema: float | None = None
    last_train_metrics: dict[str, float] | None = None
    history_records: list[dict[str, Any]] = field(default_factory=list)
    gradient_records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_snapshots: list[dict[str, Any]] = field(default_factory=list)
    examples_seen: int = 0
    tokens_seen: int = 0


def _empty_task_batch_step_payload(*, requested_task_batch_size: int) -> dict[str, Any]:
    return {
        "task_batch_size_requested": int(requested_task_batch_size),
        "task_batch_size_actual": 0,
        "task_batch_batched_count": 0,
        "task_batch_singleton_fallback_count": 0,
        "task_batch_singleton_fallback_fraction": 0.0,
        "task_batch_signature_counts": {},
    }


def _task_batch_microstep_payload(batch: TaskBatch) -> dict[str, Any]:
    diagnostics = task_batch_diagnostics(batch)
    requested = int(diagnostics["task_batch_size_requested"])
    actual = int(diagnostics["task_batch_size_actual"])
    is_fallback = requested > 1 and actual == 1
    return {
        "task_batch_size_requested": requested,
        "task_batch_size_actual": actual,
        "task_batch_batched_count": 1 if actual > 1 else 0,
        "task_batch_singleton_fallback_count": 1 if is_fallback else 0,
        "task_batch_singleton_fallback_fraction": 1.0 if is_fallback else 0.0,
        "task_batch_signature_counts": {str(diagnostics["task_batch_signature"]): 1},
    }


def _accumulate_task_batch_step_payload(
    step_payload: dict[str, Any],
    *,
    microstep_payload: Mapping[str, Any],
) -> None:
    requested = int(microstep_payload["task_batch_size_requested"])
    expected_requested = int(step_payload["task_batch_size_requested"])
    if int(step_payload["task_batch_size_actual"]) == 0:
        step_payload["task_batch_size_requested"] = requested
    elif requested != expected_requested:
        raise RuntimeError(
            "task-batch diagnostics changed requested task batch size within one optimizer step: "
            f"expected {expected_requested}, got {requested}"
        )

    actual = int(microstep_payload["task_batch_size_actual"])
    batched_count = int(step_payload["task_batch_batched_count"]) + int(
        microstep_payload["task_batch_batched_count"]
    )
    fallback_count = int(step_payload["task_batch_singleton_fallback_count"]) + int(
        microstep_payload["task_batch_singleton_fallback_count"]
    )
    step_payload["task_batch_size_actual"] = int(step_payload["task_batch_size_actual"]) + actual
    step_payload["task_batch_batched_count"] = batched_count
    step_payload["task_batch_singleton_fallback_count"] = fallback_count
    task_batch_microsteps = batched_count + fallback_count
    step_payload["task_batch_singleton_fallback_fraction"] = (
        0.0
        if task_batch_microsteps <= 0
        else float(fallback_count / float(task_batch_microsteps))
    )

    step_signature_counts = cast(dict[str, int], step_payload["task_batch_signature_counts"])
    for signature, count in cast(Mapping[str, int], microstep_payload["task_batch_signature_counts"]).items():
        key = str(signature)
        step_signature_counts[key] = step_signature_counts.get(key, 0) + int(count)


def _task_weighted_microstep_loss(
    loss: torch.Tensor,
    *,
    actual_task_count: int,
    accelerator: Any,
) -> torch.Tensor:
    resolved_task_count = int(actual_task_count)
    if resolved_task_count <= 0:
        raise RuntimeError(
            "task-batch accumulation requires a positive actual_task_count, "
            f"got {resolved_task_count}"
        )
    accumulation_steps = getattr(accelerator, "gradient_accumulation_steps", 1)
    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        accumulation_steps = 1
    return loss * float(resolved_task_count) * float(accumulation_steps)


def _normalize_accumulated_task_gradients(
    model: torch.nn.Module,
    *,
    step_total_task_count: int,
) -> None:
    resolved_task_count = int(step_total_task_count)
    if resolved_task_count <= 0:
        raise RuntimeError(
            "task-batch accumulation requires a positive step_total_task_count, "
            f"got {resolved_task_count}"
        )
    scale = 1.0 / float(resolved_task_count)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _apply_guard_update(
    state: TrainingLoopState,
    *,
    guard_update: GuardStepUpdate,
) -> None:
    state.global_step = guard_update.global_step
    state.train_elapsed_seconds = guard_update.train_elapsed_seconds
    state.nan_skip_count = guard_update.nan_skip_count
    state.examples_seen = guard_update.examples_seen
    state.tokens_seen = guard_update.tokens_seen
    state.stop_requested = guard_update.stop_requested


def _save_eval_mode_artifact(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    path: Path,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    save_eval_mode_checkpoint(
        prepared_opts,
        path=path,
        model_state_factory=lambda: accelerator.get_state_dict(model),
        global_step=global_step,
        cfg=cfg,
        restore_training=restore_training,
    )


def _maybe_save_best_checkpoint(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    output_dir: Path,
    cfg: DictConfig,
    state: TrainingLoopState,
    val_loss: float,
) -> None:
    if val_loss >= state.best_val:
        return
    state.best_val = val_loss
    state.best_val_step = float(state.global_step)
    state.best_checkpoint = output_dir / "checkpoints" / "best.pt"
    if accelerator.is_main_process:
        _save_eval_mode_artifact(
            accelerator=accelerator,
            model=model,
            prepared_opts=prepared_opts,
            path=state.best_checkpoint,
            global_step=state.global_step,
            cfg=cfg,
            restore_training=False,
        )


def _maybe_save_snapshot_checkpoint(
    *,
    checkpoint_every: int | None,
    accelerator: Any,
    model: torch.nn.Module,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    output_dir: Path,
    cfg: DictConfig,
    state: TrainingLoopState,
) -> None:
    if checkpoint_every is None or state.global_step % checkpoint_every != 0:
        return
    snapshot_checkpoint = output_dir / "checkpoints" / f"step_{state.global_step:06d}.pt"
    if accelerator.is_main_process:
        _save_eval_mode_artifact(
            accelerator=accelerator,
            model=model,
            prepared_opts=prepared_opts,
            path=snapshot_checkpoint,
            global_step=state.global_step,
            cfg=cfg,
            restore_training=True,
        )
        state.checkpoint_snapshots.append(
            {
                "step": int(state.global_step),
                "path": str(snapshot_checkpoint.resolve()),
                "elapsed_seconds": float(state.train_elapsed_seconds),
                "train_elapsed_seconds": float(state.train_elapsed_seconds),
            }
        )


def _append_step_records(
    *,
    accelerator: Any,
    history_path: Path | None,
    gradient_path: Path,
    state: TrainingLoopState,
    stage_name: str,
    train_log: Mapping[str, Any],
    train_loss: float,
    current_train_metrics: Mapping[str, float],
    first_lr: float,
    grad_norm_value: float,
    global_grad_norm_kind: str,
    pre_clip_module_grad_norms: Mapping[str, float],
    activation_norms: Mapping[str, float] | None,
    elapsed_seconds: float,
    history_val_metrics: Mapping[str, float] | None,
    loss_delta_value: float | None,
    grad_clip_threshold: float,
    grad_clip_triggered: bool,
    step_batch_payload: Mapping[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return
    train_metrics_for_history = {
        key.removeprefix("train/"): float(value)
        for key, value in train_log.items()
        if key.startswith("train/")
        and isinstance(value, (int, float))
        and key != "train/lr"
        and math.isfinite(float(value))
    }
    history_payload = history_record(
        global_step=state.global_step,
        stage_name=stage_name,
        train_loss=float(train_loss),
        train_metrics=train_metrics_for_history,
        lr=float(first_lr),
        grad_norm=None if not math.isfinite(grad_norm_value) else float(grad_norm_value),
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        val_metrics=history_val_metrics,
        train_loss_delta=loss_delta_value,
        train_loss_ema=state.loss_ema,
        grad_clip_threshold=grad_clip_threshold,
        grad_clip_triggered=grad_clip_triggered,
        task_batch_size_requested=int(step_batch_payload["task_batch_size_requested"]),
        task_batch_size_actual=int(step_batch_payload["task_batch_size_actual"]),
        task_batch_batched_count=int(step_batch_payload["task_batch_batched_count"]),
        task_batch_singleton_fallback_count=int(step_batch_payload["task_batch_singleton_fallback_count"]),
        task_batch_singleton_fallback_fraction=float(
            step_batch_payload["task_batch_singleton_fallback_fraction"]
        ),
        task_batch_signature_counts=cast(
            Mapping[str, int],
            step_batch_payload["task_batch_signature_counts"],
        ),
    )
    state.history_records.append(history_payload)
    if history_path is not None:
        append_history_record(history_path, history_payload)

    gradient_payload = gradient_history_record(
        global_step=state.global_step,
        stage_name=stage_name,
        train_loss=float(train_loss),
        train_acc=current_train_metrics.get("acc"),
        lr=float(first_lr),
        global_grad_norm=None if not math.isfinite(grad_norm_value) else float(grad_norm_value),
        global_grad_norm_kind=global_grad_norm_kind,
        module_grad_norms=pre_clip_module_grad_norms,
        activation_norms=activation_norms,
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        grad_clip_threshold=grad_clip_threshold,
        grad_clip_triggered=grad_clip_triggered,
        task_batch_size_requested=int(step_batch_payload["task_batch_size_requested"]),
        task_batch_size_actual=int(step_batch_payload["task_batch_size_actual"]),
        task_batch_batched_count=int(step_batch_payload["task_batch_batched_count"]),
        task_batch_singleton_fallback_count=int(step_batch_payload["task_batch_singleton_fallback_count"]),
        task_batch_singleton_fallback_fraction=float(
            step_batch_payload["task_batch_singleton_fallback_fraction"]
        ),
        task_batch_signature_counts=cast(
            Mapping[str, int],
            step_batch_payload["task_batch_signature_counts"],
        ),
    )
    state.gradient_records.append(gradient_payload)
    append_jsonl_record(gradient_path, gradient_payload)


def _save_stage_latest_checkpoints(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    output_dir: Path,
    stage_name: str,
    cfg: DictConfig,
    state: TrainingLoopState,
) -> None:
    state.latest_checkpoint = stage_latest_checkpoint_path(output_dir, stage_name=stage_name)
    compatibility_latest_checkpoint = canonical_latest_checkpoint_path(output_dir)
    if accelerator.is_main_process:
        _save_eval_mode_artifact(
            accelerator=accelerator,
            model=model,
            prepared_opts=prepared_opts,
            path=state.latest_checkpoint,
            global_step=state.global_step,
            cfg=cfg,
            restore_training=True,
        )
        _save_eval_mode_artifact(
            accelerator=accelerator,
            model=model,
            prepared_opts=prepared_opts,
            path=compatibility_latest_checkpoint,
            global_step=state.global_step,
            cfg=cfg,
            restore_training=True,
        )


def run_training_loop(
    *,
    cfg: DictConfig,
    task: str,
    output_dir: Path,
    history_path: Path | None,
    gradient_path: Path,
    accelerator: Any,
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    train_loader,
    val_loader,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    lr_scales: dict[str, list[float]],
    stage_configs: list[Any],
    expected_keys: set[str],
    task_batch_size: int,
    grad_accum_steps: int,
    checkpoint_every: int | None,
    max_steps: int | None,
    target_train_seconds: float | None,
    val_batches: int,
    train_start: float,
    trace_activations: bool,
    non_blocking_device_transfer: bool,
    flush_activation_trace_stats,
    run: Any,
    state: TrainingLoopState,
) -> None:
    train_iter: Iterator[TaskBatch] = iter(cast(tuple[TaskBatch, ...], ()))
    if train_loader is not None:
        train_iter = cycle_loader(train_loader)
    grad_clip_threshold = float(cfg.runtime.grad_clip)
    eval_every = int(cfg.runtime.eval_every)
    for stage in stage_configs:
        _set_optimizer_training_mode(prepared_opts, training=True)

        for stage_step in range(1, stage.steps + 1):
            model.train()
            current_base_lr = stage_base_lr(
                stage,
                step=stage_step,
                lr_min=float(cfg.optimizer.min_lr),
            )
            for opt_name, opt in prepared_opts:
                _set_optimizer_base_lr(
                    opt,
                    base_lr=current_base_lr,
                    scales=lr_scales[opt_name],
                )
            for _opt_name, opt in prepared_opts:
                opt.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_loss_count = 0
            train_metric_sums: dict[str, float] = {}
            train_metric_counts: dict[str, int] = {}
            activation_sum_sqs: dict[str, float] = {}
            activation_element_counts: dict[str, float] = {}
            step_batch_payload = _empty_task_batch_step_payload(
                requested_task_batch_size=task_batch_size,
            )
            step_train_start = time.perf_counter()
            step_total_task_count = 0
            step_examples_seen = 0
            step_tokens_seen = 0
            for _micro_step in range(grad_accum_steps):
                batch: TaskBatch = next(train_iter)
                microstep_batch_payload = _task_batch_microstep_payload(batch)
                _accumulate_task_batch_step_payload(
                    step_payload=step_batch_payload,
                    microstep_payload=microstep_batch_payload,
                )
                actual_task_count = int(microstep_batch_payload["task_batch_size_actual"])
                step_total_task_count += actual_task_count
                step_examples_seen += task_batch_examples_seen(batch)
                step_tokens_seen += task_batch_token_count(batch)
                batch = move_batch(
                    batch,
                    accelerator.device,
                    non_blocking=non_blocking_device_transfer,
                )
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        output = model(batch)
                        loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
                    accelerator.backward(
                        _task_weighted_microstep_loss(
                            loss,
                            actual_task_count=actual_task_count,
                            accelerator=accelerator,
                        )
                    )
                train_loss_sum += float(loss.detach().item()) * float(actual_task_count)
                train_loss_count += actual_task_count
                for key, value in metrics.items():
                    train_metric_sums[key] = train_metric_sums.get(key, 0.0) + (
                        float(value) * float(actual_task_count)
                    )
                    train_metric_counts[key] = train_metric_counts.get(key, 0) + actual_task_count
                batch_activation_trace_stats = flush_activation_trace_stats()
                if batch_activation_trace_stats is not None:
                    for activation_name, (activation_sum_sq, activation_count) in batch_activation_trace_stats.items():
                        activation_sum_sqs[activation_name] = activation_sum_sqs.get(
                            activation_name,
                            0.0,
                        ) + float(activation_sum_sq)
                        activation_element_counts[activation_name] = activation_element_counts.get(
                            activation_name,
                            0.0,
                        ) + float(activation_count)
            _normalize_accumulated_task_gradients(
                model,
                step_total_task_count=step_total_task_count,
            )

            local_nan_detected = not math.isfinite(train_loss_sum)
            global_nan_detected = _reduce_any_flag(
                accelerator,
                local_nan_detected,
                device=accelerator.device,
            )
            first_lr = float(prepared_opts[0][1].param_groups[0]["lr"])
            if global_nan_detected:
                guard_update = handle_non_finite_loss(
                    accelerator=accelerator,
                    prepared_opts=prepared_opts,
                    run=run,
                    stage_name=stage.name,
                    lr=first_lr,
                    history_records=state.history_records,
                    history_path=history_path,
                    gradient_records=state.gradient_records,
                    gradient_path=gradient_path,
                    train_start=train_start,
                    step_train_start=step_train_start,
                    train_elapsed_seconds=state.train_elapsed_seconds,
                    global_step=state.global_step,
                    nan_skip_count=state.nan_skip_count,
                    examples_seen=state.examples_seen,
                    tokens_seen=state.tokens_seen,
                    step_examples_seen=step_examples_seen,
                    step_tokens_seen=step_tokens_seen,
                    step_batch_payload=step_batch_payload,
                    loss_ema=state.loss_ema,
                    grad_clip_threshold=grad_clip_threshold,
                    max_steps=max_steps,
                    target_train_seconds=target_train_seconds,
                    flush_activation_trace_stats=flush_activation_trace_stats,
                )
                _apply_guard_update(state, guard_update=guard_update)
                if state.stop_requested:
                    break
                continue

            activation_norms: dict[str, float] | None = None
            if trace_activations:
                activation_sum_sqs, activation_element_counts = _reduce_keyed_weighted_scalars(
                    accelerator,
                    weighted_sums=activation_sum_sqs,
                    weights=activation_element_counts,
                    device=accelerator.device,
                )
                activation_norms = _merge_activation_norms(
                    activation_sum_sqs,
                    activation_element_counts,
                )
            pre_clip_module_grad_norms = module_grad_norms(base_model) if accelerator.is_main_process else {}
            local_grad_norm = total_grad_norm(model.parameters())
            if grad_clip_threshold > 0:
                clipped = accelerator.clip_grad_norm_(model.parameters(), grad_clip_threshold)
                local_grad_norm = normalize_grad_norm_value(clipped, fallback=local_grad_norm)
            global_grad_norm_kind = _reduce_global_grad_norm_kind(
                accelerator,
                local_kind=_global_grad_norm_kind(float(local_grad_norm)),
                device=accelerator.device,
            )
            if global_grad_norm_kind != "finite":
                guard_update = handle_non_finite_grad(
                    accelerator=accelerator,
                    prepared_opts=prepared_opts,
                    run=run,
                    stage_name=stage.name,
                    lr=first_lr,
                    history_records=state.history_records,
                    history_path=history_path,
                    gradient_records=state.gradient_records,
                    gradient_path=gradient_path,
                    train_start=train_start,
                    step_train_start=step_train_start,
                    train_elapsed_seconds=state.train_elapsed_seconds,
                    global_step=state.global_step,
                    nan_skip_count=state.nan_skip_count,
                    examples_seen=state.examples_seen,
                    tokens_seen=state.tokens_seen,
                    step_examples_seen=step_examples_seen,
                    step_tokens_seen=step_tokens_seen,
                    step_batch_payload=step_batch_payload,
                    loss_ema=state.loss_ema,
                    grad_clip_threshold=grad_clip_threshold,
                    max_steps=max_steps,
                    target_train_seconds=target_train_seconds,
                    flush_activation_trace_stats=flush_activation_trace_stats,
                    global_grad_norm_kind=global_grad_norm_kind,
                    module_grad_norms=pre_clip_module_grad_norms,
                )
                _apply_guard_update(state, guard_update=guard_update)
                if state.stop_requested:
                    break
                continue
            train_metric_sums["grad_norm"] = train_metric_sums.get("grad_norm", 0.0) + float(local_grad_norm)
            train_metric_counts["grad_norm"] = train_metric_counts.get("grad_norm", 0) + 1

            for _opt_name, opt in prepared_opts:
                opt.step()

            state.train_elapsed_seconds += time.perf_counter() - step_train_start
            state.global_step += 1
            state.examples_seen += step_examples_seen
            state.tokens_seen += step_tokens_seen
            metric_keys = sorted(set(train_metric_sums) | expected_keys)
            n_metrics = len(metric_keys)
            packed = torch.zeros(
                2 + 2 * n_metrics,
                device=accelerator.device,
                dtype=_reduction_float_dtype(accelerator.device),
            )
            packed[0] = train_loss_sum
            packed[1] = train_loss_count
            for i, key in enumerate(metric_keys):
                packed[2 + 2 * i] = train_metric_sums.get(key, 0.0)
                packed[2 + 2 * i + 1] = train_metric_counts.get(key, 0)
            reduced = accelerator.reduce(packed, reduction="sum")

            g_loss_sum = reduced[0].item()
            g_loss_count = reduced[1].item()
            train_loss = g_loss_sum / g_loss_count if g_loss_count > 0 else 0.0

            lr_values = {name: float(opt.param_groups[0]["lr"]) for name, opt in prepared_opts}
            first_lr = next(iter(lr_values.values()))
            grad_norm_value = float("nan")
            train_log: dict[str, Any] = {
                "train/loss": train_loss,
                "train/lr": first_lr,
                "train/stage": stage.name,
            }
            for name, value in lr_values.items():
                train_log[f"train/lr_{name}"] = value
            current_train_metrics: dict[str, float] = {}
            for i, key in enumerate(metric_keys):
                g_sum = reduced[2 + 2 * i].item()
                g_count = reduced[2 + 2 * i + 1].item()
                metric_mean = g_sum / g_count if g_count > 0 else float("nan")
                if key == "grad_norm":
                    grad_norm_value = float(metric_mean)
                if math.isfinite(metric_mean):
                    current_train_metrics[key] = float(metric_mean)
                    train_log[f"train/{key}"] = metric_mean
                    if key == "grad_norm":
                        state.grad_norm_sum += grad_norm_value
                        state.grad_norm_count += 1
                        state.max_grad_norm = max(state.max_grad_norm, grad_norm_value)
                        state.final_grad_norm = grad_norm_value

            current_train_loss = float(train_loss)
            loss_delta_value = train_loss_delta(
                current_train_loss,
                previous_train_loss=state.previous_train_loss,
            )
            state.loss_ema = update_loss_ema(current_train_loss, previous_ema=state.loss_ema)
            state.previous_train_loss = current_train_loss
            elapsed_seconds = time.perf_counter() - train_start
            grad_clip_triggered = bool(
                grad_clip_threshold > 0
                and math.isfinite(grad_norm_value)
                and float(grad_norm_value) > grad_clip_threshold
            )
            global_grad_norm_kind = _global_grad_norm_kind(float(grad_norm_value))
            train_log["train/loss_delta"] = loss_delta_value
            train_log["train/loss_ema"] = state.loss_ema
            train_log["train/elapsed_seconds"] = elapsed_seconds
            train_log["train/train_elapsed_seconds"] = state.train_elapsed_seconds
            train_log["train/grad_clip_threshold"] = grad_clip_threshold
            train_log["train/grad_clip_triggered"] = grad_clip_triggered
            train_log["train/task_batch_size_requested"] = int(step_batch_payload["task_batch_size_requested"])
            train_log["train/task_batch_size_actual"] = int(step_batch_payload["task_batch_size_actual"])
            train_log["train/task_batch_batched_count"] = int(step_batch_payload["task_batch_batched_count"])
            train_log["train/task_batch_singleton_fallback_count"] = int(
                step_batch_payload["task_batch_singleton_fallback_count"]
            )
            train_log["train/task_batch_singleton_fallback_fraction"] = float(
                step_batch_payload["task_batch_singleton_fallback_fraction"]
            )
            if accelerator.is_main_process:
                for module_name, module_value in pre_clip_module_grad_norms.items():
                    train_log[f"train/module_grad_norm/{module_name}"] = float(module_value)
                feature_grad = pre_clip_module_grad_norms.get("feature_encoder")
                head_grad = pre_clip_module_grad_norms.get("direct_head")
                if feature_grad is not None and float(feature_grad) > 0.0 and head_grad is not None:
                    train_log["train/module_balance/direct_head_to_feature_encoder"] = float(head_grad) / float(
                        feature_grad
                    )
                if head_grad is not None and float(head_grad) > 0.0 and feature_grad is not None:
                    train_log["train/module_balance/feature_encoder_to_direct_head"] = float(feature_grad) / float(
                        head_grad
                    )
                if activation_norms is not None:
                    for activation_name, activation_value in activation_norms.items():
                        train_log[f"train/activation_norm/{activation_name}"] = float(activation_value)
            state.final_train_loss = current_train_loss
            state.final_train_loss_ema = state.loss_ema
            state.last_train_metrics = current_train_metrics
            log_wandb_metrics(run, train_log, step=state.global_step)

            history_val_metrics: dict[str, float] | None = None
            if val_loader is not None and state.global_step % eval_every == 0:
                _ = flush_activation_trace_stats()
                _set_optimizer_training_mode(prepared_opts, training=False)
                val_metrics = _evaluate_loader(
                    model,
                    val_loader,
                    accelerator=accelerator,
                    task=task,
                    max_batches=val_batches,
                    non_blocking_device_transfer=non_blocking_device_transfer,
                )
                log_wandb_metrics(
                    run,
                    {f"val/{key}": value for key, value in val_metrics.items()},
                    step=state.global_step,
                )
                history_val_metrics = val_metrics
                state.last_val_metrics = val_metrics
                _ = flush_activation_trace_stats()
                _maybe_save_best_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    prepared_opts=prepared_opts,
                    output_dir=output_dir,
                    cfg=cfg,
                    state=state,
                    val_loss=float(val_metrics["val_loss"]),
                )
                _set_optimizer_training_mode(prepared_opts, training=True)

            _maybe_save_snapshot_checkpoint(
                checkpoint_every=checkpoint_every,
                accelerator=accelerator,
                model=model,
                prepared_opts=prepared_opts,
                output_dir=output_dir,
                cfg=cfg,
                state=state,
            )
            _append_step_records(
                accelerator=accelerator,
                history_path=history_path,
                gradient_path=gradient_path,
                state=state,
                stage_name=stage.name,
                train_log=train_log,
                train_loss=float(train_loss),
                current_train_metrics=current_train_metrics,
                first_lr=float(first_lr),
                grad_norm_value=float(grad_norm_value),
                global_grad_norm_kind=global_grad_norm_kind,
                pre_clip_module_grad_norms=pre_clip_module_grad_norms,
                activation_norms=activation_norms,
                elapsed_seconds=elapsed_seconds,
                history_val_metrics=history_val_metrics,
                loss_delta_value=loss_delta_value,
                grad_clip_threshold=grad_clip_threshold,
                grad_clip_triggered=grad_clip_triggered,
                step_batch_payload=step_batch_payload,
            )

            if max_steps is not None and state.global_step >= max_steps:
                state.stop_requested = True
            if target_train_seconds is not None and state.train_elapsed_seconds >= target_train_seconds:
                state.stop_requested = True
            if state.stop_requested:
                break

        _save_stage_latest_checkpoints(
            accelerator=accelerator,
            model=model,
            prepared_opts=prepared_opts,
            output_dir=output_dir,
            stage_name=stage.name,
            cfg=cfg,
            state=state,
        )
        if state.stop_requested:
            break
