"""Non-finite guard helpers for the trainer loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, cast

import torch

from .artifacts import append_history_record, append_jsonl_record, gradient_history_record, history_record
from .distributed import _reduce_any_flag
from .wandb import log_wandb_metrics


@dataclass(slots=True, frozen=True)
class GuardStepUpdate:
    global_step: int
    train_elapsed_seconds: float
    nan_skip_count: int
    examples_seen: int
    tokens_seen: int
    stop_requested: bool


def _merge_activation_norms(
    sum_sqs: dict[str, float],
    element_counts: dict[str, float],
) -> dict[str, float] | None:
    if not sum_sqs:
        return None
    merged: dict[str, float] = {}
    for name, value in sum_sqs.items():
        count = float(element_counts.get(name, 0.0))
        if count <= 0.0:
            continue
        merged[name] = float(math.sqrt(float(value) / count))
    return merged or None


def _accelerator_num_processes(accelerator: Any) -> int:
    raw_num_processes = getattr(accelerator, "num_processes", None)
    if isinstance(raw_num_processes, int) and raw_num_processes > 0:
        return raw_num_processes
    raw_state = getattr(accelerator, "state", None)
    state_num_processes = getattr(raw_state, "num_processes", None)
    if isinstance(state_num_processes, int) and state_num_processes > 0:
        return state_num_processes
    return 1


def _global_grad_norm_kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "pos_inf" if value > 0.0 else "neg_inf"
    return "finite"


def _reduce_global_grad_norm_kind(
    accelerator: Any,
    *,
    local_kind: str,
    device: torch.device,
) -> str:
    if local_kind not in {"finite", "nan", "pos_inf", "neg_inf"}:
        raise ValueError(f"unsupported global grad norm kind {local_kind!r}")
    if _accelerator_num_processes(accelerator) == 1:
        return local_kind
    if _reduce_any_flag(accelerator, local_kind == "nan", device=device):
        return "nan"
    if _reduce_any_flag(accelerator, local_kind == "pos_inf", device=device):
        return "pos_inf"
    if _reduce_any_flag(accelerator, local_kind == "neg_inf", device=device):
        return "neg_inf"
    return "finite"


def _non_finite_stop_requested(
    *,
    global_step: int,
    train_elapsed_seconds: float,
    max_steps: int | None,
    target_train_seconds: float | None,
) -> bool:
    if max_steps is not None and global_step >= max_steps:
        return True
    return target_train_seconds is not None and train_elapsed_seconds >= target_train_seconds


def _handle_non_finite_step(
    *,
    accelerator: Any,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    run: Any,
    stage_name: str,
    lr: float,
    guard_metric_key: str,
    wandb_log: Mapping[str, Any],
    history_records: list[dict[str, Any]],
    history_path,
    gradient_records: list[dict[str, Any]],
    gradient_path,
    train_start: float,
    step_train_start: float,
    train_elapsed_seconds: float,
    global_step: int,
    nan_skip_count: int,
    examples_seen: int,
    tokens_seen: int,
    step_examples_seen: int,
    step_tokens_seen: int,
    step_batch_payload: Mapping[str, Any],
    loss_ema: float | None,
    grad_clip_threshold: float,
    max_steps: int | None,
    target_train_seconds: float | None,
    flush_activation_trace_stats: Callable[[], Any] | None,
    global_grad_norm_kind: str | None = None,
    module_grad_norms: Mapping[str, float] | None = None,
) -> GuardStepUpdate:
    if flush_activation_trace_stats is not None:
        _ = flush_activation_trace_stats()
    updated_nan_skip_count = nan_skip_count + 1
    for _opt_name, opt in prepared_opts:
        opt.zero_grad(set_to_none=True)
    updated_train_elapsed_seconds = train_elapsed_seconds + (time.perf_counter() - step_train_start)
    updated_global_step = global_step + 1
    updated_examples_seen = examples_seen + step_examples_seen
    updated_tokens_seen = tokens_seen + step_tokens_seen
    log_wandb_metrics(run, dict(wandb_log), step=updated_global_step)
    if accelerator.is_main_process:
        elapsed_seconds = time.perf_counter() - train_start
        history_payload = history_record(
            global_step=updated_global_step,
            stage_name=stage_name,
            train_loss=float("nan"),
            train_metrics={guard_metric_key: 1.0},
            lr=float(lr),
            grad_norm=None,
            elapsed_seconds=elapsed_seconds,
            train_elapsed_seconds=updated_train_elapsed_seconds,
            val_metrics=None,
            train_loss_delta=None,
            train_loss_ema=loss_ema,
            grad_clip_threshold=grad_clip_threshold,
            grad_clip_triggered=False,
            task_batch_size_requested=int(step_batch_payload["task_batch_size_requested"]),
            task_batch_size_actual=int(step_batch_payload["task_batch_size_actual"]),
            task_batch_batched_count=int(step_batch_payload["task_batch_batched_count"]),
            task_batch_singleton_fallback_count=int(
                step_batch_payload["task_batch_singleton_fallback_count"]
            ),
            task_batch_singleton_fallback_fraction=float(
                step_batch_payload["task_batch_singleton_fallback_fraction"]
            ),
            task_batch_signature_counts=cast(
                Mapping[str, int],
                step_batch_payload["task_batch_signature_counts"],
            ),
        )
        history_records.append(history_payload)
        if history_path is not None:
            append_history_record(history_path, history_payload)
        if global_grad_norm_kind is not None:
            resolved_module_grad_norms = {} if module_grad_norms is None else module_grad_norms
            gradient_payload = gradient_history_record(
                global_step=updated_global_step,
                stage_name=stage_name,
                train_loss=float("nan"),
                train_acc=None,
                lr=float(lr),
                global_grad_norm=None,
                global_grad_norm_kind=global_grad_norm_kind,
                module_grad_norms=resolved_module_grad_norms,
                activation_norms=None,
                elapsed_seconds=elapsed_seconds,
                train_elapsed_seconds=updated_train_elapsed_seconds,
                grad_clip_threshold=grad_clip_threshold,
                grad_clip_triggered=False,
                task_batch_size_requested=int(step_batch_payload["task_batch_size_requested"]),
                task_batch_size_actual=int(step_batch_payload["task_batch_size_actual"]),
                task_batch_batched_count=int(step_batch_payload["task_batch_batched_count"]),
                task_batch_singleton_fallback_count=int(
                    step_batch_payload["task_batch_singleton_fallback_count"]
                ),
                task_batch_singleton_fallback_fraction=float(
                    step_batch_payload["task_batch_singleton_fallback_fraction"]
                ),
                task_batch_signature_counts=cast(
                    Mapping[str, int],
                    step_batch_payload["task_batch_signature_counts"],
                ),
            )
            gradient_records.append(gradient_payload)
            append_jsonl_record(gradient_path, gradient_payload)
    return GuardStepUpdate(
        global_step=updated_global_step,
        train_elapsed_seconds=updated_train_elapsed_seconds,
        nan_skip_count=updated_nan_skip_count,
        examples_seen=updated_examples_seen,
        tokens_seen=updated_tokens_seen,
        stop_requested=_non_finite_stop_requested(
            global_step=updated_global_step,
            train_elapsed_seconds=updated_train_elapsed_seconds,
            max_steps=max_steps,
            target_train_seconds=target_train_seconds,
        ),
    )


def handle_non_finite_loss(
    *,
    accelerator: Any,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    run: Any,
    stage_name: str,
    lr: float,
    history_records: list[dict[str, Any]],
    history_path,
    gradient_records: list[dict[str, Any]],
    gradient_path,
    train_start: float,
    step_train_start: float,
    train_elapsed_seconds: float,
    global_step: int,
    nan_skip_count: int,
    examples_seen: int,
    tokens_seen: int,
    step_examples_seen: int,
    step_tokens_seen: int,
    step_batch_payload: Mapping[str, Any],
    loss_ema: float | None,
    grad_clip_threshold: float,
    max_steps: int | None,
    target_train_seconds: float | None,
    flush_activation_trace_stats: Callable[[], Any] | None,
) -> GuardStepUpdate:
    return _handle_non_finite_step(
        accelerator=accelerator,
        prepared_opts=prepared_opts,
        run=run,
        stage_name=stage_name,
        lr=lr,
        guard_metric_key="nan_guard_triggered",
        wandb_log={
            "train/nan_guard_triggered": True,
            "train/nan_skip_count": float(nan_skip_count + 1),
        },
        history_records=history_records,
        history_path=history_path,
        gradient_records=gradient_records,
        gradient_path=gradient_path,
        train_start=train_start,
        step_train_start=step_train_start,
        train_elapsed_seconds=train_elapsed_seconds,
        global_step=global_step,
        nan_skip_count=nan_skip_count,
        examples_seen=examples_seen,
        tokens_seen=tokens_seen,
        step_examples_seen=step_examples_seen,
        step_tokens_seen=step_tokens_seen,
        step_batch_payload=step_batch_payload,
        loss_ema=loss_ema,
        grad_clip_threshold=grad_clip_threshold,
        max_steps=max_steps,
        target_train_seconds=target_train_seconds,
        flush_activation_trace_stats=flush_activation_trace_stats,
    )


def handle_non_finite_grad(
    *,
    accelerator: Any,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    run: Any,
    stage_name: str,
    lr: float,
    history_records: list[dict[str, Any]],
    history_path,
    gradient_records: list[dict[str, Any]],
    gradient_path,
    train_start: float,
    step_train_start: float,
    train_elapsed_seconds: float,
    global_step: int,
    nan_skip_count: int,
    examples_seen: int,
    tokens_seen: int,
    step_examples_seen: int,
    step_tokens_seen: int,
    step_batch_payload: Mapping[str, Any],
    loss_ema: float | None,
    grad_clip_threshold: float,
    max_steps: int | None,
    target_train_seconds: float | None,
    flush_activation_trace_stats: Callable[[], Any] | None,
    global_grad_norm_kind: str,
    module_grad_norms: Mapping[str, float],
) -> GuardStepUpdate:
    return _handle_non_finite_step(
        accelerator=accelerator,
        prepared_opts=prepared_opts,
        run=run,
        stage_name=stage_name,
        lr=lr,
        guard_metric_key="non_finite_grad_guard_triggered",
        wandb_log={
            "train/non_finite_grad_guard_triggered": True,
            "train/non_finite_grad_kind": global_grad_norm_kind,
            "train/nan_skip_count": float(nan_skip_count + 1),
        },
        history_records=history_records,
        history_path=history_path,
        gradient_records=gradient_records,
        gradient_path=gradient_path,
        train_start=train_start,
        step_train_start=step_train_start,
        train_elapsed_seconds=train_elapsed_seconds,
        global_step=global_step,
        nan_skip_count=nan_skip_count,
        examples_seen=examples_seen,
        tokens_seen=tokens_seen,
        step_examples_seen=step_examples_seen,
        step_tokens_seen=step_tokens_seen,
        step_batch_payload=step_batch_payload,
        loss_ema=loss_ema,
        grad_clip_threshold=grad_clip_threshold,
        max_steps=max_steps,
        target_train_seconds=target_train_seconds,
        flush_activation_trace_stats=flush_activation_trace_stats,
        global_grad_norm_kind=global_grad_norm_kind,
        module_grad_norms=module_grad_norms,
    )
