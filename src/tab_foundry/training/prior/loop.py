"""Training loop helpers for exact prior-dump training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, cast

from omegaconf import DictConfig
import torch

from tab_foundry.hardware_profiles import build_hardware_summary
from tab_foundry.model.spec import SANDWICH_FAMILY_MODEL_ARCHES
from tab_foundry.training.artifacts import (
    append_history_record,
    append_jsonl_record,
    assert_clean_training_output,
    gradient_history_record,
    history_path_from_cfg,
    history_record,
    save_eval_mode_checkpoint,
)
from tab_foundry.training.instability import (
    build_regime_budget_summary,
    build_runtime_summary,
    build_training_telemetry,
    grad_norm_summary_from_running_totals,
    gradient_history_path,
    module_grad_norms,
    normalize_grad_norm_value,
    peak_device_memory_summary,
    reset_peak_device_memory_stats,
    telemetry_path,
    tensor_batch_examples_seen,
    tensor_batch_token_count,
    total_grad_norm,
    train_loss_delta,
    training_shape_summary_from_signature_task_counts,
    update_loss_ema,
    write_training_telemetry,
)
from tab_foundry.training.losses import classification_loss, classification_z_loss
from tab_foundry.training.loss_surface import resolve_classification_z_loss_coeff
from tab_foundry.training.prior.io import stack_prior_step
from tab_foundry.training.prior.missingness import (
    _accumulate_missingness,
    _accumulate_synthetic_missingness,
    _apply_prior_missingness,
    _initial_missingness_summary,
    _prior_wandb_summary_payload,
)
from tab_foundry.training.prior_dump import PriorDumpNonFinitePolicy, PriorDumpTaskBatchReader
from tab_foundry.training.schedule import stage_base_lr
from tab_foundry.training.surface import TRAINING_BACKEND_LEGACY_PRIOR, write_training_surface_record
from tab_foundry.training.trainer_optimizer import _set_optimizer_base_lr, _set_optimizer_training_mode
from tab_foundry.training.wandb import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    training_surface_wandb_summary_payload,
    update_wandb_summary,
    wandb_identity_payload,
)
from tab_foundry.task_batching import task_batch_signature_text
from tab_foundry.types import TrainResult


_PRIOR_STAGE_NAME = "prior_dump"
_PRIOR_BATCH_NDIM = 3


def _requires_explicit_feature_types(arch: str) -> bool:
    return str(arch).strip().lower() in SANDWICH_FAMILY_MODEL_ARCHES


@dataclass(slots=True)
class PriorTrainingState:
    global_step: int = 0
    latest_checkpoint: Path | None = None
    final_train_loss: float | None = None
    final_train_acc: float | None = None
    final_train_bpc: float | None = None
    final_train_bpf: float | None = None
    final_grad_norm: float = 0.0
    grad_norm_sum: float = 0.0
    grad_norm_count: int = 0
    max_grad_norm: float = 0.0
    clipped_step_count: int = 0
    nan_skip_count: int = 0
    train_elapsed_seconds: float = 0.0
    examples_seen: int = 0
    tokens_seen: int = 0
    previous_train_loss: float | None = None
    loss_ema: float | None = None
    history_records: list[dict[str, Any]] = field(default_factory=list)
    gradient_records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_snapshots: list[dict[str, Any]] = field(default_factory=list)
    signature_task_counts: dict[str, int] = field(default_factory=dict)


def _global_grad_norm_kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "pos_inf" if value > 0.0 else "neg_inf"
    return "finite"


def _accumulate_prior_training_shape_signature(
    state: PriorTrainingState,
    *,
    x_batch: torch.Tensor,
    train_test_split_index: int,
    task_count: int,
) -> None:
    if task_count <= 0 or x_batch.ndim != _PRIOR_BATCH_NDIM:
        return
    total_rows = int(x_batch.shape[1])
    signature = task_batch_signature_text(
        (
            int(train_test_split_index),
            int(total_rows - int(train_test_split_index)),
            int(x_batch.shape[2]),
            None,
        )
    )
    state.signature_task_counts[signature] = state.signature_task_counts.get(signature, 0) + int(task_count)


def _save_eval_mode_artifact(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    path: Path,
    model: torch.nn.Module,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    save_eval_mode_checkpoint(
        prepared_opts,
        path=path,
        model_state_factory=model.state_dict,
        global_step=global_step,
        cfg=cfg,
        restore_training=restore_training,
    )


def _merge_activation_norms(
    weighted_sums: dict[str, float],
    weight_totals: dict[str, float],
) -> dict[str, float] | None:
    if not weighted_sums:
        return None
    merged: dict[str, float] = {}
    for name, value in weighted_sums.items():
        weight = float(weight_totals.get(name, 0.0))
        if weight <= 0.0:
            continue
        merged[name] = float(value / weight)
    return merged or None


def _run_prior_step_with_microbatch_retry(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_batch: torch.Tensor,
    y_train_batch: torch.Tensor,
    y_all_batch: torch.Tensor,
    feature_types_batch: list[list[str]] | None,
    train_test_split_index: int,
    loss_surface: str,
    classification_z_loss_coeff: float,
    trace_activations: bool,
    flush_activation_trace: object,
) -> tuple[float, dict[str, float], dict[str, float] | None, int, int]:
    forward_batched = getattr(model, "forward_batched", None)
    if not callable(forward_batched):
        raise RuntimeError("prior-dump training requires a model with forward_batched()")

    total_batch_size = int(x_batch.shape[0])
    if total_batch_size <= 0:
        raise RuntimeError("prior-dump training requires batch_size >= 1")

    def _attempt(microbatch_size: int) -> tuple[float, dict[str, float], dict[str, float] | None, int]:
        weighted_loss = 0.0
        weighted_metrics: dict[str, float] = {}
        activation_weighted_sums: dict[str, float] = {}
        activation_weight_totals: dict[str, float] = {}
        microbatch_count = 0
        for start in range(0, total_batch_size, microbatch_size):
            stop = min(start + microbatch_size, total_batch_size)
            microbatch_count += 1
            weight = float(stop - start) / float(total_batch_size)
            batched_kwargs: dict[str, Any] = {
                "x_all": x_batch[start:stop],
                "y_train": y_train_batch[start:stop],
                "train_test_split_index": train_test_split_index,
            }
            model_arch = str(getattr(model, "arch", "")).strip().lower()
            if _requires_explicit_feature_types(model_arch):
                if feature_types_batch is None:
                    raise RuntimeError(
                        f"{model_arch} prior-dump training requires explicit feature_types "
                        "for every task in the batch"
                    )
                batched_kwargs["feature_types"] = feature_types_batch[start:stop]
            if loss_surface == "cell_bpc":
                if model_arch == "tabfoundry_sandwich":
                    batched_kwargs["y_test"] = y_all_batch[start:stop, train_test_split_index:]
                forward_batched_cell_likelihood = getattr(model, "forward_batched_cell_likelihood", None)
                if not callable(forward_batched_cell_likelihood):
                    raise RuntimeError(
                        "cell_bpc prior-dump training requires a model with "
                        "forward_batched_cell_likelihood()"
                    )
                output = forward_batched_cell_likelihood(**batched_kwargs)
                if output.bpc is None or output.bpf is None:
                    raise RuntimeError("cell_bpc prior-dump training requires output.bpc and output.bpf")
                loss = output.bpc
                weighted_metrics["bpc"] = weighted_metrics.get("bpc", 0.0) + (
                    float(output.bpc.detach().item()) * weight
                )
                weighted_metrics["bpf"] = weighted_metrics.get("bpf", 0.0) + (
                    float(output.bpf.detach().item()) * weight
                )
                if output.aux_metrics is not None and output.aux_metrics.get("acc") is not None:
                    weighted_metrics["acc"] = weighted_metrics.get("acc", 0.0) + (
                        float(output.aux_metrics["acc"]) * weight
                    )
            else:
                logits = forward_batched(**batched_kwargs)
                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("prior-dump training requires tensor logits")
                targets = y_all_batch[start:stop, train_test_split_index:].reshape(-1).to(torch.int64)
                flat_logits = logits.reshape(-1, int(logits.shape[-1]))
                if classification_z_loss_coeff > 0.0:
                    loss = classification_loss(
                        flat_logits,
                        targets,
                        z_loss_coeff=classification_z_loss_coeff,
                    )
                    ce_loss = classification_loss(flat_logits, targets)
                    z_loss = classification_z_loss(flat_logits)
                    weighted_metrics["classification_ce_loss"] = weighted_metrics.get(
                        "classification_ce_loss",
                        0.0,
                    ) + (float(ce_loss.detach().item()) * weight)
                    weighted_metrics["classification_z_loss"] = weighted_metrics.get(
                        "classification_z_loss",
                        0.0,
                    ) + (float(z_loss.detach().item()) * weight)
                    weighted_metrics["classification_z_loss_coeff"] = float(
                        classification_z_loss_coeff
                    )
                else:
                    loss = classification_loss(flat_logits, targets)
                weighted_metrics["acc"] = weighted_metrics.get("acc", 0.0) + (
                    float(
                        (
                            logits.argmax(dim=-1)
                            == y_all_batch[start:stop, train_test_split_index:].to(torch.int64)
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    * weight
                )
            activation_norms = (
                flush_activation_trace()
                if trace_activations and callable(flush_activation_trace)
                else None
            )
            (loss * weight).backward()
            weighted_loss += float(loss.detach().item()) * weight
            if activation_norms is not None:
                for activation_name, activation_value in activation_norms.items():
                    activation_weighted_sums[activation_name] = (
                        activation_weighted_sums.get(activation_name, 0.0)
                        + (float(activation_value) * weight)
                    )
                    activation_weight_totals[activation_name] = (
                        activation_weight_totals.get(activation_name, 0.0) + weight
                    )
        return (
            weighted_loss,
            weighted_metrics,
            _merge_activation_norms(activation_weighted_sums, activation_weight_totals),
            microbatch_count,
        )

    microbatch_size = total_batch_size
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, metric_values, activation_norms, microbatch_count = _attempt(microbatch_size)
            return (
                loss_value,
                metric_values,
                activation_norms,
                microbatch_size,
                microbatch_count,
            )
        except torch.OutOfMemoryError:
            if callable(flush_activation_trace):
                _ = flush_activation_trace()
            optimizer.zero_grad(set_to_none=True)
            if x_batch.device.type == "cuda":
                torch.cuda.empty_cache()
            if microbatch_size <= 1:
                raise
            next_microbatch_size = max(1, microbatch_size // 2)
            if next_microbatch_size == microbatch_size:
                next_microbatch_size = microbatch_size - 1
            print(
                "Warning: prior-dump step hit OOM; "
                f"retrying with microbatch_size={next_microbatch_size} "
                f"(effective_batch_size={total_batch_size})",
                file=sys.stderr,
                flush=True,
            )
            microbatch_size = next_microbatch_size


def _prepare_prior_optimizer(
    *,
    optimizer_selection: Any,
    initial_lr: float,
) -> tuple[list[tuple[str, torch.optim.Optimizer]], torch.optim.Optimizer, list[float]]:
    if optimizer_selection.resolved_name not in {"schedulefree_adamw", "adamw"}:
        raise RuntimeError(
            "exact-parity prior-dump training requires optimizer 'schedulefree_adamw' or 'adamw', "
            f"resolved {optimizer_selection.resolved_name!r}"
        )
    if len(optimizer_selection.optimizers) != 1:
        raise RuntimeError(
            "exact-parity prior-dump training expects exactly one optimizer instance, "
            f"got {len(optimizer_selection.optimizers)}"
        )
    prepared_opts = list(optimizer_selection.optimizers)
    optimizer = prepared_opts[0][1]
    _set_optimizer_training_mode(prepared_opts, training=True)
    lr_scales = [1.0 for _ in optimizer.param_groups]
    _set_optimizer_base_lr(
        optimizer,
        base_lr=initial_lr,
        scales=lr_scales,
    )
    return prepared_opts, optimizer, lr_scales


def _record_non_finite_loss_step(
    *,
    state: PriorTrainingState,
    optimizer: torch.optim.Optimizer,
    run: Any,
    history_path: Path | None,
    train_start: float,
    grad_clip: float,
) -> None:
    state.nan_skip_count += 1
    optimizer.zero_grad(set_to_none=True)
    log_wandb_metrics(
        run,
        {
            "train/nan_guard_triggered": True,
            "train/nan_skip_count": float(state.nan_skip_count),
        },
        step=state.global_step,
    )
    history_payload = history_record(
        global_step=state.global_step,
        stage_name=_PRIOR_STAGE_NAME,
        train_loss=float("nan"),
        train_metrics={"nan_guard_triggered": 1.0},
        lr=float(optimizer.param_groups[0]["lr"]),
        grad_norm=None,
        elapsed_seconds=time.perf_counter() - train_start,
        train_elapsed_seconds=state.train_elapsed_seconds,
        val_metrics=None,
        train_loss_delta=None,
        train_loss_ema=state.loss_ema,
        grad_clip_threshold=float(grad_clip),
        grad_clip_triggered=False,
    )
    state.history_records.append(history_payload)
    if history_path is not None:
        append_history_record(history_path, history_payload)


def _record_non_finite_grad_step(
    *,
    state: PriorTrainingState,
    optimizer: torch.optim.Optimizer,
    run: Any,
    history_path: Path | None,
    gradient_path: Path,
    train_start: float,
    grad_clip: float,
    pre_clip_module_grad_norms: Mapping[str, float],
    activation_norms: Mapping[str, float] | None,
    global_grad_norm_kind: str,
) -> None:
    state.nan_skip_count += 1
    optimizer.zero_grad(set_to_none=True)
    elapsed_seconds = time.perf_counter() - train_start
    log_wandb_metrics(
        run,
        {
            "train/non_finite_grad_guard_triggered": True,
            "train/non_finite_grad_kind": global_grad_norm_kind,
            "train/nan_skip_count": float(state.nan_skip_count),
        },
        step=state.global_step,
    )
    history_payload = history_record(
        global_step=state.global_step,
        stage_name=_PRIOR_STAGE_NAME,
        train_loss=float("nan"),
        train_metrics={"non_finite_grad_guard_triggered": 1.0},
        lr=float(optimizer.param_groups[0]["lr"]),
        grad_norm=None,
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        val_metrics=None,
        train_loss_delta=None,
        train_loss_ema=state.loss_ema,
        grad_clip_threshold=float(grad_clip),
        grad_clip_triggered=False,
    )
    state.history_records.append(history_payload)
    if history_path is not None:
        append_history_record(history_path, history_payload)

    gradient_payload = gradient_history_record(
        global_step=state.global_step,
        stage_name=_PRIOR_STAGE_NAME,
        train_loss=float("nan"),
        train_acc=None,
        lr=float(optimizer.param_groups[0]["lr"]),
        global_grad_norm=None,
        global_grad_norm_kind=global_grad_norm_kind,
        module_grad_norms=pre_clip_module_grad_norms,
        activation_norms=activation_norms,
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        grad_clip_threshold=float(grad_clip),
        grad_clip_triggered=False,
    )
    state.gradient_records.append(gradient_payload)
    append_jsonl_record(gradient_path, gradient_payload)


def _capture_successful_step(
    state: PriorTrainingState,
    *,
    history_step_loss: float,
    history_step_metrics: Mapping[str, float],
    local_grad_norm: float,
) -> None:
    state.final_train_loss = float(history_step_loss)
    state.final_train_acc = (
        None if history_step_metrics.get("acc") is None else float(history_step_metrics["acc"])
    )
    state.final_train_bpc = (
        None if history_step_metrics.get("bpc") is None else float(history_step_metrics["bpc"])
    )
    state.final_train_bpf = (
        None if history_step_metrics.get("bpf") is None else float(history_step_metrics["bpf"])
    )
    state.final_grad_norm = float(local_grad_norm)
    state.grad_norm_sum += state.final_grad_norm
    state.grad_norm_count += 1
    state.max_grad_norm = max(state.max_grad_norm, state.final_grad_norm)


def _append_module_balance_metrics(
    train_log: dict[str, Any],
    *,
    pre_clip_module_grad_norms: Mapping[str, float],
) -> None:
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


def _prior_train_log_payload(
    *,
    state: PriorTrainingState,
    history_step_loss: float,
    history_step_metrics: Mapping[str, float],
    optimizer: torch.optim.Optimizer,
    elapsed_seconds: float,
    grad_clip: float,
    grad_clip_triggered: bool,
    prior_dump_missingness: Mapping[str, Any],
    synthetic_prior_missingness: Mapping[str, Any],
    microbatch_size_used: int,
    microbatch_count: int,
    pre_clip_module_grad_norms: Mapping[str, float],
    activation_norms: Mapping[str, float] | None,
    loss_delta_value: float | None,
) -> dict[str, Any]:
    train_log: dict[str, Any] = {
        "train/loss": float(history_step_loss),
        "train/lr": float(optimizer.param_groups[0]["lr"]),
        "train/grad_norm": float(state.final_grad_norm),
        "train/loss_delta": loss_delta_value,
        "train/loss_ema": state.loss_ema,
        "train/elapsed_seconds": float(elapsed_seconds),
        "train/train_elapsed_seconds": float(state.train_elapsed_seconds),
        "train/grad_clip_threshold": float(grad_clip),
        "train/grad_clip_triggered": grad_clip_triggered,
        "train/grad_clip_count_so_far": int(state.clipped_step_count),
        "train/grad_clip_fraction_so_far": float(state.clipped_step_count / state.global_step),
        "train/prior_dump_skipped_batch_count": int(prior_dump_missingness["skipped_batch_count"]),
        "train/prior_dump_non_finite_feature_count": int(prior_dump_missingness["non_finite_feature_count"]),
        "train/prior_dump_non_finite_label_count": int(prior_dump_missingness["non_finite_label_count"]),
        "train/synthetic_prior_masked_feature_count": int(synthetic_prior_missingness["masked_feature_count"]),
        "train/prior_dump_microbatch_size": int(microbatch_size_used),
        "train/prior_dump_microbatch_count": int(microbatch_count),
        "train/stage": _PRIOR_STAGE_NAME,
    }
    for metric_name, metric_value in history_step_metrics.items():
        train_log[f"train/{metric_name}"] = float(metric_value)
    for module_name, module_value in pre_clip_module_grad_norms.items():
        train_log[f"train/module_grad_norm/{module_name}"] = float(module_value)
    _append_module_balance_metrics(
        train_log,
        pre_clip_module_grad_norms=pre_clip_module_grad_norms,
    )
    if activation_norms is not None:
        for activation_name, activation_value in activation_norms.items():
            train_log[f"train/activation_norm/{activation_name}"] = float(activation_value)
    return train_log


def _append_step_records(
    *,
    state: PriorTrainingState,
    history_path: Path | None,
    gradient_path: Path,
    history_step_loss: float,
    history_step_metrics: Mapping[str, float],
    optimizer: torch.optim.Optimizer,
    elapsed_seconds: float,
    grad_clip: float,
    grad_clip_triggered: bool,
    pre_clip_module_grad_norms: Mapping[str, float],
    activation_norms: Mapping[str, float] | None,
    loss_delta_value: float | None,
) -> None:
    history_payload = history_record(
        global_step=state.global_step,
        stage_name=_PRIOR_STAGE_NAME,
        train_loss=history_step_loss,
        train_metrics=history_step_metrics,
        lr=float(optimizer.param_groups[0]["lr"]),
        grad_norm=state.final_grad_norm,
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        val_metrics=None,
        train_loss_delta=loss_delta_value,
        train_loss_ema=state.loss_ema,
        grad_clip_threshold=float(grad_clip),
        grad_clip_triggered=grad_clip_triggered,
    )
    state.history_records.append(history_payload)
    if history_path is not None:
        append_history_record(history_path, history_payload)

    gradient_payload = gradient_history_record(
        global_step=state.global_step,
        stage_name=_PRIOR_STAGE_NAME,
        train_loss=history_step_loss,
        train_acc=None if history_step_metrics.get("acc") is None else history_step_metrics["acc"],
        lr=float(optimizer.param_groups[0]["lr"]),
        global_grad_norm=state.final_grad_norm,
        module_grad_norms=pre_clip_module_grad_norms,
        activation_norms=activation_norms,
        elapsed_seconds=elapsed_seconds,
        train_elapsed_seconds=state.train_elapsed_seconds,
        grad_clip_threshold=float(grad_clip),
        grad_clip_triggered=grad_clip_triggered,
    )
    state.gradient_records.append(gradient_payload)
    append_jsonl_record(gradient_path, gradient_payload)


def _maybe_save_snapshot_checkpoint(
    *,
    state: PriorTrainingState,
    checkpoint_every: int,
    output_dir: Path,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    model: torch.nn.Module,
    cfg: DictConfig,
) -> None:
    if state.global_step % checkpoint_every != 0:
        return
    checkpoint_path = output_dir / "checkpoints" / f"step_{state.global_step:06d}.pt"
    _save_eval_mode_artifact(
        prepared_opts,
        path=checkpoint_path,
        model=model,
        global_step=state.global_step,
        cfg=cfg,
        restore_training=True,
    )
    state.checkpoint_snapshots.append(
        {
            "step": int(state.global_step),
            "path": str(checkpoint_path.resolve()),
            "elapsed_seconds": float(state.train_elapsed_seconds),
            "train_elapsed_seconds": float(state.train_elapsed_seconds),
        }
    )


def _maybe_print_progress(
    *,
    state: PriorTrainingState,
    eval_every: int,
    history_step_loss: float,
    history_step_metrics: Mapping[str, float],
) -> None:
    if state.global_step % eval_every != 0:
        return
    if history_step_metrics.get("bpc") is not None:
        primary_metric_fragment = f"bpc {history_step_metrics['bpc']:7.4f}"
    elif history_step_metrics.get("acc") is not None:
        primary_metric_fragment = f"acc {history_step_metrics['acc']:7.4f}"
    else:
        primary_metric_fragment = "metric     n/a"
    print(
        f"time {state.train_elapsed_seconds:7.1f}s | "
        f"step {state.global_step:4d} | "
        f"loss {history_step_loss:7.4f} | "
        f"{primary_metric_fragment}"
    )


def _save_latest_checkpoint(
    *,
    state: PriorTrainingState,
    output_dir: Path,
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    model: torch.nn.Module,
    cfg: DictConfig,
    artifacts: dict[str, Any],
) -> None:
    state.latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
    _save_eval_mode_artifact(
        prepared_opts,
        path=state.latest_checkpoint,
        model=model,
        global_step=state.global_step,
        cfg=cfg,
        restore_training=False,
    )
    artifacts["latest_checkpoint"] = str(state.latest_checkpoint.resolve())


def _build_prior_telemetry_payload(
    *,
    state: PriorTrainingState,
    output_dir: Path,
    task: str,
    loss_surface: str,
    training_surface_payload: dict[str, Any] | None,
    artifacts: Mapping[str, Any],
    missingness_summary: dict[str, Any] | None,
    run: Any,
    cfg: DictConfig,
    device: torch.device,
    train_start: float,
    success: bool,
    error: Exception | None = None,
) -> tuple[dict[str, Any], float]:
    wall_elapsed_seconds = time.perf_counter() - train_start
    hardware_summary = build_hardware_summary(device)
    runtime_summary = build_runtime_summary(
        train_elapsed_seconds=state.train_elapsed_seconds,
        wall_elapsed_seconds=wall_elapsed_seconds,
        examples_seen=state.examples_seen,
        tokens_seen=state.tokens_seen,
        peak_memory_summary=peak_device_memory_summary(device),
        total_device_vram_bytes=(
            None
            if not isinstance(hardware_summary, Mapping)
            else hardware_summary.get("total_device_vram_bytes")
        ),
    )
    regime_budget = build_regime_budget_summary(
        task=task,
        loss_surface=loss_surface,
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
        hardware_summary=hardware_summary,
        regime_budget=regime_budget,
        training_shape_summary=training_shape_summary_from_signature_task_counts(
            state.signature_task_counts
        ),
        missingness=missingness_summary,
        training_surface_record=training_surface_payload,
        wandb=wandb_identity_payload(run, cfg=cfg),
        error=error,
    )
    return telemetry_payload, wall_elapsed_seconds


def _update_prior_wandb_summary(
    *,
    run: Any,
    output_dir: Path,
    global_step: int,
    telemetry_payload: dict[str, Any],
) -> None:
    update_wandb_summary(
        run,
        _prior_wandb_summary_payload(
            output_dir=output_dir,
            global_step=global_step,
            telemetry_payload=telemetry_payload,
        ),
    )


def run_prior_training(
    cfg: DictConfig,
    *,
    prior_dump_path: Path,
    device_name: str,
    model: torch.nn.Module,
    loss_surface: str,
    optimizer_selection: Any,
    training_surface_raw_cfg: Mapping[str, Any],
    max_steps: int,
    eval_every: int,
    checkpoint_every: int,
    grad_clip: float,
    trace_activations: bool,
    prior_batch_config,
    prior_stage,
    lr_min: float,
    initial_lr: float,
    prior_missingness_config,
    prior_dump_non_finite_policy: PriorDumpNonFinitePolicy,
    spec,
) -> TrainResult:
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_path_from_cfg(cfg)
    assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = gradient_history_path(output_dir)
    telemetry_output_path = telemetry_path(output_dir)

    device = torch.device(device_name)
    enable_activation_trace = getattr(model, "enable_activation_trace", None)
    flush_activation_trace = getattr(model, "flush_activation_trace", None)
    if trace_activations and callable(enable_activation_trace):
        enable_activation_trace()
    model.to(device)
    model.train()
    classification_z_loss_coeff = resolve_classification_z_loss_coeff(
        getattr(cfg, "training", None)
    )

    training_surface_path = output_dir / "training_surface_record.json"
    run = None
    training_surface_payload: dict[str, Any] | None = None
    artifacts: dict[str, Any] = {
        "train_history_jsonl": None if history_path is None else str(history_path),
        "gradient_history_jsonl": str(gradient_path),
        "telemetry_json": str(telemetry_output_path),
        "training_surface_record_json": str(training_surface_path.resolve()),
        "checkpoints_dir": str((output_dir / "checkpoints").resolve()),
        "latest_checkpoint": None,
    }
    missingness_summary: dict[str, Any] | None = None
    state = PriorTrainingState()
    train_start = time.perf_counter()

    prior_missingness_generator = None
    if prior_missingness_config is not None:
        prior_missingness_generator = torch.Generator(device="cpu")
        prior_missingness_generator.manual_seed(int(cfg.runtime.seed))

    def _record_non_finite_batch(batch_missingness: Any) -> None:
        if missingness_summary is None:
            raise RuntimeError("prior training setup did not initialize missingness state")
        _accumulate_missingness(
            missingness_summary,
            batch_missingness=batch_missingness,
            skipped=prior_dump_non_finite_policy == "skip",
        )

    try:
        training_surface_payload = write_training_surface_record(
            training_surface_path,
            raw_cfg=training_surface_raw_cfg,
            run_dir=output_dir,
            backend=TRAINING_BACKEND_LEGACY_PRIOR,
        )
        run = init_wandb_run(
            cfg,
            enabled=bool(getattr(cfg.logging, "use_wandb", False)),
        )
        update_wandb_summary(
            run,
            training_surface_wandb_summary_payload(training_surface_payload),
        )
        reset_peak_device_memory_stats(device)
        missingness_summary = _initial_missingness_summary(
            prior_dump_path,
            prior_missingness_config=prior_missingness_config,
            prior_dump_non_finite_policy=prior_dump_non_finite_policy,
        )
        prepared_opts, optimizer, lr_scales = _prepare_prior_optimizer(
            optimizer_selection=optimizer_selection,
            initial_lr=initial_lr,
        )
        reader = PriorDumpTaskBatchReader(
            prior_dump_path,
            num_steps=max_steps,
            batch_size=prior_batch_config.batch_size,
            non_finite_policy=prior_dump_non_finite_policy,
            require_feature_types=_requires_explicit_feature_types(str(spec.arch)),
            on_non_finite_batch=_record_non_finite_batch,
        )

        for prior_step in reader:
            if missingness_summary is None:
                raise RuntimeError("prior training setup did not initialize missingness state")
            if prior_step.missingness is not None:
                _accumulate_missingness(
                    missingness_summary,
                    batch_missingness=prior_step.missingness,
                )
            step_train_start = time.perf_counter()
            if prior_stage is not None:
                current_base_lr = float(
                    stage_base_lr(prior_stage, step=int(prior_step.step_index), lr_min=lr_min)
                )
                _set_optimizer_base_lr(
                    optimizer,
                    base_lr=current_base_lr,
                    scales=lr_scales,
                )

            x_batch, y_train_batch, y_all_batch, feature_types_batch = stack_prior_step(
                prior_step,
                device=device,
            )
            step_examples_seen = tensor_batch_examples_seen(x_batch)
            _accumulate_prior_training_shape_signature(
                state,
                x_batch=x_batch,
                train_test_split_index=prior_step.train_test_split_index,
                task_count=step_examples_seen,
            )
            step_tokens_seen = tensor_batch_token_count(x_batch)
            x_batch, synthetic_missingness = _apply_prior_missingness(
                x_batch,
                prior_step=prior_step,
                generator=prior_missingness_generator,
                prior_missingness_config=prior_missingness_config,
            )
            _accumulate_synthetic_missingness(
                missingness_summary,
                batch_missingness=synthetic_missingness,
            )
            (
                history_step_loss,
                history_step_metrics,
                activation_norms,
                microbatch_size_used,
                microbatch_count,
            ) = _run_prior_step_with_microbatch_retry(
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_train_batch=y_train_batch,
                y_all_batch=y_all_batch,
                feature_types_batch=feature_types_batch,
                train_test_split_index=prior_step.train_test_split_index,
                loss_surface=loss_surface,
                classification_z_loss_coeff=classification_z_loss_coeff,
                trace_activations=trace_activations,
                flush_activation_trace=flush_activation_trace,
            )
            state.train_elapsed_seconds += time.perf_counter() - step_train_start
            state.global_step = int(prior_step.step_index)
            state.examples_seen += step_examples_seen
            state.tokens_seen += step_tokens_seen

            if not math.isfinite(history_step_loss):
                _record_non_finite_loss_step(
                    state=state,
                    optimizer=optimizer,
                    run=run,
                    history_path=history_path,
                    train_start=train_start,
                    grad_clip=grad_clip,
                )
                continue

            pre_clip_module_grad_norms = module_grad_norms(model)
            local_grad_norm = float(total_grad_norm(model.parameters()))
            if grad_clip > 0:
                clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                local_grad_norm = float(
                    normalize_grad_norm_value(clipped, fallback=local_grad_norm)
                )
            global_grad_norm_kind = _global_grad_norm_kind(local_grad_norm)
            if global_grad_norm_kind != "finite":
                _record_non_finite_grad_step(
                    state=state,
                    optimizer=optimizer,
                    run=run,
                    history_path=history_path,
                    gradient_path=gradient_path,
                    train_start=train_start,
                    grad_clip=grad_clip,
                    pre_clip_module_grad_norms=pre_clip_module_grad_norms,
                    activation_norms=activation_norms,
                    global_grad_norm_kind=global_grad_norm_kind,
                )
                continue

            grad_clip_triggered = bool(grad_clip > 0 and local_grad_norm > grad_clip)
            if grad_clip_triggered:
                state.clipped_step_count += 1

            optimizer.step()
            _capture_successful_step(
                state,
                history_step_loss=history_step_loss,
                history_step_metrics=history_step_metrics,
                local_grad_norm=local_grad_norm,
            )
            loss_delta_value = train_loss_delta(
                history_step_loss,
                previous_train_loss=state.previous_train_loss,
            )
            state.loss_ema = update_loss_ema(history_step_loss, previous_ema=state.loss_ema)
            state.previous_train_loss = history_step_loss
            elapsed_seconds = time.perf_counter() - train_start
            prior_dump_missingness = cast(dict[str, Any], missingness_summary["prior_dump"])
            synthetic_prior_missingness = cast(dict[str, Any], missingness_summary["synthetic_prior"])
            train_log = _prior_train_log_payload(
                state=state,
                history_step_loss=history_step_loss,
                history_step_metrics=history_step_metrics,
                optimizer=optimizer,
                elapsed_seconds=elapsed_seconds,
                grad_clip=grad_clip,
                grad_clip_triggered=grad_clip_triggered,
                prior_dump_missingness=prior_dump_missingness,
                synthetic_prior_missingness=synthetic_prior_missingness,
                microbatch_size_used=microbatch_size_used,
                microbatch_count=microbatch_count,
                pre_clip_module_grad_norms=pre_clip_module_grad_norms,
                activation_norms=activation_norms,
                loss_delta_value=loss_delta_value,
            )
            log_wandb_metrics(run, train_log, step=state.global_step)
            _append_step_records(
                state=state,
                history_path=history_path,
                gradient_path=gradient_path,
                history_step_loss=history_step_loss,
                history_step_metrics=history_step_metrics,
                optimizer=optimizer,
                elapsed_seconds=elapsed_seconds,
                grad_clip=grad_clip,
                grad_clip_triggered=grad_clip_triggered,
                pre_clip_module_grad_norms=pre_clip_module_grad_norms,
                activation_norms=activation_norms,
                loss_delta_value=loss_delta_value,
            )
            _maybe_save_snapshot_checkpoint(
                state=state,
                checkpoint_every=checkpoint_every,
                output_dir=output_dir,
                prepared_opts=prepared_opts,
                model=model,
                cfg=cfg,
            )
            _maybe_print_progress(
                state=state,
                eval_every=eval_every,
                history_step_loss=history_step_loss,
                history_step_metrics=history_step_metrics,
            )

        _save_latest_checkpoint(
            state=state,
            output_dir=output_dir,
            prepared_opts=prepared_opts,
            model=model,
            cfg=cfg,
            artifacts=artifacts,
        )
        telemetry_payload, wall_elapsed_seconds = _build_prior_telemetry_payload(
            state=state,
            output_dir=output_dir,
            task=str(cfg.task),
            loss_surface=loss_surface,
            training_surface_payload=training_surface_payload,
            artifacts=artifacts,
            missingness_summary=missingness_summary,
            run=run,
            cfg=cfg,
            device=device,
            train_start=train_start,
            success=True,
        )
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        _update_prior_wandb_summary(
            run=run,
            output_dir=output_dir,
            global_step=state.global_step,
            telemetry_payload=telemetry_payload,
        )
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=None,
            latest_checkpoint=state.latest_checkpoint,
            global_step=state.global_step,
            metrics={
                "final_train_loss": None if state.final_train_loss is None else float(state.final_train_loss),
                "final_train_acc": None if state.final_train_acc is None else float(state.final_train_acc),
                "final_train_bpc": None if state.final_train_bpc is None else float(state.final_train_bpc),
                "final_train_bpf": None if state.final_train_bpf is None else float(state.final_train_bpf),
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
        )
    except Exception as exc:
        telemetry_payload, _wall_elapsed_seconds = _build_prior_telemetry_payload(
            state=state,
            output_dir=output_dir,
            task=str(cfg.task),
            loss_surface=loss_surface,
            training_surface_payload=training_surface_payload,
            artifacts=artifacts,
            missingness_summary=missingness_summary,
            run=run,
            cfg=cfg,
            device=device,
            train_start=train_start,
            success=False,
            error=exc,
        )
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        _update_prior_wandb_summary(
            run=run,
            output_dir=output_dir,
            global_step=state.global_step,
            telemetry_payload=telemetry_payload,
        )
        raise
    finally:
        finish_wandb_run(run)
