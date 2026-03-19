"""Training loop."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.types import TrainResult

from .artifacts import (
    append_history_record,
    append_jsonl_record,
    assert_clean_training_output,
    gradient_history_record,
    history_path_from_cfg,
    history_record,
    save_checkpoint,
)
from .batching import move_batch
from .distributed import _reduction_float_dtype, _reduce_keyed_weighted_scalars
from .instability import (
    build_training_telemetry,
    gradient_history_path,
    module_grad_norms,
    normalize_grad_norm_value,
    telemetry_path,
    total_grad_norm,
    train_loss_delta,
    update_loss_ema,
    write_training_telemetry,
)
from .optimizer import build_optimizer
from .runtime import build_accelerator_from_runtime
from .schedule import build_stage_configs, stage_base_lr
from .surface import write_training_surface_record
from .trainer_metrics import (
    _compute_loss_and_metrics,
    _evaluate_loader,
    _expected_metric_keys,
    cycle_loader,
)
from .trainer_optimizer import (
    _optimizer_lr_scales,
    _set_optimizer_base_lr,
    _set_optimizer_training_mode,
)
from .trainer_runtime_config import (
    _checkpoint_every,
    _resolve_grad_accum_steps,
    _resolve_max_steps,
    _resolve_target_train_seconds,
)
from .trainer_summary import _training_telemetry_summary_payload, _trainer_summary_payload
from .wandb import finish_wandb_run, init_wandb_run, log_wandb_metrics, update_wandb_summary


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


def train(cfg: DictConfig) -> TrainResult:
    """Train from config."""

    task = str(cfg.task)
    seed = int(cfg.runtime.seed)
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_path_from_cfg(cfg)
    assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = gradient_history_path(output_dir)
    telemetry_output_path = telemetry_path(output_dir)
    training_surface_path = output_dir / "training_surface_record.json"

    torch.manual_seed(seed)
    grad_accum_steps = _resolve_grad_accum_steps(cfg.runtime)
    checkpoint_every = _checkpoint_every(cfg.runtime)
    max_steps = _resolve_max_steps(cfg.runtime)
    target_train_seconds = _resolve_target_train_seconds(cfg.runtime)

    accelerator = build_accelerator_from_runtime(
        cfg.runtime,
        grad_accum_steps_override=grad_accum_steps,
    )

    train_ds = build_task_dataset(
        cfg.data,
        split="train",
        task=task,
        seed=seed,
        preprocessing_cfg=cfg.get("preprocessing"),
    )
    val_ds = build_task_dataset(
        cfg.data,
        split="val",
        task=task,
        seed=seed + 1,
        preprocessing_cfg=cfg.get("preprocessing"),
    )

    train_loader = build_task_loader(
        train_ds,
        shuffle=True,
        num_workers=int(cfg.runtime.num_workers),
        seed=seed,
    )
    val_loader = build_task_loader(
        val_ds,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        seed=seed + 1,
    )

    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg: dict[str, Any] = {}
    if isinstance(raw_model_cfg, dict):
        model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    model = build_model_from_spec(model_spec)
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    base_model = accelerator.unwrap_model(model)
    trace_activations = bool(getattr(cfg.runtime, "trace_activations", False))
    enable_activation_trace = getattr(base_model, "enable_activation_trace", None)
    flush_activation_trace = getattr(base_model, "flush_activation_trace", None)
    flush_activation_trace_stats = getattr(base_model, "flush_activation_trace_stats", None)
    requires_exact_activation_trace_stats = bool(
        trace_activations
        and (
            grad_accum_steps > 1
            or _accelerator_num_processes(accelerator) > 1
        )
    )
    if (
        requires_exact_activation_trace_stats
        and not callable(flush_activation_trace_stats)
        and callable(flush_activation_trace)
    ):
        raise RuntimeError(
            "trace_activations with grad_accum_steps > 1 or multi-process execution "
            "requires flush_activation_trace_stats()"
        )
    if trace_activations and callable(enable_activation_trace):
        enable_activation_trace()

    def _flush_activation_trace_stats() -> dict[str, tuple[float, int]] | None:
        if not trace_activations:
            return None
        if callable(flush_activation_trace_stats):
            raw_snapshot = flush_activation_trace_stats()
            if raw_snapshot is None:
                return None
            return {
                str(name): (float(total_sum_sq), int(total_count))
                for name, (total_sum_sq, total_count) in raw_snapshot.items()
                if int(total_count) > 0
            }
        if callable(flush_activation_trace):
            legacy_snapshot = flush_activation_trace()
            if legacy_snapshot is None:
                return None
            if requires_exact_activation_trace_stats:
                raise RuntimeError(
                    "trace_activations with grad_accum_steps > 1 or multi-process execution "
                    "requires flush_activation_trace_stats()"
                )
            return {
                str(name): (float(value) * float(value), 1)
                for name, value in legacy_snapshot.items()
                if math.isfinite(float(value))
            }
        return None

    training_surface_payload: dict[str, Any] | None = None
    if accelerator.is_main_process:
        raw_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        training_surface_payload = write_training_surface_record(
            training_surface_path,
            raw_cfg=raw_cfg,
            run_dir=output_dir,
        )

    run = init_wandb_run(
        cfg,
        enabled=bool(getattr(cfg.logging, "use_wandb", False) and accelerator.is_main_process),
    )

    raw_stages = cast(list[dict[str, object]], OmegaConf.to_container(cfg.schedule.stages, resolve=True))
    stage_configs = build_stage_configs(raw_stages)
    if not stage_configs:
        raise RuntimeError("schedule.stages must contain at least one stage")
    train_iter = cycle_loader(train_loader)

    first_stage = stage_configs[0]
    optimizer_sel = build_optimizer(
        accelerator.unwrap_model(model),
        name=str(cfg.optimizer.name),
        lr=first_stage.lr_max,
        weight_decay=float(cfg.optimizer.weight_decay),
        extra_kwargs={"betas": tuple(cfg.optimizer.betas)},
        require_requested=bool(cfg.optimizer.require_requested),
        muon_per_parameter_lr=bool(cfg.optimizer.muon_per_parameter_lr),
        muon_lr_scale_base=float(cfg.optimizer.muon_lr_scale_base),
        muon_partition_non2d=bool(cfg.optimizer.muon_partition_non2d),
    )
    if optimizer_sel.fallback_reason is None:
        accelerator.print(
            f"[optimizer] requested={optimizer_sel.requested_name} "
            f"resolved={optimizer_sel.resolved_name}"
        )
    else:
        accelerator.print(
            f"[optimizer] requested={optimizer_sel.requested_name} "
            f"resolved={optimizer_sel.resolved_name} fallback={optimizer_sel.fallback_reason}"
        )

    prepared_opts: list[tuple[str, torch.optim.Optimizer]] = []
    lr_scales: dict[str, list[float]] = {}
    for opt_name, opt in optimizer_sel.optimizers:
        prepared = accelerator.prepare_optimizer(opt)
        prepared_opts.append((opt_name, prepared))
        lr_scales[opt_name] = _optimizer_lr_scales(prepared, base_lr=first_stage.lr_max)

    expected_keys = _expected_metric_keys(task)

    global_step = 0
    best_checkpoint: Path | None = None
    latest_checkpoint: Path | None = None
    best_val = float("inf")
    best_val_step = 0.0
    last_val_metrics: dict[str, float] | None = None
    train_start = time.perf_counter()
    train_elapsed_seconds = 0.0
    stop_requested = False
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    final_grad_norm = 0.0
    nan_skip_count = 0
    previous_train_loss: float | None = None
    loss_ema: float | None = None
    final_train_loss: float | None = None
    final_train_loss_ema: float | None = None
    last_train_metrics: dict[str, float] | None = None
    history_records: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    checkpoint_snapshots: list[dict[str, Any]] = []

    def _artifacts_payload() -> dict[str, Any]:
        return {
            "train_history_jsonl": None if history_path is None else str(history_path),
            "gradient_history_jsonl": str(gradient_path),
            "telemetry_json": str(telemetry_output_path),
            "training_surface_record_json": str(training_surface_path.resolve()),
            "checkpoints_dir": str((output_dir / "checkpoints").resolve()),
            "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint.resolve()),
            "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint.resolve()),
        }

    try:
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
                step_train_start = time.perf_counter()
                for _micro_step in range(grad_accum_steps):
                    batch = move_batch(next(train_iter), accelerator.device)
                    with accelerator.accumulate(model):
                        with accelerator.autocast():
                            output = model(batch)
                            loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
                        accelerator.backward(loss)
                    train_loss_sum += float(loss.detach().item())
                    train_loss_count += 1
                    for key, value in metrics.items():
                        train_metric_sums[key] = train_metric_sums.get(key, 0.0) + float(value)
                        train_metric_counts[key] = train_metric_counts.get(key, 0) + 1
                    batch_activation_trace_stats = _flush_activation_trace_stats()
                    if batch_activation_trace_stats is not None:
                        for activation_name, (activation_sum_sq, activation_count) in batch_activation_trace_stats.items():
                            activation_sum_sqs[activation_name] = (
                                activation_sum_sqs.get(activation_name, 0.0)
                                + float(activation_sum_sq)
                            )
                            activation_element_counts[activation_name] = (
                                activation_element_counts.get(activation_name, 0.0)
                                + float(activation_count)
                            )

                nan_detected = not math.isfinite(train_loss_sum)
                if nan_detected:
                    _ = _flush_activation_trace_stats()
                    nan_skip_count += 1
                    for _opt_name, opt in prepared_opts:
                        opt.zero_grad(set_to_none=True)
                    train_elapsed_seconds += time.perf_counter() - step_train_start
                    global_step += 1
                    nan_log: dict[str, Any] = {
                        "train/nan_guard_triggered": True,
                        "train/nan_skip_count": float(nan_skip_count),
                    }
                    log_wandb_metrics(run, nan_log, step=global_step)
                    if accelerator.is_main_process:
                        nan_history_payload = history_record(
                            global_step=global_step,
                            stage_name=stage.name,
                            train_loss=float("nan"),
                            train_metrics={"nan_guard_triggered": 1.0},
                            lr=float(prepared_opts[0][1].param_groups[0]["lr"]),
                            grad_norm=None,
                            elapsed_seconds=time.perf_counter() - train_start,
                            train_elapsed_seconds=train_elapsed_seconds,
                            val_metrics=None,
                            train_loss_delta=None,
                            train_loss_ema=loss_ema,
                            grad_clip_threshold=float(cfg.runtime.grad_clip),
                            grad_clip_triggered=False,
                        )
                        history_records.append(nan_history_payload)
                        if history_path is not None:
                            append_history_record(history_path, nan_history_payload)
                    if max_steps is not None and global_step >= max_steps:
                        stop_requested = True
                    if target_train_seconds is not None and train_elapsed_seconds >= target_train_seconds:
                        stop_requested = True
                    if stop_requested:
                        break
                    continue

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
                pre_clip_module_grad_norms = (
                    module_grad_norms(base_model) if accelerator.is_main_process else {}
                )
                local_grad_norm = total_grad_norm(model.parameters())
                if float(cfg.runtime.grad_clip) > 0:
                    clipped = accelerator.clip_grad_norm_(model.parameters(), float(cfg.runtime.grad_clip))
                    local_grad_norm = normalize_grad_norm_value(clipped, fallback=local_grad_norm)
                train_metric_sums["grad_norm"] = train_metric_sums.get("grad_norm", 0.0) + float(local_grad_norm)
                train_metric_counts["grad_norm"] = train_metric_counts.get("grad_norm", 0) + 1

                for _opt_name, opt in prepared_opts:
                    opt.step()

                train_elapsed_seconds += time.perf_counter() - step_train_start
                global_step += 1
                metric_keys = sorted(set(train_metric_sums) | expected_keys)
                # Pack loss + metric sums/counts into one tensor for a single all-reduce.
                # Layout: [loss_sum, loss_count, key0_sum, key0_count, ...]
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
                            grad_norm_sum += grad_norm_value
                            grad_norm_count += 1
                            max_grad_norm = max(max_grad_norm, grad_norm_value)
                            final_grad_norm = grad_norm_value

                current_train_loss = float(train_loss)
                loss_delta_value = train_loss_delta(
                    current_train_loss,
                    previous_train_loss=previous_train_loss,
                )
                loss_ema = update_loss_ema(current_train_loss, previous_ema=loss_ema)
                previous_train_loss = current_train_loss
                elapsed_seconds = time.perf_counter() - train_start
                grad_clip_threshold = float(cfg.runtime.grad_clip)
                grad_clip_triggered = bool(
                    grad_clip_threshold > 0
                    and math.isfinite(grad_norm_value)
                    and float(grad_norm_value) > grad_clip_threshold
                )
                global_grad_norm_kind = _global_grad_norm_kind(float(grad_norm_value))
                train_log["train/loss_delta"] = loss_delta_value
                train_log["train/loss_ema"] = loss_ema
                train_log["train/elapsed_seconds"] = elapsed_seconds
                train_log["train/train_elapsed_seconds"] = train_elapsed_seconds
                train_log["train/grad_clip_threshold"] = grad_clip_threshold
                train_log["train/grad_clip_triggered"] = grad_clip_triggered
                if accelerator.is_main_process:
                    for module_name, module_value in pre_clip_module_grad_norms.items():
                        train_log[f"train/module_grad_norm/{module_name}"] = float(module_value)
                    feature_grad = pre_clip_module_grad_norms.get("feature_encoder")
                    head_grad = pre_clip_module_grad_norms.get("direct_head")
                    if feature_grad is not None and float(feature_grad) > 0.0 and head_grad is not None:
                        train_log["train/module_balance/direct_head_to_feature_encoder"] = float(head_grad) / float(feature_grad)
                    if head_grad is not None and float(head_grad) > 0.0 and feature_grad is not None:
                        train_log["train/module_balance/feature_encoder_to_direct_head"] = float(feature_grad) / float(head_grad)
                    if activation_norms is not None:
                        for activation_name, activation_value in activation_norms.items():
                            train_log[f"train/activation_norm/{activation_name}"] = float(activation_value)
                final_train_loss = current_train_loss
                final_train_loss_ema = loss_ema
                last_train_metrics = current_train_metrics
                log_wandb_metrics(run, train_log, step=global_step)

                history_val_metrics: dict[str, float] | None = None
                if global_step % int(cfg.runtime.eval_every) == 0:
                    _ = _flush_activation_trace_stats()
                    _set_optimizer_training_mode(prepared_opts, training=False)
                    val_metrics = _evaluate_loader(
                        model,
                        val_loader,
                        accelerator=accelerator,
                        task=task,
                        max_batches=int(cfg.runtime.val_batches),
                    )
                    log_wandb_metrics(
                        run,
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=global_step,
                    )
                    history_val_metrics = val_metrics
                    last_val_metrics = val_metrics
                    _ = _flush_activation_trace_stats()

                    if val_metrics["val_loss"] < best_val:
                        best_val = val_metrics["val_loss"]
                        best_val_step = float(global_step)
                        best_checkpoint = output_dir / "checkpoints" / "best.pt"
                        if accelerator.is_main_process:
                            save_checkpoint(
                                best_checkpoint,
                                model_state=accelerator.get_state_dict(model),
                                global_step=global_step,
                                cfg=cfg,
                            )
                    _set_optimizer_training_mode(prepared_opts, training=True)

                if checkpoint_every is not None and global_step % checkpoint_every == 0:
                    snapshot_checkpoint = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                    if accelerator.is_main_process:
                        save_checkpoint(
                            snapshot_checkpoint,
                            model_state=accelerator.get_state_dict(model),
                            global_step=global_step,
                            cfg=cfg,
                        )
                        checkpoint_snapshots.append(
                            {
                                "step": int(global_step),
                                "path": str(snapshot_checkpoint.resolve()),
                                "elapsed_seconds": float(train_elapsed_seconds),
                                "train_elapsed_seconds": float(train_elapsed_seconds),
                            }
                        )

                if accelerator.is_main_process:
                    train_metrics_for_history = {
                        key.removeprefix("train/"): float(value)
                        for key, value in train_log.items()
                        if key.startswith("train/")
                        and isinstance(value, (int, float))
                        and key != "train/lr"
                        and math.isfinite(float(value))
                    }
                    history_payload = history_record(
                        global_step=global_step,
                        stage_name=stage.name,
                        train_loss=float(train_loss),
                        train_metrics=train_metrics_for_history,
                        lr=float(first_lr),
                        grad_norm=None if not math.isfinite(grad_norm_value) else float(grad_norm_value),
                        elapsed_seconds=elapsed_seconds,
                        train_elapsed_seconds=train_elapsed_seconds,
                        val_metrics=history_val_metrics,
                        train_loss_delta=loss_delta_value,
                        train_loss_ema=loss_ema,
                        grad_clip_threshold=grad_clip_threshold,
                        grad_clip_triggered=grad_clip_triggered,
                    )
                    history_records.append(history_payload)
                    if history_path is not None:
                        append_history_record(history_path, history_payload)

                    gradient_payload = gradient_history_record(
                        global_step=global_step,
                        stage_name=stage.name,
                        train_loss=float(train_loss),
                        train_acc=current_train_metrics.get("acc"),
                        lr=float(first_lr),
                        global_grad_norm=None
                        if not math.isfinite(grad_norm_value)
                        else float(grad_norm_value),
                        global_grad_norm_kind=global_grad_norm_kind,
                        module_grad_norms=pre_clip_module_grad_norms,
                        activation_norms=activation_norms,
                        elapsed_seconds=elapsed_seconds,
                        train_elapsed_seconds=train_elapsed_seconds,
                        grad_clip_threshold=grad_clip_threshold,
                        grad_clip_triggered=grad_clip_triggered,
                    )
                    gradient_records.append(gradient_payload)
                    append_jsonl_record(gradient_path, gradient_payload)

                if max_steps is not None and global_step >= max_steps:
                    stop_requested = True
                if target_train_seconds is not None and train_elapsed_seconds >= target_train_seconds:
                    stop_requested = True
                if stop_requested:
                    break

            latest_checkpoint = output_dir / "checkpoints" / f"latest_{stage.name}.pt"
            if accelerator.is_main_process:
                save_checkpoint(
                    latest_checkpoint,
                    model_state=accelerator.get_state_dict(model),
                    global_step=global_step,
                    cfg=cfg,
                )
            if stop_requested:
                break

        accelerator.wait_for_everyone()

        wall_elapsed_seconds = time.perf_counter() - train_start
        if best_checkpoint is None and latest_checkpoint is not None:
            best_checkpoint = output_dir / "checkpoints" / "best.pt"
            if accelerator.is_main_process:
                save_checkpoint(
                    best_checkpoint,
                    model_state=accelerator.get_state_dict(model),
                    global_step=global_step,
                    cfg=cfg,
                )

        result = TrainResult(
            output_dir=output_dir,
            best_checkpoint=best_checkpoint,
            latest_checkpoint=latest_checkpoint,
            global_step=global_step,
            metrics={
                "best_val_loss": float(best_val),
                "best_val_step": float(best_val_step),
                "final_val_loss": float(last_val_metrics["val_loss"])
                if last_val_metrics is not None
                else float(best_val),
                "final_grad_norm": float(final_grad_norm),
                "mean_grad_norm": float(grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else 0.0,
                "max_grad_norm": float(max_grad_norm),
                "train_elapsed_seconds": float(train_elapsed_seconds),
                "wall_elapsed_seconds": float(wall_elapsed_seconds),
                "nan_skip_count": float(nan_skip_count),
            },
        )
        if accelerator.is_main_process:
            telemetry_payload = build_training_telemetry(
                run_dir=output_dir,
                success=True,
                artifacts=_artifacts_payload(),
                checkpoint_snapshots=checkpoint_snapshots,
                history_records=history_records,
                gradient_records=gradient_records,
                training_surface_record=training_surface_payload,
            )
            write_training_telemetry(telemetry_output_path, telemetry_payload)
            update_wandb_summary(
                run,
                _training_telemetry_summary_payload(telemetry_payload=telemetry_payload),
            )
        update_wandb_summary(
            run,
            _trainer_summary_payload(
                output_dir=output_dir,
                optimizer_requested_name=optimizer_sel.requested_name,
                optimizer_resolved_name=optimizer_sel.resolved_name,
                optimizer_fallback_reason=optimizer_sel.fallback_reason,
                global_step=global_step,
                best_checkpoint=best_checkpoint,
                latest_checkpoint=latest_checkpoint,
                best_val=best_val,
                best_val_step=best_val_step,
                final_train_loss=final_train_loss,
                final_train_loss_ema=final_train_loss_ema,
                last_train_metrics=last_train_metrics,
                last_val_metrics=last_val_metrics,
                final_grad_norm=final_grad_norm,
                grad_norm_sum=grad_norm_sum,
                grad_norm_count=grad_norm_count,
                max_grad_norm=max_grad_norm,
                train_elapsed_seconds=train_elapsed_seconds,
                wall_elapsed_seconds=wall_elapsed_seconds,
                nan_skip_count=nan_skip_count,
            ),
        )
        return result
    except Exception as exc:
        if accelerator.is_main_process:
            telemetry_payload = build_training_telemetry(
                run_dir=output_dir,
                success=False,
                artifacts=_artifacts_payload(),
                checkpoint_snapshots=checkpoint_snapshots,
                history_records=history_records,
                gradient_records=gradient_records,
                training_surface_record=training_surface_payload,
                error=exc,
            )
            write_training_telemetry(telemetry_output_path, telemetry_payload)
            update_wandb_summary(
                run,
                _training_telemetry_summary_payload(telemetry_payload=telemetry_payload),
            )
        update_wandb_summary(
            run,
            _trainer_summary_payload(
                output_dir=output_dir,
                optimizer_requested_name=optimizer_sel.requested_name,
                optimizer_resolved_name=optimizer_sel.resolved_name,
                optimizer_fallback_reason=optimizer_sel.fallback_reason,
                global_step=global_step,
                best_checkpoint=best_checkpoint,
                latest_checkpoint=latest_checkpoint,
                best_val=best_val,
                best_val_step=best_val_step,
                final_train_loss=final_train_loss,
                final_train_loss_ema=final_train_loss_ema,
                last_train_metrics=last_train_metrics,
                last_val_metrics=last_val_metrics,
                final_grad_norm=final_grad_norm,
                grad_norm_sum=grad_norm_sum,
                grad_norm_count=grad_norm_count,
                max_grad_norm=max_grad_norm,
                train_elapsed_seconds=train_elapsed_seconds,
                wall_elapsed_seconds=time.perf_counter() - train_start,
                nan_skip_count=nan_skip_count,
                error=exc,
            ),
        )
        raise
    finally:
        finish_wandb_run(run)
