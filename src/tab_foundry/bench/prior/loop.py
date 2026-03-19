"""Training loop helpers for exact prior-dump training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.types import TrainResult


_PRIOR_STAGE_NAME = "prior_dump"


@dataclass(frozen=True)
class PriorTrainingDeps:
    resolve_prior_training_device_name: Any
    history_path_from_cfg: Any
    assert_clean_training_output: Any
    gradient_history_path: Any
    telemetry_path: Any
    build_model_from_spec: Any
    write_training_surface_record: Any
    init_wandb_run: Any
    finish_wandb_run: Any
    log_wandb_metrics: Any
    update_prior_wandb_summary: Any
    initial_missingness_summary: Any
    build_optimizer: Any
    optimizer_kwargs: Any
    set_optimizer_training_mode: Any
    set_optimizer_base_lr: Any
    stage_base_lr: Any
    accumulate_missingness: Any
    apply_prior_missingness: Any
    accumulate_synthetic_missingness: Any
    prior_dump_task_batch_reader: Any
    stack_prior_step: Any
    classification_loss: Any
    module_grad_norms: Any
    total_grad_norm: Any
    normalize_grad_norm_value: Any
    train_loss_delta: Any
    update_loss_ema: Any
    history_record: Any
    append_history_record: Any
    gradient_history_record: Any
    append_jsonl_record: Any
    save_eval_mode_checkpoint: Any
    build_training_telemetry: Any
    write_training_telemetry: Any


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
    deps: PriorTrainingDeps,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_batch: torch.Tensor,
    y_train_batch: torch.Tensor,
    y_all_batch: torch.Tensor,
    train_test_split_index: int,
    trace_activations: bool,
    flush_activation_trace: object,
) -> tuple[float, float, dict[str, float] | None, int, int]:
    forward_batched = getattr(model, "forward_batched", None)
    if not callable(forward_batched):
        raise RuntimeError("prior-dump training requires a model with forward_batched()")

    total_batch_size = int(x_batch.shape[0])
    if total_batch_size <= 0:
        raise RuntimeError("prior-dump training requires batch_size >= 1")

    def _attempt(microbatch_size: int) -> tuple[float, float, dict[str, float] | None, int]:
        weighted_loss = 0.0
        weighted_acc = 0.0
        activation_weighted_sums: dict[str, float] = {}
        activation_weight_totals: dict[str, float] = {}
        microbatch_count = 0
        for start in range(0, total_batch_size, microbatch_size):
            stop = min(start + microbatch_size, total_batch_size)
            microbatch_count += 1
            weight = float(stop - start) / float(total_batch_size)
            logits = forward_batched(
                x_all=x_batch[start:stop],
                y_train=y_train_batch[start:stop],
                train_test_split_index=train_test_split_index,
            )
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("prior-dump training requires tensor logits")
            activation_norms = (
                flush_activation_trace()
                if trace_activations and callable(flush_activation_trace)
                else None
            )
            targets = y_all_batch[start:stop, train_test_split_index:].reshape(-1).to(torch.int64)
            loss = deps.classification_loss(
                logits.reshape(-1, int(logits.shape[-1])),
                targets,
            )
            (loss * weight).backward()
            weighted_loss += float(loss.detach().item()) * weight
            weighted_acc += (
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
            weighted_acc,
            _merge_activation_norms(activation_weighted_sums, activation_weight_totals),
            microbatch_count,
        )

    microbatch_size = total_batch_size
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, acc_value, activation_norms, microbatch_count = _attempt(microbatch_size)
            return (
                loss_value,
                acc_value,
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


def run_prior_training(
    cfg: DictConfig,
    *,
    prior_dump_path: Path,
    spec,
    staged_surface,
    max_steps: int,
    eval_every: int,
    checkpoint_every: int,
    grad_clip: float,
    trace_activations: bool,
    prior_batch_config,
    prior_stage,
    lr_min: float,
    prior_missingness_config,
    prior_dump_non_finite_policy: str,
    deps: PriorTrainingDeps,
) -> TrainResult:
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = deps.history_path_from_cfg(cfg)
    deps.assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = deps.gradient_history_path(output_dir)
    telemetry_output_path = deps.telemetry_path(output_dir)

    device = torch.device(
        deps.resolve_prior_training_device_name(
            cfg,
            spec=spec,
            staged_surface=staged_surface,
        )
    )
    model = deps.build_model_from_spec(spec)
    enable_activation_trace = getattr(model, "enable_activation_trace", None)
    flush_activation_trace = getattr(model, "flush_activation_trace", None)
    if trace_activations and callable(enable_activation_trace):
        enable_activation_trace()
    model.to(device)
    model.train()

    raw_cfg = cast(dict[str, object], OmegaConf.to_container(cfg, resolve=True))
    raw_training_cfg = raw_cfg.get("training")
    if not isinstance(raw_training_cfg, dict):
        raw_training_cfg = {}
        raw_cfg["training"] = raw_training_cfg
    raw_training_cfg["prior_dump_batch_size"] = int(prior_batch_config.batch_size)
    raw_training_cfg["prior_dump_lr_scale_rule"] = str(prior_batch_config.lr_scale_rule)
    raw_training_cfg["prior_dump_batch_reference_size"] = int(prior_batch_config.reference_batch_size)
    raw_training_cfg["effective_lr_scale_factor"] = float(prior_batch_config.effective_lr_scale_factor)
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    if isinstance(raw_optimizer_cfg, dict) and raw_optimizer_cfg.get("min_lr") is not None:
        raw_optimizer_cfg["min_lr"] = float(lr_min)
    raw_schedule_cfg = raw_cfg.get("schedule")
    if isinstance(raw_schedule_cfg, dict):
        raw_stages = raw_schedule_cfg.get("stages")
        if isinstance(raw_stages, list):
            for stage in raw_stages:
                if isinstance(stage, dict) and stage.get("lr_max") is not None:
                    stage["lr_max"] = float(stage["lr_max"]) * float(prior_batch_config.effective_lr_scale_factor)
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
    history_records: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    checkpoint_snapshots: list[dict[str, Any]] = []
    missingness_summary: dict[str, Any] | None = None

    initial_lr = (
        float(deps.stage_base_lr(prior_stage, step=1, lr_min=lr_min))
        if prior_stage is not None
        else float(lr_min)
    )

    prepared_opts: list[tuple[str, torch.optim.Optimizer]] = []
    optimizer: torch.optim.Optimizer | None = None
    lr_scales: list[float] = []

    history_step_loss = 0.0
    history_step_acc = 0.0
    global_step = 0
    latest_checkpoint: Path | None = None
    final_grad_norm = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    clipped_step_count = 0
    train_start = time.perf_counter()
    train_elapsed_seconds = 0.0
    previous_train_loss: float | None = None
    loss_ema: float | None = None
    prior_missingness_generator = None
    if prior_missingness_config is not None:
        prior_missingness_generator = torch.Generator(device="cpu")
        prior_missingness_generator.manual_seed(int(cfg.runtime.seed))

    def _record_non_finite_batch(batch_missingness) -> None:
        if missingness_summary is None:
            raise RuntimeError("prior training setup did not initialize missingness state")
        deps.accumulate_missingness(
            missingness_summary,
            batch_missingness=batch_missingness,
            skipped=prior_dump_non_finite_policy == "skip",
        )

    try:
        training_surface_payload = deps.write_training_surface_record(
            training_surface_path,
            raw_cfg=raw_cfg,
            run_dir=output_dir,
        )
        run = deps.init_wandb_run(
            cfg,
            enabled=bool(getattr(cfg.logging, "use_wandb", False)),
        )
        missingness_summary = deps.initial_missingness_summary(
            prior_dump_path,
            prior_missingness_config=prior_missingness_config,
            prior_dump_non_finite_policy=prior_dump_non_finite_policy,
        )

        optimizer_selection = deps.build_optimizer(
            model,
            name=str(cfg.optimizer.name),
            lr=initial_lr,
            weight_decay=float(cfg.optimizer.weight_decay),
            extra_kwargs=deps.optimizer_kwargs(cfg),
            require_requested=bool(cfg.optimizer.require_requested),
            muon_per_parameter_lr=bool(getattr(cfg.optimizer, "muon_per_parameter_lr", True)),
            muon_lr_scale_base=float(getattr(cfg.optimizer, "muon_lr_scale_base", 0.2)),
            muon_partition_non2d=bool(getattr(cfg.optimizer, "muon_partition_non2d", True)),
        )
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
        deps.set_optimizer_training_mode(prepared_opts, training=True)
        lr_scales = [1.0 for _ in optimizer.param_groups]
        deps.set_optimizer_base_lr(
            optimizer,
            base_lr=initial_lr,
            scales=lr_scales,
        )

        reader = deps.prior_dump_task_batch_reader(
            prior_dump_path,
            num_steps=max_steps,
            batch_size=prior_batch_config.batch_size,
            non_finite_policy=prior_dump_non_finite_policy,
            on_non_finite_batch=_record_non_finite_batch,
        )

        for prior_step in reader:
            if optimizer is None or missingness_summary is None:
                raise RuntimeError("prior training setup did not initialize optimizer and missingness state")
            if prior_step.missingness is not None:
                deps.accumulate_missingness(
                    missingness_summary,
                    batch_missingness=prior_step.missingness,
                )
            step_train_start = time.perf_counter()
            if prior_stage is not None:
                current_base_lr = float(
                    deps.stage_base_lr(prior_stage, step=int(prior_step.step_index), lr_min=lr_min)
                )
                deps.set_optimizer_base_lr(
                    optimizer,
                    base_lr=current_base_lr,
                    scales=lr_scales,
                )
            forward_batched = getattr(model, "forward_batched", None)
            if not callable(forward_batched):
                raise RuntimeError("prior-dump training requires a model with forward_batched()")
            x_batch, y_train_batch, y_all_batch = deps.stack_prior_step(prior_step, device=device)
            x_batch, synthetic_missingness = deps.apply_prior_missingness(
                x_batch,
                prior_step=prior_step,
                generator=prior_missingness_generator,
                prior_missingness_config=prior_missingness_config,
            )
            deps.accumulate_synthetic_missingness(
                missingness_summary,
                batch_missingness=synthetic_missingness,
            )
            (
                history_step_loss,
                history_step_acc,
                activation_norms,
                microbatch_size_used,
                microbatch_count,
            ) = _run_prior_step_with_microbatch_retry(
                deps=deps,
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_train_batch=y_train_batch,
                y_all_batch=y_all_batch,
                train_test_split_index=prior_step.train_test_split_index,
                trace_activations=trace_activations,
                flush_activation_trace=flush_activation_trace,
            )

            pre_clip_module_grad_norms = deps.module_grad_norms(model)
            local_grad_norm = deps.total_grad_norm(model.parameters())
            clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            local_grad_norm = deps.normalize_grad_norm_value(clipped, fallback=local_grad_norm)
            grad_clip_triggered = bool(local_grad_norm > grad_clip)
            if grad_clip_triggered:
                clipped_step_count += 1

            optimizer.step()
            step_train_duration = time.perf_counter() - step_train_start
            train_elapsed_seconds += step_train_duration
            global_step = int(prior_step.step_index)

            final_grad_norm = float(local_grad_norm)
            grad_norm_sum += final_grad_norm
            grad_norm_count += 1
            max_grad_norm = max(max_grad_norm, final_grad_norm)

            loss_delta_value = deps.train_loss_delta(
                history_step_loss,
                previous_train_loss=previous_train_loss,
            )
            loss_ema = deps.update_loss_ema(history_step_loss, previous_ema=loss_ema)
            previous_train_loss = history_step_loss
            elapsed_seconds = time.perf_counter() - train_start
            prior_dump_missingness = cast(dict[str, Any], missingness_summary["prior_dump"])
            synthetic_prior_missingness = cast(dict[str, Any], missingness_summary["synthetic_prior"])
            train_log: dict[str, Any] = {
                "train/loss": float(history_step_loss),
                "train/acc": float(history_step_acc),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": float(final_grad_norm),
                "train/loss_delta": loss_delta_value,
                "train/loss_ema": loss_ema,
                "train/elapsed_seconds": float(elapsed_seconds),
                "train/train_elapsed_seconds": float(train_elapsed_seconds),
                "train/grad_clip_threshold": float(grad_clip),
                "train/grad_clip_triggered": grad_clip_triggered,
                "train/grad_clip_count_so_far": int(clipped_step_count),
                "train/grad_clip_fraction_so_far": float(clipped_step_count / global_step),
                "train/prior_dump_skipped_batch_count": int(prior_dump_missingness["skipped_batch_count"]),
                "train/prior_dump_non_finite_feature_count": int(prior_dump_missingness["non_finite_feature_count"]),
                "train/prior_dump_non_finite_label_count": int(prior_dump_missingness["non_finite_label_count"]),
                "train/synthetic_prior_masked_feature_count": int(synthetic_prior_missingness["masked_feature_count"]),
                "train/prior_dump_microbatch_size": int(microbatch_size_used),
                "train/prior_dump_microbatch_count": int(microbatch_count),
                "train/stage": _PRIOR_STAGE_NAME,
            }
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
            deps.log_wandb_metrics(run, train_log, step=global_step)
            history_payload = deps.history_record(
                global_step=global_step,
                stage_name=_PRIOR_STAGE_NAME,
                train_loss=history_step_loss,
                train_metrics={"acc": history_step_acc},
                lr=float(optimizer.param_groups[0]["lr"]),
                grad_norm=final_grad_norm,
                elapsed_seconds=elapsed_seconds,
                train_elapsed_seconds=train_elapsed_seconds,
                val_metrics=None,
                train_loss_delta=loss_delta_value,
                train_loss_ema=loss_ema,
                grad_clip_threshold=float(grad_clip),
                grad_clip_triggered=grad_clip_triggered,
            )
            history_records.append(history_payload)
            if history_path is not None:
                deps.append_history_record(history_path, history_payload)

            gradient_payload = deps.gradient_history_record(
                global_step=global_step,
                stage_name=_PRIOR_STAGE_NAME,
                train_loss=history_step_loss,
                train_acc=history_step_acc,
                lr=float(optimizer.param_groups[0]["lr"]),
                global_grad_norm=final_grad_norm,
                module_grad_norms=pre_clip_module_grad_norms,
                activation_norms=activation_norms,
                elapsed_seconds=elapsed_seconds,
                train_elapsed_seconds=train_elapsed_seconds,
                grad_clip_threshold=float(grad_clip),
                grad_clip_triggered=grad_clip_triggered,
            )
            gradient_records.append(gradient_payload)
            deps.append_jsonl_record(gradient_path, gradient_payload)

            if global_step % checkpoint_every == 0:
                checkpoint_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                deps.save_eval_mode_checkpoint(
                    prepared_opts,
                    path=checkpoint_path,
                    model=model,
                    global_step=global_step,
                    cfg=cfg,
                    restore_training=True,
                )
                checkpoint_snapshots.append(
                    {
                        "step": int(global_step),
                        "path": str(checkpoint_path.resolve()),
                        "elapsed_seconds": float(train_elapsed_seconds),
                        "train_elapsed_seconds": float(train_elapsed_seconds),
                    }
                )
            if global_step % eval_every == 0:
                print(
                    f"time {train_elapsed_seconds:7.1f}s | "
                    f"step {global_step:4d} | "
                    f"loss {history_step_loss:7.4f} | "
                    f"acc {history_step_acc:7.4f}"
                )

        latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
        deps.save_eval_mode_checkpoint(
            prepared_opts,
            path=latest_checkpoint,
            model=model,
            global_step=global_step,
            cfg=cfg,
            restore_training=False,
        )
        artifacts["latest_checkpoint"] = str(latest_checkpoint.resolve())
        wall_elapsed_seconds = time.perf_counter() - train_start
        telemetry_payload = deps.build_training_telemetry(
            run_dir=output_dir,
            success=True,
            artifacts=artifacts,
            checkpoint_snapshots=checkpoint_snapshots,
            history_records=history_records,
            gradient_records=gradient_records,
            missingness=missingness_summary,
            training_surface_record=training_surface_payload,
        )
        deps.write_training_telemetry(telemetry_output_path, telemetry_payload)
        deps.update_prior_wandb_summary(
            run,
            output_dir=output_dir,
            global_step=global_step,
            telemetry_payload=telemetry_payload,
        )
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=None,
            latest_checkpoint=latest_checkpoint,
            global_step=global_step,
            metrics={
                "final_train_loss": float(history_step_loss),
                "final_train_acc": float(history_step_acc),
                "final_grad_norm": float(final_grad_norm),
                "mean_grad_norm": float(grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else 0.0,
                "max_grad_norm": float(max_grad_norm),
                "train_elapsed_seconds": float(train_elapsed_seconds),
                "wall_elapsed_seconds": float(wall_elapsed_seconds),
            },
        )
    except Exception as exc:
        telemetry_payload = deps.build_training_telemetry(
            run_dir=output_dir,
            success=False,
            artifacts=artifacts,
            checkpoint_snapshots=checkpoint_snapshots,
            history_records=history_records,
            gradient_records=gradient_records,
            missingness=missingness_summary,
            training_surface_record=training_surface_payload,
            error=exc,
        )
        deps.write_training_telemetry(telemetry_output_path, telemetry_payload)
        deps.update_prior_wandb_summary(
            run,
            output_dir=output_dir,
            global_step=global_step,
            telemetry_payload=telemetry_payload,
        )
        raise
    finally:
        deps.finish_wandb_run(run)
