"""Training loop."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.task_batching import move_batch, resolve_task_batch_size
from tab_foundry.types import TrainResult

from .artifacts import assert_clean_training_output, save_checkpoint, save_eval_mode_checkpoint
from .instability import (
    gradient_history_path,
    normalize_grad_norm_value,
    reset_peak_device_memory_stats,
    telemetry_path,
)
from .loss_surface import configure_model_loss_surface, resolve_training_loss_surface
from .optimizer import build_optimizer
from .runtime import build_accelerator_from_runtime
from .schedule import build_stage_configs
from .surface import TRAINING_BACKEND_MANIFEST, write_training_surface_record
from .trainer_finalize import finalize_training_run
from .trainer_guards import _accelerator_num_processes
from .trainer_loop import TrainingLoopState, run_training_loop
from .trainer_metrics import _compute_loss_and_metrics, _evaluate_loader, _expected_metric_keys
from .trainer_optimizer import _optimizer_lr_scales, _set_optimizer_training_mode
from .trainer_runtime_config import (
    _checkpoint_every,
    _resolve_activation_checkpointing,
    _resolve_grad_accum_steps,
    _resolve_max_steps,
    _resolve_target_train_seconds,
    _resolve_val_batches,
)
from .trainer_summary import _trainer_summary_payload
from .task_batching_validation import validate_task_batching_support
from .wandb import (
    finish_wandb_run,
    init_wandb_run,
    training_surface_wandb_summary_payload,
    update_wandb_summary,
)

__all__ = [
    "_compute_loss_and_metrics",
    "_resolve_grad_accum_steps",
    "_trainer_summary_payload",
    "move_batch",
    "normalize_grad_norm_value",
    "train",
]


def train(cfg: DictConfig) -> TrainResult:
    """Train from config."""

    task = str(cfg.task)
    seed = int(cfg.runtime.seed)
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "train_history.jsonl"
    if hasattr(cfg, "runtime"):
        from .artifacts import history_path_from_cfg

        resolved_history_path = history_path_from_cfg(cfg)
        if resolved_history_path is not None:
            history_path = resolved_history_path
    assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = gradient_history_path(output_dir)
    telemetry_output_path = telemetry_path(output_dir)
    training_surface_path = output_dir / "training_surface_record.json"

    torch.manual_seed(seed)
    grad_accum_steps = _resolve_grad_accum_steps(cfg.runtime)
    checkpoint_every = _checkpoint_every(cfg.runtime)
    max_steps = _resolve_max_steps(cfg.runtime)
    target_train_seconds = _resolve_target_train_seconds(cfg.runtime)
    val_batches = _resolve_val_batches(cfg.runtime)
    task_batch_size = resolve_task_batch_size(cfg.get("training"))

    accelerator = build_accelerator_from_runtime(
        cfg.runtime,
        grad_accum_steps_override=grad_accum_steps,
        dataloader_even_batches_override=False if task_batch_size > 1 else None,
    )

    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg: dict[str, Any] = {}
    if isinstance(raw_model_cfg, dict):
        model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    loss_surface = resolve_training_loss_surface(
        cfg.get("training"),
        model_spec=model_spec,
        backend=TRAINING_BACKEND_MANIFEST,
    )

    train_ds = build_task_dataset(
        cfg.data,
        split="train",
        task=task,
        seed=seed,
        preprocessing_cfg=cfg.get("preprocessing"),
    )
    validate_task_batching_support(
        train_ds,
        task_batch_size=task_batch_size,
        model_spec=model_spec,
        shuffle=True,
        context="training train split",
    )
    train_loader = build_task_loader(
        train_ds,
        shuffle=True,
        num_workers=int(cfg.runtime.num_workers),
        seed=seed,
        task_batch_size=task_batch_size,
    )
    val_loader = None
    if val_batches > 0:
        val_ds = build_task_dataset(
            cfg.data,
            split="val",
            task=task,
            seed=seed + 1,
            preprocessing_cfg=cfg.get("preprocessing"),
        )
        validate_task_batching_support(
            val_ds,
            task_batch_size=task_batch_size,
            model_spec=model_spec,
            shuffle=False,
            context="training val split",
        )
        val_loader = build_task_loader(
            val_ds,
            shuffle=False,
            num_workers=int(cfg.runtime.num_workers),
            seed=seed + 1,
            task_batch_size=task_batch_size,
        )
    model = build_model_from_spec(model_spec)
    configure_model_loss_surface(model, loss_surface=loss_surface)
    activation_checkpointing = _resolve_activation_checkpointing(cfg.runtime)
    if activation_checkpointing:
        enable_activation_checkpointing = getattr(model, "enable_activation_checkpointing", None)
        if not callable(enable_activation_checkpointing):
            raise RuntimeError(
                "runtime.activation_checkpointing=true requires a model with "
                "enable_activation_checkpointing(); this runtime hook is currently "
                "implemented by tabfoundry_staged and tabfoundry_sandwich"
            )
        enable_activation_checkpointing()
    if val_loader is None:
        model, train_loader = accelerator.prepare(model, train_loader)
    else:
        model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    base_model = accelerator.unwrap_model(model)
    trace_activations = bool(getattr(cfg.runtime, "trace_activations", False))
    enable_activation_trace = getattr(base_model, "enable_activation_trace", None)
    flush_activation_trace = getattr(base_model, "flush_activation_trace", None)
    flush_activation_trace_stats = getattr(base_model, "flush_activation_trace_stats", None)
    requires_exact_activation_trace_stats = bool(
        trace_activations and (grad_accum_steps > 1 or _accelerator_num_processes(accelerator) > 1)
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
                if torch.isfinite(torch.tensor(float(value)))
            }
        return None

    run = None
    training_surface_payload: dict[str, Any] | None = None
    optimizer_requested_name = str(cfg.optimizer.name)
    optimizer_resolved_name = optimizer_requested_name
    optimizer_fallback_reason: str | None = None
    prepared_opts: list[tuple[str, torch.optim.Optimizer]] = []
    lr_scales: dict[str, list[float]] = {}
    state = TrainingLoopState()
    train_start = time.perf_counter()

    def _artifacts_payload() -> dict[str, Any]:
        return {
            "train_history_jsonl": None if history_path is None else str(history_path),
            "gradient_history_jsonl": str(gradient_path),
            "telemetry_json": str(telemetry_output_path),
            "training_surface_record_json": str(training_surface_path.resolve()),
            "checkpoints_dir": str((output_dir / "checkpoints").resolve()),
            "best_checkpoint": (
                None if state.best_checkpoint is None else str(state.best_checkpoint.resolve())
            ),
            "latest_checkpoint": (
                None if state.latest_checkpoint is None else str(state.latest_checkpoint.resolve())
            ),
        }

    try:
        if accelerator.is_main_process:
            raw_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            raw_training_cfg = raw_cfg.get("training")
            if not isinstance(raw_training_cfg, dict):
                raw_training_cfg = {}
                raw_cfg["training"] = raw_training_cfg
            raw_training_cfg["loss_surface"] = loss_surface
            training_surface_payload = write_training_surface_record(
                training_surface_path,
                raw_cfg=raw_cfg,
                run_dir=output_dir,
                backend=TRAINING_BACKEND_MANIFEST,
            )

        run = init_wandb_run(
            cfg,
            enabled=bool(getattr(cfg.logging, "use_wandb", False) and accelerator.is_main_process),
        )
        update_wandb_summary(
            run,
            training_surface_wandb_summary_payload(training_surface_payload),
        )
        reset_peak_device_memory_stats(accelerator.device)

        raw_stages = cast(list[dict[str, object]], OmegaConf.to_container(cfg.schedule.stages, resolve=True))
        stage_configs = build_stage_configs(raw_stages)
        if not stage_configs:
            raise RuntimeError("schedule.stages must contain at least one stage")

        first_stage = stage_configs[0]
        optimizer_sel = build_optimizer(
            accelerator.unwrap_model(model),
            name=optimizer_requested_name,
            lr=first_stage.lr_max,
            weight_decay=float(cfg.optimizer.weight_decay),
            extra_kwargs={"betas": tuple(cfg.optimizer.betas)},
            require_requested=bool(cfg.optimizer.require_requested),
            muon_per_parameter_lr=bool(cfg.optimizer.muon_per_parameter_lr),
            muon_lr_scale_base=float(cfg.optimizer.muon_lr_scale_base),
            muon_partition_non2d=bool(cfg.optimizer.muon_partition_non2d),
        )
        optimizer_requested_name = optimizer_sel.requested_name
        optimizer_resolved_name = optimizer_sel.resolved_name
        optimizer_fallback_reason = optimizer_sel.fallback_reason
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

        for opt_name, opt in optimizer_sel.optimizers:
            prepared = accelerator.prepare_optimizer(opt)
            prepared_opts.append((opt_name, prepared))
            lr_scales[opt_name] = _optimizer_lr_scales(prepared, base_lr=first_stage.lr_max)

        run_training_loop(
            cfg=cfg,
            task=task,
            output_dir=output_dir,
            history_path=history_path,
            gradient_path=gradient_path,
            accelerator=accelerator,
            model=model,
            base_model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            prepared_opts=prepared_opts,
            lr_scales=lr_scales,
            stage_configs=stage_configs,
            expected_keys=_expected_metric_keys(task),
            task_batch_size=task_batch_size,
            grad_accum_steps=grad_accum_steps,
            checkpoint_every=checkpoint_every,
            max_steps=max_steps,
            target_train_seconds=target_train_seconds,
            val_batches=val_batches,
            train_start=train_start,
            trace_activations=trace_activations,
            flush_activation_trace_stats=_flush_activation_trace_stats,
            run=run,
            compute_loss_and_metrics=_compute_loss_and_metrics,
            evaluate_loader=_evaluate_loader,
            move_batch_fn=move_batch,
            normalize_grad_norm_value_fn=normalize_grad_norm_value,
            state=state,
        )
        accelerator.wait_for_everyone()

        if state.best_checkpoint is None and state.latest_checkpoint is not None:
            state.best_checkpoint = output_dir / "checkpoints" / "best.pt"
            if accelerator.is_main_process:
                save_eval_mode_checkpoint(
                    prepared_opts,
                    path=state.best_checkpoint,
                    model_state_factory=lambda: accelerator.get_state_dict(model),
                    global_step=state.global_step,
                    cfg=cfg,
                    restore_training=False,
                    set_optimizer_training_mode_fn=_set_optimizer_training_mode,
                    save_checkpoint_fn=save_checkpoint,
                )

        result, _wall_elapsed_seconds, _task_batching_summary = finalize_training_run(
            accelerator=accelerator,
            output_dir=output_dir,
            task=task,
            cfg=cfg,
            run=run,
            training_surface_payload=training_surface_payload,
            telemetry_output_path=telemetry_output_path,
            artifacts=_artifacts_payload(),
            optimizer_requested_name=optimizer_requested_name,
            optimizer_resolved_name=optimizer_resolved_name,
            optimizer_fallback_reason=optimizer_fallback_reason,
            loss_surface=loss_surface,
            state=state,
            train_start=train_start,
            success=True,
        )
        if result is None:
            raise RuntimeError("successful training finalization did not return a TrainResult")
        return result
    except Exception as exc:
        _result, _wall_elapsed_seconds, _task_batching_summary = finalize_training_run(
            accelerator=accelerator,
            output_dir=output_dir,
            task=task,
            cfg=cfg,
            run=run,
            training_surface_payload=training_surface_payload,
            telemetry_output_path=telemetry_output_path,
            artifacts=_artifacts_payload(),
            optimizer_requested_name=optimizer_requested_name,
            optimizer_resolved_name=optimizer_resolved_name,
            optimizer_fallback_reason=optimizer_fallback_reason,
            loss_surface=loss_surface,
            state=state,
            train_start=train_start,
            success=False,
            error=exc,
        )
        raise
    finally:
        finish_wandb_run(run)
