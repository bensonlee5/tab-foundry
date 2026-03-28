"""Prior-dump training for exact-parity tabfoundry classification models."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from tab_foundry.device import resolve_device
from tab_foundry.training.prior.config import (
    _model_spec_from_cfg,
    _optimizer_kwargs,
    _resolve_lr,
    _resolve_prior_dump_batch_config,
    _resolve_prior_dump_non_finite_policy,
    _resolve_prior_schedule,
    _resolve_prior_wandb_run_name,
    _validate_prior_training_model_spec,
    DEFAULT_BATCH_SIZE as _CONFIG_DEFAULT_BATCH_SIZE,
)
from tab_foundry.training.prior.io import save_eval_mode_checkpoint as _save_eval_mode_checkpoint_impl
from tab_foundry.training.prior.io import stack_prior_step as _stack_prior_step
from tab_foundry.training.prior.loop import PriorTrainingDeps, run_prior_training
from tab_foundry.training.prior.settings import PriorMissingnessConfig, PriorRuntimeConfig
from tab_foundry.training.prior.missingness import (
    _accumulate_missingness,
    _accumulate_synthetic_missingness,
    _apply_prior_missingness,
    _initial_missingness_summary,
    _prior_wandb_summary_payload,
)
from tab_foundry.training.prior.runtime import (
    resolve_prior_training_device_name as _resolve_prior_training_device_name_impl,
    seed_prior_training,
)
from tab_foundry.training.prior.wandb import update_prior_wandb_summary
from tab_foundry.training.prior_dump import PriorDumpTaskBatchReader
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import ModelBuildSpec
from tab_foundry.training.artifacts import (
    append_jsonl_record,
    append_history_record,
    assert_clean_training_output,
    gradient_history_record,
    history_path_from_cfg,
    history_record,
    save_checkpoint,
)
from tab_foundry.training.instability import (
    build_regime_budget_summary,
    build_runtime_summary,
    build_training_telemetry,
    gradient_history_path,
    module_grad_norms,
    normalize_grad_norm_value,
    objective_metric_for_task,
    peak_device_memory_summary,
    reset_peak_device_memory_stats,
    tensor_batch_examples_seen,
    tensor_batch_token_count,
    telemetry_path,
    total_grad_norm,
    train_loss_delta,
    update_loss_ema,
    write_training_telemetry,
)
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import build_optimizer
from tab_foundry.training.schedule import stage_base_lr
from tab_foundry.training.surface import write_training_surface_record
from tab_foundry.training.trainer_optimizer import _set_optimizer_base_lr, _set_optimizer_training_mode
from tab_foundry.training.wandb import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    training_surface_wandb_summary_payload,
    update_wandb_summary,
    wandb_identity_payload,
)
from tab_foundry.types import TrainResult


DEFAULT_PRIOR_DUMP_PATH = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_BATCH_SIZE = _CONFIG_DEFAULT_BATCH_SIZE
DEFAULT_EXPERIMENT = "cls_benchmark_linear_simple_prior"


def _resolve_prior_training_device_name(
    cfg: DictConfig,
    *,
    spec: ModelBuildSpec,
    staged_surface,
) -> str:
    return _resolve_prior_training_device_name_impl(
        cfg,
        spec=spec,
        staged_surface=staged_surface,
        resolve_device_fn=resolve_device,
    )


def _save_eval_mode_checkpoint(
    prepared_opts,
    *,
    path: Path,
    model,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    _save_eval_mode_checkpoint_impl(
        prepared_opts,
        path=path,
        model=model,
        global_step=global_step,
        cfg=cfg,
        restore_training=restore_training,
        set_optimizer_training_mode_fn=_set_optimizer_training_mode,
        save_checkpoint_fn=save_checkpoint,
    )


def _build_prior_training_deps() -> PriorTrainingDeps:
    return PriorTrainingDeps(
        resolve_prior_training_device_name=_resolve_prior_training_device_name,
        history_path_from_cfg=history_path_from_cfg,
        assert_clean_training_output=assert_clean_training_output,
        gradient_history_path=gradient_history_path,
        telemetry_path=telemetry_path,
        build_model_from_spec=build_model_from_spec,
        write_training_surface_record=write_training_surface_record,
        init_wandb_run=init_wandb_run,
        finish_wandb_run=finish_wandb_run,
        log_wandb_metrics=log_wandb_metrics,
        update_wandb_summary=update_wandb_summary,
        training_surface_wandb_summary_payload=training_surface_wandb_summary_payload,
        update_prior_wandb_summary=lambda run, *, output_dir, global_step, telemetry_payload: update_prior_wandb_summary(
            run,
            output_dir=output_dir,
            global_step=global_step,
            telemetry_payload=telemetry_payload,
            prior_wandb_summary_payload_fn=_prior_wandb_summary_payload,
            update_wandb_summary_fn=update_wandb_summary,
        ),
        wandb_identity_payload=wandb_identity_payload,
        initial_missingness_summary=_initial_missingness_summary,
        build_optimizer=build_optimizer,
        optimizer_kwargs=_optimizer_kwargs,
        set_optimizer_training_mode=_set_optimizer_training_mode,
        set_optimizer_base_lr=_set_optimizer_base_lr,
        stage_base_lr=stage_base_lr,
        accumulate_missingness=_accumulate_missingness,
        apply_prior_missingness=_apply_prior_missingness,
        accumulate_synthetic_missingness=_accumulate_synthetic_missingness,
        prior_dump_task_batch_reader=PriorDumpTaskBatchReader,
        stack_prior_step=_stack_prior_step,
        classification_loss=classification_loss,
        module_grad_norms=module_grad_norms,
        total_grad_norm=total_grad_norm,
        normalize_grad_norm_value=normalize_grad_norm_value,
        train_loss_delta=train_loss_delta,
        update_loss_ema=update_loss_ema,
        history_record=history_record,
        append_history_record=append_history_record,
        gradient_history_record=gradient_history_record,
        append_jsonl_record=append_jsonl_record,
        save_eval_mode_checkpoint=_save_eval_mode_checkpoint,
        build_runtime_summary=build_runtime_summary,
        build_regime_budget_summary=build_regime_budget_summary,
        build_training_telemetry=build_training_telemetry,
        objective_metric_for_task=objective_metric_for_task,
        peak_device_memory_summary=peak_device_memory_summary,
        reset_peak_device_memory_stats=reset_peak_device_memory_stats,
        tensor_batch_examples_seen=tensor_batch_examples_seen,
        tensor_batch_token_count=tensor_batch_token_count,
        write_training_telemetry=write_training_telemetry,
    )


def train_tabfoundry_simple_prior(
    cfg: DictConfig,
    *,
    prior_dump_path: Path = DEFAULT_PRIOR_DUMP_PATH,
    batch_size: int | None = None,
) -> TrainResult:
    """Train an exact-parity staged/simple classifier on the nanoTabPFN prior dump."""

    if str(cfg.task).strip().lower() != "classification":
        raise ValueError(f"prior-dump training requires task='classification', got {cfg.task!r}")

    spec = _model_spec_from_cfg(cfg)
    staged_surface = _validate_prior_training_model_spec(spec)
    if str(cfg.runtime.mixed_precision).strip().lower() != "no":
        raise ValueError(
            "exact-parity prior-dump training requires runtime.mixed_precision='no', "
            f"got {cfg.runtime.mixed_precision!r}"
        )

    seed_prior_training(int(cfg.runtime.seed))

    runtime_config = PriorRuntimeConfig.from_runtime_cfg(getattr(cfg, "runtime", None))
    max_steps = int(runtime_config.max_steps)
    eval_every = int(runtime_config.eval_every)
    checkpoint_every = int(runtime_config.checkpoint_every)
    grad_clip = float(cfg.runtime.grad_clip)
    trace_activations = bool(runtime_config.trace_activations)
    if grad_clip <= 0:
        raise ValueError(f"runtime.grad_clip must be > 0 for prior-dump training, got {grad_clip}")

    prior_batch_config = _resolve_prior_dump_batch_config(cfg, batch_size_override=batch_size)
    lr_min = _resolve_lr(cfg) * prior_batch_config.effective_lr_scale_factor
    training_cfg = getattr(cfg, "training", None)
    prior_missingness = PriorMissingnessConfig.from_training_overrides(
        None if training_cfg is None else getattr(training_cfg, "overrides", None)
    )
    prior_missingness_config = None if prior_missingness is None else prior_missingness.to_runtime_dict()
    prior_dump_non_finite_policy = _resolve_prior_dump_non_finite_policy(cfg)
    prior_stage = _resolve_prior_schedule(
        cfg,
        max_steps=max_steps,
        lr_min=lr_min,
        lr_scale_factor=prior_batch_config.effective_lr_scale_factor,
    )
    cfg.logging.run_name = _resolve_prior_wandb_run_name(cfg)

    return run_prior_training(
        cfg,
        prior_dump_path=prior_dump_path,
        spec=spec,
        staged_surface=staged_surface,
        max_steps=max_steps,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        grad_clip=grad_clip,
        trace_activations=trace_activations,
        prior_batch_config=prior_batch_config,
        prior_stage=prior_stage,
        lr_min=lr_min,
        prior_missingness_config=prior_missingness_config,
        prior_dump_non_finite_policy=prior_dump_non_finite_policy,
        deps=_build_prior_training_deps(),
    )
