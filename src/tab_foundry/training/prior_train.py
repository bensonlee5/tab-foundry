"""Prior-dump training for exact-parity tabfoundry classification models."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.training.loss_surface import configure_model_loss_surface, resolve_training_loss_surface
from tab_foundry.training.optimizer import build_optimizer
from tab_foundry.training.prior.config import (
    _model_spec_from_cfg,
    _optimizer_kwargs,
    _resolve_lr,
    _resolve_prior_dump_batch_config,
    _resolve_prior_dump_non_finite_policy,
    _resolve_prior_schedule,
    _resolve_prior_wandb_run_name,
    _validate_prior_training_model_spec,
    build_prior_training_surface_raw_cfg,
    DEFAULT_BATCH_SIZE as _CONFIG_DEFAULT_BATCH_SIZE,
)
from tab_foundry.training.prior.loop import run_prior_training
from tab_foundry.training.prior.runtime import (
    resolve_prior_training_device_name,
    seed_prior_training,
)
from tab_foundry.training.prior.settings import PriorMissingnessConfig, PriorRuntimeConfig
from tab_foundry.training.schedule import stage_base_lr
from tab_foundry.training.surface import TRAINING_BACKEND_LEGACY_PRIOR
from tab_foundry.types import TrainResult


DEFAULT_PRIOR_DUMP_PATH = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_BATCH_SIZE = _CONFIG_DEFAULT_BATCH_SIZE
DEFAULT_EXPERIMENT = "cls_benchmark_linear_simple_prior"


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

    device_name = resolve_prior_training_device_name(
        cfg,
        spec=spec,
        staged_surface=staged_surface,
    )
    model = build_model_from_spec(spec)
    loss_surface = resolve_training_loss_surface(
        getattr(cfg, "training", None),
        model_spec=spec,
        backend=TRAINING_BACKEND_LEGACY_PRIOR,
    )
    configure_model_loss_surface(model, loss_surface=loss_surface)
    initial_lr = (
        float(stage_base_lr(prior_stage, step=1, lr_min=lr_min))
        if prior_stage is not None
        else float(lr_min)
    )
    optimizer_selection = build_optimizer(
        model,
        name=str(cfg.optimizer.name),
        lr=initial_lr,
        weight_decay=float(cfg.optimizer.weight_decay),
        extra_kwargs=_optimizer_kwargs(cfg),
        require_requested=bool(cfg.optimizer.require_requested),
        muon_per_parameter_lr=bool(getattr(cfg.optimizer, "muon_per_parameter_lr", True)),
        muon_lr_scale_base=float(getattr(cfg.optimizer, "muon_lr_scale_base", 0.2)),
        muon_partition_non2d=bool(getattr(cfg.optimizer, "muon_partition_non2d", True)),
    )
    training_surface_raw_cfg = build_prior_training_surface_raw_cfg(
        cfg,
        loss_surface=loss_surface,
        prior_batch_config=prior_batch_config,
        lr_min=lr_min,
    )

    return run_prior_training(
        cfg,
        prior_dump_path=prior_dump_path,
        device_name=device_name,
        model=model,
        loss_surface=loss_surface,
        optimizer_selection=optimizer_selection,
        training_surface_raw_cfg=training_surface_raw_cfg,
        max_steps=max_steps,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        grad_clip=grad_clip,
        trace_activations=trace_activations,
        prior_batch_config=prior_batch_config,
        prior_stage=prior_stage,
        lr_min=lr_min,
        initial_lr=initial_lr,
        prior_missingness_config=prior_missingness_config,
        prior_dump_non_finite_policy=prior_dump_non_finite_policy,
        spec=spec,
    )
