"""Checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)
from tab_foundry.task_batching import (
    resolve_task_batch_size,
)
from tab_foundry.types import EvalResult

from .runtime import build_accelerator_from_runtime
from .task_batching_validation import validate_task_batching_support
from .trainer_metrics import _compute_loss_and_metrics, _evaluate_loader
from .wandb import finish_wandb_run, init_wandb_run, log_wandb_metrics, update_wandb_summary


def _checkpoint_config_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    cfg_payload = payload.get("config")
    return cfg_payload if isinstance(cfg_payload, dict) else {}


def _checkpoint_config_section(
    payload: dict[str, Any],
    cfg: DictConfig,
    *,
    section: str,
) -> DictConfig | None:
    checkpoint_cfg = _checkpoint_config_mapping(payload)
    checkpoint_section = checkpoint_cfg.get(section)
    if isinstance(checkpoint_section, dict):
        return OmegaConf.create(checkpoint_section)

    fallback = cfg.get(section)
    if isinstance(fallback, DictConfig):
        return fallback
    if isinstance(fallback, dict):
        return OmegaConf.create(fallback)
    return None


def _checkpoint_preprocessing_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> DictConfig | None:
    return _checkpoint_config_section(payload, cfg, section="preprocessing")


def _checkpoint_training_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> DictConfig | None:
    return _checkpoint_config_section(payload, cfg, section="training")


def _checkpoint_data_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> DictConfig:
    data_cfg = _checkpoint_config_section(payload, cfg, section="data")
    if data_cfg is None:
        raise RuntimeError("evaluate_checkpoint requires data config from checkpoint or runtime cfg")
    return data_cfg


def _checkpoint_dataset_seed(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> int:
    checkpoint_cfg = _checkpoint_config_mapping(payload)
    checkpoint_runtime_cfg = checkpoint_cfg.get("runtime")
    if isinstance(checkpoint_runtime_cfg, dict):
        raw_seed = checkpoint_runtime_cfg.get("seed")
        if raw_seed is not None:
            try:
                return int(raw_seed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Unsupported checkpoint runtime seed value: {raw_seed!r}") from exc
    return int(cfg.runtime.seed)


def _checkpoint_model_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> ModelBuildSpec:
    checkpoint_cfg = _checkpoint_config_mapping(payload)
    task_raw = checkpoint_cfg.get("task", cfg.task)
    task = str(task_raw).strip().lower()
    if task != "classification":
        raise RuntimeError(
            "Only classification checkpoints are supported in this branch; "
            f"got task={task!r}."
        )

    raw_fallback = OmegaConf.to_container(cfg.model, resolve=True)
    fallback_model_cfg: dict[str, Any] = {}
    if isinstance(raw_fallback, dict):
        fallback_model_cfg = {str(key): value for key, value in raw_fallback.items()}
    model_cfg = checkpoint_cfg.get("model")
    primary_model_cfg: dict[str, Any] = {}
    if isinstance(model_cfg, dict):
        primary_model_cfg = {str(key): value for key, value in model_cfg.items()}
    model_state = payload.get("model")
    state_dict = model_state if isinstance(model_state, dict) else None
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=primary_model_cfg,
        fallback=fallback_model_cfg,
        state_dict=state_dict,
    )


def _resolved_checkpoint_step(payload: Mapping[str, Any]) -> int:
    raw_value = payload.get("global_step")
    if raw_value is None:
        return 0
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


def _evaluation_summary_payload(
    *,
    checkpoint: Path,
    split: str,
    max_batches: int,
    global_step: int,
    metrics: Mapping[str, float] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run": {
            "checkpoint": str(checkpoint),
            "split": split,
            "global_step": int(global_step),
        },
        "eval": {
            "max_batches": int(max_batches),
        },
    }
    if metrics is not None:
        summary["metrics"] = {
            str(key): float(value)
            for key, value in metrics.items()
        }
    if error is not None:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    return summary


def evaluate_checkpoint(cfg: DictConfig) -> EvalResult:
    """Evaluate a saved checkpoint."""

    checkpoint = Path(str(cfg.eval.checkpoint)).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")

    model_spec = _checkpoint_model_settings(payload, cfg)
    data_cfg = _checkpoint_data_settings(payload, cfg)
    preprocessing_cfg = _checkpoint_preprocessing_settings(payload, cfg)
    training_cfg = _checkpoint_training_settings(payload, cfg)
    dataset_seed = _checkpoint_dataset_seed(payload, cfg)
    task = model_spec.task
    eval_step = _resolved_checkpoint_step(payload)
    task_batch_size = resolve_task_batch_size(training_cfg)
    model = build_model_from_spec(model_spec)
    model.load_state_dict(payload["model"])

    split = str(cfg.eval.split)
    max_batches = int(cfg.eval.max_batches)
    ds = build_task_dataset(
        data_cfg,
        split=split,
        task=task,
        seed=dataset_seed,
        preprocessing_cfg=preprocessing_cfg,
    )
    validate_task_batching_support(
        ds,
        task_batch_size=task_batch_size,
        model_spec=model_spec,
        shuffle=False,
        context=f"evaluation split={split!r}",
    )
    loader = build_task_loader(
        ds,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        seed=dataset_seed,
        task_batch_size=task_batch_size,
    )

    accelerator = build_accelerator_from_runtime(
        cfg.runtime,
        dataloader_even_batches_override=False if task_batch_size > 1 else None,
    )
    model, loader = accelerator.prepare(model, loader)
    logging_cfg = cfg.get("logging")
    use_wandb = bool(getattr(logging_cfg, "use_wandb", False)) if logging_cfg is not None else False
    run = init_wandb_run(
        cfg,
        enabled=bool(use_wandb and accelerator.is_main_process),
    )

    metric_name = "acc"

    try:
        eval_metrics = _evaluate_loader(
            model,
            loader,
            accelerator=accelerator,
            task=task,
            max_batches=max_batches,
            compute_loss_and_metrics=_compute_loss_and_metrics,
        )
        result_metrics = {
            "loss": float(eval_metrics["val_loss"]),
            metric_name: float(eval_metrics[metric_name]),
        }
        log_wandb_metrics(
            run,
            {f"eval/{key}": value for key, value in result_metrics.items()},
            step=eval_step,
        )
        update_wandb_summary(
            run,
            _evaluation_summary_payload(
                checkpoint=checkpoint,
                split=split,
                max_batches=max_batches,
                global_step=eval_step,
                metrics=result_metrics,
            ),
        )
        return EvalResult(
            checkpoint=checkpoint,
            metrics=result_metrics,
        )
    except Exception as exc:
        update_wandb_summary(
            run,
            _evaluation_summary_payload(
                checkpoint=checkpoint,
                split=split,
                max_batches=max_batches,
                global_step=eval_step,
                error=exc,
            ),
        )
        raise
    finally:
        finish_wandb_run(run)
