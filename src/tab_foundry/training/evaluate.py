"""Checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.missingness import validate_missingness_runtime_policy
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)
from tab_foundry.preprocessing import resolve_preprocessing_surface
from tab_foundry.types import EvalResult

from .batching import move_batch
from .distributed import _global_mean_from_local
from .runtime import build_accelerator_from_runtime
from .trainer import _compute_loss_and_metrics
from .wandb import finish_wandb_run, init_wandb_run, log_wandb_metrics, update_wandb_summary


def _checkpoint_preprocessing_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> DictConfig | None:
    cfg_payload = payload.get("config")
    checkpoint_cfg = cfg_payload if isinstance(cfg_payload, dict) else {}
    checkpoint_preprocessing_cfg = checkpoint_cfg.get("preprocessing")
    if isinstance(checkpoint_preprocessing_cfg, dict):
        return OmegaConf.create(checkpoint_preprocessing_cfg)

    fallback = cfg.get("preprocessing")
    if isinstance(fallback, DictConfig):
        return fallback
    if isinstance(fallback, dict):
        return OmegaConf.create(fallback)
    return None


def _checkpoint_model_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> ModelBuildSpec:
    cfg_payload = payload.get("config")
    checkpoint_cfg = cfg_payload if isinstance(cfg_payload, dict) else {}
    task_raw = checkpoint_cfg.get("task", cfg.task)
    task = str(task_raw).strip().lower()
    if task not in {"classification", "regression"}:
        raise RuntimeError(f"Unsupported checkpoint task value: {task!r}")

    raw_overrides = None
    eval_cfg = cfg.get("eval")
    if isinstance(eval_cfg, DictConfig):
        model_overrides_cfg = eval_cfg.get("model_overrides")
        if model_overrides_cfg is not None:
            raw_overrides = OmegaConf.to_container(model_overrides_cfg, resolve=True)
    elif isinstance(eval_cfg, dict):
        raw_overrides = eval_cfg.get("model_overrides")
    explicit_model_overrides: dict[str, Any] | None = None
    if raw_overrides is not None:
        if not isinstance(raw_overrides, dict):
            raise RuntimeError("eval.model_overrides must be a mapping when provided")
        explicit_model_overrides = {str(key): value for key, value in raw_overrides.items()}
    model_cfg = checkpoint_cfg.get("model")
    primary_model_cfg: dict[str, Any] = {}
    if isinstance(model_cfg, dict):
        primary_model_cfg = {str(key): value for key, value in model_cfg.items()}
    model_state = payload.get("model")
    state_dict = model_state if isinstance(model_state, dict) else None
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=primary_model_cfg,
        explicit_overrides=explicit_model_overrides,
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
    preprocessing_cfg = _checkpoint_preprocessing_settings(payload, cfg)
    raw_data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg = None if not isinstance(raw_data_cfg, dict) else {str(key): value for key, value in raw_data_cfg.items()}
    data_surface = resolve_data_surface(data_cfg)
    raw_preprocessing_cfg = (
        None if preprocessing_cfg is None else OmegaConf.to_container(preprocessing_cfg, resolve=True)
    )
    preprocessing_cfg_payload = (
        None
        if not isinstance(raw_preprocessing_cfg, dict)
        else {str(key): value for key, value in raw_preprocessing_cfg.items()}
    )
    preprocessing_surface = resolve_preprocessing_surface(
        preprocessing_cfg_payload
    )
    validate_missingness_runtime_policy(
        missingness_mode=getattr(model_spec, "missingness_mode", "none"),
        allow_missing_values=data_surface.allow_missing_values,
        impute_missing=preprocessing_surface.impute_missing,
        context="evaluate_checkpoint",
    )
    task = model_spec.task
    eval_step = _resolved_checkpoint_step(payload)
    model = build_model_from_spec(model_spec)
    model.load_state_dict(payload["model"])

    split = str(cfg.eval.split)
    max_batches = int(cfg.eval.max_batches)
    ds = build_task_dataset(
        cfg.data,
        split=split,
        task=task,
        seed=int(cfg.runtime.seed),
        preprocessing_cfg=preprocessing_cfg,
    )
    loader = build_task_loader(
        ds,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        seed=int(cfg.runtime.seed),
    )

    accelerator = build_accelerator_from_runtime(cfg.runtime)
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    logging_cfg = cfg.get("logging")
    use_wandb = bool(getattr(logging_cfg, "use_wandb", False)) if logging_cfg is not None else False
    run = init_wandb_run(
        cfg,
        enabled=bool(use_wandb and accelerator.is_main_process),
    )

    loss_sum = 0.0
    score_sum = 0.0
    count = 0

    metric_name = "acc" if task == "classification" else "rmse"

    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                batch = move_batch(batch, accelerator.device)
                with accelerator.autocast():
                    output = model(batch)
                    loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
                loss_sum += float(loss.item())
                score_sum += metrics.get(metric_name, 0.0)
                count += 1

        dev = accelerator.device
        loss_value = _global_mean_from_local(
            accelerator, local_sum=loss_sum, local_count=count, device=dev, default=float("inf"),
        )
        metric_value = _global_mean_from_local(
            accelerator, local_sum=score_sum, local_count=count, device=dev, default=0.0,
        )
        result_metrics = {
            "loss": loss_value,
            metric_name: metric_value,
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
