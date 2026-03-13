"""Checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)
from tab_foundry.types import EvalResult

from .batching import move_batch
from .distributed import _global_mean_from_local
from .runtime import build_accelerator_from_runtime
from .trainer import _compute_loss_and_metrics


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


def evaluate_checkpoint(cfg: DictConfig) -> EvalResult:
    """Evaluate a saved checkpoint."""

    checkpoint = Path(str(cfg.eval.checkpoint)).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")

    model_spec = _checkpoint_model_settings(payload, cfg)
    task = model_spec.task
    model = build_model_from_spec(model_spec)
    model.load_state_dict(payload["model"])

    split = str(cfg.eval.split)
    ds = build_task_dataset(
        cfg.data,
        split=split,
        task=task,
        seed=int(cfg.runtime.seed),
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

    loss_sum = 0.0
    score_sum = 0.0
    count = 0

    metric_name = "acc" if task == "classification" else "rmse"

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= int(cfg.eval.max_batches):
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
    return EvalResult(
        checkpoint=checkpoint,
        metrics={
            "loss": loss_value,
            metric_name: metric_value,
        },
    )
