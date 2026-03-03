"""Checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from tab_foundry.data.dataset import CauchyParquetTaskDataset
from tab_foundry.model.factory import build_model
from tab_foundry.types import EvalResult

from .batching import collate_task_batch, move_batch
from .distributed import _global_mean_from_local
from .runtime import build_accelerator_from_runtime
from .trainer import _compute_loss_and_metrics


def _checkpoint_model_settings(
    payload: dict[str, Any],
    cfg: DictConfig,
) -> tuple[str, int, int, int, str, int]:
    cfg_payload = payload.get("config")
    checkpoint_cfg = cfg_payload if isinstance(cfg_payload, dict) else {}
    task_raw = checkpoint_cfg.get("task", cfg.task)
    task = str(task_raw).strip().lower()
    if task not in {"classification", "regression"}:
        raise RuntimeError(f"Unsupported checkpoint task value: {task!r}")

    model_cfg = checkpoint_cfg.get("model")
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    d_col = int(model_cfg.get("d_col", cfg.model.d_col))
    d_icl = int(model_cfg.get("d_icl", cfg.model.d_icl))
    feature_group_size = int(model_cfg.get("feature_group_size", cfg.model.feature_group_size))
    many_class_train_mode = str(
        model_cfg.get("many_class_train_mode", cfg.model.many_class_train_mode)
    )
    max_mixed_radix_digits = int(
        model_cfg.get("max_mixed_radix_digits", cfg.model.max_mixed_radix_digits)
    )
    return (
        task,
        d_col,
        d_icl,
        feature_group_size,
        many_class_train_mode,
        max_mixed_radix_digits,
    )


def evaluate_checkpoint(cfg: DictConfig) -> EvalResult:
    """Evaluate a saved checkpoint."""

    checkpoint = Path(str(cfg.eval.checkpoint)).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")

    (
        task,
        d_col,
        d_icl,
        feature_group_size,
        many_class_train_mode,
        max_mixed_radix_digits,
    ) = _checkpoint_model_settings(payload, cfg)
    model = build_model(
        task=task,
        d_col=d_col,
        d_icl=d_icl,
        feature_group_size=feature_group_size,
        many_class_train_mode=many_class_train_mode,
        max_mixed_radix_digits=max_mixed_radix_digits,
    )
    model.load_state_dict(payload["model"])

    split = str(cfg.eval.split)
    ds = CauchyParquetTaskDataset(
        manifest_path=Path(str(cfg.data.manifest_path)),
        split=split,
        task=task,
        train_row_cap=(int(cfg.data.train_row_cap) if cfg.data.train_row_cap is not None else None),
        test_row_cap=(int(cfg.data.test_row_cap) if cfg.data.test_row_cap is not None else None),
        seed=int(cfg.runtime.seed),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        collate_fn=collate_task_batch,
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
