"""Training loop."""

from __future__ import annotations

from collections.abc import Iterator
import math
import os
from pathlib import Path
from typing import Any, Literal, cast

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.model.factory import build_model_from_spec, model_build_spec_from_mappings
from tab_foundry.model.tabiclv2 import ClassificationOutput, RegressionOutput
from tab_foundry.types import TaskBatch, TrainResult

from .batching import collate_task_batch, move_batch
from .distributed import _global_mean_from_local
from .losses import classification_loss, hierarchical_nll_loss, quantile_pinball_loss
from .optimizer import build_optimizer
from .runtime import build_accelerator_from_runtime
from .schedule import build_stage_configs


def _cycle(loader: DataLoader[TaskBatch]) -> Iterator[TaskBatch]:
    while True:
        yield from loader


def _compute_loss_and_metrics(
    output: ClassificationOutput | RegressionOutput,
    batch: TaskBatch,
    *,
    task: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if task == "classification":
        if not isinstance(output, ClassificationOutput):
            raise TypeError("classification run expected ClassificationOutput")
        n_test = int(batch.y_test.shape[0])
        if n_test <= 0:
            raise RuntimeError("classification batch has zero test labels")

        if output.logits is not None:
            logits = output.logits[:, : output.num_classes]
            target = batch.y_test.to(torch.int64)
            loss = classification_loss(logits, target)
            acc = (logits.argmax(dim=-1) == target).float().mean().item()
            cls_metrics = {"acc": float(acc)}
            if output.aux_metrics is not None:
                cls_metrics.update(output.aux_metrics)
            return loss, cls_metrics

        if output.class_probs is not None:
            probs = output.class_probs
            target = batch.y_test.to(torch.int64)
            loss = hierarchical_nll_loss(probs, target)
            acc = (probs.argmax(dim=-1) == target).float().mean().item()
            cls_metrics = {"acc": float(acc)}
            if output.aux_metrics is not None:
                cls_metrics.update(output.aux_metrics)
            return loss, cls_metrics

        if output.path_logits is None or output.path_targets is None:
            raise RuntimeError("many-class output missing class_probs and path terms")
        if len(output.path_logits) != len(output.path_targets):
            raise RuntimeError("path_logits and path_targets length mismatch")

        counts = (
            output.path_sample_counts
            if output.path_sample_counts is not None
            else [int(logits.shape[0]) for logits in output.path_logits]
        )
        if len(counts) != len(output.path_logits):
            raise RuntimeError("path_sample_counts length mismatch")
        weighted_total: torch.Tensor | None = None
        total_edges = 0
        for logits, targets, sample_count in zip(
            output.path_logits, output.path_targets, counts, strict=True
        ):
            count_i = int(sample_count)
            if count_i <= 0:
                continue
            term = classification_loss(logits, targets.to(torch.int64))
            contrib = term * float(count_i)
            weighted_total = contrib if weighted_total is None else weighted_total + contrib
            total_edges += count_i
        if weighted_total is None or total_edges <= 0 or n_test <= 0:
            raise RuntimeError("path-based many-class output has no valid terms")
        loss = weighted_total / float(n_test)
        path_metrics: dict[str, float] = {}
        if output.aux_metrics is not None:
            path_metrics.update(output.aux_metrics)
        return loss, path_metrics

    if not isinstance(output, RegressionOutput):
        raise TypeError("regression run expected RegressionOutput")
    target = batch.y_test.to(torch.float32)
    levels = output.quantile_levels
    if levels is None:
        levels = torch.arange(1, 1000, device=target.device, dtype=torch.float32) / 1000.0
    loss = quantile_pinball_loss(output.quantiles, target, quantile_levels=levels)
    pred_mean = output.quantiles.mean(dim=-1)
    rmse = torch.sqrt(torch.mean((pred_mean - target) ** 2)).item()
    return loss, {"rmse": float(rmse)}


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[TaskBatch],
    *,
    accelerator: Accelerator,
    task: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    score_sum = 0.0
    count = 0
    metric_name = "acc" if task == "classification" else "rmse"

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            batch = move_batch(batch, accelerator.device)
            with accelerator.autocast():
                output = model(batch)
                loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
            loss_sum += float(loss.detach().item())
            score_sum += float(metrics[metric_name])
            count += 1

    model.train()
    dev = accelerator.device
    val_loss = _global_mean_from_local(
        accelerator, local_sum=loss_sum, local_count=count, device=dev, default=float("inf"),
    )
    val_score = _global_mean_from_local(
        accelerator, local_sum=score_sum, local_count=count, device=dev, default=0.0,
    )
    return {"val_loss": val_loss, metric_name: val_score}


def _wandb_init(cfg: DictConfig, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except Exception:
        return None

    mode: Literal["online", "offline"] = "online" if os.getenv("WANDB_API_KEY") else "offline"
    cfg_payload = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    return wandb.init(
        project=str(cfg.logging.project),
        name=str(cfg.logging.run_name),
        mode=mode,
        config=cfg_payload,
    )


def _save_checkpoint(
    path: Path,
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    global_step: int,
    cfg: DictConfig,
) -> None:
    if not accelerator.is_main_process:
        return
    payload = {
        "model": accelerator.get_state_dict(model),
        "global_step": global_step,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _resolve_grad_accum_steps(cfg: DictConfig) -> int:
    value = int(getattr(cfg, "grad_accum_steps", 1))
    if value <= 0:
        raise ValueError(f"runtime.grad_accum_steps must be >= 1, got {value}")
    return value


def _optimizer_lr_scales(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
) -> list[float]:
    if base_lr <= 0:
        return [1.0 for _ in optimizer.param_groups]
    return [float(group["lr"]) / float(base_lr) for group in optimizer.param_groups]


def _set_optimizer_base_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    scales: list[float],
) -> None:
    if len(scales) != len(optimizer.param_groups):
        raise RuntimeError("lr scales count does not match optimizer param groups")
    for group, scale in zip(optimizer.param_groups, scales, strict=True):
        group["lr"] = float(base_lr) * float(scale)


def _cosine_base_lr(*, step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if total_steps <= 1:
        return float(lr_min)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lr_min) + (float(lr_max) - float(lr_min)) * cosine


def _expected_metric_keys(task: str) -> set[str]:
    if task == "classification":
        return {
            "acc",
            "many_class_nodes_visited",
            "many_class_avg_path_depth",
            "many_class_empty_nodes",
        }
    return {"rmse"}


def train(cfg: DictConfig) -> TrainResult:
    """Train from config."""

    task = str(cfg.task)
    seed = int(cfg.runtime.seed)
    torch.manual_seed(seed)
    grad_accum_steps = _resolve_grad_accum_steps(cfg.runtime)

    accelerator = build_accelerator_from_runtime(
        cfg.runtime,
        grad_accum_steps_override=grad_accum_steps,
    )

    train_ds = PackedParquetTaskDataset(
        manifest_path=Path(str(cfg.data.manifest_path)),
        split="train",
        task=task,
        train_row_cap=(int(cfg.data.train_row_cap) if cfg.data.train_row_cap is not None else None),
        test_row_cap=(int(cfg.data.test_row_cap) if cfg.data.test_row_cap is not None else None),
        seed=seed,
    )
    val_ds = PackedParquetTaskDataset(
        manifest_path=Path(str(cfg.data.manifest_path)),
        split="val",
        task=task,
        train_row_cap=(int(cfg.data.train_row_cap) if cfg.data.train_row_cap is not None else None),
        test_row_cap=(int(cfg.data.test_row_cap) if cfg.data.test_row_cap is not None else None),
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=int(cfg.runtime.num_workers),
        collate_fn=collate_task_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        collate_fn=collate_task_batch,
    )

    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg: dict[str, Any] = {}
    if isinstance(raw_model_cfg, dict):
        model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    model = build_model_from_spec(model_spec)
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run = _wandb_init(cfg, enabled=bool(cfg.logging.use_wandb and accelerator.is_main_process))

    raw_stages = cast(list[dict[str, object]], OmegaConf.to_container(cfg.schedule.stages, resolve=True))
    stage_configs = build_stage_configs(raw_stages)
    if not stage_configs:
        raise RuntimeError("schedule.stages must contain at least one stage")
    train_iter = _cycle(train_loader)

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
    if run is not None and accelerator.is_main_process:
        run.log({"optimizer/fallback": 1.0 if optimizer_sel.fallback_reason else 0.0}, step=0)

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

    for stage in stage_configs:
        for opt_name, opt in prepared_opts:
            _set_optimizer_base_lr(
                opt,
                base_lr=stage.lr_max,
                scales=lr_scales[opt_name],
            )

        for stage_step in range(1, stage.steps + 1):
            model.train()
            for _opt_name, opt in prepared_opts:
                opt.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_loss_count = 0
            train_metric_sums: dict[str, float] = {}
            train_metric_counts: dict[str, int] = {}
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

            if float(cfg.runtime.grad_clip) > 0:
                accelerator.clip_grad_norm_(model.parameters(), float(cfg.runtime.grad_clip))

            for _opt_name, opt in prepared_opts:
                opt.step()
            stage_base_lr = _cosine_base_lr(
                step=stage_step,
                total_steps=stage.steps,
                lr_max=stage.lr_max,
                lr_min=float(cfg.optimizer.min_lr),
            )
            for opt_name, opt in prepared_opts:
                _set_optimizer_base_lr(
                    opt,
                    base_lr=stage_base_lr,
                    scales=lr_scales[opt_name],
                )

            global_step += 1
            metric_keys = sorted(set(train_metric_sums) | expected_keys)
            # Pack loss + metric sums/counts into one tensor for a single all-reduce.
            # Layout: [loss_sum, loss_count, key0_sum, key0_count, ...]
            n_metrics = len(metric_keys)
            packed = torch.zeros(2 + 2 * n_metrics, device=accelerator.device, dtype=torch.float64)
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
            train_log: dict[str, float | str] = {
                "train/loss": train_loss,
                "train/lr": first_lr,
                "stage": stage.name,
                "step": float(global_step),
            }
            for name, value in lr_values.items():
                train_log[f"train/lr_{name}"] = value
            for i, key in enumerate(metric_keys):
                g_sum = reduced[2 + 2 * i].item()
                g_count = reduced[2 + 2 * i + 1].item()
                metric_mean = g_sum / g_count if g_count > 0 else float("nan")
                if math.isfinite(metric_mean):
                    train_log[f"train/{key}"] = metric_mean

            if run is not None and accelerator.is_main_process:
                run.log(train_log, step=global_step)

            if global_step % int(cfg.runtime.eval_every) == 0:
                val_metrics = _evaluate_loader(
                    model,
                    val_loader,
                    accelerator=accelerator,
                    task=task,
                    max_batches=int(cfg.runtime.val_batches),
                )
                if run is not None and accelerator.is_main_process:
                    run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

                if val_metrics["val_loss"] < best_val:
                    best_val = val_metrics["val_loss"]
                    best_checkpoint = output_dir / "checkpoints" / "best.pt"
                    _save_checkpoint(
                        best_checkpoint,
                        accelerator=accelerator,
                        model=model,
                        global_step=global_step,
                        cfg=cfg,
                    )

        latest_checkpoint = output_dir / "checkpoints" / f"latest_{stage.name}.pt"
        _save_checkpoint(
            latest_checkpoint,
            accelerator=accelerator,
            model=model,
            global_step=global_step,
            cfg=cfg,
        )

    accelerator.wait_for_everyone()
    if run is not None and accelerator.is_main_process:
        run.finish()

    return TrainResult(
        output_dir=output_dir,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        global_step=global_step,
        metrics={"best_val_loss": float(best_val)},
    )
