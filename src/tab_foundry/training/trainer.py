"""Training loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import math
from numbers import Real
import os
from pathlib import Path
import time
from typing import Any, Literal, cast

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from tab_foundry.data.factory import build_task_dataset, build_task_loader
from tab_foundry.model.architectures.tabfoundry import ClassificationOutput, RegressionOutput
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.types import TaskBatch, TrainResult

from .artifacts import (
    append_history_record,
    assert_clean_training_output,
    history_path_from_cfg,
    history_record,
    save_checkpoint,
)
from .batching import move_batch
from .distributed import _global_mean_from_local, _reduction_float_dtype
from .losses import classification_loss, hierarchical_nll_loss, quantile_pinball_loss
from .optimizer import build_optimizer
from .runtime import build_accelerator_from_runtime
from .schedule import build_stage_configs, stage_base_lr


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

    api_key = _resolve_wandb_api_key()
    mode: Literal["online", "offline"] = "online" if api_key else "offline"
    cfg_payload = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    return wandb.init(
        project=str(cfg.logging.project),
        name=str(cfg.logging.run_name),
        mode=mode,
        config=cfg_payload,
    )


def _resolve_wandb_api_key() -> str | None:
    value = os.getenv("WANDB_API_KEY")
    if value is not None:
        normalized = value.strip()
        if normalized:
            return normalized

    file_override = os.getenv("WANDB_API_KEY_FILE")
    candidate = Path(file_override).expanduser() if file_override else Path("~/.wandb/wandb_api_key.txt").expanduser()
    try:
        normalized = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not normalized:
        return None
    os.environ["WANDB_API_KEY"] = normalized
    return normalized


def _resolve_grad_accum_steps(cfg: DictConfig) -> int:
    value = int(getattr(cfg, "grad_accum_steps", 1))
    if value <= 0:
        raise ValueError(f"runtime.grad_accum_steps must be >= 1, got {value}")
    return value


def _checkpoint_every(cfg: DictConfig) -> int | None:
    raw_value = getattr(cfg, "checkpoint_every", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.checkpoint_every must be >= 1, got {value}")
    return value


def _resolve_max_steps(runtime_cfg: DictConfig) -> int | None:
    raw_value = getattr(runtime_cfg, "max_steps", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.max_steps must be >= 1, got {value}")
    return value


def _resolve_target_train_seconds(runtime_cfg: DictConfig) -> float | None:
    raw_value = getattr(runtime_cfg, "target_train_seconds", None)
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"runtime.target_train_seconds must be > 0, got {value}")
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


def _set_optimizer_training_mode(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    training: bool,
) -> None:
    method_name = "train" if training else "eval"
    for _name, optimizer in prepared_opts:
        method = getattr(optimizer, method_name, None)
        if callable(method):
            method()


def _expected_metric_keys(task: str) -> set[str]:
    if task == "classification":
        return {
            "acc",
            "grad_norm",
            "many_class_nodes_visited",
            "many_class_avg_path_depth",
            "many_class_empty_nodes",
        }
    return {"rmse", "grad_norm"}


def _total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    found_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        norm = float(torch.linalg.vector_norm(grad).item())
        total_sq += norm * norm
        found_grad = True
    if not found_grad:
        return 0.0
    return math.sqrt(total_sq)


def _normalize_grad_norm_value(value: object, *, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, torch.Tensor):
        value_f = float(value.detach().item())
        return value_f if math.isfinite(value_f) else float(fallback)
    if isinstance(value, Real):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else float(fallback)
    return float(fallback)


def train(cfg: DictConfig) -> TrainResult:
    """Train from config."""

    task = str(cfg.task)
    seed = int(cfg.runtime.seed)
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_path_from_cfg(cfg)
    assert_clean_training_output(output_dir, history_path=history_path)

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
    )
    val_ds = build_task_dataset(
        cfg.data,
        split="val",
        task=task,
        seed=seed + 1,
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
    best_val_step = 0.0
    last_val_metrics: dict[str, float] | None = None
    train_start = time.perf_counter()
    train_elapsed_seconds = 0.0
    stop_requested = False
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    final_grad_norm = 0.0

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

            local_grad_norm = _total_grad_norm(model.parameters())
            if float(cfg.runtime.grad_clip) > 0:
                clipped = accelerator.clip_grad_norm_(model.parameters(), float(cfg.runtime.grad_clip))
                local_grad_norm = _normalize_grad_norm_value(clipped, fallback=local_grad_norm)
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
                    if key == "grad_norm":
                        grad_norm_value = float(metric_mean)
                        grad_norm_sum += grad_norm_value
                        grad_norm_count += 1
                        max_grad_norm = max(max_grad_norm, grad_norm_value)
                        final_grad_norm = grad_norm_value

            if run is not None and accelerator.is_main_process:
                run.log(train_log, step=global_step)

            history_val_metrics: dict[str, float] | None = None
            if global_step % int(cfg.runtime.eval_every) == 0:
                _set_optimizer_training_mode(prepared_opts, training=False)
                val_metrics = _evaluate_loader(
                    model,
                    val_loader,
                    accelerator=accelerator,
                    task=task,
                    max_batches=int(cfg.runtime.val_batches),
                )
                if run is not None and accelerator.is_main_process:
                    run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                history_val_metrics = val_metrics
                last_val_metrics = val_metrics

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

            if history_path is not None and accelerator.is_main_process:
                train_metrics_for_history = {
                    key.removeprefix("train/"): float(value)
                    for key, value in train_log.items()
                    if key.startswith("train/")
                    and isinstance(value, (int, float))
                    and key != "train/lr"
                    and math.isfinite(float(value))
                }
                append_history_record(
                    history_path,
                    history_record(
                        global_step=global_step,
                        stage_name=stage.name,
                        train_loss=float(train_loss),
                        train_metrics=train_metrics_for_history,
                        lr=float(first_lr),
                        grad_norm=None if not math.isfinite(grad_norm_value) else float(grad_norm_value),
                        elapsed_seconds=time.perf_counter() - train_start,
                        train_elapsed_seconds=train_elapsed_seconds,
                        val_metrics=history_val_metrics,
                    ),
                )

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
    if run is not None and accelerator.is_main_process:
        run.finish()

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

    return TrainResult(
        output_dir=output_dir,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        global_step=global_step,
        metrics={
            "best_val_loss": float(best_val),
            "best_val_step": float(best_val_step),
            "final_val_loss": float(last_val_metrics["val_loss"]) if last_val_metrics is not None else float(best_val),
            "final_grad_norm": float(final_grad_norm),
            "mean_grad_norm": float(grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else 0.0,
            "max_grad_norm": float(max_grad_norm),
            "train_elapsed_seconds": float(train_elapsed_seconds),
            "wall_elapsed_seconds": float(wall_elapsed_seconds),
        },
    )
