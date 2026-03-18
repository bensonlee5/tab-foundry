"""Prior-dump training for exact-parity tabfoundry classification models."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.bench.nanotabpfn import resolve_device
from tab_foundry.bench.prior.config import (
    _model_spec_from_cfg,
    _optimizer_kwargs,
    _resolve_lr,
    _resolve_positive_int,
    _resolve_prior_dump_batch_config,
    _resolve_prior_dump_non_finite_policy,
    _resolve_prior_missingness_config,
    _resolve_prior_schedule,
    _resolve_prior_wandb_run_name,
    _resolve_runtime_bool,
    _validate_prior_training_model_spec,
    DEFAULT_BATCH_SIZE as _CONFIG_DEFAULT_BATCH_SIZE,
)
from tab_foundry.bench.prior_dump import (
    PriorDumpBatchMissingness,
    PriorDumpTaskBatchReader,
)
from tab_foundry.bench.prior.missingness import (
    _accumulate_missingness,
    _accumulate_synthetic_missingness,
    _apply_prior_missingness,
    _initial_missingness_summary,
    _prior_wandb_summary_payload,
)
from tab_foundry.config import compose_config
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    ResolvedStageSurface,
)
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
    build_training_telemetry,
    gradient_history_path,
    module_grad_norms,
    normalize_grad_norm_value,
    telemetry_path,
    total_grad_norm,
    train_loss_delta,
    update_loss_ema,
    write_training_telemetry,
)
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import build_optimizer
from tab_foundry.training.surface import write_training_surface_record
from tab_foundry.training.schedule import stage_base_lr
from tab_foundry.training.trainer import _set_optimizer_base_lr, _set_optimizer_training_mode
from tab_foundry.training.wandb import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    update_wandb_summary,
)
from tab_foundry.types import TrainResult


DEFAULT_PRIOR_DUMP_PATH = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_BATCH_SIZE = _CONFIG_DEFAULT_BATCH_SIZE
DEFAULT_EXPERIMENT = "cls_benchmark_linear_simple_prior"
_PRIOR_STAGE_NAME = "prior_dump"


def _resolve_prior_training_device_name(
    cfg: DictConfig,
    *,
    spec: ModelBuildSpec,
    staged_surface: ResolvedStageSurface | None,
) -> str:
    requested_device = str(getattr(cfg.runtime, "device", "auto") or "auto").strip()
    resolved_device = resolve_device(requested_device)
    if (
        resolved_device != "mps"
        or spec.arch != "tabfoundry_staged"
        or staged_surface is None
        or staged_surface.row_pool != "row_cls"
    ):
        return resolved_device

    row_pool_layers = int(staged_surface.row_pool_config.n_layers or 0)
    if row_pool_layers <= 1:
        return resolved_device

    print(
        "Warning: exact prior-dump training requested "
        f"runtime.device={requested_device!r} resolved to 'mps', but staged "
        f"row_pool='row_cls' with tfrow_n_layers={row_pool_layers} is unstable on MPS; "
        "falling back to CPU for this run.",
        file=sys.stderr,
        flush=True,
    )
    cfg.runtime.device = "cpu"
    return "cpu"


def _stack_prior_step(
    prior_step,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prior_step.x_batch is not None and prior_step.y_batch is not None:
        x_batch = prior_step.x_batch.to(device=device, dtype=torch.float32)
        y_batch = prior_step.y_batch.to(device=device, dtype=torch.float32)
        return (
            x_batch,
            y_batch[:, : prior_step.train_test_split_index],
            y_batch,
        )

    tasks = prior_step.tasks
    if len(tasks) <= 0:
        raise RuntimeError(f"prior dump step {prior_step.step_index} produced no tasks")
    first_x_all = torch.cat([tasks[0].x_train, tasks[0].x_test], dim=0)
    row_count = int(first_x_all.shape[0])
    feature_count = int(first_x_all.shape[1])
    for task in tasks[1:]:
        x_all = torch.cat([task.x_train, task.x_test], dim=0)
        if tuple(x_all.shape) != (row_count, feature_count):
            raise RuntimeError(
                "exact nanoTabPFN parity requires rectangular prior batches, but got "
                f"mixed x_all shapes {(row_count, feature_count)} and {tuple(x_all.shape)}"
            )
        y_all = torch.cat([task.y_train, task.y_test], dim=0)
        if tuple(y_all.shape) != (row_count,):
            raise RuntimeError(
                "exact nanoTabPFN parity requires matching label lengths across the batch, but got "
                f"{row_count} and {tuple(y_all.shape)}"
            )
    x_batch = torch.stack(
        [torch.cat([task.x_train, task.x_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    y_train_batch = torch.stack([task.y_train for task in tasks], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    y_all_batch = torch.stack(
        [torch.cat([task.y_train, task.y_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    return x_batch, y_train_batch, y_all_batch


def _save_eval_mode_checkpoint(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    path: Path,
    model: torch.nn.Module,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    _set_optimizer_training_mode(prepared_opts, training=False)
    save_checkpoint(
        path,
        model_state=model.state_dict(),
        global_step=global_step,
        cfg=cfg,
    )
    if restore_training:
        _set_optimizer_training_mode(prepared_opts, training=True)


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

    seed = int(cfg.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_path_from_cfg(cfg)
    assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = gradient_history_path(output_dir)
    telemetry_output_path = telemetry_path(output_dir)

    max_steps = _resolve_positive_int(getattr(cfg.runtime, "max_steps", None), name="runtime.max_steps")
    eval_every = _resolve_positive_int(getattr(cfg.runtime, "eval_every", None), name="runtime.eval_every")
    checkpoint_every = _resolve_positive_int(
        getattr(cfg.runtime, "checkpoint_every", None),
        name="runtime.checkpoint_every",
    )
    grad_clip = float(cfg.runtime.grad_clip)
    trace_activations = _resolve_runtime_bool(
        getattr(cfg.runtime, "trace_activations", False),
        name="runtime.trace_activations",
    )
    if grad_clip <= 0:
        raise ValueError(f"runtime.grad_clip must be > 0 for prior-dump training, got {grad_clip}")

    prior_batch_config = _resolve_prior_dump_batch_config(cfg, batch_size_override=batch_size)
    lr_min = _resolve_lr(cfg) * prior_batch_config.effective_lr_scale_factor
    device = torch.device(
        _resolve_prior_training_device_name(
            cfg,
            spec=spec,
            staged_surface=staged_surface,
        )
    )
    model = build_model_from_spec(spec)
    enable_activation_trace = getattr(model, "enable_activation_trace", None)
    flush_activation_trace = getattr(model, "flush_activation_trace", None)
    if trace_activations and callable(enable_activation_trace):
        enable_activation_trace()
    model.to(device)
    model.train()
    cfg.logging.run_name = _resolve_prior_wandb_run_name(cfg)
    raw_cfg = cast(dict[str, object], OmegaConf.to_container(cfg, resolve=True))
    raw_training_cfg = raw_cfg.get("training")
    if not isinstance(raw_training_cfg, dict):
        raw_training_cfg = {}
        raw_cfg["training"] = raw_training_cfg
    raw_training_cfg["prior_dump_batch_size"] = int(prior_batch_config.batch_size)
    raw_training_cfg["prior_dump_lr_scale_rule"] = str(prior_batch_config.lr_scale_rule)
    raw_training_cfg["prior_dump_batch_reference_size"] = int(
        prior_batch_config.reference_batch_size
    )
    raw_training_cfg["effective_lr_scale_factor"] = float(
        prior_batch_config.effective_lr_scale_factor
    )
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    if isinstance(raw_optimizer_cfg, dict) and raw_optimizer_cfg.get("min_lr") is not None:
        raw_optimizer_cfg["min_lr"] = float(lr_min)
    raw_schedule_cfg = raw_cfg.get("schedule")
    if isinstance(raw_schedule_cfg, dict):
        raw_stages = raw_schedule_cfg.get("stages")
        if isinstance(raw_stages, list):
            for stage in raw_stages:
                if isinstance(stage, dict) and stage.get("lr_max") is not None:
                    stage["lr_max"] = float(stage["lr_max"]) * float(
                        prior_batch_config.effective_lr_scale_factor
                    )
    training_surface_path = output_dir / "training_surface_record.json"
    training_surface_payload = write_training_surface_record(
        training_surface_path,
        raw_cfg=raw_cfg,
        run_dir=output_dir,
    )
    run = init_wandb_run(
        cfg,
        enabled=bool(getattr(cfg.logging, "use_wandb", False)),
    )
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
    prior_missingness_config = _resolve_prior_missingness_config(cfg)
    prior_dump_non_finite_policy = _resolve_prior_dump_non_finite_policy(cfg)
    missingness_summary = _initial_missingness_summary(
        prior_dump_path,
        prior_missingness_config=prior_missingness_config,
        prior_dump_non_finite_policy=prior_dump_non_finite_policy,
    )

    prior_stage = _resolve_prior_schedule(
        cfg,
        max_steps=max_steps,
        lr_min=lr_min,
        lr_scale_factor=prior_batch_config.effective_lr_scale_factor,
    )
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
    _set_optimizer_training_mode(prepared_opts, training=True)
    lr_scales = [1.0 for _ in optimizer.param_groups]
    _set_optimizer_base_lr(
        optimizer,
        base_lr=initial_lr,
        scales=lr_scales,
    )

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
        prior_missingness_generator.manual_seed(seed)

    def _record_non_finite_batch(batch_missingness: PriorDumpBatchMissingness) -> None:
        _accumulate_missingness(
            missingness_summary,
            batch_missingness=batch_missingness,
            skipped=prior_dump_non_finite_policy == "skip",
        )

    reader = PriorDumpTaskBatchReader(
        prior_dump_path,
        num_steps=max_steps,
        batch_size=prior_batch_config.batch_size,
        non_finite_policy=prior_dump_non_finite_policy,
        on_non_finite_batch=_record_non_finite_batch,
    )

    try:
        for prior_step in reader:
            if prior_step.missingness is not None:
                _accumulate_missingness(
                    missingness_summary,
                    batch_missingness=prior_step.missingness,
                )
            optimizer.zero_grad(set_to_none=True)
            step_train_start = time.perf_counter()
            if prior_stage is not None:
                current_base_lr = float(
                    stage_base_lr(prior_stage, step=int(prior_step.step_index), lr_min=lr_min)
                )
                _set_optimizer_base_lr(
                    optimizer,
                    base_lr=current_base_lr,
                    scales=lr_scales,
                )
            forward_batched = getattr(model, "forward_batched", None)
            if not callable(forward_batched):
                raise RuntimeError(
                    "prior-dump training requires a model with forward_batched()"
                )
            x_batch, y_train_batch, y_all_batch = _stack_prior_step(prior_step, device=device)
            x_batch, synthetic_missingness = _apply_prior_missingness(
                x_batch,
                prior_step=prior_step,
                generator=prior_missingness_generator,
                prior_missingness_config=prior_missingness_config,
            )
            _accumulate_synthetic_missingness(
                missingness_summary,
                batch_missingness=synthetic_missingness,
            )
            logits = forward_batched(
                x_all=x_batch,
                y_train=y_train_batch,
                train_test_split_index=prior_step.train_test_split_index,
            )
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("prior-dump training requires tensor logits")
            activation_norms = (
                flush_activation_trace() if trace_activations and callable(flush_activation_trace) else None
            )
            targets = y_all_batch[:, prior_step.train_test_split_index:].reshape(-1).to(torch.int64)
            loss = classification_loss(
                logits.reshape(-1, int(logits.shape[-1])),
                targets,
            )
            loss.backward()
            history_step_loss = float(loss.detach().item())
            history_step_acc = float(
                (
                    logits.argmax(dim=-1)
                    == y_all_batch[:, prior_step.train_test_split_index:].to(torch.int64)
                )
                .float()
                .mean()
                .item()
            )

            pre_clip_module_grad_norms = module_grad_norms(model)
            local_grad_norm = total_grad_norm(model.parameters())
            clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            local_grad_norm = normalize_grad_norm_value(clipped, fallback=local_grad_norm)
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

            loss_delta_value = train_loss_delta(
                history_step_loss,
                previous_train_loss=previous_train_loss,
            )
            loss_ema = update_loss_ema(history_step_loss, previous_ema=loss_ema)
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
            log_wandb_metrics(run, train_log, step=global_step)
            history_payload = history_record(
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
                append_history_record(history_path, history_payload)

            gradient_payload = gradient_history_record(
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
            append_jsonl_record(gradient_path, gradient_payload)

            if global_step % checkpoint_every == 0:
                checkpoint_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                _save_eval_mode_checkpoint(
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
        _save_eval_mode_checkpoint(
            prepared_opts,
            path=latest_checkpoint,
            model=model,
            global_step=global_step,
            cfg=cfg,
            restore_training=False,
        )
        artifacts["latest_checkpoint"] = str(latest_checkpoint.resolve())
        wall_elapsed_seconds = time.perf_counter() - train_start
        telemetry_payload = build_training_telemetry(
            run_dir=output_dir,
            success=True,
            artifacts=artifacts,
            checkpoint_snapshots=checkpoint_snapshots,
            history_records=history_records,
            gradient_records=gradient_records,
            missingness=missingness_summary,
            training_surface_record=training_surface_payload,
        )
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        update_wandb_summary(
            run,
            _prior_wandb_summary_payload(
                output_dir=output_dir,
                global_step=global_step,
                telemetry_payload=telemetry_payload,
            ),
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
        telemetry_payload = build_training_telemetry(
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
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        update_wandb_summary(
            run,
            _prior_wandb_summary_payload(
                output_dir=output_dir,
                global_step=global_step,
                telemetry_payload=telemetry_payload,
            ),
        )
        raise
    finally:
        finish_wandb_run(run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an exact-parity tabfoundry classifier on the nanoTabPFN prior dump"
    )
    parser.add_argument(
        "--prior-dump",
        default=str(DEFAULT_PRIOR_DUMP_PATH),
        help="Path to the nanoTabPFN prior dump (.h5)",
    )
    parser.add_argument("overrides", nargs="*", help="Optional Hydra override strings")
    return parser


def _compose_prior_cfg(overrides: Sequence[str]) -> DictConfig:
    resolved_overrides = list(overrides)
    if not any(str(override).startswith("experiment=") for override in resolved_overrides):
        resolved_overrides.insert(0, f"experiment={DEFAULT_EXPERIMENT}")
    return compose_config(resolved_overrides)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _compose_prior_cfg(args.overrides)
    result = train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=Path(str(args.prior_dump)),
    )
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
