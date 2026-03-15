"""Prior-dump training for exact-parity tabfoundry classification models."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Sequence, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.bench.nanotabpfn import resolve_device
from tab_foundry.bench.prior_dump import PriorDumpTaskBatchReader
from tab_foundry.config import compose_config
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import ModelBuildSpec, model_build_spec_from_mappings
from tab_foundry.training.artifacts import (
    append_history_record,
    assert_clean_training_output,
    history_path_from_cfg,
    history_record,
    save_checkpoint,
)
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import build_optimizer
from tab_foundry.training.surface import write_training_surface_record
from tab_foundry.training.trainer import (
    _normalize_grad_norm_value,
    _set_optimizer_training_mode,
    _total_grad_norm,
)
from tab_foundry.types import TrainResult


DEFAULT_PRIOR_DUMP_PATH = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_BATCH_SIZE = 32
DEFAULT_EXPERIMENT = "cls_benchmark_linear_simple_prior"
_PRIOR_STAGE_NAME = "prior_dump"


def _resolve_positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be int-compatible, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be >= 1, got {resolved}")
    return resolved


def _resolve_lr(cfg: DictConfig) -> float:
    raw_lr = getattr(cfg.optimizer, "min_lr", None)
    if raw_lr is None:
        raise ValueError("exact-parity prior training requires optimizer.min_lr")
    lr = float(raw_lr)
    if lr <= 0:
        raise ValueError(f"optimizer.min_lr must be > 0, got {lr}")
    return lr


def _optimizer_kwargs(cfg: DictConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    raw_betas = getattr(cfg.optimizer, "betas", None)
    if raw_betas is not None:
        kwargs["betas"] = tuple(float(value) for value in raw_betas)
    return kwargs


def _model_spec_from_cfg(cfg: DictConfig):
    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("cfg.model must resolve to a mapping")
    primary_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    return model_build_spec_from_mappings(
        task=str(cfg.task),
        primary=primary_cfg,
    )


def _validate_prior_training_model_spec(spec: ModelBuildSpec) -> None:
    if spec.arch not in {"tabfoundry_simple", "tabfoundry_staged"}:
        raise ValueError(
            "prior-dump training requires model.arch in {'tabfoundry_simple', 'tabfoundry_staged'}, "
            f"got {spec.arch!r}"
        )
    if spec.arch != "tabfoundry_staged":
        return

    surface = resolve_staged_surface(spec)
    if surface.head == "many_class":
        raise ValueError(
            "prior-dump training requires a staged recipe with forward_batched() tensor logits, "
            f"got model.stage={surface.stage!r}"
        )


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
    optimizer: torch.optim.Optimizer,
    *,
    path: Path,
    model: torch.nn.Module,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    prepared_opts = [("schedulefree_adamw", optimizer)]
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
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TrainResult:
    """Train an exact-parity staged/simple classifier on the nanoTabPFN prior dump."""

    if str(cfg.task).strip().lower() != "classification":
        raise ValueError(f"prior-dump training requires task='classification', got {cfg.task!r}")

    spec = _model_spec_from_cfg(cfg)
    _validate_prior_training_model_spec(spec)
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

    max_steps = _resolve_positive_int(getattr(cfg.runtime, "max_steps", None), name="runtime.max_steps")
    eval_every = _resolve_positive_int(getattr(cfg.runtime, "eval_every", None), name="runtime.eval_every")
    checkpoint_every = _resolve_positive_int(
        getattr(cfg.runtime, "checkpoint_every", None),
        name="runtime.checkpoint_every",
    )
    grad_clip = float(cfg.runtime.grad_clip)
    if grad_clip <= 0:
        raise ValueError(f"runtime.grad_clip must be > 0 for prior-dump training, got {grad_clip}")

    lr = _resolve_lr(cfg)
    model = build_model_from_spec(spec)
    device = torch.device(resolve_device(str(cfg.runtime.device)))
    model.to(device)
    model.train()
    raw_cfg = cast(dict[str, object], OmegaConf.to_container(cfg, resolve=True))
    write_training_surface_record(
        output_dir / "training_surface_record.json",
        raw_cfg=raw_cfg,
        run_dir=output_dir,
    )

    optimizer_selection = build_optimizer(
        model,
        name=str(cfg.optimizer.name),
        lr=lr,
        weight_decay=float(cfg.optimizer.weight_decay),
        extra_kwargs=_optimizer_kwargs(cfg),
        require_requested=bool(cfg.optimizer.require_requested),
        muon_per_parameter_lr=bool(getattr(cfg.optimizer, "muon_per_parameter_lr", True)),
        muon_lr_scale_base=float(getattr(cfg.optimizer, "muon_lr_scale_base", 0.2)),
        muon_partition_non2d=bool(getattr(cfg.optimizer, "muon_partition_non2d", True)),
    )
    if optimizer_selection.resolved_name != "schedulefree_adamw":
        raise RuntimeError(
            "exact-parity prior-dump training requires optimizer 'schedulefree_adamw', "
            f"resolved {optimizer_selection.resolved_name!r}"
        )
    if len(optimizer_selection.optimizers) != 1:
        raise RuntimeError(
            "exact-parity prior-dump training expects exactly one optimizer instance, "
            f"got {len(optimizer_selection.optimizers)}"
        )
    optimizer = optimizer_selection.optimizers[0][1]
    prepared_opts = [("schedulefree_adamw", optimizer)]
    _set_optimizer_training_mode(prepared_opts, training=True)

    history_step_loss = 0.0
    history_step_acc = 0.0
    global_step = 0
    latest_checkpoint: Path | None = None
    final_grad_norm = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    train_start = time.perf_counter()
    train_elapsed_seconds = 0.0
    reader = PriorDumpTaskBatchReader(
        prior_dump_path,
        num_steps=max_steps,
        batch_size=batch_size,
    )

    for prior_step in reader:
        optimizer.zero_grad(set_to_none=True)
        step_train_start = time.perf_counter()
        forward_batched = getattr(model, "forward_batched", None)
        if not callable(forward_batched):
            raise RuntimeError(
                "prior-dump training requires a model with forward_batched()"
            )
        x_batch, y_train_batch, y_all_batch = _stack_prior_step(prior_step, device=device)
        logits = forward_batched(
            x_all=x_batch,
            y_train=y_train_batch,
            train_test_split_index=prior_step.train_test_split_index,
        )
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("prior-dump training requires tensor logits")
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

        local_grad_norm = _total_grad_norm(model.parameters())
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        local_grad_norm = _normalize_grad_norm_value(clipped, fallback=local_grad_norm)

        optimizer.step()
        step_train_duration = time.perf_counter() - step_train_start
        train_elapsed_seconds += step_train_duration
        global_step = int(prior_step.step_index)

        final_grad_norm = float(local_grad_norm)
        grad_norm_sum += final_grad_norm
        grad_norm_count += 1
        max_grad_norm = max(max_grad_norm, final_grad_norm)

        if history_path is not None:
            append_history_record(
                history_path,
                history_record(
                    global_step=global_step,
                    stage_name=_PRIOR_STAGE_NAME,
                    train_loss=history_step_loss,
                    train_metrics={"acc": history_step_acc},
                    lr=float(optimizer.param_groups[0]["lr"]),
                    grad_norm=final_grad_norm,
                    elapsed_seconds=time.perf_counter() - train_start,
                    train_elapsed_seconds=train_elapsed_seconds,
                    val_metrics=None,
                ),
            )

        if global_step % checkpoint_every == 0:
            _save_eval_mode_checkpoint(
                optimizer,
                path=output_dir / "checkpoints" / f"step_{global_step:06d}.pt",
                model=model,
                global_step=global_step,
                cfg=cfg,
                restore_training=True,
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
        optimizer,
        path=latest_checkpoint,
        model=model,
        global_step=global_step,
        cfg=cfg,
        restore_training=False,
    )
    wall_elapsed_seconds = time.perf_counter() - train_start
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
