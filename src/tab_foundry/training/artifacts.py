"""Training artifact helpers shared across training entrypoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf
import torch

from .checkpoint_paths import (
    canonical_latest_checkpoint_path,
    checkpoint_dir,
    resolve_latest_checkpoint_path,
    stage_latest_checkpoint_path,
)
from .instability import gradient_history_path, telemetry_path
from .trainer_optimizer import _set_optimizer_training_mode

__all__ = [
    "canonical_latest_checkpoint_path",
    "checkpoint_dir",
    "resolve_latest_checkpoint_path",
    "stage_latest_checkpoint_path",
]


def history_path_from_cfg(cfg: DictConfig) -> Path | None:
    """Resolve the optional training history JSONL path from config."""

    raw_path = getattr(cfg.logging, "history_jsonl_path", None)
    if raw_path is None:
        return None
    value = str(raw_path).strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def assert_clean_training_output(output_dir: Path, *, history_path: Path | None) -> None:
    """Reject reuse of a training output directory with existing artifacts."""

    checkpoint_output_dir = checkpoint_dir(output_dir)
    history_is_dirty = False
    if history_path is not None and history_path.exists():
        history_is_dirty = history_path.is_dir() or history_path.stat().st_size > 0
    checkpoint_paths = (
        sorted(checkpoint_output_dir.glob("*.pt")) if checkpoint_output_dir.exists() else []
    )
    extra_artifacts = {
        "gradient_history": gradient_history_path(output_dir),
        "telemetry": telemetry_path(output_dir),
        "training_surface_record": output_dir / "training_surface_record.json",
    }
    dirty_extras = {
        name: path
        for name, path in extra_artifacts.items()
        if path.exists() and (path.is_dir() or path.stat().st_size > 0)
    }
    if not history_is_dirty and not checkpoint_paths and not dirty_extras:
        return
    found_artifacts: list[str] = []
    if history_is_dirty and history_path is not None:
        found_artifacts.append(f"history={history_path}")
    if checkpoint_paths:
        found_artifacts.append(f"checkpoints={checkpoint_output_dir}")
    for name, path in sorted(dirty_extras.items()):
        found_artifacts.append(f"{name}={path}")
    artifact_summary = ", ".join(found_artifacts)
    raise RuntimeError(
        "runtime.output_dir is not resume-safe: found existing training artifacts "
        f"({artifact_summary}); remove prior history/checkpoints or choose a fresh "
        "runtime.output_dir"
    )


def _history_value(metrics: Mapping[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def history_record(
    *,
    global_step: int,
    stage_name: str,
    train_loss: float,
    train_metrics: Mapping[str, float],
    lr: float,
    grad_norm: float | None,
    elapsed_seconds: float,
    train_elapsed_seconds: float,
    val_metrics: Mapping[str, float] | None,
    train_loss_delta: float | None = None,
    train_loss_ema: float | None = None,
    grad_clip_threshold: float | None = None,
    grad_clip_triggered: bool | None = None,
    task_batch_size_requested: int | None = None,
    task_batch_size_actual: int | None = None,
    task_batch_batched_count: int | None = None,
    task_batch_singleton_fallback_count: int | None = None,
    task_batch_singleton_fallback_fraction: float | None = None,
    task_batch_signature_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build one history JSONL record with the standard training schema."""

    record: dict[str, Any] = {
        "step": int(global_step),
        "stage": stage_name,
        "train_loss": float(train_loss),
        "train_acc": _history_value(train_metrics, "acc"),
        "lr": float(lr),
        "grad_norm": None
        if grad_norm is None or not math.isfinite(float(grad_norm))
        else float(grad_norm),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "train_elapsed_seconds": max(0.0, float(train_elapsed_seconds)),
        "val_loss": None,
        "val_acc": None,
        "train_loss_delta": None
        if train_loss_delta is None or not math.isfinite(float(train_loss_delta))
        else float(train_loss_delta),
        "train_loss_ema": None
        if train_loss_ema is None or not math.isfinite(float(train_loss_ema))
        else float(train_loss_ema),
        "grad_clip_threshold": None
        if grad_clip_threshold is None or not math.isfinite(float(grad_clip_threshold))
        else float(grad_clip_threshold),
        "grad_clip_triggered": None
        if grad_clip_triggered is None
        else bool(grad_clip_triggered),
        "task_batch_size_requested": None
        if task_batch_size_requested is None
        else int(task_batch_size_requested),
        "task_batch_size_actual": None
        if task_batch_size_actual is None
        else int(task_batch_size_actual),
        "task_batch_batched_count": None
        if task_batch_batched_count is None
        else int(task_batch_batched_count),
        "task_batch_singleton_fallback_count": None
        if task_batch_singleton_fallback_count is None
        else int(task_batch_singleton_fallback_count),
        "task_batch_singleton_fallback_fraction": None
        if task_batch_singleton_fallback_fraction is None
        or not math.isfinite(float(task_batch_singleton_fallback_fraction))
        else float(task_batch_singleton_fallback_fraction),
    }
    if task_batch_signature_counts is not None:
        record["task_batch_signature_counts"] = {
            str(name): int(count)
            for name, count in sorted(task_batch_signature_counts.items())
        }
    if val_metrics is not None:
        record["val_loss"] = _history_value(val_metrics, "val_loss")
        record["val_acc"] = _history_value(val_metrics, "acc")
    return record


def gradient_history_record(
    *,
    global_step: int,
    stage_name: str,
    train_loss: float,
    train_acc: float | None,
    lr: float,
    global_grad_norm: float | None,
    global_grad_norm_kind: str | None = None,
    module_grad_norms: Mapping[str, float],
    activation_norms: Mapping[str, float] | None = None,
    elapsed_seconds: float,
    train_elapsed_seconds: float,
    grad_clip_threshold: float | None,
    grad_clip_triggered: bool | None,
    task_batch_size_requested: int | None = None,
    task_batch_size_actual: int | None = None,
    task_batch_batched_count: int | None = None,
    task_batch_singleton_fallback_count: int | None = None,
    task_batch_singleton_fallback_fraction: float | None = None,
    task_batch_signature_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build one detailed module-gradient record for JSONL output."""

    resolved_global_grad_norm = None
    if global_grad_norm is not None:
        value_f = float(global_grad_norm)
        if math.isfinite(value_f):
            resolved_global_grad_norm = value_f

    inferred_global_grad_norm_kind: str | None = None
    if global_grad_norm is not None:
        value_f = float(global_grad_norm)
        if math.isnan(value_f):
            inferred_global_grad_norm_kind = "nan"
        elif math.isinf(value_f):
            inferred_global_grad_norm_kind = "pos_inf" if value_f > 0.0 else "neg_inf"
        elif math.isfinite(value_f):
            inferred_global_grad_norm_kind = "finite"
    if global_grad_norm_kind is None:
        if inferred_global_grad_norm_kind is None:
            raise ValueError("global_grad_norm_kind is required when global_grad_norm is None")
        resolved_global_grad_norm_kind = inferred_global_grad_norm_kind
    else:
        resolved_global_grad_norm_kind = str(global_grad_norm_kind)
        if resolved_global_grad_norm_kind not in {"finite", "nan", "pos_inf", "neg_inf"}:
            raise ValueError(
                "global_grad_norm_kind must be one of 'finite', 'nan', 'pos_inf', or 'neg_inf'"
            )
        if (
            inferred_global_grad_norm_kind is not None
            and inferred_global_grad_norm_kind != resolved_global_grad_norm_kind
        ):
            raise ValueError(
                "global_grad_norm_kind does not match the provided global_grad_norm value"
            )

    record: dict[str, Any] = {
        "step": int(global_step),
        "stage": stage_name,
        "train_loss": float(train_loss),
        "train_acc": None
        if train_acc is None or not math.isfinite(float(train_acc))
        else float(train_acc),
        "lr": float(lr),
        # Keep the short alias aligned with train_history.jsonl for easier
        # ad hoc inspection while preserving the explicit global name.
        "grad_norm": resolved_global_grad_norm,
        "global_grad_norm": resolved_global_grad_norm,
        "global_grad_norm_kind": resolved_global_grad_norm_kind,
        "module_grad_norms": {
            str(name): float(value)
            for name, value in sorted(module_grad_norms.items())
            if math.isfinite(float(value))
        },
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "train_elapsed_seconds": max(0.0, float(train_elapsed_seconds)),
        "grad_clip_threshold": None
        if grad_clip_threshold is None or not math.isfinite(float(grad_clip_threshold))
        else float(grad_clip_threshold),
        "grad_clip_triggered": None
        if grad_clip_triggered is None
        else bool(grad_clip_triggered),
        "task_batch_size_requested": None
        if task_batch_size_requested is None
        else int(task_batch_size_requested),
        "task_batch_size_actual": None
        if task_batch_size_actual is None
        else int(task_batch_size_actual),
        "task_batch_batched_count": None
        if task_batch_batched_count is None
        else int(task_batch_batched_count),
        "task_batch_singleton_fallback_count": None
        if task_batch_singleton_fallback_count is None
        else int(task_batch_singleton_fallback_count),
        "task_batch_singleton_fallback_fraction": None
        if task_batch_singleton_fallback_fraction is None
        or not math.isfinite(float(task_batch_singleton_fallback_fraction))
        else float(task_batch_singleton_fallback_fraction),
    }
    if task_batch_signature_counts is not None:
        record["task_batch_signature_counts"] = {
            str(name): int(count)
            for name, count in sorted(task_batch_signature_counts.items())
        }
    if activation_norms is not None:
        record["activation_norms"] = {
            str(name): float(value)
            for name, value in sorted(activation_norms.items())
            if math.isfinite(float(value))
        }
    return record


def append_history_record(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one standard training history record to JSONL output."""

    append_jsonl_record(path, payload)


def append_jsonl_record(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON object to a newline-delimited artifact file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True)
        handle.write("\n")


def checkpoint_payload(
    *,
    model_state: Mapping[str, Any],
    global_step: int,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Build the standard checkpoint payload used by train/eval/export."""

    return {
        "model": dict(model_state),
        "global_step": int(global_step),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def save_checkpoint(
    path: Path,
    *,
    model_state: Mapping[str, Any],
    global_step: int,
    cfg: DictConfig,
) -> None:
    """Write a standard checkpoint payload to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model_state=model_state,
            global_step=global_step,
            cfg=cfg,
        ),
        path,
    )


def save_eval_mode_checkpoint(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    path: Path,
    model_state_factory: Callable[[], Mapping[str, Any]],
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    """Save a checkpoint while optimizers are in eval mode."""

    _set_optimizer_training_mode(prepared_opts, training=False)
    try:
        save_checkpoint(
            path,
            model_state=model_state_factory(),
            global_step=global_step,
            cfg=cfg,
        )
    finally:
        if restore_training:
            _set_optimizer_training_mode(prepared_opts, training=True)
