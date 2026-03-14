"""Training artifact helpers shared across training entrypoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf
import torch


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

    checkpoint_dir = output_dir / "checkpoints"
    history_is_dirty = False
    if history_path is not None and history_path.exists():
        history_is_dirty = history_path.is_dir() or history_path.stat().st_size > 0
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    if not history_is_dirty and not checkpoint_paths:
        return
    found_artifacts: list[str] = []
    if history_is_dirty and history_path is not None:
        found_artifacts.append(f"history={history_path}")
    if checkpoint_paths:
        found_artifacts.append(f"checkpoints={checkpoint_dir}")
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
) -> dict[str, float | int | str | None]:
    """Build one history JSONL record with the standard training schema."""

    record: dict[str, float | int | str | None] = {
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
    }
    if val_metrics is not None:
        record["val_loss"] = _history_value(val_metrics, "val_loss")
        record["val_acc"] = _history_value(val_metrics, "acc")
    return record


def append_history_record(path: Path, payload: Mapping[str, float | int | str | None]) -> None:
    """Append one standard training history record to JSONL output."""

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
