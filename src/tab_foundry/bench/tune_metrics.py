"""Shared history-derived metrics for tuning workflows."""

from __future__ import annotations

import math
from typing import Any, Sequence, cast

from omegaconf import OmegaConf

from tab_foundry.bench.artifacts import load_history
from tab_foundry.training.instability import grad_norm_summary_from_values
from tab_foundry.training.schedule import build_stage_configs, warmup_steps_for_stage


def finite_history_values(history: Sequence[dict[str, Any]], key: str) -> list[float]:
    """Collect finite numeric history values for one key."""

    values: list[float] = []
    for record in history:
        raw = record.get(key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def post_warmup_variance(history: Sequence[dict[str, Any]], *, raw_cfg: Any) -> float:
    """Measure train-loss variance after the first schedule warmup."""

    raw_schedule = None
    if isinstance(raw_cfg, dict):
        raw_schedule = raw_cfg.get("schedule")
    else:
        raw_schedule = getattr(raw_cfg, "schedule", None)
    warmup_steps = 0
    if raw_schedule is not None:
        raw_payload = OmegaConf.to_container(raw_schedule, resolve=True)
        if isinstance(raw_payload, dict):
            raw_stages = raw_payload.get("stages")
            if isinstance(raw_stages, list):
                normalized_stages = [
                    {str(key): value for key, value in stage.items()}
                    for stage in raw_stages
                    if isinstance(stage, dict)
                ]
                if normalized_stages:
                    stage_configs = build_stage_configs(normalized_stages)
                    if stage_configs:
                        warmup_steps = warmup_steps_for_stage(stage_configs[0])
    losses = [
        float(record["train_loss"])
        for record in history
        if int(record["step"]) > warmup_steps and math.isfinite(float(record["train_loss"]))
    ]
    if len(losses) < 2:
        return float("inf")
    mean = sum(losses) / float(len(losses))
    return sum((loss - mean) ** 2 for loss in losses) / float(len(losses))


def history_summary(history: Sequence[dict[str, Any]], *, raw_cfg: Any) -> dict[str, float | None]:
    """Summarize training stability metrics from one history sequence."""

    grad_norms = finite_history_values(history, "grad_norm")
    last_record = cast(dict[str, Any], history[-1])
    train_elapsed = float(last_record.get("train_elapsed_seconds", 0.0))
    wall_elapsed = float(last_record.get("elapsed_seconds", train_elapsed))
    return {
        "post_warmup_train_loss_var": post_warmup_variance(history, raw_cfg=raw_cfg),
        **grad_norm_summary_from_values(grad_norms),
        "train_elapsed_seconds": train_elapsed if math.isfinite(train_elapsed) else None,
        "wall_elapsed_seconds": wall_elapsed if math.isfinite(wall_elapsed) else None,
    }


def load_history_summary(path, *, raw_cfg: Any) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Load a history file and summarize it for tuning decisions."""

    history = load_history(path)
    return history, history_summary(history, raw_cfg=raw_cfg)
