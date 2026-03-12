"""Training schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


_VALID_LR_SCHEDULES = frozenset({"cosine", "linear"})


@dataclass(slots=True)
class StageConfig:
    """One pretraining stage configuration."""

    name: str
    steps: int
    lr_max: float
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.0


def warmup_steps_for_stage(stage: StageConfig) -> int:
    """Resolve the integer warmup length for one stage."""

    if stage.steps <= 1 or stage.warmup_ratio <= 0.0:
        return 0
    return min(stage.steps - 1, max(1, int(math.ceil(float(stage.steps) * float(stage.warmup_ratio)))))


def stage_base_lr(stage: StageConfig, *, step: int, lr_min: float) -> float:
    """Return the base LR to use for a 1-based optimizer step within a stage."""

    if step <= 0:
        raise ValueError(f"stage step must be >= 1, got {step}")
    if step > stage.steps:
        raise ValueError(f"stage step must be <= stage.steps ({stage.steps}), got {step}")
    if stage.steps <= 1:
        return float(stage.lr_max)

    if stage.lr_schedule == "cosine":
        progress = float(step - 1) / float(stage.steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(lr_min) + (float(stage.lr_max) - float(lr_min)) * cosine

    warmup_steps = warmup_steps_for_stage(stage)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(stage.lr_max) * (float(step) / float(warmup_steps))

    decay_steps = stage.steps - warmup_steps
    if decay_steps <= 1:
        return float(lr_min)
    decay_step = step - warmup_steps - 1
    progress = min(max(float(decay_step) / float(decay_steps - 1), 0.0), 1.0)
    return float(stage.lr_max) + (float(lr_min) - float(stage.lr_max)) * progress


def build_stage_configs(cfg_stages: list[dict[str, object]]) -> list[StageConfig]:
    """Parse stage configs from OmegaConf-resolved list."""

    stages: list[StageConfig] = []
    for raw in cfg_stages:
        steps_raw = raw["steps"]
        lr_raw = raw["lr_max"]
        lr_schedule_raw = raw.get("lr_schedule", "cosine")
        warmup_ratio_raw = raw.get("warmup_ratio", 0.0)
        if not isinstance(steps_raw, int):
            raise ValueError(f"stage steps must be int, got {type(steps_raw)!r}")
        if not isinstance(lr_raw, (int, float)):
            raise ValueError(f"stage lr_max must be float, got {type(lr_raw)!r}")
        if not isinstance(lr_schedule_raw, str):
            raise ValueError(f"stage lr_schedule must be str, got {type(lr_schedule_raw)!r}")
        lr_schedule = lr_schedule_raw.strip().lower()
        if lr_schedule not in _VALID_LR_SCHEDULES:
            raise ValueError(
                f"stage lr_schedule must be one of {sorted(_VALID_LR_SCHEDULES)!r}, got {lr_schedule_raw!r}"
            )
        if not isinstance(warmup_ratio_raw, (int, float)):
            raise ValueError(f"stage warmup_ratio must be float, got {type(warmup_ratio_raw)!r}")
        warmup_ratio = float(warmup_ratio_raw)
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"stage warmup_ratio must be in [0, 1), got {warmup_ratio!r}")
        stages.append(
            StageConfig(
                name=str(raw["name"]),
                steps=steps_raw,
                lr_max=float(lr_raw),
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
            )
        )
    return stages
