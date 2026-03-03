"""Training schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StageConfig:
    """One pretraining stage configuration."""

    name: str
    steps: int
    lr_max: float


def build_stage_configs(cfg_stages: list[dict[str, object]]) -> list[StageConfig]:
    """Parse stage configs from OmegaConf-resolved list."""

    stages: list[StageConfig] = []
    for raw in cfg_stages:
        steps_raw = raw["steps"]
        lr_raw = raw["lr_max"]
        if not isinstance(steps_raw, int):
            raise ValueError(f"stage steps must be int, got {type(steps_raw)!r}")
        if not isinstance(lr_raw, (int, float)):
            raise ValueError(f"stage lr_max must be float, got {type(lr_raw)!r}")
        stages.append(
            StageConfig(
                name=str(raw["name"]),
                steps=steps_raw,
                lr_max=float(lr_raw),
            )
        )
    return stages
