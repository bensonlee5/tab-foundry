"""Shared Hydra config composition helpers."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def config_dir() -> Path:
    """Return the repo config directory."""

    return Path(__file__).resolve().parents[2] / "configs"


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the root config with optional Hydra overrides."""

    resolved_overrides = list(overrides or [])
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        return compose(config_name="config", overrides=resolved_overrides)
