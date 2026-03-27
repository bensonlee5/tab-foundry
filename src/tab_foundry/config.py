"""Shared Hydra config composition helpers."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .repo_paths import repo_root


def config_dir() -> Path:
    """Return the repo config directory."""

    return repo_root() / "configs"


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the root config with optional Hydra overrides."""

    resolved_overrides = list(overrides or [])
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cfg = compose(config_name="config", overrides=resolved_overrides)
    explicit_model_arch = any(
        str(override).strip().startswith(prefix)
        for override in resolved_overrides
        for prefix in ("model.arch=", "+model.arch=", "++model.arch=")
    )
    staged_only_override = any(
        str(override).strip().startswith(prefix)
        for override in resolved_overrides
        for prefix in (
            "model.stage=",
            "+model.stage=",
            "++model.stage=",
            "model.stage_label=",
            "+model.stage_label=",
            "++model.stage_label=",
            "model.module_overrides",
            "+model.module_overrides",
            "++model.module_overrides",
        )
    )
    if not explicit_model_arch and staged_only_override:
        cfg.model.arch = "tabfoundry_staged"
    return cfg
