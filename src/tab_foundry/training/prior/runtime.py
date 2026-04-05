"""Runtime helpers for exact prior-dump training."""

from __future__ import annotations

import random

import numpy as np
from omegaconf import DictConfig
import torch

from tab_foundry.model.architectures.tabfoundry_staged.resolved import ResolvedStageSurface
from tab_foundry.model.spec import ModelBuildSpec
from tab_foundry.training.runtime import resolve_training_device_name


def resolve_prior_training_device_name(
    cfg: DictConfig,
    *,
    spec: ModelBuildSpec,
    staged_surface: ResolvedStageSurface | None,
) -> str:
    del spec, staged_surface
    return resolve_training_device_name(cfg.runtime)


def seed_prior_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
