"""Runtime helpers for exact prior-dump training."""

from __future__ import annotations

import random
import sys

import numpy as np
from omegaconf import DictConfig
import torch

from tab_foundry.model.architectures.tabfoundry_staged.resolved import ResolvedStageSurface
from tab_foundry.model.spec import ModelBuildSpec


def resolve_prior_training_device_name(
    cfg: DictConfig,
    *,
    spec: ModelBuildSpec,
    staged_surface: ResolvedStageSurface | None,
    resolve_device_fn,
) -> str:
    requested_device = str(getattr(cfg.runtime, "device", "auto") or "auto").strip()
    resolved_device = resolve_device_fn(requested_device)
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


def seed_prior_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
