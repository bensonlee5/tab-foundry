"""Optimizer factory with Muon fallback."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class OptimizerSelection:
    """Container for optimizer and selection metadata."""

    optimizers: list[tuple[str, torch.optim.Optimizer]]
    requested_name: str
    resolved_name: str
    fallback_reason: str | None = None


def _muon_lr_for_param(param: nn.Parameter, *, base_lr: float, scale_base: float) -> float:
    if param.ndim < 2:
        return base_lr
    n = int(param.shape[0])
    m = int(param.numel() // max(1, n))
    scale = float(scale_base) * math.sqrt(float(max(n, m)))
    return base_lr * scale


def _build_muon_params(
    params: list[nn.Parameter],
    *,
    base_lr: float,
    per_parameter_lr: bool,
    scale_base: float,
) -> list[nn.Parameter] | list[dict[str, Any]]:
    if not per_parameter_lr:
        return params
    return [
        {
            "params": [param],
            "lr": _muon_lr_for_param(param, base_lr=base_lr, scale_base=scale_base),
        }
        for param in params
    ]


def _embedding_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            ids.add(id(module.weight))
    return ids


def _partition_muon_params(
    model: nn.Module,
    params: list[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    embedding_ids = _embedding_param_ids(model)
    muon_params = [p for p in params if p.ndim == 2 and id(p) not in embedding_ids]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in params if id(p) not in muon_ids]
    return muon_params, adamw_params


def build_optimizer(
    model: nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
    extra_kwargs: dict[str, Any] | None = None,
    require_requested: bool = False,
    muon_per_parameter_lr: bool = True,
    muon_lr_scale_base: float = 0.2,
    muon_partition_non2d: bool = True,
) -> OptimizerSelection:
    """Build optimizer from config name."""

    extra_kwargs = extra_kwargs or {}
    params = [p for p in model.parameters() if p.requires_grad]
    requested = name.strip().lower()

    if requested == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra_kwargs)
        return OptimizerSelection(
            optimizers=[("adamw", opt)],
            requested_name=requested,
            resolved_name="adamw",
            fallback_reason=None,
        )

    if requested == "muon":
        try:
            from muon import Muon  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            if require_requested:
                raise RuntimeError(
                    "Requested optimizer 'muon' is unavailable and optimizer.require_requested=true."
                ) from exc
            fallback_reason = "muon_unavailable"
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra_kwargs)
            return OptimizerSelection(
                optimizers=[("adamw", opt)],
                requested_name=requested,
                resolved_name="adamw",
                fallback_reason=fallback_reason,
            )

        muon_sig = inspect.signature(Muon)
        allowed_muon_keys = set(muon_sig.parameters.keys())
        muon_kwargs = {k: v for k, v in extra_kwargs.items() if k in allowed_muon_keys}
        muon_source_params = params
        adamw_tail_params: list[nn.Parameter] = []
        if muon_partition_non2d:
            muon_source_params, adamw_tail_params = _partition_muon_params(model, params)
        if not muon_source_params:
            fallback_reason = "muon_no_eligible_params"
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra_kwargs)
            return OptimizerSelection(
                optimizers=[("adamw", opt)],
                requested_name=requested,
                resolved_name="adamw",
                fallback_reason=fallback_reason,
            )
        muon_params = _build_muon_params(
            muon_source_params,
            base_lr=lr,
            per_parameter_lr=muon_per_parameter_lr,
            scale_base=muon_lr_scale_base,
        )
        optimizers: list[tuple[str, torch.optim.Optimizer]] = []
        try:
            muon_opt = Muon(muon_params, lr=lr, weight_decay=weight_decay, **muon_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Muon optimizer: {exc}") from exc
        optimizers.append(("muon", muon_opt))
        if adamw_tail_params:
            adamw_tail = torch.optim.AdamW(
                adamw_tail_params,
                lr=lr,
                weight_decay=weight_decay,
                **extra_kwargs,
            )
            optimizers.append(("adamw", adamw_tail))
        resolved = "muon+adamw" if len(optimizers) == 2 else "muon"
        return OptimizerSelection(
            optimizers=optimizers,
            requested_name=requested,
            resolved_name=resolved,
            fallback_reason=None,
        )

    raise ValueError(f"Unsupported optimizer name: {name!r}")
