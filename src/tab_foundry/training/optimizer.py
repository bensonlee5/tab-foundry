"""Optimizer factory with Muon fallback."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from types import MethodType
from typing import Any

import torch
from torch import nn

from tab_foundry.model.components.rational import rational_parameter_ids


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


def _partition_rational_params(
    model: nn.Module,
    params: list[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    rational_ids = rational_parameter_ids(model)
    if not rational_ids:
        return params, []
    rational_params = [param for param in params if id(param) in rational_ids]
    rational_param_ids_set = {id(param) for param in rational_params}
    base_params = [param for param in params if id(param) not in rational_param_ids_set]
    return base_params, rational_params


def _build_adamw_param_groups(
    model: nn.Module,
    params: list[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> list[nn.Parameter] | list[dict[str, Any]]:
    base_params, rational_params = _partition_rational_params(model, params)
    if not rational_params:
        return params
    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    param_groups.append(
        {
            "params": rational_params,
            "lr": lr,
            "weight_decay": 0.0,
        }
    )
    return param_groups


def _wrap_step_to_ignore_unused_closure(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    step_sig = inspect.signature(type(optimizer).step)
    if "closure" in step_sig.parameters:
        return optimizer

    original_step = optimizer.step

    def _step_with_optional_closure(
        self: torch.optim.Optimizer,
        closure: Any = None,
    ) -> Any:
        del self, closure
        return original_step()

    setattr(optimizer, "step", MethodType(_step_with_optional_closure, optimizer))
    return optimizer


def _wrap_muon_step(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Wrap Muon step() so inactive params are skipped and closure remains optional."""

    step_sig = inspect.signature(type(optimizer).step)
    accepts_closure = "closure" in step_sig.parameters
    original_step = optimizer.step

    def _step_with_optional_closure_and_active_params_only(
        self: torch.optim.Optimizer,
        closure: Any = None,
    ) -> Any:
        del self
        original_group_params = [list(group["params"]) for group in optimizer.param_groups]
        any_active = False
        try:
            for group, original_params in zip(
                optimizer.param_groups,
                original_group_params,
                strict=True,
            ):
                active_params = [param for param in original_params if param.grad is not None]
                group["params"] = active_params
                any_active = any_active or bool(active_params)
            if not any_active:
                return None
            if accepts_closure:
                if closure is None:
                    return original_step()
                return original_step(closure)
            del closure
            return original_step()
        finally:
            for group, original_params in zip(
                optimizer.param_groups,
                original_group_params,
                strict=True,
            ):
                group["params"] = original_params

    setattr(
        optimizer,
        "step",
        MethodType(_step_with_optional_closure_and_active_params_only, optimizer),
    )
    return optimizer


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
    adamw_params = _build_adamw_param_groups(model, params, lr=lr, weight_decay=weight_decay)
    requested = name.strip().lower()
    if requested != "muon" and "momentum" in extra_kwargs:
        raise ValueError(
            "optimizer.momentum is only supported for the requested optimizer 'muon'; "
            f"got requested optimizer {requested!r}"
        )

    if requested == "adamw":
        opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, **extra_kwargs)
        return OptimizerSelection(
            optimizers=[("adamw", opt)],
            requested_name=requested,
            resolved_name="adamw",
            fallback_reason=None,
        )

    if requested == "schedulefree_adamw":
        try:
            import schedulefree  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            if require_requested:
                raise RuntimeError(
                    "Requested optimizer 'schedulefree_adamw' is unavailable and "
                    "optimizer.require_requested=true."
                ) from exc
            fallback_reason = "schedulefree_unavailable"
            opt = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=weight_decay,
                **extra_kwargs,
            )
            return OptimizerSelection(
                optimizers=[("adamw", opt)],
                requested_name=requested,
                resolved_name="adamw",
                fallback_reason=fallback_reason,
            )

        optimizer_cls = schedulefree.AdamWScheduleFree
        optimizer_sig = inspect.signature(optimizer_cls)
        allowed_keys = set(optimizer_sig.parameters.keys())
        allowed_kwargs = {key: value for key, value in extra_kwargs.items() if key in allowed_keys}
        opt = optimizer_cls(adamw_params, lr=lr, weight_decay=weight_decay, **allowed_kwargs)
        return OptimizerSelection(
            optimizers=[("schedulefree_adamw", opt)],
            requested_name=requested,
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        )

    if requested == "muon":
        try:
            import muon as muon_module  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            if require_requested:
                raise RuntimeError(
                    "Requested optimizer 'muon' is unavailable and optimizer.require_requested=true."
                ) from exc
            fallback_reason = "muon_unavailable"
            fallback_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "momentum"}
            opt = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=weight_decay,
                **fallback_extra_kwargs,
            )
            return OptimizerSelection(
                optimizers=[("adamw", opt)],
                requested_name=requested,
                resolved_name="adamw",
                fallback_reason=fallback_reason,
            )

        distributed_muon_cls = getattr(muon_module, "Muon", None)
        if not callable(distributed_muon_cls):
            raise RuntimeError("Installed muon package does not export Muon.")

        muon_source_params = params
        adamw_tail_params: list[nn.Parameter] = []
        if muon_partition_non2d:
            muon_source_params, adamw_tail_params = _partition_muon_params(model, params)
            if not muon_source_params:
                fallback_reason = "muon_no_eligible_params"
                fallback_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "momentum"}
                opt = torch.optim.AdamW(
                    adamw_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **fallback_extra_kwargs,
                )
                return OptimizerSelection(
                    optimizers=[("adamw", opt)],
                    requested_name=requested,
                    resolved_name="adamw",
                fallback_reason=fallback_reason,
            )
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        use_single_device_muon = not dist_ready
        muon_cls = distributed_muon_cls
        if use_single_device_muon:
            single_device_muon_cls = getattr(muon_module, "SingleDeviceMuon", None)
            if not callable(single_device_muon_cls):
                if require_requested:
                    raise RuntimeError(
                        "Requested optimizer 'muon' requires SingleDeviceMuon when no distributed "
                        "process group is initialized."
                    )
                fallback_reason = "muon_single_device_unavailable"
                fallback_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "momentum"}
                opt = torch.optim.AdamW(
                    adamw_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **fallback_extra_kwargs,
                )
                return OptimizerSelection(
                    optimizers=[("adamw", opt)],
                    requested_name=requested,
                    resolved_name="adamw",
                    fallback_reason=fallback_reason,
                )
            muon_cls = single_device_muon_cls

        muon_sig = inspect.signature(muon_cls)
        allowed_muon_keys = set(muon_sig.parameters.keys())
        muon_kwargs = {k: v for k, v in extra_kwargs.items() if k in allowed_muon_keys}
        muon_params = _build_muon_params(
            muon_source_params,
            base_lr=lr,
            per_parameter_lr=muon_per_parameter_lr,
            scale_base=muon_lr_scale_base,
        )
        optimizers: list[tuple[str, torch.optim.Optimizer]] = []
        try:
            muon_opt = muon_cls(muon_params, lr=lr, weight_decay=weight_decay, **muon_kwargs)
        except Exception as exc:
            raise RuntimeError("Muon initialization failed for requested optimizer 'muon'.") from exc
        muon_opt = _wrap_muon_step(muon_opt)
        optimizers.append(("muon", muon_opt))
        if adamw_tail_params:
            adamw_tail_groups = _build_adamw_param_groups(
                model,
                adamw_tail_params,
                lr=lr,
                weight_decay=weight_decay,
            )
            adamw_tail_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k != "momentum"}
            adamw_tail = torch.optim.AdamW(
                adamw_tail_groups,
                lr=lr,
                weight_decay=weight_decay,
                **adamw_tail_extra_kwargs,
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
