"""Optimizer helpers shared by trainer entrypoints."""

from __future__ import annotations

import torch


def _optimizer_lr_scales(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
) -> list[float]:
    if base_lr <= 0:
        return [1.0 for _ in optimizer.param_groups]
    return [float(group["lr"]) / float(base_lr) for group in optimizer.param_groups]


def _set_optimizer_base_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    scales: list[float],
) -> None:
    if len(scales) != len(optimizer.param_groups):
        raise RuntimeError("lr scales count does not match optimizer param groups")
    for group, scale in zip(optimizer.param_groups, scales, strict=True):
        group["lr"] = float(base_lr) * float(scale)


def _set_optimizer_training_mode(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    training: bool,
) -> None:
    method_name = "train" if training else "eval"
    for _name, optimizer in prepared_opts:
        method = getattr(optimizer, method_name, None)
        if callable(method):
            method()
