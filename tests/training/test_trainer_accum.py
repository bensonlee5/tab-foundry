from __future__ import annotations

import pytest
from omegaconf import OmegaConf
import torch

from tab_foundry.training.distributed import (
    _global_mean_from_local,
    _reduce_count_scalar,
    _reduce_sum_scalar,
)
from tab_foundry.training.trainer import _resolve_grad_accum_steps


class _FakeReduceAccelerator:
    def __init__(self, factor: int) -> None:
        self.factor = int(factor)
        self.device = torch.device("cpu")

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        return tensor * self.factor


def test_resolve_grad_accum_steps_default() -> None:
    cfg = OmegaConf.create({})
    assert _resolve_grad_accum_steps(cfg) == 1


def test_resolve_grad_accum_steps_positive_value() -> None:
    cfg = OmegaConf.create({"grad_accum_steps": 4})
    assert _resolve_grad_accum_steps(cfg) == 4


def test_resolve_grad_accum_steps_rejects_non_positive() -> None:
    cfg = OmegaConf.create({"grad_accum_steps": 0})
    with pytest.raises(ValueError, match="must be >= 1"):
        _ = _resolve_grad_accum_steps(cfg)


def test_reduce_scalar_helpers() -> None:
    accelerator = _FakeReduceAccelerator(factor=3)
    assert _reduce_sum_scalar(accelerator, 2.5, device=accelerator.device) == pytest.approx(7.5)
    assert _reduce_count_scalar(accelerator, 4, device=accelerator.device) == 12


def test_global_mean_from_local_with_reduction() -> None:
    accelerator = _FakeReduceAccelerator(factor=4)
    value = _global_mean_from_local(
        accelerator,
        local_sum=3.0,
        local_count=2,
        device=accelerator.device,
        default=float("nan"),
    )
    assert value == pytest.approx(1.5)
