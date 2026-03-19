from __future__ import annotations

import tab_foundry.training.distributed as distributed_module
import pytest
from omegaconf import OmegaConf
import torch

from tab_foundry.training.distributed import (
    _global_mean_from_local,
    _reduce_keyed_weighted_scalars,
    _reduce_count_scalar,
    _reduce_sum_scalar,
)
from tab_foundry.training.trainer import _resolve_grad_accum_steps


class _FakeReduceAccelerator:
    def __init__(self, factor: int) -> None:
        self.factor = int(factor)
        self.device = torch.device("cpu")
        self.is_main_process = True

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


def test_reduce_keyed_weighted_scalars_uses_union_of_rank_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _FakeReduceAccelerator(factor=1)
    remote_keys = ["c", "b"]
    remote_packed = torch.tensor([0.0, 0.0, 3.0, 1.5, 4.0, 2.0], dtype=torch.float64)

    def _gather_object(local_keys: list[str]) -> list[list[str]]:
        return [list(local_keys), remote_keys]

    def _broadcast_object_list(object_list: list[object], from_process: int = 0) -> None:
        assert from_process == 0
        assert object_list[0] == ["a", "b", "c"]

    def _reduce(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        return tensor + remote_packed.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(distributed_module, "gather_object", _gather_object)
    monkeypatch.setattr(distributed_module, "broadcast_object_list", _broadcast_object_list)
    monkeypatch.setattr(accelerator, "reduce", _reduce)

    reduced_sums, reduced_weights = _reduce_keyed_weighted_scalars(
        accelerator,
        weighted_sums={"b": 2.0, "a": 1.0},
        weights={"a": 0.5},
        device=accelerator.device,
    )

    assert reduced_sums == pytest.approx({"a": 1.0, "b": 5.0, "c": 4.0})
    assert reduced_weights == pytest.approx({"a": 0.5, "b": 1.5, "c": 2.0})
