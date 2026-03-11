from __future__ import annotations

import importlib.util
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tab_foundry.training.optimizer import (
    _build_muon_params,
    _muon_lr_for_param,
    _partition_muon_params,
    build_optimizer,
)


def test_optimizer_unknown_name_raises() -> None:
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError):
        _ = build_optimizer(
            model,
            name="unknown",
            lr=1e-3,
            weight_decay=0.0,
            extra_kwargs={},
        )


def test_muon_selection_or_fallback() -> None:
    model = nn.Linear(4, 2)
    muon_available = importlib.util.find_spec("muon") is not None

    sel = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={"betas": (0.9, 0.95)},
        require_requested=False,
    )

    if muon_available:
        assert sel.resolved_name.startswith("muon")
        assert sel.fallback_reason is None
        assert len(sel.optimizers) >= 1
    else:
        assert len(sel.optimizers) == 1
        assert isinstance(sel.optimizers[0][1], torch.optim.AdamW)
        assert sel.resolved_name == "adamw"
        assert sel.fallback_reason == "muon_unavailable"


def test_muon_required_behavior() -> None:
    model = nn.Linear(4, 2)
    muon_available = importlib.util.find_spec("muon") is not None

    if muon_available:
        sel = build_optimizer(
            model,
            name="muon",
            lr=1e-3,
            weight_decay=0.0,
            extra_kwargs={},
            require_requested=True,
        )
        assert sel.resolved_name.startswith("muon")
    else:
        with pytest.raises(RuntimeError):
            _ = build_optimizer(
                model,
                name="muon",
                lr=1e-3,
                weight_decay=0.0,
                extra_kwargs={},
                require_requested=True,
            )


def test_muon_lr_scale_for_matrix_and_vector_params() -> None:
    matrix = nn.Parameter(torch.zeros((8, 4)))
    vector = nn.Parameter(torch.zeros((8,)))
    base_lr = 1e-3
    matrix_lr = _muon_lr_for_param(matrix, base_lr=base_lr, scale_base=0.2)
    vector_lr = _muon_lr_for_param(vector, base_lr=base_lr, scale_base=0.2)
    assert matrix_lr == pytest.approx(base_lr * 0.2 * math.sqrt(8.0))
    assert vector_lr == pytest.approx(base_lr)


def test_muon_param_group_builder() -> None:
    params = [
        nn.Parameter(torch.zeros((4, 4))),
        nn.Parameter(torch.zeros((4,))),
    ]

    grouped = _build_muon_params(
        params,
        base_lr=1e-3,
        per_parameter_lr=True,
        scale_base=0.2,
    )
    assert isinstance(grouped, list)
    assert isinstance(grouped[0], dict)
    first_group = grouped[0]
    second_group = grouped[1]
    assert isinstance(first_group, dict)
    assert isinstance(second_group, dict)
    assert first_group["lr"] == pytest.approx(1e-3 * 0.2 * math.sqrt(4.0))
    assert second_group["lr"] == pytest.approx(1e-3)


def test_partition_muon_params_excludes_embeddings_and_non_2d() -> None:
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.linear = nn.Linear(8, 4)
            self.bias_only = nn.Parameter(torch.zeros(4))

    model = _Toy()
    params = [p for p in model.parameters() if p.requires_grad]
    muon_params, adamw_params = _partition_muon_params(model, params)
    assert all(p.ndim == 2 for p in muon_params)
    assert all(id(p) != id(model.embed.weight) for p in muon_params)
    assert set(id(p) for p in muon_params).isdisjoint(set(id(p) for p in adamw_params))
    assert set(id(p) for p in muon_params).union(set(id(p) for p in adamw_params)) == set(
        id(p) for p in params
    )


def test_schedulefree_optimizer_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeScheduleFreeAdamW:
        def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float]) -> None:
            self.params = list(params)
            self.lr = lr
            self.weight_decay = weight_decay
            self.betas = betas
            self.param_groups = [{"lr": lr}]

    monkeypatch.setitem(
        sys.modules,
        "schedulefree",
        SimpleNamespace(AdamWScheduleFree=_FakeScheduleFreeAdamW),
    )

    model = nn.Linear(4, 2)
    sel = build_optimizer(
        model,
        name="schedulefree_adamw",
        lr=4.0e-3,
        weight_decay=0.0,
        extra_kwargs={"betas": (0.9, 0.95)},
        require_requested=True,
    )

    assert sel.resolved_name == "schedulefree_adamw"
    assert sel.fallback_reason is None
    assert sel.optimizers[0][0] == "schedulefree_adamw"
