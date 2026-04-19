from __future__ import annotations

import builtins
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tab_foundry.model.components.rational import RationalActivation, rational_parameter_ids
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


def test_non_muon_optimizer_rejects_momentum_extra_kwarg() -> None:
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="optimizer\\.momentum is only supported"):
        _ = build_optimizer(
            model,
            name="adamw",
            lr=1e-3,
            weight_decay=0.0,
            extra_kwargs={"betas": (0.9, 0.95), "momentum": 0.95},
        )


def test_muon_missing_dependency_falls_back_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 2)
    original_import = builtins.__import__

    def _missing_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "muon":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_import)

    sel = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={"betas": (0.9, 0.95)},
        require_requested=False,
    )

    assert len(sel.optimizers) == 1
    assert isinstance(sel.optimizers[0][1], torch.optim.AdamW)
    assert sel.resolved_name == "adamw"
    assert sel.fallback_reason == "muon_unavailable"


def test_muon_required_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 2)
    original_import = builtins.__import__

    def _missing_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "muon":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_import)

    try:
        sel = build_optimizer(
            model,
            name="muon",
            lr=1e-3,
            weight_decay=0.0,
            extra_kwargs={},
            require_requested=True,
        )
        assert sel.resolved_name.startswith("muon")
    except RuntimeError as exc:
        assert "Requested optimizer 'muon' is unavailable" in str(exc)


@pytest.mark.parametrize("require_requested", [False, True])
def test_muon_init_failures_raise(
    monkeypatch: pytest.MonkeyPatch,
    require_requested: bool,
) -> None:
    class _RaisingMuon:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated muon init bug")

    monkeypatch.setitem(
        sys.modules,
        "muon",
        SimpleNamespace(Muon=_RaisingMuon, SingleDeviceMuon=_RaisingMuon),
    )

    model = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match="Muon initialization failed"):
        _ = build_optimizer(
            model,
            name="muon",
            lr=1e-3,
            weight_decay=0.0,
            extra_kwargs={},
            require_requested=require_requested,
        )


def test_muon_no_eligible_params_falls_back_without_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EmbeddingOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.bias = nn.Parameter(torch.zeros(8))

    monkeypatch.setitem(sys.modules, "muon", SimpleNamespace(Muon=object, SingleDeviceMuon=object))

    model = _EmbeddingOnly()
    sel = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={},
        require_requested=False,
    )

    assert len(sel.optimizers) == 1
    assert isinstance(sel.optimizers[0][1], torch.optim.AdamW)
    assert sel.resolved_name == "adamw"
    assert sel.fallback_reason == "muon_no_eligible_params"


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


def test_muon_single_device_step_supports_per_parameter_lr_without_process_group() -> None:
    _ = pytest.importorskip("muon")

    model = nn.Linear(8, 4)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    selection = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={},
        require_requested=True,
        muon_per_parameter_lr=True,
        muon_partition_non2d=True,
    )

    assert selection.resolved_name == "muon+adamw"
    for _name, optimizer in selection.optimizers:
        optimizer.step(None)


def test_muon_step_skips_inactive_2d_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMuon:
        def __init__(self, params, lr: float, weight_decay: float) -> None:
            del lr
            self.param_groups: list[dict[str, object]] = []
            self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}
            self.seen_param_ids_per_step: list[list[int]] = []
            raw_params = list(params)
            if raw_params and isinstance(raw_params[0], dict):
                for raw_group in raw_params:
                    group = dict(raw_group)
                    group["params"] = list(group["params"])
                    group.setdefault("weight_decay", weight_decay)
                    group.setdefault("momentum", 0.95)
                    self.param_groups.append(group)
            else:
                self.param_groups.append(
                    {
                        "params": list(raw_params),
                        "weight_decay": weight_decay,
                        "momentum": 0.95,
                    }
                )

        def step(self) -> None:
            seen: list[int] = []
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        raise RuntimeError("Muon received a parameter with grad=None")
                    if param not in self.state:
                        self.state[param] = {"momentum_buffer": torch.zeros_like(param)}
                    seen.append(id(param))
            self.seen_param_ids_per_step.append(seen)

    class _PartiallyUsed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Linear(8, 4, bias=False)
            self.unused = nn.Linear(8, 4, bias=False)
            self.bias = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.used(x) + self.bias

    monkeypatch.setitem(
        sys.modules,
        "muon",
        SimpleNamespace(Muon=_FakeMuon, SingleDeviceMuon=_FakeMuon),
    )

    model = _PartiallyUsed()
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    selection = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={},
        require_requested=True,
        muon_per_parameter_lr=True,
        muon_partition_non2d=True,
    )

    muon_opt = next(optimizer for name, optimizer in selection.optimizers if name == "muon")
    for _name, optimizer in selection.optimizers:
        optimizer.step(None)

    assert selection.resolved_name == "muon+adamw"
    assert muon_opt.seen_param_ids_per_step == [[id(model.used.weight)]]
    assert model.used.weight in muon_opt.state
    assert model.unused.weight not in muon_opt.state


def test_muon_step_is_noop_when_all_muon_params_are_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMuon:
        def __init__(self, params, lr: float, weight_decay: float) -> None:
            del lr
            self.param_groups: list[dict[str, object]] = []
            self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}
            self.seen_param_ids_per_step: list[list[int]] = []
            raw_params = list(params)
            if raw_params and isinstance(raw_params[0], dict):
                for raw_group in raw_params:
                    group = dict(raw_group)
                    group["params"] = list(group["params"])
                    group.setdefault("weight_decay", weight_decay)
                    group.setdefault("momentum", 0.95)
                    self.param_groups.append(group)
            else:
                self.param_groups.append(
                    {
                        "params": list(raw_params),
                        "weight_decay": weight_decay,
                        "momentum": 0.95,
                    }
                )

        def step(self) -> None:
            seen: list[int] = []
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        raise RuntimeError("Muon received a parameter with grad=None")
                    if param not in self.state:
                        self.state[param] = {"momentum_buffer": torch.zeros_like(param)}
                    seen.append(id(param))
            self.seen_param_ids_per_step.append(seen)

    class _InactiveMuonParams(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unused = nn.Linear(8, 4, bias=False)
            self.bias = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros((x.shape[0], 4)) + self.bias

    monkeypatch.setitem(
        sys.modules,
        "muon",
        SimpleNamespace(Muon=_FakeMuon, SingleDeviceMuon=_FakeMuon),
    )

    model = _InactiveMuonParams()
    x = torch.randn(4, 8)
    loss = model(x).pow(2).mean()
    loss.backward()

    selection = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.0,
        extra_kwargs={},
        require_requested=True,
        muon_per_parameter_lr=True,
        muon_partition_non2d=True,
    )

    muon_opt = next(optimizer for name, optimizer in selection.optimizers if name == "muon")
    for _name, optimizer in selection.optimizers:
        optimizer.step(None)

    assert selection.resolved_name == "muon+adamw"
    assert muon_opt.seen_param_ids_per_step == []
    assert model.unused.weight not in muon_opt.state


def test_muon_forwards_real_momentum_kwarg_to_muon_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeMuon:
        def __init__(self, params, lr: float, weight_decay: float, momentum: float) -> None:
            captured["params"] = list(params)
            captured["lr"] = lr
            captured["weight_decay"] = weight_decay
            captured["momentum"] = momentum
            self.param_groups = [{"params": list(params), "lr": lr, "momentum": momentum}]

        def step(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "muon",
        SimpleNamespace(Muon=_FakeMuon, SingleDeviceMuon=_FakeMuon),
    )

    model = nn.Linear(8, 4, bias=False)
    selection = build_optimizer(
        model,
        name="muon",
        lr=1e-3,
        weight_decay=0.01,
        extra_kwargs={"betas": (0.9, 0.95), "momentum": 0.975},
        require_requested=True,
        muon_per_parameter_lr=False,
        muon_partition_non2d=False,
    )

    assert selection.resolved_name == "muon"
    assert captured["momentum"] == pytest.approx(0.975)
    assert captured["lr"] == pytest.approx(1e-3)
    assert captured["weight_decay"] == pytest.approx(0.01)


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


def test_adamw_optimizer_groups_rational_params_without_weight_decay() -> None:
    class _ToyRational(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_in = nn.Linear(4, 4)
            self.activation = RationalActivation()
            self.linear_out = nn.Linear(4, 2)

    model = _ToyRational()
    selection = build_optimizer(
        model,
        name="adamw",
        lr=1e-3,
        weight_decay=0.1,
        extra_kwargs={},
        require_requested=True,
    )

    optimizer = selection.optimizers[0][1]
    rational_ids = rational_parameter_ids(model)

    assert len(optimizer.param_groups) == 2
    assert any(group["weight_decay"] == pytest.approx(0.1) for group in optimizer.param_groups)
    zero_decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == pytest.approx(0.0)
    )
    assert {id(param) for param in zero_decay_group["params"]} == rational_ids


def test_adamw_optimizer_keeps_single_param_group_without_rational_params() -> None:
    selection = build_optimizer(
        nn.Linear(4, 2),
        name="adamw",
        lr=1e-3,
        weight_decay=0.1,
        extra_kwargs={},
        require_requested=True,
    )

    assert len(selection.optimizers[0][1].param_groups) == 1


def test_schedulefree_optimizer_groups_rational_params_without_weight_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeScheduleFreeAdamW:
        def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float]) -> None:
            del betas
            self.param_groups: list[dict[str, object]] = []
            raw_params = list(params)
            if raw_params and isinstance(raw_params[0], dict):
                for raw_group in raw_params:
                    group = dict(raw_group)
                    group["params"] = list(group["params"])
                    group.setdefault("lr", lr)
                    group.setdefault("weight_decay", weight_decay)
                    self.param_groups.append(group)
            else:
                self.param_groups.append(
                    {
                        "params": list(raw_params),
                        "lr": lr,
                        "weight_decay": weight_decay,
                    }
                )

    class _ToyRational(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_in = nn.Linear(4, 4)
            self.activation = RationalActivation()
            self.linear_out = nn.Linear(4, 2)

    monkeypatch.setitem(
        sys.modules,
        "schedulefree",
        SimpleNamespace(AdamWScheduleFree=_FakeScheduleFreeAdamW),
    )

    model = _ToyRational()
    selection = build_optimizer(
        model,
        name="schedulefree_adamw",
        lr=1e-3,
        weight_decay=0.1,
        extra_kwargs={"betas": (0.9, 0.95)},
        require_requested=True,
    )

    optimizer = selection.optimizers[0][1]
    rational_ids = rational_parameter_ids(model)

    assert len(optimizer.param_groups) == 2
    zero_decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == pytest.approx(0.0)
    )
    assert {id(param) for param in zero_decay_group["params"]} == rational_ids
