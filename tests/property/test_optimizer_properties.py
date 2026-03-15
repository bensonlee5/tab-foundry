from __future__ import annotations

import math
import builtins

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch
from torch import nn

from tab_foundry.training.optimizer import (
    _build_muon_params,
    _muon_lr_for_param,
    _partition_muon_params,
    build_optimizer,
)


@settings(deadline=None, max_examples=35)
@given(
    n=st.integers(min_value=1, max_value=64),
    m=st.integers(min_value=1, max_value=64),
    larger_n=st.integers(min_value=1, max_value=128),
    larger_m=st.integers(min_value=1, max_value=128),
    base_lr=st.floats(min_value=1.0e-6, max_value=1.0e-1, allow_nan=False, allow_infinity=False),
    scale_base=st.floats(min_value=1.0e-3, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_muon_lr_scaling_is_base_for_vectors_and_monotonic_for_larger_matrices(
    n: int,
    m: int,
    larger_n: int,
    larger_m: int,
    base_lr: float,
    scale_base: float,
) -> None:
    small = nn.Parameter(torch.zeros((n, m)))
    vector = nn.Parameter(torch.zeros((n,)))
    bigger_dim = max(max(n, m), max(larger_n, larger_m))
    large = nn.Parameter(torch.zeros((bigger_dim, bigger_dim)))

    small_lr = _muon_lr_for_param(small, base_lr=base_lr, scale_base=scale_base)
    large_lr = _muon_lr_for_param(large, base_lr=base_lr, scale_base=scale_base)
    vector_lr = _muon_lr_for_param(vector, base_lr=base_lr, scale_base=scale_base)

    assert vector_lr == pytest.approx(base_lr)
    assert small_lr == pytest.approx(base_lr * scale_base * math.sqrt(float(max(n, m))))
    assert large_lr >= small_lr


@settings(deadline=None, max_examples=30)
@given(
    matrix_rows=st.integers(min_value=1, max_value=16),
    matrix_cols=st.integers(min_value=1, max_value=16),
    vector_size=st.integers(min_value=1, max_value=16),
    base_lr=st.floats(min_value=1.0e-6, max_value=1.0e-1, allow_nan=False, allow_infinity=False),
    scale_base=st.floats(min_value=1.0e-3, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_build_muon_params_preserves_param_identity_and_count(
    matrix_rows: int,
    matrix_cols: int,
    vector_size: int,
    base_lr: float,
    scale_base: float,
) -> None:
    params = [
        nn.Parameter(torch.zeros((matrix_rows, matrix_cols))),
        nn.Parameter(torch.zeros((vector_size,))),
    ]

    raw = _build_muon_params(
        params,
        base_lr=base_lr,
        per_parameter_lr=False,
        scale_base=scale_base,
    )
    grouped = _build_muon_params(
        params,
        base_lr=base_lr,
        per_parameter_lr=True,
        scale_base=scale_base,
    )

    assert raw == params
    assert len(grouped) == len(params)
    grouped_params = [group["params"][0] for group in grouped]
    assert [id(param) for param in grouped_params] == [id(param) for param in params]
    assert grouped[0]["lr"] == pytest.approx(
        _muon_lr_for_param(params[0], base_lr=base_lr, scale_base=scale_base)
    )
    assert grouped[1]["lr"] == pytest.approx(base_lr)


@settings(deadline=None, max_examples=25)
@given(
    vocab=st.integers(min_value=2, max_value=32),
    embed_dim=st.integers(min_value=1, max_value=16),
    hidden=st.integers(min_value=1, max_value=16),
)
def test_partition_muon_params_is_lossless_and_disjoint(
    vocab: int,
    embed_dim: int,
    hidden: int,
) -> None:
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, embed_dim)
            self.linear = nn.Linear(embed_dim, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.bias_only = nn.Parameter(torch.zeros(hidden))

    model = _Toy()
    params = [param for param in model.parameters() if param.requires_grad]

    muon_params, adamw_params = _partition_muon_params(model, params)

    muon_ids = {id(param) for param in muon_params}
    adamw_ids = {id(param) for param in adamw_params}
    all_ids = {id(param) for param in params}

    assert muon_ids.isdisjoint(adamw_ids)
    assert muon_ids | adamw_ids == all_ids
    assert all(param.ndim == 2 for param in muon_params)
    assert id(model.embed.weight) not in muon_ids


@settings(deadline=None, max_examples=10)
@given(requested=st.sampled_from(("muon", "schedulefree_adamw")))
def test_build_optimizer_fallback_metadata_is_coherent_when_optional_optimizer_is_unavailable(
    requested: str,
) -> None:
    model = nn.Linear(4, 2)
    missing_module = "muon" if requested == "muon" else "schedulefree"
    original_import = builtins.__import__

    with pytest.MonkeyPatch.context() as monkeypatch:
        def _missing_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == missing_module:
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _missing_import)

        selection = build_optimizer(
            model,
            name=requested,
            lr=1.0e-3,
            weight_decay=0.0,
            extra_kwargs={},
            require_requested=False,
        )

    assert selection.requested_name == requested
    assert selection.resolved_name == "adamw"
    assert len(selection.optimizers) == 1
    assert isinstance(selection.optimizers[0][1], torch.optim.AdamW)
    if requested == "muon":
        assert selection.fallback_reason == "muon_unavailable"
    else:
        assert selection.fallback_reason == "schedulefree_unavailable"
