from __future__ import annotations

import torch

from tab_foundry.training.compile_dispatch import SignatureFamilyCompileDispatcher
from tab_foundry.training.runtime import CompilePolicy
from tab_foundry.types import TaskBatch


def _batch(
    *,
    n_train: int = 8,
    n_test: int = 3,
    n_features: int = 5,
    num_classes: int = 2,
) -> TaskBatch:
    x_train = torch.randn(n_train, n_features)
    y_train = torch.randint(0, num_classes, (n_train,), dtype=torch.int64)
    x_test = torch.randn(n_test, n_features)
    y_test = torch.randint(0, num_classes, (n_test,), dtype=torch.int64)
    return TaskBatch(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={},
        num_classes=num_classes,
    )


class _Model(torch.nn.Module):
    def forward(self, batch: TaskBatch) -> torch.Tensor:
        return batch.x_test


def _compile_policy() -> CompilePolicy:
    return CompilePolicy(enabled=True, backend="eager", mode="default", dynamic=True)


def test_signature_family_dispatcher_compiles_once_per_exact_signature(
    monkeypatch,
) -> None:
    compile_calls: list[tuple[int, int, int, int | None]] = []

    def _fake_compile(model, **_kwargs):
        recorded = False

        def _wrapped(batch: TaskBatch):
            nonlocal recorded
            if not recorded:
                compile_calls.append(
                    (8, 3, 5, int(batch.num_classes) if batch.num_classes is not None else None)
                )
                recorded = True
            return model(batch)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)
    dispatcher = SignatureFamilyCompileDispatcher(
        _Model(),
        compile_policy=_compile_policy(),
        max_families=1,
    )

    first = _batch(num_classes=2)
    second = _batch(num_classes=4)
    repeated = _batch(num_classes=2)

    _ = dispatcher(first)
    _ = dispatcher(second)
    _ = dispatcher(repeated)

    assert compile_calls == [(8, 3, 5, 2), (8, 3, 5, 4)]
    assert dispatcher.summary() == {
        "family_cache_hits": 1,
        "family_cache_misses": 2,
        "compiled_family_count": 1,
        "family_switch_count": 0,
        "uncached_family_call_count": 0,
        "compiled_families": ["8x3x5"],
        "family_call_counts": {"8x3x5": 3},
    }


def test_signature_family_dispatcher_keeps_family_cap_on_coarse_key(
    monkeypatch,
) -> None:
    compile_calls: list[tuple[int, int, int, int | None]] = []

    def _fake_compile(model, **_kwargs):
        def _wrapped(batch: TaskBatch):
            compile_calls.append(
                (
                    int(batch.x_train.shape[0]),
                    int(batch.x_test.shape[0]),
                    int(batch.x_train.shape[1]),
                    int(batch.num_classes) if batch.num_classes is not None else None,
                )
            )
            return model(batch)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)
    dispatcher = SignatureFamilyCompileDispatcher(
        _Model(),
        compile_policy=_compile_policy(),
        max_families=1,
    )

    _ = dispatcher(_batch(num_classes=2))
    _ = dispatcher(_batch(num_classes=4))
    _ = dispatcher(_batch(n_train=9, num_classes=2))

    assert compile_calls == [(8, 3, 5, 2), (8, 3, 5, 4)]
    assert dispatcher.summary() == {
        "family_cache_hits": 0,
        "family_cache_misses": 3,
        "compiled_family_count": 1,
        "family_switch_count": 1,
        "uncached_family_call_count": 1,
        "compiled_families": ["8x3x5"],
        "family_call_counts": {"8x3x5": 2, "9x3x5": 1},
    }
