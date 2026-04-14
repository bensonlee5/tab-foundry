"""Shape-family-aware torch.compile dispatch helpers."""

from __future__ import annotations

from typing import Any

import torch

from tab_foundry.task_batching import task_batch_signature
from tab_foundry.types import TaskBatch

from .runtime import CompilePolicy

SignatureFamily = tuple[int, int, int]


def _signature_family(batch: TaskBatch) -> SignatureFamily:
    signature = task_batch_signature(batch)
    return int(signature[0]), int(signature[1]), int(signature[2])


def _signature_family_text(family: SignatureFamily) -> str:
    return f"{int(family[0])}x{int(family[1])}x{int(family[2])}"


class SignatureFamilyCompileDispatcher:
    """Compile one callable per shape family and dispatch lazily at runtime."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        compile_policy: CompilePolicy,
        max_families: int,
    ) -> None:
        if not compile_policy.enabled:
            raise ValueError("SignatureFamilyCompileDispatcher requires compile_policy.enabled=True")
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            raise RuntimeError("runtime.compile_model=true requires torch.compile support")
        self._model = model
        self._compile_fn = compile_fn
        self._compile_kwargs = compile_policy.torch_compile_kwargs()
        self._max_families = int(max_families)
        self._compiled_by_family: dict[SignatureFamily, Any] = {}
        self._family_hits = 0
        self._family_misses = 0
        self._family_switches = 0
        self._uncached_family_calls = 0
        self._last_family: SignatureFamily | None = None
        self._family_call_counts: dict[str, int] = {}

    def __call__(self, batch: TaskBatch):
        family = _signature_family(batch)
        family_text = _signature_family_text(family)
        if self._last_family is not None and family != self._last_family:
            self._family_switches += 1
        self._last_family = family
        self._family_call_counts[family_text] = self._family_call_counts.get(family_text, 0) + 1

        compiled = self._compiled_by_family.get(family)
        if compiled is not None:
            self._family_hits += 1
            return compiled(batch)

        self._family_misses += 1
        if len(self._compiled_by_family) >= self._max_families:
            self._uncached_family_calls += 1
            return self._model(batch)

        compiled = self._compile_fn(self._model, **self._compile_kwargs)
        self._compiled_by_family[family] = compiled
        return compiled(batch)

    def summary(self) -> dict[str, Any]:
        compiled_families = sorted(_signature_family_text(family) for family in self._compiled_by_family)
        return {
            "family_cache_hits": int(self._family_hits),
            "family_cache_misses": int(self._family_misses),
            "compiled_family_count": int(len(self._compiled_by_family)),
            "family_switch_count": int(self._family_switches),
            "uncached_family_call_count": int(self._uncached_family_calls),
            "compiled_families": compiled_families,
            "family_call_counts": dict(sorted(self._family_call_counts.items())),
        }
