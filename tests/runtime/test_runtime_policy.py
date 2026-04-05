from __future__ import annotations

from omegaconf import OmegaConf
import pytest

import tab_foundry.training.runtime as training_runtime_module
from tab_foundry.training.runtime import (
    resolve_cpu_mode,
    resolve_grad_accum_steps,
    resolve_mixed_precision,
)


def test_runtime_cpu_resolution() -> None:
    cfg = OmegaConf.create({"device": "cpu", "mixed_precision": "no"})
    assert resolve_cpu_mode(cfg) is True
    assert resolve_mixed_precision(cfg) == "no"


def test_runtime_auto_resolution_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cpu")
    cfg = OmegaConf.create({"device": "auto", "mixed_precision": "bf16"})
    assert resolve_cpu_mode(cfg) is True
    assert resolve_mixed_precision(cfg) == "bf16"
    assert resolve_mixed_precision(cfg, override="no") == "no"


def test_runtime_auto_resolution_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cuda")
    cfg = OmegaConf.create({"device": "auto", "mixed_precision": "bf16"})
    assert resolve_cpu_mode(cfg) is False


def test_runtime_auto_resolution_rejects_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "mps")
    cfg = OmegaConf.create({"device": "auto", "mixed_precision": "bf16"})

    with pytest.raises(ValueError, match="runtime.device='auto' resolved to 'mps'"):
        _ = resolve_cpu_mode(cfg)


def test_runtime_grad_accum_resolution() -> None:
    cfg = OmegaConf.create({"grad_accum_steps": 8})
    assert resolve_grad_accum_steps(cfg) == 8
    assert resolve_grad_accum_steps(cfg, override=3) == 3
