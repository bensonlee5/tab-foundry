from __future__ import annotations

from contextlib import nullcontext

from omegaconf import OmegaConf
import pytest

import tab_foundry.training.runtime as training_runtime_module
from tab_foundry.training.runtime import (
    resolve_compile_policy,
    resolve_compile_model,
    resolve_compile_shape_dispatch_policy,
    resolve_cpu_mode,
    resolve_cuda_graph_capture_policy,
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


def test_runtime_compile_model_resolution_coerces_bool_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(training_runtime_module.torch, "compile", lambda model, **_kwargs: model)
    cfg = OmegaConf.create(
        {
            "device": "auto",
            "mixed_precision": "bf16",
            "compile_model": "yes",
            "compile_dynamic": "on",
            "trace_activations": "false",
        }
    )

    policy = resolve_compile_policy(cfg)

    assert policy.enabled is True
    assert policy.dynamic is True
    assert policy.backend == "inductor"
    assert policy.mode == "max-autotune-no-cudagraphs"
    assert policy.torch_compile_kwargs() == {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
    }
    assert resolve_compile_model(cfg) is True


def test_runtime_compile_model_rejects_invalid_bool() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": "maybe",
        }
    )

    with pytest.raises(ValueError, match="runtime.compile_model must be boolean-compatible"):
        _ = resolve_compile_model(cfg)


def test_runtime_compile_dynamic_rejects_invalid_bool() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "compile_dynamic": "maybe",
            "trace_activations": False,
        }
    )

    with pytest.raises(ValueError, match="runtime.compile_dynamic must be boolean-compatible"):
        _ = resolve_compile_policy(cfg)


def test_runtime_compile_policy_rejects_invalid_backend() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "compile_backend": "nvfuser",
            "trace_activations": False,
        }
    )

    with pytest.raises(ValueError, match="runtime.compile_backend must be one of"):
        _ = resolve_compile_policy(cfg)


def test_runtime_compile_policy_rejects_invalid_mode() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "compile_mode": "turbo",
            "trace_activations": False,
        }
    )

    with pytest.raises(ValueError, match="runtime.compile_mode must be one of"):
        _ = resolve_compile_policy(cfg)


def test_runtime_compile_model_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(training_runtime_module.torch, "compile", lambda model, **_kwargs: model)
    cfg = OmegaConf.create(
        {
            "device": "auto",
            "mixed_precision": "bf16",
            "compile_model": True,
            "trace_activations": False,
        }
    )

    with pytest.raises(ValueError, match="requires runtime.device to resolve to 'cuda'"):
        _ = resolve_compile_model(cfg)


def test_runtime_compile_model_rejects_activation_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module.torch, "compile", lambda model, **_kwargs: model)
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "trace_activations": True,
        }
    )

    with pytest.raises(ValueError, match="requires runtime.trace_activations=false"):
        _ = resolve_compile_model(cfg)


def test_runtime_compile_model_requires_torch_compile_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module.torch, "compile", None)
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "trace_activations": False,
        }
    )

    with pytest.raises(RuntimeError, match="requires torch.compile support"):
        _ = resolve_compile_model(cfg)


def test_runtime_compile_dynamic_is_inert_when_compile_is_disabled() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "mixed_precision": "no",
            "compile_model": False,
            "compile_dynamic": True,
        }
    )

    policy = resolve_compile_policy(cfg)

    assert policy.enabled is False
    assert policy.dynamic is True
    assert policy.torch_compile_kwargs() == {}


def test_runtime_compile_shape_dispatch_resolution() -> None:
    cfg = OmegaConf.create(
        {
            "compile_shape_dispatch_mode": "signature_family",
            "compile_shape_dispatch_max_families": 12,
        }
    )

    assert resolve_compile_shape_dispatch_policy(cfg) == ("signature_family", 12)


def test_runtime_cuda_graph_capture_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(training_runtime_module.torch, "compile", lambda model, **_kwargs: model)
    monkeypatch.setattr(training_runtime_module.torch.cuda, "CUDAGraph", object, raising=False)
    monkeypatch.setattr(
        training_runtime_module.torch.cuda,
        "graph",
        lambda *_args, **_kwargs: nullcontext(),
        raising=False,
    )
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "compile_backend": "eager",
            "compile_dynamic": True,
            "compile_shape_dispatch_mode": "signature_family",
            "compile_shape_dispatch_max_families": 12,
            "cuda_graph_capture_mode": "signature_family",
            "cuda_graph_max_families": 5,
            "trace_activations": False,
        }
    )

    policy = resolve_cuda_graph_capture_policy(cfg)

    assert policy.enabled is True
    assert policy.mode == "signature_family"
    assert policy.max_families == 5


def test_runtime_cuda_graph_capture_requires_signature_family_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "cuda")
    monkeypatch.setattr(training_runtime_module.torch, "compile", lambda model, **_kwargs: model)
    monkeypatch.setattr(training_runtime_module.torch.cuda, "CUDAGraph", object, raising=False)
    monkeypatch.setattr(
        training_runtime_module.torch.cuda,
        "graph",
        lambda *_args, **_kwargs: nullcontext(),
        raising=False,
    )
    cfg = OmegaConf.create(
        {
            "device": "cuda",
            "mixed_precision": "bf16",
            "compile_model": True,
            "compile_backend": "eager",
            "compile_dynamic": True,
            "compile_shape_dispatch_mode": "off",
            "cuda_graph_capture_mode": "signature_family",
            "trace_activations": False,
        }
    )

    with pytest.raises(
        ValueError,
        match="runtime.compile_shape_dispatch_mode='signature_family'",
    ):
        _ = resolve_cuda_graph_capture_policy(cfg)
