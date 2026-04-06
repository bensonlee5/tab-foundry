from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tab_foundry.training.trainer as trainer_module

from tests.support.train_eval_smoke_cases import (
    _FakeAccelerator,
    _TraceableStageLocalClassifier,
    _classification_cfg,
    _install_classification_fakes,
)


class _CompileTrackingAccelerator(_FakeAccelerator):
    def __init__(self, *, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def prepare(self, *items: object) -> object:
        self._events.append("prepare")
        return super().prepare(*items)


class _CountingProfiler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_train_compile_model_runs_after_checkpointing_and_before_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _install_classification_fakes(monkeypatch)

    class _CompileTrackingModel(_TraceableStageLocalClassifier):
        def enable_activation_checkpointing(self) -> None:
            events.append("activation_checkpointing")
            super().enable_activation_checkpointing()

    model = _CompileTrackingModel()
    fake_spec = SimpleNamespace(task="classification", arch="tabfoundry_sandwich")
    monkeypatch.setattr(trainer_module, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _CompileTrackingAccelerator(events=events),
    )
    monkeypatch.setattr(
        trainer_module,
        "configure_model_loss_surface",
        lambda *args, **kwargs: events.append("loss_surface"),
    )

    def _fake_compile(compiled_model, **kwargs):
        assert compiled_model is model
        assert kwargs == {"mode": "max-autotune-no-cudagraphs"}
        events.append("compile")
        return compiled_model

    monkeypatch.setattr(trainer_module.torch, "compile", _fake_compile)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.device = "cuda"
    cfg.runtime.compile_model = True
    cfg.runtime.activation_checkpointing = True

    result = trainer_module.train(cfg)
    training_surface = json.loads(
        (result.output_dir / "training_surface_record.json").read_text(encoding="utf-8")
    )

    assert events[:4] == ["loss_surface", "activation_checkpointing", "compile", "prepare"]
    assert model.activation_checkpointing_enabled is True
    assert training_surface["runtime"]["compile_model"] is True


def test_train_compile_model_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compile_calls = 0
    _install_classification_fakes(monkeypatch)

    def _unexpected_compile(model, **_kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return model

    monkeypatch.setattr(trainer_module.torch, "compile", _unexpected_compile)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.compile_model = False

    _ = trainer_module.train(cfg)

    assert compile_calls == 0


def test_train_profiler_steps_once_per_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    profiler = _CountingProfiler()
    cfg = _classification_cfg(tmp_path)
    cfg.schedule.stages = [{"name": "stage1", "steps": 2, "lr_max": 1.0e-3}]

    result = trainer_module.train(cfg, profiler=profiler)

    assert result.global_step == 2
    assert profiler.steps == 2
