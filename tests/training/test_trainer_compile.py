from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tab_foundry.cli.groups.train as train_cli_module
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


class _FakeKeyAverages(list[SimpleNamespace]):
    def table(self, **_kwargs: object) -> str:
        return "fake profiler table"


class _FakeProfiler:
    def __init__(self) -> None:
        self.steps = 0

    def __enter__(self) -> "_FakeProfiler":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def step(self) -> None:
        self.steps += 1

    def key_averages(self) -> _FakeKeyAverages:
        return _FakeKeyAverages(
            [
                SimpleNamespace(
                    key="aten::matmul",
                    count=2,
                    self_cpu_time_total=10.0,
                    cpu_time_total=20.0,
                    self_cuda_time_total=100.0,
                    cuda_time_total=120.0,
                    cpu_memory_usage=0,
                    cuda_memory_usage=256,
                    flops=1024.0,
                ),
                SimpleNamespace(
                    key="aten::copy_",
                    count=1,
                    self_cpu_time_total=5.0,
                    cpu_time_total=7.0,
                    self_cuda_time_total=20.0,
                    cuda_time_total=25.0,
                    cpu_memory_usage=128,
                    cuda_memory_usage=512,
                    flops=0.0,
                ),
                SimpleNamespace(
                    key="ProfilerStep*",
                    count=2,
                    self_cpu_time_total=0.0,
                    cpu_time_total=0.0,
                    self_cuda_time_total=0.0,
                    cuda_time_total=0.0,
                    cpu_memory_usage=0,
                    cuda_memory_usage=0,
                    flops=0.0,
                ),
            ]
        )


def _assert_train_compile_model_order_and_kwargs(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compile_backend: str,
    compile_mode: str,
    compile_dynamic: bool,
    expected_compile_kwargs: dict[str, object],
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
        assert kwargs == expected_compile_kwargs
        events.append("compile")
        return compiled_model

    monkeypatch.setattr(trainer_module.torch, "compile", _fake_compile)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.device = "cuda"
    cfg.runtime.compile_model = True
    cfg.runtime.compile_dynamic = compile_dynamic
    cfg.runtime.compile_backend = compile_backend
    cfg.runtime.compile_mode = compile_mode
    cfg.runtime.activation_checkpointing = True

    result = trainer_module.train(cfg)
    training_surface = json.loads(
        (result.output_dir / "training_surface_record.json").read_text(encoding="utf-8")
    )

    assert events[:4] == ["loss_surface", "activation_checkpointing", "compile", "prepare"]
    assert model.activation_checkpointing_enabled is True
    assert training_surface["runtime"]["compile_model"] is True
    assert training_surface["runtime"]["compile_dynamic"] is compile_dynamic
    assert training_surface["runtime"]["compile_backend"] == compile_backend
    assert training_surface["runtime"]["compile_mode"] == compile_mode


def test_train_compile_model_runs_after_checkpointing_and_before_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _assert_train_compile_model_order_and_kwargs(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
        compile_dynamic=False,
        expected_compile_kwargs={"backend": "inductor", "mode": "max-autotune-no-cudagraphs"},
    )


@pytest.mark.parametrize(
    ("compile_backend", "compile_mode", "compile_dynamic", "expected_compile_kwargs"),
    [
        (
            "inductor",
            "max-autotune-no-cudagraphs",
            False,
            {"backend": "inductor", "mode": "max-autotune-no-cudagraphs"},
        ),
        ("eager", "reduce-overhead", False, {"backend": "eager"}),
        ("aot_eager", "default", False, {"backend": "aot_eager"}),
        ("eager", "default", True, {"backend": "eager", "dynamic": True}),
        (
            "inductor",
            "default",
            True,
            {"backend": "inductor", "mode": "default", "dynamic": True},
        ),
    ],
)
def test_train_compile_model_uses_expected_compile_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compile_backend: str,
    compile_mode: str,
    compile_dynamic: bool,
    expected_compile_kwargs: dict[str, object],
) -> None:
    _assert_train_compile_model_order_and_kwargs(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        expected_compile_kwargs=expected_compile_kwargs,
    )


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
    cfg.runtime.compile_dynamic = True

    _ = trainer_module.train(cfg)

    assert compile_calls == 0


def test_train_signature_family_compile_dispatch_compiles_after_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _install_classification_fakes(monkeypatch)
    model = _TraceableStageLocalClassifier()
    fake_spec = SimpleNamespace(task="classification", arch="tabfoundry_sandwich")
    monkeypatch.setattr(trainer_module, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _CompileTrackingAccelerator(events=events),
    )

    def _fake_compile(compiled_model, **_kwargs):
        assert compiled_model is model
        events.append("compile")
        return compiled_model

    monkeypatch.setattr(trainer_module.torch, "compile", _fake_compile)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.device = "cuda"
    cfg.runtime.compile_model = True
    cfg.runtime.compile_backend = "eager"
    cfg.runtime.compile_dynamic = True
    cfg.runtime.compile_shape_dispatch_mode = "signature_family"
    cfg.runtime.compile_shape_dispatch_max_families = 4
    cfg.runtime.val_batches = 0
    cfg.schedule.stages = [{"name": "stage1", "steps": 1, "lr_max": 1.0e-3}]

    _ = trainer_module.train(cfg)

    assert events.index("prepare") < events.index("compile")
    assert events.count("compile") == 1


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


def test_train_profile_command_writes_structured_profiler_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.output_dir = str(tmp_path / "profile_run")
    fake_profiler = _FakeProfiler()
    monkeypatch.setattr(train_cli_module, "compose_config", lambda _overrides: cfg)
    monkeypatch.setattr(
        train_cli_module.torch.profiler,
        "profile",
        lambda **_kwargs: fake_profiler,
    )
    monkeypatch.setattr(
        train_cli_module.torch.cuda,
        "is_available",
        lambda: False,
    )

    def _fake_run_training(profile_cfg: object, *, profiler: object) -> SimpleNamespace:
        assert profile_cfg.runtime.checkpoint_every is None
        assert profile_cfg.runtime.profile_step_timing is True
        assert profiler is fake_profiler
        profiler.step()
        return SimpleNamespace(
            output_dir=Path(profile_cfg.runtime.output_dir),
            global_step=4,
            metrics={},
        )

    monkeypatch.setattr(train_cli_module, "run_training", _fake_run_training)

    result = train_cli_module._run_training_profile_command(
        overrides=(),
        max_steps=4,
        wait=1,
        warmup=1,
        active=2,
        repeat=1,
    )

    profile_dir = Path(cfg.runtime.output_dir) / "torch_profiler"
    assert result == 0
    assert (profile_dir / "key_averages.txt").read_text(encoding="utf-8") == "fake profiler table"
    payload = json.loads((profile_dir / "profile_summary.json").read_text(encoding="utf-8"))
    assert payload["schedule"]["expected_profiled_step_count"] == 2
    assert payload["profiled_step_count"] == 2
    assert payload["operator_class_totals"]["compute"]["flops"] == 1024.0
    assert payload["operator_class_totals"]["memory_movement"]["cuda_memory_allocated_bytes"] == 512
    assert payload["top_operators"][0]["name"] == "aten::matmul"
