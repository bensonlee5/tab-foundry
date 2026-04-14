from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

import tab_foundry.research.tf_rd_022_compile_first as compile_first_module
from tab_foundry.config import compose_config


def test_tf_rd_022_compile_first_cfg_inherits_the_policy_surface_with_a_compile_only_delta() -> None:
    base_cfg = compose_config(
        ["experiment=cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1"]
    )
    compile_cfg = compile_first_module.tf_rd_022_compile_first_cfg()

    base_payload = OmegaConf.to_container(base_cfg, resolve=True)
    compile_payload = OmegaConf.to_container(compile_cfg, resolve=True)
    assert isinstance(base_payload, dict)
    assert isinstance(compile_payload, dict)
    assert compile_payload["training"] == base_payload["training"]
    assert compile_payload["optimizer"] == base_payload["optimizer"]
    assert compile_payload["schedule"] == base_payload["schedule"]
    assert compile_payload["model"] == base_payload["model"]
    assert compile_payload["data"] == base_payload["data"]

    base_runtime = dict(base_payload["runtime"])
    compile_runtime = dict(compile_payload["runtime"])
    assert base_runtime.pop("compile_model") is True
    assert compile_runtime.pop("compile_model") is True
    assert base_runtime.pop("compile_dynamic") is True
    assert compile_runtime.pop("compile_dynamic") is False
    assert base_runtime.pop("compile_backend") == "eager"
    assert base_runtime["compile_mode"] == "max-autotune-no-cudagraphs"
    assert compile_runtime.pop("compile_backend") == "inductor"
    assert compile_runtime["compile_mode"] == "max-autotune-no-cudagraphs"
    assert base_runtime.pop("compile_shape_dispatch_mode") == "signature_family"
    assert compile_runtime.pop("compile_shape_dispatch_mode") == "off"
    assert base_runtime.pop("output_dir") == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1"
    )
    assert compile_runtime.pop("output_dir") == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1"
    )
    assert compile_runtime == base_runtime
    assert compile_payload["logging"]["run_name"] == (
        "cls-benchmark-sandwich-classification-evolution-tf-rd-022-policy-compile-v1"
    )
    assert compile_payload["logging"]["history_jsonl_path"] == (
        "outputs/cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_v1/train_history.jsonl"
    )


def test_tf_rd_022_compile_profile_cfg_applies_short_run_overrides(tmp_path: Path) -> None:
    cfg = compile_first_module.tf_rd_022_compile_profile_cfg(
        output_dir=tmp_path / "profile_run",
        max_steps=24,
    )

    assert bool(cfg.runtime.compile_model) is True
    assert bool(cfg.runtime.compile_dynamic) is False
    assert str(cfg.runtime.compile_backend) == "inductor"
    assert str(cfg.runtime.compile_mode) == "max-autotune-no-cudagraphs"
    assert int(cfg.runtime.max_steps) == 24
    assert int(cfg.runtime.eval_every) == 24
    assert int(cfg.runtime.checkpoint_every) == 24
    assert str(cfg.runtime.output_dir) == str((tmp_path / "profile_run").resolve())
    assert str(cfg.logging.run_name).endswith("-profile-short")


def test_run_tf_rd_022_compile_profile_writes_metadata_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeKeyAverages:
        def table(self, **kwargs: object) -> str:
            captured["table_kwargs"] = kwargs
            return "fake profiler summary"

    class _FakeProfiler:
        def __enter__(self) -> _FakeProfiler:
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            captured["exited"] = True

        def key_averages(self) -> _FakeKeyAverages:
            return _FakeKeyAverages()

        def step(self) -> None:
            return None

    def _fake_schedule(**kwargs: int) -> str:
        captured["schedule_kwargs"] = kwargs
        return "fake_schedule"

    def _fake_trace_handler(path: str):
        captured["trace_handler_path"] = path

        def _handler(_profiler: object) -> None:
            return None

        return _handler

    def _fake_profile(**kwargs: object) -> _FakeProfiler:
        captured["profile_kwargs"] = kwargs
        return _FakeProfiler()

    def _fake_train(cfg, *, profiler) -> SimpleNamespace:
        captured["train_output_dir"] = str(cfg.runtime.output_dir)
        captured["train_run_name"] = str(cfg.logging.run_name)
        assert profiler is not None
        return SimpleNamespace(output_dir=Path(str(cfg.runtime.output_dir)).resolve())

    monkeypatch.setattr(compile_first_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(compile_first_module.torch.profiler, "schedule", _fake_schedule)
    monkeypatch.setattr(
        compile_first_module.torch.profiler,
        "tensorboard_trace_handler",
        _fake_trace_handler,
    )
    monkeypatch.setattr(compile_first_module.torch.profiler, "profile", _fake_profile)
    monkeypatch.setattr(compile_first_module, "train", _fake_train)

    metadata = compile_first_module.run_tf_rd_022_compile_profile(
        tmp_path / "profile_output",
        max_steps=12,
        wait=2,
        warmup=3,
        active=4,
        repeat=5,
    )

    trace_dir = tmp_path / "profile_output" / "torch_profiler"
    summary_path = trace_dir / "key_averages.txt"
    metadata_path = trace_dir / "profile_metadata.json"
    persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert captured["entered"] is True
    assert captured["exited"] is True
    assert captured["schedule_kwargs"] == {"wait": 2, "warmup": 3, "active": 4, "repeat": 5}
    assert str(captured["train_output_dir"]) == str((tmp_path / "profile_output").resolve())
    assert str(captured["train_run_name"]).endswith("-profile-short")
    assert summary_path.read_text(encoding="utf-8") == "fake profiler summary"
    assert metadata == persisted_metadata
    assert metadata["summary_path"] == str(summary_path)
    assert metadata["trace_dir"] == str(trace_dir)
    assert metadata["max_steps"] == 12
    assert metadata["schedule"] == {"wait": 2, "warmup": 3, "active": 4, "repeat": 5}
