from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import torch

import tab_foundry.training.trainer as trainer_module
from tab_foundry.training.surface import build_training_surface_record

from tests.support.train_eval_smoke_cases import (
    _TinyClassifier,
    _classification_cfg,
    _install_classification_fakes,
)


def test_build_training_surface_record_persists_initial_checkpoint_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry_staged"},
            "data": {"source": "manifest", "manifest_path": str(tmp_path / "manifest.parquet")},
            "training": {"initial_checkpoint_path": str(checkpoint_path)},
        },
        run_dir=tmp_path / "run",
    )

    assert record["training"]["initial_checkpoint_path"] == str(
        checkpoint_path.expanduser().resolve()
    )


def test_train_initial_checkpoint_path_loads_model_weights_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    captured: dict[str, object] = {}
    checkpoint_model = _TinyClassifier()
    for tensor in checkpoint_model.parameters():
        tensor.data.fill_(3.5)
    checkpoint_path = tmp_path / "init.pt"
    torch.save({"model": checkpoint_model.state_dict(), "optimizer": {"state": {"ignored": 1}}}, checkpoint_path)

    class _CapturingModel(_TinyClassifier):
        pass

    built_model = _CapturingModel()
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: built_model)
    monkeypatch.setattr(
        trainer_module,
        "model_build_spec_from_mappings",
        lambda **_kwargs: SimpleNamespace(task="classification", arch="tabfoundry_simple"),
    )

    def _fake_run_training_loop(**kwargs):
        model = kwargs["base_model"]
        captured["weight_before_training"] = model.linear.weight.detach().clone()
        state = kwargs["state"]
        state.global_step = 1
        state.latest_checkpoint = kwargs["output_dir"] / "checkpoints" / "latest.pt"
        state.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, state.latest_checkpoint)

    monkeypatch.setattr(trainer_module, "run_training_loop", _fake_run_training_loop)
    cfg = _classification_cfg(tmp_path)
    cfg.training = {
        "task_batch_size": 1,
        "loss_surface": "classification",
        "initial_checkpoint_path": str(checkpoint_path),
    }

    _ = trainer_module.train(cfg)

    assert torch.equal(
        captured["weight_before_training"],
        checkpoint_model.linear.weight.detach(),
    )
    surface_record = json.loads(
        (tmp_path / "outputs" / "training_surface_record.json").read_text(encoding="utf-8")
    )
    assert surface_record["training"]["initial_checkpoint_path"] == str(checkpoint_path.resolve())
