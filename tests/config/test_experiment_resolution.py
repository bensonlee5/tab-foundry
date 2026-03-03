from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(*overrides: str):
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_cls_workstation_task_resolution() -> None:
    cfg = _compose("experiment=cls_workstation")
    assert str(cfg.task) == "classification"
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_reg_workstation_task_resolution() -> None:
    cfg = _compose("experiment=reg_workstation")
    assert str(cfg.task) == "regression"
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_reg_smoke_task_and_runtime_resolution() -> None:
    cfg = _compose("experiment=reg_smoke")
    assert str(cfg.task) == "regression"
    assert str(cfg.runtime.device) == "cpu"
    assert str(cfg.runtime.mixed_precision) == "no"
    assert int(cfg.runtime.grad_accum_steps) == 1
    assert int(cfg.model.feature_group_size) == 32
    assert str(cfg.model.many_class_train_mode) == "path_nll"
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_cls_smoke_optimizer_resolution() -> None:
    cfg = _compose("experiment=cls_smoke")
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_runtime_smoke_override_resolution() -> None:
    cfg = _compose("runtime=smoke")
    assert str(cfg.runtime.mixed_precision) == "no"


def test_cls_smoke_adamw_override_resolution() -> None:
    cfg = _compose("experiment=cls_smoke", "optimizer=adamw")
    assert str(cfg.optimizer.name) == "adamw"
    assert bool(cfg.optimizer.require_requested) is False
