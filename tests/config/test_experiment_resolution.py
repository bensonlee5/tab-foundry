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
    assert int(cfg.model.feature_group_size) == 1
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_reg_workstation_task_resolution() -> None:
    cfg = _compose("experiment=reg_workstation")
    assert str(cfg.task) == "regression"
    assert int(cfg.model.feature_group_size) == 1
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_reg_smoke_task_and_runtime_resolution() -> None:
    cfg = _compose("experiment=reg_smoke")
    assert str(cfg.task) == "regression"
    assert str(cfg.runtime.device) == "cpu"
    assert str(cfg.runtime.mixed_precision) == "no"
    assert int(cfg.runtime.grad_accum_steps) == 1
    assert int(cfg.model.feature_group_size) == 1
    assert str(cfg.model.many_class_train_mode) == "path_nll"
    assert int(cfg.model.many_class_base) == 10
    assert bool(cfg.model.use_digit_position_embed) is True
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True


def test_cls_smoke_optimizer_resolution() -> None:
    cfg = _compose("experiment=cls_smoke")
    assert str(cfg.optimizer.name) == "muon"
    assert bool(cfg.optimizer.require_requested) is True
    assert int(cfg.model.feature_group_size) == 1
    assert cfg.logging.history_jsonl_path is None


def test_runtime_smoke_override_resolution() -> None:
    cfg = _compose("runtime=smoke")
    assert str(cfg.runtime.mixed_precision) == "no"
    assert cfg.runtime.checkpoint_every is None


def test_cls_smoke_adamw_override_resolution() -> None:
    cfg = _compose("experiment=cls_smoke", "optimizer=adamw")
    assert str(cfg.optimizer.name) == "adamw"
    assert bool(cfg.optimizer.require_requested) is False


def test_cls_benchmark_linear_resolution() -> None:
    cfg = _compose("experiment=cls_benchmark_linear")
    assert str(cfg.task) == "classification"
    assert int(cfg.model.feature_group_size) == 1
    assert str(cfg.optimizer.name) == "adamw"
    assert bool(cfg.optimizer.require_requested) is False
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert int(cfg.runtime.max_steps) == 400
    assert float(cfg.runtime.target_train_seconds) == 330.0
    stage = cfg.schedule.stages[0]
    assert str(stage["lr_schedule"]) == "linear"
    assert float(stage["warmup_ratio"]) == 0.05
