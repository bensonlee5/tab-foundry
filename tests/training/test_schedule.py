from __future__ import annotations

import pytest

from tab_foundry.training.schedule import StageConfig, build_stage_configs, stage_base_lr


def test_stage_base_lr_cosine_preserves_old_shape() -> None:
    stage = StageConfig(name="stage1", steps=3, lr_max=1.0e-3)

    assert stage_base_lr(stage, step=1, lr_min=1.0e-4) == pytest.approx(1.0e-3)
    assert stage_base_lr(stage, step=3, lr_min=1.0e-4) == pytest.approx(1.0e-4)


def test_stage_base_lr_linear_with_warmup() -> None:
    stage = StageConfig(
        name="stage1",
        steps=40,
        lr_max=1.0e-3,
        lr_schedule="linear",
        warmup_ratio=0.05,
    )

    assert stage_base_lr(stage, step=1, lr_min=1.0e-4) == pytest.approx(5.0e-4)
    assert stage_base_lr(stage, step=2, lr_min=1.0e-4) == pytest.approx(1.0e-3)
    assert stage_base_lr(stage, step=40, lr_min=1.0e-4) == pytest.approx(1.0e-4)


def test_stage_base_lr_linear_decay_without_warmup() -> None:
    stage = StageConfig(
        name="stage1",
        steps=5,
        lr_max=1.0e-3,
        lr_schedule="linear",
        warmup_ratio=0.0,
    )

    assert stage_base_lr(stage, step=1, lr_min=1.0e-4) == pytest.approx(1.0e-3)
    assert stage_base_lr(stage, step=3, lr_min=1.0e-4) == pytest.approx(5.5e-4)
    assert stage_base_lr(stage, step=5, lr_min=1.0e-4) == pytest.approx(1.0e-4)


def test_build_stage_configs_accepts_linear_warmup_fields() -> None:
    stages = build_stage_configs(
        [
            {
                "name": "stage1",
                "steps": 10,
                "lr_max": 8.0e-4,
                "lr_schedule": "linear",
                "warmup_ratio": 0.05,
            }
        ]
    )

    assert stages[0].lr_schedule == "linear"
    assert stages[0].warmup_ratio == pytest.approx(0.05)


def test_build_stage_configs_rejects_bad_warmup_ratio() -> None:
    with pytest.raises(ValueError, match="warmup_ratio"):
        _ = build_stage_configs(
            [{"name": "stage1", "steps": 10, "lr_max": 8.0e-4, "warmup_ratio": 1.0}]
        )
