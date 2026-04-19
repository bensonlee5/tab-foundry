from __future__ import annotations

import math

import pytest

from tab_foundry.research.sweep.transfer import (
    nearest_realizable_effective_batch,
    resolve_transfer_schedule,
    round_half_up,
)


def test_round_half_up_and_nearest_realizable_effective_batch() -> None:
    assert round_half_up(625.5) == 626
    assert round_half_up(625.49) == 625
    assert nearest_realizable_effective_batch(target_effective_batch=100.79, task_batch_size=16) == 96
    assert nearest_realizable_effective_batch(target_effective_batch=104.0, task_batch_size=16) == 112


def test_resolve_transfer_schedule_regime_b_scales_fixed_batch_formula_exactly() -> None:
    schedule = resolve_transfer_schedule(
        regime_label="B",
        base_lr_max=1.0e-3,
        base_momentum=0.95,
        base_effective_batch=64,
        base_effective_budget=625 * 64,
        target_effective_budget=2500 * 64,
        task_batch_size=16,
        fixed_effective_batch=64,
    )

    assert schedule["regime_label"] == "B"
    assert schedule["formula_label"] == "Theorem 2 fixed-batch transfer"
    assert schedule["realized_effective_batch"] == 64
    assert schedule["grad_accum_steps"] == 4
    assert schedule["max_steps"] == 2500
    assert schedule["realized_effective_budget"] == 2500 * 64
    assert schedule["budget_drift"] == pytest.approx(0.0)
    assert schedule["batch_drift"] == pytest.approx(0.0)
    assert schedule["target_alpha"] == pytest.approx(0.025)
    assert schedule["target_momentum"] == pytest.approx(0.975)
    assert schedule["target_lr_max"] == pytest.approx(1.0e-3 * (4.0 ** -0.75))
    assert schedule["min_lr"] == pytest.approx(schedule["target_lr_max"] * 1.0e-3)


def test_resolve_transfer_schedule_regime_d_rounds_batch_and_budget_deterministically() -> None:
    schedule = resolve_transfer_schedule(
        regime_label="D",
        base_lr_max=1.0e-3,
        base_momentum=0.95,
        base_effective_batch=80,
        base_effective_budget=625 * 64,
        target_effective_budget=2500 * 64,
        task_batch_size=16,
    )

    expected_target_batch = 80.0 * (4.0 ** (1.0 / 6.0))
    expected_target_alpha = (1.0 - 0.95) * (4.0 ** (-1.0 / 3.0))
    expected_target_lr = 1.0e-3 * (4.0 ** (-7.0 / 12.0))

    assert schedule["regime_label"] == "D"
    assert schedule["formula_label"] == "Theorem 3 joint-transfer proxy"
    assert schedule["target_effective_batch"] == pytest.approx(expected_target_batch)
    assert schedule["realized_effective_batch"] == 96
    assert schedule["grad_accum_steps"] == 6
    assert schedule["max_steps"] == 1667
    assert schedule["realized_effective_budget"] == 160032
    assert schedule["budget_drift"] == pytest.approx(32.0 / 160000.0)
    assert schedule["batch_drift"] == pytest.approx(96.0 / expected_target_batch - 1.0)
    assert schedule["target_alpha"] == pytest.approx(expected_target_alpha)
    assert schedule["target_momentum"] == pytest.approx(1.0 - expected_target_alpha)
    assert schedule["target_lr_max"] == pytest.approx(expected_target_lr)
    assert math.isclose(schedule["min_lr"], expected_target_lr * 1.0e-3)


def test_resolve_transfer_schedule_rejects_excess_effective_budget_drift() -> None:
    with pytest.raises(RuntimeError, match="effective-budget drift"):
        resolve_transfer_schedule(
            regime_label="B",
            base_lr_max=1.0e-3,
            base_momentum=0.99,
            base_effective_batch=64,
            base_effective_budget=625 * 64,
            target_effective_budget=50,
            task_batch_size=16,
            fixed_effective_batch=64,
        )
