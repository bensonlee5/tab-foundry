from __future__ import annotations

import math
from copy import deepcopy

import pytest

from tab_foundry.research.sweep.row_dependencies import resolve_dynamic_training_overrides
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


def test_resolve_transfer_schedule_regime_d_t1_from_shared_anchor_rounds_to_80() -> None:
    schedule = resolve_transfer_schedule(
        regime_label="D",
        base_lr_max=1.0e-3,
        base_momentum=0.95,
        base_effective_batch=64,
        base_effective_budget=625 * 64,
        target_effective_budget=2500 * 64,
        task_batch_size=16,
    )

    expected_target_batch = 64.0 * (4.0 ** (1.0 / 6.0))
    expected_target_alpha = (1.0 - 0.95) * (4.0 ** (-1.0 / 3.0))
    expected_target_lr = 1.0e-3 * (4.0 ** (-7.0 / 12.0))

    assert schedule["regime_label"] == "D"
    assert schedule["formula_label"] == "Theorem 3 joint-transfer proxy"
    assert schedule["target_effective_batch"] == pytest.approx(expected_target_batch)
    assert schedule["realized_effective_batch"] == 80
    assert schedule["grad_accum_steps"] == 5
    assert schedule["max_steps"] == 2000
    assert schedule["realized_effective_budget"] == 160000
    assert schedule["budget_drift"] == pytest.approx(0.0)
    assert schedule["batch_drift"] == pytest.approx(80.0 / expected_target_batch - 1.0)
    assert schedule["target_alpha"] == pytest.approx(expected_target_alpha)
    assert schedule["target_momentum"] == pytest.approx(1.0 - expected_target_alpha)
    assert schedule["target_lr_max"] == pytest.approx(expected_target_lr)
    assert math.isclose(schedule["min_lr"], expected_target_lr * 1.0e-3)


def test_resolve_transfer_schedule_regime_d_t2_from_shared_anchor_rounds_to_96() -> None:
    schedule = resolve_transfer_schedule(
        regime_label="D",
        base_lr_max=1.0e-3,
        base_momentum=0.95,
        base_effective_batch=64,
        base_effective_budget=625 * 64,
        target_effective_budget=5000 * 64,
        task_batch_size=16,
    )

    expected_target_batch = 64.0 * (8.0 ** (1.0 / 6.0))
    expected_target_alpha = (1.0 - 0.95) * (8.0 ** (-1.0 / 3.0))
    expected_target_lr = 1.0e-3 * (8.0 ** (-7.0 / 12.0))

    assert schedule["regime_label"] == "D"
    assert schedule["target_effective_batch"] == pytest.approx(expected_target_batch)
    assert schedule["realized_effective_batch"] == 96
    assert schedule["grad_accum_steps"] == 6
    assert schedule["max_steps"] == 3333
    assert schedule["realized_effective_budget"] == 319968
    assert schedule["budget_drift"] == pytest.approx(-32.0 / 320000.0)
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


def test_resolve_dynamic_training_overrides_shared_anchor_transfer_resolves_from_anchor_row() -> None:
    anchor_row = {
        "order": 1,
        "delta_ref": "delta_anchor",
        "delta_id": "delta_anchor",
        "run_id": "anchor_run",
        "training": {
            "task_batch_size": 16,
            "overrides": {
                "optimizer": {
                    "name": "muon",
                    "momentum": 0.95,
                    "min_lr": 1.0e-6,
                },
                "runtime": {
                    "grad_accum_steps": 4,
                    "max_steps": 625,
                },
                "schedule": {
                    "stages": [
                        {
                            "steps": 625,
                            "lr_max": 1.0e-3,
                        }
                    ]
                },
            },
        },
        "reuse_train_artifact": {
            "run_dir": "outputs/staged_ladder/research/lmo/anchor_run/train",
        },
        "imported_baseline_provenance": {
            "source_sweep_id": "tf_rd_009_muon_ns_one_epoch_medium_v1",
            "source_order": 9,
        },
        "transfer_context": {
            "regime_label": "carry_lowbatch",
            "target_budget_label": "T0",
        },
    }
    target_row = {
        "order": 2,
        "delta_ref": "delta_target",
        "delta_id": "delta_target",
        "training": {
            "task_batch_size": 16,
            "overrides": {
                "optimizer": {
                    "name": "muon",
                    "min_lr": 1.0e-6,
                },
                "runtime": {
                    "grad_accum_steps": 4,
                    "max_steps": 625,
                },
                "schedule": {
                    "stages": [
                        {
                            "steps": 625,
                            "lr_max": 1.0e-3,
                        }
                    ]
                },
            },
        },
        "transfer_context": {
            "phase": "validation",
            "regime_label": "D",
            "formula_label": "Theorem 3 joint-transfer proxy",
            "base_budget_label": "T0",
            "target_budget_label": "T1",
            "candidate_label": "shared_anchor_regime_d",
        },
        "dynamic_training_overrides": {
            "transfer_schedule": {
                "kind": "shared_anchor_transfer",
                "anchor_order": 1,
                "anchor_sweep_id": "tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1",
                "anchor_label": "carry_lowbatch_shared_anchor_t0",
                "regime_label": "D",
                "base_effective_budget": 625 * 64,
                "target_effective_budget": 2500 * 64,
                "min_lr_ratio": 1.0e-3,
                "max_budget_drift": 0.02,
            }
        },
        "notes": [],
    }
    queue = {
        "sweep_id": "tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1",
        "rows": [anchor_row, target_row],
    }

    queue_row = deepcopy(target_row)
    materialized_row = deepcopy(target_row)

    resolve_dynamic_training_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )

    for resolved_row in (queue_row, materialized_row):
        assert resolved_row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 5
        assert resolved_row["training"]["overrides"]["runtime"]["max_steps"] == 2000
        assert resolved_row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 2000
        assert resolved_row["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == pytest.approx(
            1.0e-3 * (4.0 ** (-7.0 / 12.0))
        )
        assert resolved_row["training"]["overrides"]["optimizer"]["min_lr"] == pytest.approx(
            1.0e-6 * (4.0 ** (-7.0 / 12.0))
        )
        assert resolved_row["training"]["overrides"]["optimizer"]["momentum"] == pytest.approx(
            1.0 - ((1.0 - 0.95) * (4.0 ** (-1.0 / 3.0)))
        )
        assert resolved_row["transfer_resolution"]["shared_anchor_provenance"]["anchor_order"] == 1
        assert (
            resolved_row["transfer_resolution"]["shared_anchor_provenance"]["anchor_sweep_id"]
            == "tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1"
        )
        assert (
            resolved_row["transfer_resolution"]["shared_anchor_provenance"]["anchor_run_dir"]
            == "outputs/staged_ladder/research/lmo/anchor_run/train"
        )
        assert resolved_row["transfer_resolution"]["resolution_reason"] == "shared_anchor"
