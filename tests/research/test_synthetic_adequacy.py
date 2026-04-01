from __future__ import annotations

import math
from pathlib import Path

import pytest

from tab_foundry.research.synthetic_adequacy import (
    LABEL_TARGET_LOG_LOSS_PER_TEST_CELL,
    load_synthetic_adequacy_spec,
    prediction_variance_per_test_cell,
    summarize_replicate_predictions,
    synthetic_adequacy_spec_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tf_rd_010_synthetic_adequacy_spec_is_registered() -> None:
    path = synthetic_adequacy_spec_path("tf_rd_010_synthetic_adequacy_v2", repo_root=REPO_ROOT)
    spec = load_synthetic_adequacy_spec("tf_rd_010_synthetic_adequacy_v2", repo_root=REPO_ROOT)

    assert path.exists()
    assert spec.adequacy_id == "tf_rd_010_synthetic_adequacy_v2"
    assert spec.status == "ready"
    assert spec.metric_definition == LABEL_TARGET_LOG_LOSS_PER_TEST_CELL
    assert spec.blocked_sweeps == (
        "tf_rd_010_classification_evolution_medium_v4",
        "tf_rd_010_classification_evolution_large_v2",
    )
    assert [block.block_id for block in spec.blocks] == [
        "latent_target_canary_easy_v2",
        "production_control_v4",
    ]
    assert spec.blocks[0].corpus_ref == "tf_rd_010_latent_target_canary_v2"
    assert spec.blocks[1].corpus_ref == "tf_rd_010_dagzoo_medium_control_v4"
    assert spec.blocks[0].predictors == ("chance", "logistic_regression")
    assert spec.blocks[1].predictors == ("chance", "sandwich")
    assert "generator_problem" in spec.decision_buckets
    assert "training_regime_problem" in spec.decision_buckets
    assert "inconclusive" in spec.decision_buckets


def test_synthetic_adequacy_metrics_measure_log_loss_and_variance() -> None:
    replicate_probabilities = [
        [[0.90, 0.10], [0.20, 0.80]],
        [[0.80, 0.20], [0.25, 0.75]],
        [[0.85, 0.15], [0.15, 0.85]],
    ]
    targets = [0, 1]

    summary = summarize_replicate_predictions(replicate_probabilities, targets)

    expected_mean_log_loss = sum(
        (
            -math.log(0.90) - math.log(0.80),
            -math.log(0.80) - math.log(0.75),
            -math.log(0.85) - math.log(0.85),
        )
    ) / 3.0 / 2.0
    assert summary["mean_log_loss_per_test_cell"] == pytest.approx(expected_mean_log_loss)
    assert summary["std_log_loss_per_test_cell"] > 0.0
    assert summary["prediction_variance_per_test_cell"] == pytest.approx(
        prediction_variance_per_test_cell(replicate_probabilities)
    )
    assert "mean_teacher_excess_log_loss_per_test_cell" not in summary
    assert "teacher_optimal_log_loss_per_test_cell" not in summary
