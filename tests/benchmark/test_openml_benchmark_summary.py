from __future__ import annotations

# ruff: noqa: F401

from tests.support.openml_benchmark_compare_cases import (
    test_collect_checkpoint_snapshots_falls_back_to_latest_checkpoint_when_no_step_snapshots,
    test_build_comparison_summary_preserves_model_identity_metadata,
    test_build_comparison_summary_uses_log_loss_as_classification_best_step,
    test_collect_checkpoint_snapshots_prefers_train_elapsed_seconds,
    test_collect_checkpoint_snapshots_supports_plain_training_output,
)
