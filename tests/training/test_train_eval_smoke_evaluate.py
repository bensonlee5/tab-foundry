from __future__ import annotations

# ruff: noqa: F401

from tests.support.train_eval_smoke_cases import (
    test_evaluate_checkpoint_caps_by_task_count_without_overshooting,
    test_evaluate_checkpoint_logs_wandb_metrics_for_classification,
    test_evaluate_checkpoint_processes_first_task_batch_even_when_it_exceeds_cap,
    test_evaluate_checkpoint_rejects_regression_checkpoint,
    test_evaluate_checkpoint_smoke,
    test_evaluate_checkpoint_weights_metrics_by_actual_task_batch_size,
    test_evaluate_loader_caps_by_task_count_without_overshooting,
    test_evaluate_loader_processes_first_task_batch_even_when_it_exceeds_cap,
    test_evaluate_loader_weights_metrics_by_actual_task_batch_size,
)
