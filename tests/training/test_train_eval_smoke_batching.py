from __future__ import annotations

# ruff: noqa: F401

from tests.support.train_eval_smoke_cases import (
    test_train_aggregates_task_batching_telemetry_across_grad_accum_steps,
    test_train_allows_singleton_true_many_class_fallback_preflight,
    test_train_allows_task_batching_for_low_class_many_class_surface,
    test_train_disables_even_batch_padding_for_task_batching,
    test_train_grad_accum_streams_move_and_forward_in_lockstep,
    test_train_history_weights_microstep_metrics_by_actual_task_count,
    test_train_rejects_task_batching_for_non_manifest_loader,
    test_train_rejects_tensor_batched_true_many_class_surface_before_loader,
    test_train_smoke_task_batching_manifest_loader_emits_batching_telemetry,
    test_train_task_batch_grad_accum_matches_all_tasks_reference_update,
)
