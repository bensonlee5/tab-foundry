from __future__ import annotations

# ruff: noqa: F401

from tests.support.train_eval_smoke_cases import (
    test_build_stage_configs_rejects_non_int_steps,
    test_build_stage_configs_rejects_non_numeric_lr,
    test_build_stage_configs_validates_payloads,
    test_train_activation_checkpointing_enables_supported_model,
    test_train_activation_checkpointing_requires_supported_model,
    test_train_history_uses_linear_schedule_values,
    test_train_can_sample_module_grad_norms_and_record_step_timing,
    test_train_rejects_existing_checkpoint_artifacts,
    test_train_rejects_non_empty_history_jsonl,
    test_train_smoke_runs_end_to_end,
    test_train_smoke_runs_end_to_end_with_tabfoundry_sandwich,
    test_train_smoke_saves_fallback_best_checkpoint_in_eval_mode,
    test_train_smoke_saves_in_loop_checkpoints_in_eval_mode,
    test_train_smoke_skips_validation_loader_when_val_batches_is_zero,
    test_train_smoke_writes_history_jsonl,
    test_train_smoke_writes_step_snapshots,
)
