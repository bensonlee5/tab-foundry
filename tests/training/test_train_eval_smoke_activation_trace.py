from __future__ import annotations

# ruff: noqa: F401

from tests.support.train_eval_smoke_cases import (
    test_train_aggregates_activation_norms_across_grad_accum_with_exact_trace_sizes,
    test_train_records_non_finite_global_grad_norm_kinds,
    test_train_reduces_activation_norms_across_accelerator_ranks,
    test_train_skips_activation_rank_reduction_when_tracing_disabled,
    test_train_skips_optimizer_step_when_remote_rank_reports_nan,
    test_train_trace_activations_handles_context_disabled_surface,
    test_train_trace_activations_requires_raw_stats_for_grad_accum,
)
