from __future__ import annotations

# ruff: noqa: F401

from tests.support_research.system_delta_execute import (
    test_ensure_nanotabpfn_python_keeps_existing_usable_interpreter,
    test_ensure_nanotabpfn_python_preserves_fallback_symlink_path,
    test_ensure_nanotabpfn_python_rewrites_existing_interpreter_without_torch,
    test_execute_sweep_applies_overrides_and_promotes_first_row,
    test_execute_sweep_promotes_anchor_before_queue_write_and_matrix_sync,
    test_execute_sweep_passes_resolved_auto_device_to_run_row,
    test_execute_sweep_recovers_partial_anchor_promotion_before_running_later_rows,
    test_execute_sweep_rejects_anchor_only_resume_without_anchor,
    test_execute_sweep_rejects_mps_device_programmatically,
    test_execute_sweep_requires_explicit_sweep_id,
    test_execute_sweep_uses_completed_parent_delta_ref,
    test_execute_sweep_uses_same_invocation_parent_delta_ref,
    test_main_allows_omitting_prior_dump,
    test_main_passes_reuse_nanotabpfn_only,
    test_main_preserves_tab_foundry_python_symlink_path,
    test_main_rejects_explicit_missing_prior_dump,
    test_main_rejects_mps_device_for_sweeps,
    test_runner_imports_benchmark_runtime_from_comparison_runtime,
    test_select_queue_rows_defaults_to_ready_rows,
    test_select_queue_rows_requires_include_completed_for_explicit_completed_rows,
    test_select_queue_rows_requires_include_completed_for_explicit_screened_rows,
)
