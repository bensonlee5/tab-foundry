from __future__ import annotations

# ruff: noqa: F401

from tests.support.prior_train_cases import (
    test_prior_dump_reader_rejects_nan_or_inf_inputs_by_default,
    test_prior_dump_reader_rejects_mixed_feature_widths,
    test_prior_dump_reader_rejects_mixed_split_positions,
    test_prior_dump_reader_rejects_non_binary_dump,
    test_prior_dump_reader_rejects_nonfinite_padded_batch_cells_by_default,
    test_prior_dump_reader_reports_inf_labels_as_nonfinite,
    test_prior_dump_reader_skip_policy_errors_when_full_cycle_is_nonfinite,
    test_prior_dump_reader_skips_nonfinite_batches_when_requested,
    test_prior_dump_reader_slices_tasks_from_batch,
)
