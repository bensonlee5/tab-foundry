from __future__ import annotations

import functools
import inspect

from . import prior_train_cases as cases


def _export(names: list[str]) -> None:
    for name in names:
        case = getattr(cases, name)

        def _wrapped(*args, __case=case, **kwargs):
            return __case(*args, **kwargs)

        functools.update_wrapper(_wrapped, case)
        _wrapped.__name__ = name
        _wrapped.__qualname__ = name
        _wrapped.__signature__ = inspect.signature(case)
        globals()[name] = _wrapped


_export(
    [
        "test_prior_dump_reader_slices_tasks_from_batch",
        "test_prior_dump_reader_uses_first_split_value_in_batch",
        "test_prior_dump_reader_rejects_non_binary_dump",
        "test_prior_dump_reader_rejects_nan_or_inf_inputs_by_default",
        "test_prior_dump_reader_rejects_nonfinite_padded_batch_cells_by_default",
        "test_prior_dump_reader_reports_inf_labels_as_nonfinite",
        "test_prior_dump_reader_skips_nonfinite_batches_when_requested",
        "test_prior_dump_reader_skip_policy_errors_when_full_cycle_is_nonfinite",
    ]
)
