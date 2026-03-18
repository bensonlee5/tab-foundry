from __future__ import annotations

import functools
import inspect

from . import nanotabpfn_compare_cases as cases


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
        "test_collect_checkpoint_snapshots_prefers_train_elapsed_seconds",
        "test_collect_checkpoint_snapshots_supports_plain_training_output",
        "test_build_comparison_summary_preserves_model_identity_metadata",
        "test_build_comparison_summary_uses_log_loss_as_classification_best_step",
    ]
)
