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
        "test_train_tabfoundry_simple_prior_rejects_staged_many_class_before_io",
        "test_train_tabfoundry_simple_prior_writes_failure_telemetry_for_nonfinite_inputs",
        "test_train_tabfoundry_simple_prior_writes_failure_telemetry_for_nonfinite_labels",
        "test_train_tabfoundry_simple_prior_injects_synthetic_missingness",
    ]
)
