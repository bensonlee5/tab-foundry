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
        "test_load_openml_benchmark_datasets_matches_notebook_filters",
        "test_load_openml_benchmark_datasets_fails_on_bundle_drift",
        "test_load_openml_benchmark_datasets_requires_bundle_new_instances_match",
        "test_load_openml_benchmark_datasets_fails_on_selection_drift",
    ]
)
