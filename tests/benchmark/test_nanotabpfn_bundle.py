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
        "test_load_benchmark_bundle_requires_full_selection",
        "test_explicit_benchmark_bundle_paths_accept_checked_in_legacy_and_medium_binary_bundles",
        "test_default_benchmark_bundle_path_resolves_to_medium_binary_bundle",
    ]
)
