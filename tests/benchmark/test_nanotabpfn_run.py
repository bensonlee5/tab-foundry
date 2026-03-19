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
        "test_run_nanotabpfn_benchmark_orchestrates_external_helper",
        "test_run_nanotabpfn_benchmark_explicit_large_bundle_allows_missing_inputs",
        "test_run_nanotabpfn_benchmark_honors_nondefault_bundle_path",
        "test_run_nanotabpfn_benchmark_skips_legacy_record_derivation_failure",
        "test_run_nanotabpfn_benchmark_includes_control_baseline_annotation",
        "test_run_nanotabpfn_benchmark_rejects_unknown_control_baseline",
    ]
)
