from __future__ import annotations

import functools
import inspect

from . import benchmark_run_registry_cases as cases


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
        "test_derive_benchmark_run_record_extracts_diagnostics_and_model_size",
        "test_derive_benchmark_run_record_captures_optional_training_surface_label",
        "test_derive_benchmark_run_record_includes_optional_sweep_metadata",
        "test_derive_benchmark_run_record_uses_manifest_path_from_resolved_data_surface",
        "test_derive_benchmark_run_record_falls_back_to_best_benchmark_step_checkpoint",
    ]
)
