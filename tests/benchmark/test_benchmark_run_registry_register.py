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
        "test_register_benchmark_run_writes_repo_relative_entry_and_deltas",
        "test_register_benchmark_run_rejects_unknown_parent",
        "test_register_benchmark_run_main_parses_cli_and_defaults_config_profile",
    ]
)
