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
        "test_resolve_prior_wandb_run_name_prefers_queue_run_id_from_output_dir",
        "test_train_tabfoundry_simple_prior_logs_wandb_metrics_and_summary",
        "test_train_tabfoundry_simple_prior_logs_wandb_failure_summary",
        "test_train_tabfoundry_simple_prior_closes_wandb_for_setup_failures",
    ]
)
