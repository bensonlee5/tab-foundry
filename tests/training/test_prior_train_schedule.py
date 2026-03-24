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
        "test_train_tabfoundry_simple_prior_keeps_constant_lr_when_schedule_is_disabled",
        "test_train_tabfoundry_simple_prior_applies_linear_decay_schedule",
        "test_train_tabfoundry_simple_prior_applies_linear_warmup_decay_schedule",
        "test_train_tabfoundry_simple_prior_scales_lr_with_prior_dump_batch_size",
        "test_train_tabfoundry_simple_prior_rejects_mismatched_schedule_steps",
    ]
)
