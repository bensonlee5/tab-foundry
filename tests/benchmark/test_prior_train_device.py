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
        "test_resolve_prior_training_device_name_falls_back_for_multilayer_row_cls_on_mps",
        "test_resolve_prior_training_device_name_keeps_mps_for_target_column",
        "test_resolve_prior_training_device_name_keeps_mps_for_single_layer_row_cls",
        "test_train_tabfoundry_staged_prior_falls_back_to_cpu_for_multilayer_row_cls_on_mps",
    ]
)
