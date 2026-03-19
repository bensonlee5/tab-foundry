from __future__ import annotations

import functools
import inspect

from . import manifest_and_dataset_cases as cases


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
        "test_remap_labels_uses_train_only",
        "test_remap_labels_filters_unseen_test_classes",
        "test_dataset_raises_when_unseen_filter_removes_all_test_rows",
        "test_dataset_keeps_nan_features_when_impute_missing_is_false_but_still_remaps_labels",
        "test_dataset_rejects_missing_inputs_by_default",
        "test_dataset_and_reference_consumer_share_runtime_preprocessing_semantics",
    ]
)
