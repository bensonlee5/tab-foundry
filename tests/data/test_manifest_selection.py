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
        "test_manifest_and_dataset_loading",
        "test_manifest_include_all_tracks_missing_filter_metadata",
        "test_manifest_accepted_only_excludes_unaccepted_records",
        "test_manifest_forbid_any_excludes_datasets_with_nan_or_inf",
        "test_manifest_accepted_only_requires_at_least_one_record",
        "test_manifest_rejects_selected_dataset_index_missing_from_packed_split",
    ]
)
