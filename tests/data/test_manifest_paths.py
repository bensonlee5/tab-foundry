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
        "test_manifest_dataset_id_and_split_are_stable_across_root_paths",
        "test_manifest_prefers_canonical_dagzoo_dataset_id_across_root_paths",
        "test_manifest_canonical_dagzoo_identity_key_is_unique_across_request_runs",
        "test_manifest_keeps_root_derived_dataset_id_for_non_dagzoo_hex_metadata_id",
        "test_manifest_dataset_id_is_unique_across_nested_runs_with_same_root",
        "test_dataset_resolves_relative_paths_from_manifest_location",
        "test_manifest_paths_are_relative_to_manifest_dir",
        "test_manifest_multi_root_order_is_deterministic",
        "test_manifest_canonical_dagzoo_multi_root_order_is_deterministic",
        "test_manifest_handles_null_n_features_in_metadata",
        "test_dataset_rejects_metadata_checksum_mismatch",
        "test_dataset_error_includes_dataset_identity_key_for_canonical_dagzoo_manifest_rows",
    ]
)
