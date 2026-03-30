from __future__ import annotations

# ruff: noqa: F401

from tests.support.manifest_and_dataset_cases import (
    test_dataset_error_includes_dataset_identity_key_for_canonical_dagzoo_manifest_rows,
    test_dataset_rejects_metadata_checksum_mismatch,
    test_dataset_resolves_relative_paths_from_manifest_location,
    test_manifest_canonical_dagzoo_identity_key_is_unique_across_request_runs,
    test_manifest_canonical_dagzoo_multi_root_order_is_deterministic,
    test_manifest_dataset_id_and_split_are_stable_across_root_paths,
    test_manifest_dataset_id_is_unique_across_nested_runs_with_same_root,
    test_manifest_handles_null_n_features_in_metadata,
    test_manifest_keeps_root_derived_dataset_id_for_non_dagzoo_hex_metadata_id,
    test_manifest_multi_root_order_is_deterministic,
    test_manifest_paths_are_relative_to_manifest_dir,
    test_manifest_prefers_canonical_dagzoo_dataset_id_across_root_paths,
)
