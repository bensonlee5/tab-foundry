from __future__ import annotations

import functools
import inspect

from . import exporter_cases as cases


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
        "test_validate_export_rejects_tabfoundry_simple_manifest_that_breaks_constructor_invariants",
        "test_validate_export_rejects_tabfoundry_staged_manifest_model_inference_feature_group_size_mismatch",
        "test_validate_export_accepts_tabfoundry_staged_manifest_when_feature_group_size_matches",
        "test_validate_export_detects_weights_checksum_tamper",
        "test_validate_export_rejects_unsupported_schema",
        "test_validate_export_rejects_old_manifest_arch",
        "test_validate_export_rejects_old_inference_model_arch",
        "test_validate_export_v2_rejects_inference_model_arch_mismatch",
        "test_validate_export_v2_rejects_inference_model_stage_mismatch",
        "test_validate_export_rejects_invalid_input_normalization",
        "test_validate_export_rejects_manifest_model_tamper_with_stale_manifest_sha256",
        "test_validate_export_rejects_manifest_inference_tamper_with_stale_manifest_sha256",
        "test_validate_export_rejects_quantile_levels_for_classification",
        "test_validate_export_rejects_fixed_inference_contract_drift",
        "test_validate_export_rejects_fixed_preprocessor_contract_drift",
        "test_validate_export_rejects_manifest_preprocessor_tamper_with_stale_manifest_sha256",
        "test_validate_export_requires_manifest_sha256_for_v3",
        "test_validate_export_rejects_malformed_manifest_sha256_for_v3",
    ]
)
