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
        "test_export_bundle_defaults_to_v3_and_embeds_single_manifest",
        "test_export_bundle_accepts_smooth_tail_input_normalization",
        "test_export_bundle_supports_explicit_v2",
        "test_export_bundle_defaults_omitted_feature_group_size_to_one_when_weights_match",
        "test_export_bundle_rejects_legacy_grouped_weights_when_feature_group_size_is_omitted",
        "test_export_bundle_supports_explicit_nondefault_feature_group_size",
        "test_export_bundle_supports_tabfoundry_simple_checkpoint",
        "test_export_bundle_round_trips_staged_arch_and_stage",
        "test_export_bundle_round_trips_additive_staged_surface_fields",
        "test_export_manifest_embeds_policy_only_preprocessor",
        "test_export_manifest_embeds_resolved_nondefault_preprocessing_policy",
        "test_export_checkpoint_uses_explicit_weights_only_false",
        "test_package_version_prefers_installed_metadata",
        "test_package_version_falls_back_to_pyproject_when_metadata_missing",
        "test_package_version_raises_when_resolution_fails",
    ]
)
