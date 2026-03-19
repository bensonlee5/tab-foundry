"""Contracts and validators for inference export bundles."""

from __future__ import annotations

from ._contracts.common import (
    canonicalize_v3_manifest_payload,
    compute_v3_manifest_sha256,
    read_json_dict,
)
from ._contracts.inference import validate_inference_config_dict
from ._contracts.manifest import validate_manifest_dict
from ._contracts.models import (
    EXPECTED_GROUP_SHIFTS,
    EXPECTED_MANY_CLASS_THRESHOLD,
    EXPECTED_MISSING_VALUE_ALL_NAN_FILL,
    EXPECTED_V2_FEATURE_ORDER_POLICY,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SUPPORTED_SCHEMA_VERSIONS,
    SUPPORTED_TASKS,
    ExportClassificationLabelPolicy,
    ExportFiles,
    ExportManifest,
    ExportMissingValuePolicy,
    ExportModelSpec,
    ExportPreprocessorState,
    ExportWeights,
    InferenceConfig,
    LegacyPreprocessorState,
    ProducerInfo,
    ValidatedBundle,
)
from ._contracts.preprocessor import validate_preprocessor_state_dict


__all__ = [
    "EXPECTED_GROUP_SHIFTS",
    "EXPECTED_MANY_CLASS_THRESHOLD",
    "EXPECTED_MISSING_VALUE_ALL_NAN_FILL",
    "EXPECTED_V2_FEATURE_ORDER_POLICY",
    "SCHEMA_VERSION_V2",
    "SCHEMA_VERSION_V3",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SUPPORTED_TASKS",
    "ExportClassificationLabelPolicy",
    "ExportFiles",
    "ExportManifest",
    "ExportMissingValuePolicy",
    "ExportModelSpec",
    "ExportPreprocessorState",
    "ExportWeights",
    "InferenceConfig",
    "LegacyPreprocessorState",
    "ProducerInfo",
    "ValidatedBundle",
    "canonicalize_v3_manifest_payload",
    "compute_v3_manifest_sha256",
    "read_json_dict",
    "validate_inference_config_dict",
    "validate_manifest_dict",
    "validate_preprocessor_state_dict",
]
