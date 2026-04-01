"""Preprocessor-state validation for export bundles."""

from __future__ import annotations

from .common import _validate_payload_model
from .models import (
    SCHEMA_VERSION_V3,
    ExportClassificationLabelPolicy,
    ExportMissingValuePolicy,
    ExportPreprocessorState,
    _ClassificationLabelPolicyPayload,
    _ExportPreprocessorStatePayload,
)


def _classification_policy_from_payload(
    payload: _ClassificationLabelPolicyPayload,
) -> ExportClassificationLabelPolicy:
    return ExportClassificationLabelPolicy(
        mapping=str(payload.mapping),
        unseen_test_label=str(payload.unseen_test_label),
    )


def _export_preprocessor_state_from_payload(
    payload: _ExportPreprocessorStatePayload,
) -> ExportPreprocessorState:
    return ExportPreprocessorState(
        feature_order_policy=str(payload.feature_order_policy),
        missing_value_policy=ExportMissingValuePolicy(
            strategy=str(payload.missing_value_policy.strategy),
            all_nan_fill=float(payload.missing_value_policy.all_nan_fill),
            impute_missing=bool(payload.missing_value_policy.impute_missing),
        ),
        classification_label_policy=_classification_policy_from_payload(
            payload.classification_label_policy
        ),
        dtype_policy=payload.dtype_policy.model_dump(),
    )


def validate_preprocessor_state_dict(
    payload: dict[str, object],
    *,
    schema_version: str,
    task: str = "classification",
) -> ExportPreprocessorState:
    if schema_version != SCHEMA_VERSION_V3:
        raise ValueError(f"Unsupported schema version: {schema_version!r}")
    if task != "classification":
        raise ValueError(f"Unsupported preprocessor_state task: {task!r}")
    validated_export_payload = _validate_payload_model(
        _ExportPreprocessorStatePayload,
        payload,
        context="preprocessor_state",
    )
    return _export_preprocessor_state_from_payload(validated_export_payload)
