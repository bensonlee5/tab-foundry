"""Manifest validation for export bundles."""

from __future__ import annotations

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings

from .common import _validate_payload_model
from .inference import _inference_config_from_payload
from .models import (
    SCHEMA_VERSION_V3,
    ExportManifest,
    ExportModelSpec,
    ExportWeights,
    ProducerInfo,
    _ExportPreprocessorStatePayload,
    _InferenceConfigPayload,
    _ManifestModelPayloadV3,
    _ManifestPayloadV3,
)
from .preprocessor import (
    _export_preprocessor_state_from_payload,
)


_SHA256_HEX_LENGTH = 64


def _validate_exact_keys(
    payload: object,
    *,
    payload_model: type[object],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        return
    model_fields = getattr(payload_model, "model_fields", {})
    required = {name for name, field in model_fields.items() if field.is_required()}
    optional = set(model_fields) - required
    actual = {str(key) for key in payload}
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing={missing}")
        if extra:
            detail_parts.append(f"extra={extra}")
        raise ValueError(f"{context} keys mismatch: {' '.join(detail_parts)}")


def _validate_manifest_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("manifest.manifest_sha256 must be a 64-char hex digest")
    if len(value) != _SHA256_HEX_LENGTH:
        raise ValueError("manifest.manifest_sha256 must be a 64-char hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("manifest.manifest_sha256 must be a 64-char hex digest") from exc
    return value


def _producer_info_from_payload(payload: _ManifestPayloadV3) -> ProducerInfo:
    producer = payload.producer
    return ProducerInfo(
        name=str(producer.name),
        version=str(producer.version),
        git_sha=None if producer.git_sha is None else str(producer.git_sha),
    )


def _validate_model_spec(
    payload: _ManifestModelPayloadV3,
    *,
    task: str,
) -> ExportModelSpec:
    model_spec = model_build_spec_from_mappings(
        task=task,
        primary=payload.model_dump(exclude_none=True),
    )
    try:
        _ = build_model_from_spec(model_spec)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return ExportModelSpec.from_build_spec(model_spec, arch=str(payload.arch))


def validate_manifest_dict(payload: dict[str, object]) -> ExportManifest:
    schema_version_raw = payload.get("schema_version")
    if schema_version_raw != SCHEMA_VERSION_V3:
        raise ValueError(f"Unsupported schema version: {schema_version_raw!r}")

    if "manifest_sha256" not in payload:
        raise ValueError(
            "manifest.manifest_sha256 is required for tab-foundry-export-v3 bundles; "
            "older v3 bundles must be regenerated"
        )
    _validate_exact_keys(payload, payload_model=_ManifestPayloadV3, context="manifest")
    _ = _validate_manifest_sha256(payload.get("manifest_sha256"))
    model_raw = payload.get("model")
    _validate_exact_keys(model_raw, payload_model=_ManifestModelPayloadV3, context="manifest.model")
    _validate_exact_keys(
        payload.get("inference"),
        payload_model=_InferenceConfigPayload,
        context="manifest.inference",
    )
    _validate_exact_keys(
        payload.get("preprocessor"),
        payload_model=_ExportPreprocessorStatePayload,
        context="manifest.preprocessor",
    )
    validated_v3 = _validate_payload_model(
        _ManifestPayloadV3,
        payload,
        context="manifest",
    )
    task = str(validated_v3.task)
    model = _validate_model_spec(validated_v3.model, task=task)
    inference = _inference_config_from_payload(validated_v3.inference)
    if inference.task != task:
        raise ValueError("manifest.task and manifest.inference.task mismatch")
    if inference.model_arch != model.arch:
        raise ValueError("manifest.model.arch and manifest.inference.model_arch mismatch")
    if inference.model_stage != model.stage:
        raise ValueError("manifest.model.stage and manifest.inference.model_stage mismatch")
    if inference.feature_group_size != model.feature_group_size:
        raise ValueError("feature_group_size mismatch between manifest.model and manifest.inference")
    preprocessor = _export_preprocessor_state_from_payload(validated_v3.preprocessor)
    weights = ExportWeights(
        file=str(validated_v3.weights.file),
        sha256=str(validated_v3.weights.sha256),
    )
    return ExportManifest(
        schema_version=str(validated_v3.schema_version),
        producer=_producer_info_from_payload(validated_v3),
        task=task,
        model=model,
        created_at_utc=str(validated_v3.created_at_utc),
        manifest_sha256=str(validated_v3.manifest_sha256),
        inference=inference,
        preprocessor=preprocessor,
        weights=weights,
    )
