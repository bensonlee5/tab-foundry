"""Inference-config validation for export bundles."""

from __future__ import annotations

from tab_foundry.model.spec import STAGED_MODEL_ARCH, model_build_spec_from_mappings

from .common import _validate_payload_model
from .models import _InferenceConfigPayload, InferenceConfig


def _inference_config_from_payload(validated_payload: _InferenceConfigPayload) -> InferenceConfig:
    model_stage = None
    if validated_payload.model_stage is not None:
        model_stage = str(validated_payload.model_stage).strip().lower()
        _ = model_build_spec_from_mappings(
            task=str(validated_payload.task),
            primary={"arch": str(validated_payload.model_arch), "stage": model_stage},
        )
    elif str(validated_payload.model_arch) == STAGED_MODEL_ARCH:
        _ = model_build_spec_from_mappings(
            task=str(validated_payload.task),
            primary={"arch": str(validated_payload.model_arch)},
        )

    return InferenceConfig(
        task=str(validated_payload.task),
        model_arch=str(validated_payload.model_arch),
        model_stage=model_stage,
        group_shifts=[int(value) for value in validated_payload.group_shifts],
        feature_group_size=int(validated_payload.feature_group_size),
        many_class_threshold=int(validated_payload.many_class_threshold),
        many_class_inference_mode=str(validated_payload.many_class_inference_mode),
        quantile_levels=None,
    )


def validate_inference_config_dict(payload: dict[str, object]) -> InferenceConfig:
    validated_payload = _validate_payload_model(
        _InferenceConfigPayload,
        payload,
        context="inference_config",
    )
    return _inference_config_from_payload(validated_payload)
