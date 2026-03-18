"""Inference-config validation for export bundles."""

from __future__ import annotations

from .common import _as_str, _require_keys, _validate_payload_model
from .models import (
    EXPECTED_GROUP_SHIFTS,
    EXPECTED_MANY_CLASS_THRESHOLD,
    SUPPORTED_MANY_CLASS_INFERENCE_MODES,
    SUPPORTED_TASKS,
    _InferenceConfigPayload,
    InferenceConfig,
)
from tab_foundry.model.spec import STAGED_MODEL_ARCH, SUPPORTED_MODEL_ARCHES, model_build_spec_from_mappings


def validate_inference_config_dict(payload: dict[str, object]) -> InferenceConfig:
    task = _as_str(payload["task"], context="inference_config.task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported inference_config task: {task!r}")

    keys = {
        "task",
        "model_arch",
        "group_shifts",
        "feature_group_size",
        "many_class_threshold",
        "many_class_inference_mode",
    }
    optional_keys = {"model_stage"}
    if task == "regression":
        keys.add("quantile_levels")
    elif "quantile_levels" in payload:
        keys.add("quantile_levels")
    _require_keys(payload, keys=keys, context="inference_config", optional_keys=optional_keys)

    validated_payload = _validate_payload_model(
        _InferenceConfigPayload,
        payload,
        context="inference_config",
    )

    model_arch = _as_str(
        validated_payload.model_arch,
        context="inference_config.model_arch",
    )
    if model_arch not in SUPPORTED_MODEL_ARCHES:
        raise ValueError(f"Unsupported inference model_arch: {model_arch!r}")
    model_stage_raw = validated_payload.model_stage
    if model_stage_raw is not None and model_arch != STAGED_MODEL_ARCH:
        raise ValueError(
            "inference_config.model_stage is only valid when model_arch='tabfoundry_staged'"
        )
    model_stage = None
    if model_stage_raw is not None:
        model_stage = str(model_stage_raw).strip().lower()
        _ = model_build_spec_from_mappings(
            task=task,
            primary={"arch": model_arch, "stage": model_stage},
        )

    group_shifts = [int(v) for v in validated_payload.group_shifts]
    if group_shifts != EXPECTED_GROUP_SHIFTS:
        raise ValueError(
            f"inference_config.group_shifts must equal {EXPECTED_GROUP_SHIFTS}, got {group_shifts!r}"
        )

    feature_group_size = int(validated_payload.feature_group_size)
    if feature_group_size <= 0:
        raise ValueError("inference_config.feature_group_size must be positive")

    many_class_threshold = int(validated_payload.many_class_threshold)
    if many_class_threshold != EXPECTED_MANY_CLASS_THRESHOLD:
        raise ValueError(
            "inference_config.many_class_threshold must equal "
            f"{EXPECTED_MANY_CLASS_THRESHOLD}, got {many_class_threshold}"
        )

    many_class_inference_mode = _as_str(
        validated_payload.many_class_inference_mode,
        context="inference_config.many_class_inference_mode",
    )
    if many_class_inference_mode not in SUPPORTED_MANY_CLASS_INFERENCE_MODES:
        raise ValueError(
            "inference_config.many_class_inference_mode must be one of "
            f"{SUPPORTED_MANY_CLASS_INFERENCE_MODES}, got {many_class_inference_mode!r}"
        )

    quantile_levels_raw = validated_payload.quantile_levels
    quantile_levels: list[float] | None = None
    if quantile_levels_raw is not None:
        if task != "regression":
            raise ValueError("inference_config.quantile_levels is only valid for regression")
        quantile_levels = [float(v) for v in quantile_levels_raw]
        if len(quantile_levels) != 999:
            raise ValueError("inference_config.quantile_levels must contain 999 values")
    elif task == "regression":
        raise ValueError("inference_config.quantile_levels is required for regression")

    return InferenceConfig(
        task=task,
        model_arch=model_arch,
        model_stage=model_stage,
        group_shifts=group_shifts,
        feature_group_size=feature_group_size,
        many_class_threshold=many_class_threshold,
        many_class_inference_mode=many_class_inference_mode,
        quantile_levels=quantile_levels,
    )
