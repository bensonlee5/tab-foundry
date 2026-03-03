"""Contracts and validators for inference export bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json


SCHEMA_VERSION_V1 = "tab-foundry-export-v1"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION_V1,)
SUPPORTED_TASKS = ("classification", "regression")
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
SUPPORTED_MANY_CLASS_INFERENCE_MODES = ("full_probs",)


@dataclass(slots=True)
class ProducerInfo:
    name: str
    version: str
    git_sha: str | None


@dataclass(slots=True)
class ExportModelSpec:
    arch: str
    d_col: int
    d_icl: int
    feature_group_size: int
    many_class_train_mode: str
    max_mixed_radix_digits: int


@dataclass(slots=True)
class ExportFiles:
    weights: str
    inference_config: str
    preprocessor_state: str


@dataclass(slots=True)
class ExportManifest:
    schema_version: str
    producer: ProducerInfo
    task: str
    model: ExportModelSpec
    files: ExportFiles
    checksums: dict[str, str]
    created_at_utc: str


@dataclass(slots=True)
class InferenceConfig:
    task: str
    model_arch: str
    group_shifts: list[int]
    feature_group_size: int
    many_class_threshold: int
    many_class_inference_mode: str
    quantile_levels: list[float] | None


@dataclass(slots=True)
class PreprocessorState:
    feature_order_policy: str
    missing_value_policy: dict[str, Any]
    classification_label_policy: dict[str, Any]
    dtype_policy: dict[str, Any]


@dataclass(slots=True)
class ValidatedBundle:
    manifest: ExportManifest
    inference_config: InferenceConfig
    preprocessor_state: PreprocessorState


def read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def _require_keys(payload: dict[str, Any], *, keys: set[str], context: str) -> None:
    actual = set(payload.keys())
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{context} keys mismatch: {', '.join(details)}")


def _as_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{context} must be int")
    return int(value)


def _as_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be str")
    return value


def validate_manifest_dict(payload: dict[str, Any]) -> ExportManifest:
    _require_keys(
        payload,
        keys={"schema_version", "producer", "task", "model", "files", "checksums", "created_at_utc"},
        context="manifest",
    )

    schema_version = _as_str(payload["schema_version"], context="manifest.schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema version: {schema_version!r}")

    producer_raw = payload["producer"]
    if not isinstance(producer_raw, dict):
        raise ValueError("manifest.producer must be object")
    _require_keys(
        producer_raw,
        keys={"name", "version", "git_sha"},
        context="manifest.producer",
    )
    git_sha_raw = producer_raw["git_sha"]
    if git_sha_raw is not None and not isinstance(git_sha_raw, str):
        raise ValueError("manifest.producer.git_sha must be string or null")
    producer = ProducerInfo(
        name=_as_str(producer_raw["name"], context="manifest.producer.name"),
        version=_as_str(producer_raw["version"], context="manifest.producer.version"),
        git_sha=git_sha_raw,
    )

    task = _as_str(payload["task"], context="manifest.task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported manifest task: {task!r}")

    model_raw = payload["model"]
    if not isinstance(model_raw, dict):
        raise ValueError("manifest.model must be object")
    _require_keys(
        model_raw,
        keys={
            "arch",
            "d_col",
            "d_icl",
            "feature_group_size",
            "many_class_train_mode",
            "max_mixed_radix_digits",
        },
        context="manifest.model",
    )
    arch = _as_str(model_raw["arch"], context="manifest.model.arch")
    if arch != "tabiclv2":
        raise ValueError(f"Unsupported model arch: {arch!r}")
    many_class_train_mode = _as_str(
        model_raw["many_class_train_mode"],
        context="manifest.model.many_class_train_mode",
    )
    if many_class_train_mode not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
        raise ValueError(f"Unsupported many_class_train_mode: {many_class_train_mode!r}")

    model = ExportModelSpec(
        arch=arch,
        d_col=_as_int(model_raw["d_col"], context="manifest.model.d_col"),
        d_icl=_as_int(model_raw["d_icl"], context="manifest.model.d_icl"),
        feature_group_size=_as_int(
            model_raw["feature_group_size"],
            context="manifest.model.feature_group_size",
        ),
        many_class_train_mode=many_class_train_mode,
        max_mixed_radix_digits=_as_int(
            model_raw["max_mixed_radix_digits"],
            context="manifest.model.max_mixed_radix_digits",
        ),
    )
    if model.d_col <= 0 or model.d_icl <= 0:
        raise ValueError("manifest model dimensions must be positive")
    if model.feature_group_size <= 0 or model.max_mixed_radix_digits <= 0:
        raise ValueError("manifest model grouping/mixed-radix parameters must be positive")

    files_raw = payload["files"]
    if not isinstance(files_raw, dict):
        raise ValueError("manifest.files must be object")
    _require_keys(
        files_raw,
        keys={"weights", "inference_config", "preprocessor_state"},
        context="manifest.files",
    )
    files = ExportFiles(
        weights=_as_str(files_raw["weights"], context="manifest.files.weights"),
        inference_config=_as_str(
            files_raw["inference_config"],
            context="manifest.files.inference_config",
        ),
        preprocessor_state=_as_str(
            files_raw["preprocessor_state"],
            context="manifest.files.preprocessor_state",
        ),
    )

    checksums_raw = payload["checksums"]
    if not isinstance(checksums_raw, dict):
        raise ValueError("manifest.checksums must be object")
    _require_keys(
        checksums_raw,
        keys={"weights", "inference_config", "preprocessor_state"},
        context="manifest.checksums",
    )
    checksums: dict[str, str] = {}
    for key in ("weights", "inference_config", "preprocessor_state"):
        value = _as_str(checksums_raw[key], context=f"manifest.checksums.{key}")
        if len(value) != 64:
            raise ValueError(f"manifest.checksums.{key} must be a 64-char hex digest")
        checksums[key] = value

    created_at_utc = _as_str(payload["created_at_utc"], context="manifest.created_at_utc")
    try:
        datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("manifest.created_at_utc must be ISO8601") from exc

    return ExportManifest(
        schema_version=schema_version,
        producer=producer,
        task=task,
        model=model,
        files=files,
        checksums=checksums,
        created_at_utc=created_at_utc,
    )


def validate_inference_config_dict(payload: dict[str, Any]) -> InferenceConfig:
    keys = {
        "task",
        "model_arch",
        "group_shifts",
        "feature_group_size",
        "many_class_threshold",
        "many_class_inference_mode",
    }
    if "quantile_levels" in payload:
        keys.add("quantile_levels")
    _require_keys(payload, keys=keys, context="inference_config")

    task = _as_str(payload["task"], context="inference_config.task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported inference_config task: {task!r}")

    model_arch = _as_str(payload["model_arch"], context="inference_config.model_arch")
    if model_arch != "tabiclv2":
        raise ValueError(f"Unsupported inference model_arch: {model_arch!r}")

    group_shifts_raw = payload["group_shifts"]
    if not isinstance(group_shifts_raw, list) or any(not isinstance(v, int) for v in group_shifts_raw):
        raise ValueError("inference_config.group_shifts must be list[int]")
    group_shifts = [int(v) for v in group_shifts_raw]

    feature_group_size = _as_int(
        payload["feature_group_size"],
        context="inference_config.feature_group_size",
    )
    if feature_group_size <= 0:
        raise ValueError("inference_config.feature_group_size must be positive")

    many_class_threshold = _as_int(
        payload["many_class_threshold"],
        context="inference_config.many_class_threshold",
    )
    if many_class_threshold <= 0:
        raise ValueError("inference_config.many_class_threshold must be positive")

    many_class_inference_mode = _as_str(
        payload["many_class_inference_mode"],
        context="inference_config.many_class_inference_mode",
    )
    if many_class_inference_mode not in SUPPORTED_MANY_CLASS_INFERENCE_MODES:
        raise ValueError(f"Unsupported many_class_inference_mode: {many_class_inference_mode!r}")

    quantile_levels_raw = payload.get("quantile_levels")
    quantile_levels: list[float] | None = None
    if quantile_levels_raw is not None:
        if task != "regression":
            raise ValueError("inference_config.quantile_levels is only valid for regression")
        if not isinstance(quantile_levels_raw, list):
            raise ValueError("inference_config.quantile_levels must be list[float]")
        quantile_levels = [float(v) for v in quantile_levels_raw]
        if len(quantile_levels) != 999:
            raise ValueError("inference_config.quantile_levels must contain 999 values")

    return InferenceConfig(
        task=task,
        model_arch=model_arch,
        group_shifts=group_shifts,
        feature_group_size=feature_group_size,
        many_class_threshold=many_class_threshold,
        many_class_inference_mode=many_class_inference_mode,
        quantile_levels=quantile_levels,
    )


def validate_preprocessor_state_dict(payload: dict[str, Any]) -> PreprocessorState:
    _require_keys(
        payload,
        keys={
            "feature_order_policy",
            "missing_value_policy",
            "classification_label_policy",
            "dtype_policy",
        },
        context="preprocessor_state",
    )

    feature_order_policy = _as_str(
        payload["feature_order_policy"],
        context="preprocessor_state.feature_order_policy",
    )

    missing_value_policy = payload["missing_value_policy"]
    if not isinstance(missing_value_policy, dict):
        raise ValueError("preprocessor_state.missing_value_policy must be object")
    _require_keys(
        missing_value_policy,
        keys={"strategy", "all_nan_fill"},
        context="preprocessor_state.missing_value_policy",
    )

    classification_label_policy = payload["classification_label_policy"]
    if not isinstance(classification_label_policy, dict):
        raise ValueError("preprocessor_state.classification_label_policy must be object")
    _require_keys(
        classification_label_policy,
        keys={"mapping", "unseen_test_label"},
        context="preprocessor_state.classification_label_policy",
    )

    dtype_policy = payload["dtype_policy"]
    if not isinstance(dtype_policy, dict):
        raise ValueError("preprocessor_state.dtype_policy must be object")
    _require_keys(
        dtype_policy,
        keys={"features", "classification_labels", "regression_targets"},
        context="preprocessor_state.dtype_policy",
    )

    return PreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy=missing_value_policy,
        classification_label_policy=classification_label_policy,
        dtype_policy=dtype_policy,
    )
