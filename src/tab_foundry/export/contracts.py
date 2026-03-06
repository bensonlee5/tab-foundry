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
SUPPORTED_UNSEEN_TEST_LABEL_POLICIES = ("filter",)
EXPECTED_GROUP_SHIFTS = [0, 1, 3]
EXPECTED_MANY_CLASS_THRESHOLD = 10
EXPECTED_FEATURE_ORDER_POLICY = "lexicographic_f_columns"
EXPECTED_MISSING_VALUE_STRATEGY = "train_mean"
EXPECTED_MISSING_VALUE_ALL_NAN_FILL = 0.0
EXPECTED_DTYPE_POLICY = {
    "features": "float32",
    "classification_labels": "int64",
    "regression_targets": "float32",
}
DEFAULT_TFCOL_N_HEADS = 8
DEFAULT_TFCOL_N_LAYERS = 3
DEFAULT_TFCOL_N_INDUCING = 128
DEFAULT_TFROW_N_HEADS = 8
DEFAULT_TFROW_N_LAYERS = 3
DEFAULT_TFROW_CLS_TOKENS = 4
DEFAULT_TFICL_N_HEADS = 8
DEFAULT_TFICL_N_LAYERS = 12
DEFAULT_TFICL_FF_EXPANSION = 2
DEFAULT_MANY_CLASS_BASE = 10
DEFAULT_HEAD_HIDDEN_DIM = 1024
DEFAULT_USE_DIGIT_POSITION_EMBED = True


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
    tfcol_n_heads: int
    tfcol_n_layers: int
    tfcol_n_inducing: int
    tfrow_n_heads: int
    tfrow_n_layers: int
    tfrow_cls_tokens: int
    tficl_n_heads: int
    tficl_n_layers: int
    tficl_ff_expansion: int
    many_class_base: int
    head_hidden_dim: int
    use_digit_position_embed: bool


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


def _require_keys(
    payload: dict[str, Any],
    *,
    keys: set[str],
    context: str,
    optional_keys: set[str] | None = None,
) -> None:
    optional = optional_keys if optional_keys is not None else set()
    actual = set(payload.keys())
    missing = sorted(keys - actual)
    extra = sorted(actual - (keys | optional))
    if missing or extra:
        missing = sorted(keys - actual)
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


def _as_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be bool")
    return value


def _as_float(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be float-compatible")
    return float(value)


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
        optional_keys={
            "tfcol_n_heads",
            "tfcol_n_layers",
            "tfcol_n_inducing",
            "tfrow_n_heads",
            "tfrow_n_layers",
            "tfrow_cls_tokens",
            "tficl_n_heads",
            "tficl_n_layers",
            "tficl_ff_expansion",
            "many_class_base",
            "head_hidden_dim",
            "use_digit_position_embed",
        },
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
        tfcol_n_heads=_as_int(
            model_raw.get("tfcol_n_heads", DEFAULT_TFCOL_N_HEADS),
            context="manifest.model.tfcol_n_heads",
        ),
        tfcol_n_layers=_as_int(
            model_raw.get("tfcol_n_layers", DEFAULT_TFCOL_N_LAYERS),
            context="manifest.model.tfcol_n_layers",
        ),
        tfcol_n_inducing=_as_int(
            model_raw.get("tfcol_n_inducing", DEFAULT_TFCOL_N_INDUCING),
            context="manifest.model.tfcol_n_inducing",
        ),
        tfrow_n_heads=_as_int(
            model_raw.get("tfrow_n_heads", DEFAULT_TFROW_N_HEADS),
            context="manifest.model.tfrow_n_heads",
        ),
        tfrow_n_layers=_as_int(
            model_raw.get("tfrow_n_layers", DEFAULT_TFROW_N_LAYERS),
            context="manifest.model.tfrow_n_layers",
        ),
        tfrow_cls_tokens=_as_int(
            model_raw.get("tfrow_cls_tokens", DEFAULT_TFROW_CLS_TOKENS),
            context="manifest.model.tfrow_cls_tokens",
        ),
        tficl_n_heads=_as_int(
            model_raw.get("tficl_n_heads", DEFAULT_TFICL_N_HEADS),
            context="manifest.model.tficl_n_heads",
        ),
        tficl_n_layers=_as_int(
            model_raw.get("tficl_n_layers", DEFAULT_TFICL_N_LAYERS),
            context="manifest.model.tficl_n_layers",
        ),
        tficl_ff_expansion=_as_int(
            model_raw.get("tficl_ff_expansion", DEFAULT_TFICL_FF_EXPANSION),
            context="manifest.model.tficl_ff_expansion",
        ),
        many_class_base=_as_int(
            model_raw.get("many_class_base", DEFAULT_MANY_CLASS_BASE),
            context="manifest.model.many_class_base",
        ),
        head_hidden_dim=_as_int(
            model_raw.get("head_hidden_dim", DEFAULT_HEAD_HIDDEN_DIM),
            context="manifest.model.head_hidden_dim",
        ),
        use_digit_position_embed=_as_bool(
            model_raw.get("use_digit_position_embed", DEFAULT_USE_DIGIT_POSITION_EMBED),
            context="manifest.model.use_digit_position_embed",
        ),
    )
    if model.d_col <= 0 or model.d_icl <= 0:
        raise ValueError("manifest model dimensions must be positive")
    if model.feature_group_size <= 0 or model.max_mixed_radix_digits <= 0:
        raise ValueError("manifest model grouping/mixed-radix parameters must be positive")
    for name, value in (
        ("tfcol_n_heads", model.tfcol_n_heads),
        ("tfcol_n_layers", model.tfcol_n_layers),
        ("tfcol_n_inducing", model.tfcol_n_inducing),
        ("tfrow_n_heads", model.tfrow_n_heads),
        ("tfrow_n_layers", model.tfrow_n_layers),
        ("tfrow_cls_tokens", model.tfrow_cls_tokens),
        ("tficl_n_heads", model.tficl_n_heads),
        ("tficl_n_layers", model.tficl_n_layers),
        ("tficl_ff_expansion", model.tficl_ff_expansion),
        ("many_class_base", model.many_class_base),
        ("head_hidden_dim", model.head_hidden_dim),
    ):
        if value <= 0:
            raise ValueError(f"manifest.model.{name} must be positive")
    if model.many_class_base <= 1:
        raise ValueError("manifest.model.many_class_base must be >= 2")

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
        checksum_value = _as_str(checksums_raw[key], context=f"manifest.checksums.{key}")
        if len(checksum_value) != 64:
            raise ValueError(f"manifest.checksums.{key} must be a 64-char hex digest")
        checksums[key] = checksum_value

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
    if task == "regression":
        keys.add("quantile_levels")
    elif "quantile_levels" in payload:
        keys.add("quantile_levels")
    _require_keys(payload, keys=keys, context="inference_config")

    model_arch = _as_str(payload["model_arch"], context="inference_config.model_arch")
    if model_arch != "tabiclv2":
        raise ValueError(f"Unsupported inference model_arch: {model_arch!r}")

    group_shifts_raw = payload["group_shifts"]
    if not isinstance(group_shifts_raw, list) or any(not isinstance(v, int) for v in group_shifts_raw):
        raise ValueError("inference_config.group_shifts must be list[int]")
    group_shifts = [int(v) for v in group_shifts_raw]
    if group_shifts != EXPECTED_GROUP_SHIFTS:
        raise ValueError(
            f"inference_config.group_shifts must equal {EXPECTED_GROUP_SHIFTS}, got {group_shifts!r}"
        )

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
    if many_class_threshold != EXPECTED_MANY_CLASS_THRESHOLD:
        raise ValueError(
            "inference_config.many_class_threshold must equal "
            f"{EXPECTED_MANY_CLASS_THRESHOLD}, got {many_class_threshold}"
        )

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
    elif task == "regression":
        raise ValueError("inference_config.quantile_levels is required for regression")

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
    if feature_order_policy != EXPECTED_FEATURE_ORDER_POLICY:
        raise ValueError(
            "preprocessor_state.feature_order_policy must equal "
            f"{EXPECTED_FEATURE_ORDER_POLICY!r}"
        )

    missing_value_policy = payload["missing_value_policy"]
    if not isinstance(missing_value_policy, dict):
        raise ValueError("preprocessor_state.missing_value_policy must be object")
    _require_keys(
        missing_value_policy,
        keys={"strategy", "all_nan_fill"},
        context="preprocessor_state.missing_value_policy",
    )
    strategy = _as_str(
        missing_value_policy["strategy"],
        context="preprocessor_state.missing_value_policy.strategy",
    )
    if strategy != EXPECTED_MISSING_VALUE_STRATEGY:
        raise ValueError(
            "preprocessor_state.missing_value_policy.strategy must equal "
            f"{EXPECTED_MISSING_VALUE_STRATEGY!r}"
        )
    all_nan_fill = _as_float(
        missing_value_policy["all_nan_fill"],
        context="preprocessor_state.missing_value_policy.all_nan_fill",
    )
    if all_nan_fill != EXPECTED_MISSING_VALUE_ALL_NAN_FILL:
        raise ValueError(
            "preprocessor_state.missing_value_policy.all_nan_fill must equal "
            f"{EXPECTED_MISSING_VALUE_ALL_NAN_FILL}"
        )

    classification_label_policy = payload["classification_label_policy"]
    if not isinstance(classification_label_policy, dict):
        raise ValueError("preprocessor_state.classification_label_policy must be object")
    _require_keys(
        classification_label_policy,
        keys={"mapping", "unseen_test_label"},
        context="preprocessor_state.classification_label_policy",
    )
    mapping = _as_str(
        classification_label_policy["mapping"],
        context="preprocessor_state.classification_label_policy.mapping",
    )
    if mapping != "train_only_remap":
        raise ValueError("preprocessor_state.classification_label_policy.mapping must be train_only_remap")
    unseen_test_label = _as_str(
        classification_label_policy["unseen_test_label"],
        context="preprocessor_state.classification_label_policy.unseen_test_label",
    )
    if unseen_test_label not in SUPPORTED_UNSEEN_TEST_LABEL_POLICIES:
        allowed = ", ".join(SUPPORTED_UNSEEN_TEST_LABEL_POLICIES)
        raise ValueError(
            "preprocessor_state.classification_label_policy.unseen_test_label must be one of: "
            f"{allowed}"
        )

    dtype_policy = payload["dtype_policy"]
    if not isinstance(dtype_policy, dict):
        raise ValueError("preprocessor_state.dtype_policy must be object")
    _require_keys(
        dtype_policy,
        keys={"features", "classification_labels", "regression_targets"},
        context="preprocessor_state.dtype_policy",
    )
    for key, expected in EXPECTED_DTYPE_POLICY.items():
        actual = _as_str(dtype_policy[key], context=f"preprocessor_state.dtype_policy.{key}")
        if actual != expected:
            raise ValueError(
                f"preprocessor_state.dtype_policy.{key} must equal {expected!r}, got {actual!r}"
            )

    return PreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy=missing_value_policy,
        classification_label_policy=classification_label_policy,
        dtype_policy=dtype_policy,
    )
