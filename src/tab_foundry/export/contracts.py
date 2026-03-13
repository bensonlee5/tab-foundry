"""Contracts and validators for inference export bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.spec import (
    ModelBuildSpec,
    SUPPORTED_MANY_CLASS_TRAIN_MODES,
    model_build_spec_from_mappings,
)
from tab_foundry.preprocessing import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
)


SCHEMA_VERSION_V2 = "tab-foundry-export-v2"
SCHEMA_VERSION_V3 = "tab-foundry-export-v3"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION_V2, SCHEMA_VERSION_V3)
SUPPORTED_TASKS = ("classification", "regression")
SUPPORTED_MANY_CLASS_INFERENCE_MODES = ("full_probs",)
EXPECTED_GROUP_SHIFTS = [0, 1, 3]
EXPECTED_MANY_CLASS_THRESHOLD = 10
EXPECTED_V2_FEATURE_ORDER_POLICY = "lexicographic_f_columns"
EXPECTED_MISSING_VALUE_ALL_NAN_FILL = 0.0
SUPPORTED_MODEL_ARCHES = ("tabfoundry",)


@dataclass(slots=True)
class ProducerInfo:
    name: str
    version: str
    git_sha: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportModelSpec:
    arch: str
    d_col: int
    d_icl: int
    input_normalization: str
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

    @classmethod
    def from_build_spec(
        cls,
        spec: ModelBuildSpec,
        *,
        arch: str = "tabfoundry",
    ) -> "ExportModelSpec":
        return cls(
            arch=str(arch),
            d_col=int(spec.d_col),
            d_icl=int(spec.d_icl),
            input_normalization=str(spec.input_normalization),
            feature_group_size=int(spec.feature_group_size),
            many_class_train_mode=str(spec.many_class_train_mode),
            max_mixed_radix_digits=int(spec.max_mixed_radix_digits),
            tfcol_n_heads=int(spec.tfcol_n_heads),
            tfcol_n_layers=int(spec.tfcol_n_layers),
            tfcol_n_inducing=int(spec.tfcol_n_inducing),
            tfrow_n_heads=int(spec.tfrow_n_heads),
            tfrow_n_layers=int(spec.tfrow_n_layers),
            tfrow_cls_tokens=int(spec.tfrow_cls_tokens),
            tficl_n_heads=int(spec.tficl_n_heads),
            tficl_n_layers=int(spec.tficl_n_layers),
            tficl_ff_expansion=int(spec.tficl_ff_expansion),
            many_class_base=int(spec.many_class_base),
            head_hidden_dim=int(spec.head_hidden_dim),
            use_digit_position_embed=bool(spec.use_digit_position_embed),
        )

    def to_build_spec(self, task: str) -> ModelBuildSpec:
        return model_build_spec_from_mappings(
            task=task,
            primary={
                "d_col": self.d_col,
                "d_icl": self.d_icl,
                "input_normalization": self.input_normalization,
                "feature_group_size": self.feature_group_size,
                "many_class_train_mode": self.many_class_train_mode,
                "max_mixed_radix_digits": self.max_mixed_radix_digits,
                "tfcol_n_heads": self.tfcol_n_heads,
                "tfcol_n_layers": self.tfcol_n_layers,
                "tfcol_n_inducing": self.tfcol_n_inducing,
                "tfrow_n_heads": self.tfrow_n_heads,
                "tfrow_n_layers": self.tfrow_n_layers,
                "tfrow_cls_tokens": self.tfrow_cls_tokens,
                "tficl_n_heads": self.tficl_n_heads,
                "tficl_n_layers": self.tficl_n_layers,
                "tficl_ff_expansion": self.tficl_ff_expansion,
                "many_class_base": self.many_class_base,
                "head_hidden_dim": self.head_hidden_dim,
                "use_digit_position_embed": self.use_digit_position_embed,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportFiles:
    weights: str
    inference_config: str
    preprocessor_state: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportWeights:
    file: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class InferenceConfig:
    task: str
    model_arch: str
    group_shifts: list[int]
    feature_group_size: int
    many_class_threshold: int
    many_class_inference_mode: str
    quantile_levels: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        if self.quantile_levels is None:
            payload.pop("quantile_levels", None)
        return payload


@dataclass(slots=True)
class LegacyPreprocessorState:
    feature_order_policy: str
    missing_value_policy: dict[str, Any]
    classification_label_policy: dict[str, Any]
    dtype_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportMissingValuePolicy:
    strategy: str
    all_nan_fill: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportClassificationLabelPolicy:
    mapping: str
    unseen_test_label: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportPreprocessorState:
    feature_order_policy: str
    missing_value_policy: ExportMissingValuePolicy
    classification_label_policy: ExportClassificationLabelPolicy | None
    dtype_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True)
class ExportManifest:
    schema_version: str
    producer: ProducerInfo
    task: str
    model: ExportModelSpec
    created_at_utc: str
    manifest_sha256: str | None = None
    inference: InferenceConfig | None = None
    preprocessor: LegacyPreprocessorState | ExportPreprocessorState | None = None
    weights: ExportWeights | None = None
    files: ExportFiles | None = None
    checksums: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "producer": self.producer.to_dict(),
            "task": self.task,
            "model": self.model.to_dict(),
            "created_at_utc": self.created_at_utc,
        }
        if self.schema_version == SCHEMA_VERSION_V3:
            if self.inference is None or self.preprocessor is None or self.weights is None:
                raise RuntimeError("v3 manifest requires inference, preprocessor, and weights")
            payload["inference"] = self.inference.to_dict()
            payload["preprocessor"] = self.preprocessor.to_dict()
            payload["weights"] = self.weights.to_dict()
            if self.manifest_sha256 is not None:
                payload["manifest_sha256"] = self.manifest_sha256
            return payload
        if self.files is None or self.checksums is None:
            raise RuntimeError("v2 manifest requires files and checksums")
        payload["files"] = self.files.to_dict()
        payload["checksums"] = dict(self.checksums)
        return payload


@dataclass(slots=True)
class ValidatedBundle:
    manifest: ExportManifest
    inference_config: InferenceConfig
    preprocessor_state: LegacyPreprocessorState | ExportPreprocessorState


def read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def canonicalize_v3_manifest_payload(payload: dict[str, Any]) -> bytes:
    schema_version = _as_str(payload.get("schema_version"), context="manifest.schema_version")
    if schema_version != SCHEMA_VERSION_V3:
        raise ValueError(
            "canonicalize_v3_manifest_payload requires a tab-foundry-export-v3 payload, "
            f"got {schema_version!r}"
        )
    canonical_payload = dict(payload)
    canonical_payload.pop("manifest_sha256", None)
    try:
        return json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ValueError("v3 manifest contains non-canonical JSON values") from exc


def compute_v3_manifest_sha256(payload: dict[str, Any]) -> str:
    return sha256(canonicalize_v3_manifest_payload(payload)).hexdigest()


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


def _validate_hex_digest(value: Any, *, context: str) -> str:
    digest = _as_str(value, context=context)
    if len(digest) != 64:
        raise ValueError(f"{context} must be a 64-char hex digest")
    return digest


def _validate_input_normalization(value: Any, *, context: str) -> str:
    input_normalization = _as_str(value, context=context).strip().lower()
    if input_normalization not in SUPPORTED_INPUT_NORMALIZATION_MODES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_INPUT_NORMALIZATION_MODES}, "
            f"got {input_normalization!r}"
        )
    return input_normalization


def _validate_many_class_train_mode(value: Any, *, context: str) -> str:
    many_class_train_mode = _as_str(value, context=context).strip().lower()
    if many_class_train_mode not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_MANY_CLASS_TRAIN_MODES}, "
            f"got {many_class_train_mode!r}"
        )
    return many_class_train_mode


def _manifest_model_primary_dict(model_raw: dict[str, Any]) -> dict[str, Any]:
    primary: dict[str, Any] = {
        "d_col": _as_int(model_raw["d_col"], context="manifest.model.d_col"),
        "d_icl": _as_int(model_raw["d_icl"], context="manifest.model.d_icl"),
        "feature_group_size": _as_int(
            model_raw["feature_group_size"],
            context="manifest.model.feature_group_size",
        ),
        "many_class_train_mode": _validate_many_class_train_mode(
            model_raw["many_class_train_mode"],
            context="manifest.model.many_class_train_mode",
        ),
        "max_mixed_radix_digits": _as_int(
            model_raw["max_mixed_radix_digits"],
            context="manifest.model.max_mixed_radix_digits",
        ),
    }
    optional_fields: tuple[tuple[str, str], ...] = (
        ("input_normalization", "manifest.model.input_normalization"),
        ("tfcol_n_heads", "manifest.model.tfcol_n_heads"),
        ("tfcol_n_layers", "manifest.model.tfcol_n_layers"),
        ("tfcol_n_inducing", "manifest.model.tfcol_n_inducing"),
        ("tfrow_n_heads", "manifest.model.tfrow_n_heads"),
        ("tfrow_n_layers", "manifest.model.tfrow_n_layers"),
        ("tfrow_cls_tokens", "manifest.model.tfrow_cls_tokens"),
        ("tficl_n_heads", "manifest.model.tficl_n_heads"),
        ("tficl_n_layers", "manifest.model.tficl_n_layers"),
        ("tficl_ff_expansion", "manifest.model.tficl_ff_expansion"),
        ("many_class_base", "manifest.model.many_class_base"),
        ("head_hidden_dim", "manifest.model.head_hidden_dim"),
    )
    for field_name, context in optional_fields:
        if field_name not in model_raw:
            continue
        if field_name == "input_normalization":
            primary[field_name] = _validate_input_normalization(model_raw[field_name], context=context)
        else:
            primary[field_name] = _as_int(model_raw[field_name], context=context)
    if "use_digit_position_embed" in model_raw:
        primary["use_digit_position_embed"] = _as_bool(
            model_raw["use_digit_position_embed"],
            context="manifest.model.use_digit_position_embed",
        )
    return primary


def _validate_producer_info(payload: Any) -> ProducerInfo:
    if not isinstance(payload, dict):
        raise ValueError("manifest.producer must be object")
    _require_keys(
        payload,
        keys={"name", "version", "git_sha"},
        context="manifest.producer",
    )
    git_sha_raw = payload["git_sha"]
    if git_sha_raw is not None and not isinstance(git_sha_raw, str):
        raise ValueError("manifest.producer.git_sha must be string or null")
    return ProducerInfo(
        name=_as_str(payload["name"], context="manifest.producer.name"),
        version=_as_str(payload["version"], context="manifest.producer.version"),
        git_sha=git_sha_raw,
    )


def _validate_model_spec(
    payload: Any,
    *,
    task: str,
    schema_version: str,
) -> ExportModelSpec:
    if not isinstance(payload, dict):
        raise ValueError("manifest.model must be object")
    required_model_keys = {
        "arch",
        "d_col",
        "d_icl",
        "feature_group_size",
        "many_class_train_mode",
        "max_mixed_radix_digits",
    }
    optional_model_keys = {
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
    }
    if schema_version == SCHEMA_VERSION_V3:
        required_model_keys.add("input_normalization")
    else:
        optional_model_keys.add("input_normalization")
    _require_keys(
        payload,
        keys=required_model_keys,
        context="manifest.model",
        optional_keys=optional_model_keys,
    )
    arch = _as_str(payload["arch"], context="manifest.model.arch")
    if arch not in SUPPORTED_MODEL_ARCHES:
        raise ValueError(f"Unsupported model arch: {arch!r}")
    model_spec = model_build_spec_from_mappings(
        task=task,
        primary=_manifest_model_primary_dict(payload),
    )
    return ExportModelSpec.from_build_spec(model_spec, arch=arch)


def _validate_created_at_utc(value: Any) -> str:
    created_at_utc = _as_str(value, context="manifest.created_at_utc")
    try:
        datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("manifest.created_at_utc must be ISO8601") from exc
    return created_at_utc


def validate_manifest_dict(payload: dict[str, Any]) -> ExportManifest:
    schema_version = _as_str(payload.get("schema_version"), context="manifest.schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema version: {schema_version!r}")

    common_keys = {"schema_version", "producer", "task", "model", "created_at_utc"}
    if schema_version == SCHEMA_VERSION_V3:
        if "manifest_sha256" not in payload:
            raise ValueError(
                "manifest.manifest_sha256 is required for tab-foundry-export-v3 bundles; "
                "older v3 bundles must be regenerated"
            )
        _require_keys(
            payload,
            keys=common_keys | {"manifest_sha256", "inference", "preprocessor", "weights"},
            context="manifest",
        )
    else:
        _require_keys(
            payload,
            keys=common_keys | {"files", "checksums"},
            context="manifest",
        )

    producer = _validate_producer_info(payload["producer"])
    task = _as_str(payload["task"], context="manifest.task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported manifest task: {task!r}")
    model = _validate_model_spec(payload["model"], task=task, schema_version=schema_version)
    created_at_utc = _validate_created_at_utc(payload["created_at_utc"])
    manifest_sha256: str | None = None

    if schema_version == SCHEMA_VERSION_V3:
        manifest_sha256 = _validate_hex_digest(
            payload["manifest_sha256"],
            context="manifest.manifest_sha256",
        )
        inference_raw = payload["inference"]
        if not isinstance(inference_raw, dict):
            raise ValueError("manifest.inference must be object")
        inference = validate_inference_config_dict(inference_raw)
        if inference.task != task:
            raise ValueError("manifest.task and manifest.inference.task mismatch")
        if inference.feature_group_size != model.feature_group_size:
            raise ValueError("feature_group_size mismatch between manifest.model and manifest.inference")
        preprocessor_raw = payload["preprocessor"]
        if not isinstance(preprocessor_raw, dict):
            raise ValueError("manifest.preprocessor must be object")
        preprocessor = validate_preprocessor_state_dict(
            preprocessor_raw,
            schema_version=schema_version,
            task=task,
        )
        if not isinstance(preprocessor, ExportPreprocessorState):
            raise RuntimeError("v3 manifest preprocessor must validate to export preprocessor state")
        weights_raw = payload["weights"]
        if not isinstance(weights_raw, dict):
            raise ValueError("manifest.weights must be object")
        _require_keys(
            weights_raw,
            keys={"file", "sha256"},
            context="manifest.weights",
        )
        weights = ExportWeights(
            file=_as_str(weights_raw["file"], context="manifest.weights.file"),
            sha256=_validate_hex_digest(weights_raw["sha256"], context="manifest.weights.sha256"),
        )
        return ExportManifest(
            schema_version=schema_version,
            producer=producer,
            task=task,
            model=model,
            created_at_utc=created_at_utc,
            manifest_sha256=manifest_sha256,
            inference=inference,
            preprocessor=preprocessor,
            weights=weights,
        )

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
        inference_config=_as_str(files_raw["inference_config"], context="manifest.files.inference_config"),
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
    checksums = {
        key: _validate_hex_digest(value, context=f"manifest.checksums.{key}")
        for key, value in checksums_raw.items()
    }
    return ExportManifest(
        schema_version=schema_version,
        producer=producer,
        task=task,
        model=model,
        created_at_utc=created_at_utc,
        files=files,
        checksums=checksums,
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
    if model_arch not in SUPPORTED_MODEL_ARCHES:
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
        raise ValueError(
            "inference_config.many_class_inference_mode must be one of "
            f"{SUPPORTED_MANY_CLASS_INFERENCE_MODES}, got {many_class_inference_mode!r}"
        )

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


def _validate_dtype_policy(dtype_policy: Any) -> dict[str, str]:
    if not isinstance(dtype_policy, dict):
        raise ValueError("preprocessor_state.dtype_policy must be object")
    _require_keys(
        dtype_policy,
        keys=set(DTYPE_POLICY.keys()),
        context="preprocessor_state.dtype_policy",
    )
    normalized: dict[str, str] = {}
    for key, expected in DTYPE_POLICY.items():
        actual = _as_str(dtype_policy[key], context=f"preprocessor_state.dtype_policy.{key}")
        if actual != expected:
            raise ValueError(
                f"preprocessor_state.dtype_policy.{key} must equal {expected!r}, got {actual!r}"
            )
        normalized[key] = actual
    return normalized


def _validate_v2_preprocessor_state(payload: dict[str, Any]) -> LegacyPreprocessorState:
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
    if feature_order_policy != EXPECTED_V2_FEATURE_ORDER_POLICY:
        raise ValueError(
            "preprocessor_state.feature_order_policy must equal "
            f"{EXPECTED_V2_FEATURE_ORDER_POLICY!r}"
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
    if strategy != MISSING_VALUE_STRATEGY_TRAIN_MEAN:
        raise ValueError(
            "preprocessor_state.missing_value_policy.strategy must equal "
            f"{MISSING_VALUE_STRATEGY_TRAIN_MEAN!r}"
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
    if mapping != CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP:
        raise ValueError(
            "preprocessor_state.classification_label_policy.mapping must be "
            f"{CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP!r}"
        )
    unseen_test_label = _as_str(
        classification_label_policy["unseen_test_label"],
        context="preprocessor_state.classification_label_policy.unseen_test_label",
    )
    if unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
        raise ValueError(
            "preprocessor_state.classification_label_policy.unseen_test_label must equal "
            f"{UNSEEN_TEST_LABEL_POLICY_FILTER!r}"
        )

    return LegacyPreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy={
            "strategy": strategy,
            "all_nan_fill": all_nan_fill,
        },
        classification_label_policy={
            "mapping": mapping,
            "unseen_test_label": unseen_test_label,
        },
        dtype_policy=_validate_dtype_policy(payload["dtype_policy"]),
    )


def _validate_v3_missing_value_policy(payload: Any) -> ExportMissingValuePolicy:
    if not isinstance(payload, dict):
        raise ValueError("preprocessor_state.missing_value_policy must be object")
    _require_keys(
        payload,
        keys={"strategy", "all_nan_fill"},
        context="preprocessor_state.missing_value_policy",
    )
    strategy = _as_str(
        payload["strategy"],
        context="preprocessor_state.missing_value_policy.strategy",
    )
    if strategy != MISSING_VALUE_STRATEGY_TRAIN_MEAN:
        raise ValueError(
            "preprocessor_state.missing_value_policy.strategy must equal "
            f"{MISSING_VALUE_STRATEGY_TRAIN_MEAN!r}"
        )
    all_nan_fill = _as_float(
        payload["all_nan_fill"],
        context="preprocessor_state.missing_value_policy.all_nan_fill",
    )
    if all_nan_fill != EXPECTED_MISSING_VALUE_ALL_NAN_FILL:
        raise ValueError(
            "preprocessor_state.missing_value_policy.all_nan_fill must equal "
            f"{EXPECTED_MISSING_VALUE_ALL_NAN_FILL}"
        )
    return ExportMissingValuePolicy(
        strategy=strategy,
        all_nan_fill=all_nan_fill,
    )


def _validate_v3_classification_label_policy(
    payload: Any,
    *,
    task: str,
) -> ExportClassificationLabelPolicy | None:
    if task == "regression":
        if payload is not None:
            raise ValueError("preprocessor_state.classification_label_policy must be null for regression")
        return None
    if not isinstance(payload, dict):
        raise ValueError("preprocessor_state.classification_label_policy must be object for classification")
    _require_keys(
        payload,
        keys={"mapping", "unseen_test_label"},
        context="preprocessor_state.classification_label_policy",
    )
    mapping = _as_str(
        payload["mapping"],
        context="preprocessor_state.classification_label_policy.mapping",
    )
    if mapping != CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP:
        raise ValueError(
            "preprocessor_state.classification_label_policy.mapping must equal "
            f"{CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP!r}"
        )
    unseen_test_label = _as_str(
        payload["unseen_test_label"],
        context="preprocessor_state.classification_label_policy.unseen_test_label",
    )
    if unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
        raise ValueError(
            "preprocessor_state.classification_label_policy.unseen_test_label must equal "
            f"{UNSEEN_TEST_LABEL_POLICY_FILTER!r}"
        )
    return ExportClassificationLabelPolicy(
        mapping=mapping,
        unseen_test_label=unseen_test_label,
    )


def _validate_v3_preprocessor_state(
    payload: dict[str, Any],
    *,
    task: str,
) -> ExportPreprocessorState:
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
    if feature_order_policy != FEATURE_ORDER_POLICY_POSITIONAL:
        raise ValueError(
            "preprocessor_state.feature_order_policy must equal "
            f"{FEATURE_ORDER_POLICY_POSITIONAL!r}"
        )
    return ExportPreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy=_validate_v3_missing_value_policy(payload["missing_value_policy"]),
        classification_label_policy=_validate_v3_classification_label_policy(
            payload["classification_label_policy"],
            task=task,
        ),
        dtype_policy=_validate_dtype_policy(payload["dtype_policy"]),
    )


def validate_preprocessor_state_dict(
    payload: dict[str, Any],
    *,
    schema_version: str = SCHEMA_VERSION_V2,
    task: str = "classification",
) -> LegacyPreprocessorState | ExportPreprocessorState:
    if schema_version == SCHEMA_VERSION_V2:
        return _validate_v2_preprocessor_state(payload)
    if schema_version == SCHEMA_VERSION_V3:
        return _validate_v3_preprocessor_state(payload, task=task)
    raise ValueError(f"Unsupported schema version: {schema_version!r}")
