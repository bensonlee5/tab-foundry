"""Public data structures and strict payload models for export contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from tab_foundry.hashing import SHA256_HEX_LENGTH
from tab_foundry.model.spec import (
    SANDWICH_MODEL_ARCH,
    STAGED_MODEL_ARCH,
    SUPPORTED_MODEL_ARCHES,
)
from tab_foundry.preprocessing import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
)


SCHEMA_VERSION_V3: Final = "tab-foundry-export-v3"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION_V3,)
SUPPORTED_TASKS = ("classification",)
SUPPORTED_MANY_CLASS_INFERENCE_MODES = ("full_probs",)
EXPECTED_GROUP_SHIFTS = [0, 1, 3]
EXPECTED_MANY_CLASS_THRESHOLD = 10
EXPECTED_MISSING_VALUE_ALL_NAN_FILL = 0.0


class _ContractsPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


def _validate_created_at_utc(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be ISO8601") from exc
    return value


def _validate_hex_digest(value: str) -> str:
    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError("must be a 64-char hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("must be a 64-char hex digest") from exc
    return value


class _ManifestModelPayloadV3(_ContractsPayloadModel):
    arch: StrictStr
    stage: StrictStr | None = None
    d_col: StrictInt
    d_icl: StrictInt
    input_normalization: StrictStr
    feature_group_size: StrictInt
    many_class_train_mode: StrictStr
    max_mixed_radix_digits: StrictInt
    norm_type: StrictStr | None = None
    tfcol_n_heads: StrictInt | None = None
    tfcol_n_layers: StrictInt | None = None
    tfcol_n_inducing: StrictInt | None = None
    tfrow_n_heads: StrictInt | None = None
    tfrow_n_layers: StrictInt | None = None
    tfrow_cls_tokens: StrictInt | None = None
    tfrow_norm: StrictStr | None = None
    tficl_n_heads: StrictInt | None = None
    tficl_n_layers: StrictInt | None = None
    tficl_ff_expansion: StrictInt | None = None
    many_class_base: StrictInt | None = None
    head_hidden_dim: StrictInt | None = None
    use_digit_position_embed: StrictBool | None = None
    sandwich_latents: StrictInt | None = None
    sandwich_layers: StrictInt | None = None
    sandwich_heads: StrictInt | None = None
    sandwich_ff_expansion: StrictInt | None = None
    sandwich_activation: StrictStr | None = None
    sandwich_block_norm: StrictStr | None = None
    sandwich_summary_tokens_per_axis: StrictInt | None = None
    sandwich_self_attention_per_cross: StrictInt | None = None
    sandwich_pre_row_attention_layers: StrictInt | None = None
    sandwich_pre_column_attention_layers: StrictInt | None = None
    sandwich_pre_column_inducing_tokens: StrictInt | None = None
    feature_type_conditioning: StrictStr | None = None
    floating_likelihood: StrictStr | None = None
    integer_likelihood: StrictStr | None = None
    stage_label: StrictStr | None = None
    module_overrides: dict[StrictStr, Any] | None = None
    staged_dropout: FiniteFloat | None = None
    pre_encoder_clip: FiniteFloat | None = None


class _InferenceConfigPayload(_ContractsPayloadModel):
    task: Literal["classification"]
    model_arch: StrictStr
    model_stage: StrictStr | None = None
    group_shifts: list[StrictInt]
    feature_group_size: StrictInt
    many_class_threshold: StrictInt
    many_class_inference_mode: Literal["full_probs"]
    quantile_levels: list[FiniteFloat] | None = None

    @field_validator("model_arch")
    @classmethod
    def _validate_model_arch(cls, value: str) -> str:
        if value not in SUPPORTED_MODEL_ARCHES:
            raise ValueError(f"Unsupported inference model_arch: {value!r}")
        return value

    @field_validator("group_shifts")
    @classmethod
    def _validate_group_shifts(cls, value: list[int]) -> list[int]:
        if list(value) != EXPECTED_GROUP_SHIFTS:
            raise ValueError(
                f"inference_config.group_shifts must equal {EXPECTED_GROUP_SHIFTS}, got {list(value)!r}"
            )
        return list(value)

    @field_validator("feature_group_size")
    @classmethod
    def _validate_feature_group_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("inference_config.feature_group_size must be positive")
        return int(value)

    @field_validator("many_class_threshold")
    @classmethod
    def _validate_many_class_threshold(cls, value: int) -> int:
        if value != EXPECTED_MANY_CLASS_THRESHOLD:
            raise ValueError(
                "inference_config.many_class_threshold must equal "
                f"{EXPECTED_MANY_CLASS_THRESHOLD}, got {value}"
            )
        return int(value)

    @field_validator("quantile_levels")
    @classmethod
    def _reject_quantile_levels(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            raise ValueError(
                "inference_config.quantile_levels is not supported in this branch; "
                "regression export has been removed pending a staged rebuild."
            )
        return value

    @model_validator(mode="after")
    def _validate_model_stage(self) -> "_InferenceConfigPayload":
        if self.model_stage is not None and self.model_arch != STAGED_MODEL_ARCH:
            raise ValueError(
                "inference_config.model_stage is only valid when model_arch='tabfoundry_staged'"
            )
        return self


class _ProducerInfoPayload(_ContractsPayloadModel):
    name: StrictStr
    version: StrictStr
    git_sha: StrictStr | None = None


class _ExportWeightsPayload(_ContractsPayloadModel):
    file: StrictStr
    sha256: StrictStr

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        return _validate_hex_digest(value)


class _DtypePolicyPayload(_ContractsPayloadModel):
    features: StrictStr
    classification_labels: StrictStr
    regression_targets: StrictStr

    @model_validator(mode="after")
    def _validate_expected_values(self) -> "_DtypePolicyPayload":
        payload = self.model_dump()
        for key, expected in DTYPE_POLICY.items():
            actual = str(payload[key])
            if actual != expected:
                raise ValueError(
                    f"preprocessor_state.dtype_policy.{key} must equal {expected!r}, got {actual!r}"
                )
        return self


class _ExportMissingValuePolicyPayload(_ContractsPayloadModel):
    strategy: StrictStr
    all_nan_fill: float
    impute_missing: StrictBool = True

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        if value != MISSING_VALUE_STRATEGY_TRAIN_MEAN:
            raise ValueError(
                "preprocessor_state.missing_value_policy.strategy must equal "
                f"{MISSING_VALUE_STRATEGY_TRAIN_MEAN!r}"
            )
        return value

    @field_validator("all_nan_fill")
    @classmethod
    def _validate_all_nan_fill(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("preprocessor_state.missing_value_policy.all_nan_fill must be finite")
        return float(value)


class _ClassificationLabelPolicyPayload(_ContractsPayloadModel):
    mapping: StrictStr
    unseen_test_label: StrictStr

    @model_validator(mode="after")
    def _validate_expected_values(self) -> "_ClassificationLabelPolicyPayload":
        if self.mapping != CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP:
            raise ValueError(
                "preprocessor_state.classification_label_policy.mapping must equal "
                f"{CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP!r}"
            )
        if self.unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
            raise ValueError(
                "preprocessor_state.classification_label_policy.unseen_test_label must equal "
                f"{UNSEEN_TEST_LABEL_POLICY_FILTER!r}"
            )
        return self


class _ExportPreprocessorStatePayload(_ContractsPayloadModel):
    feature_order_policy: StrictStr
    missing_value_policy: _ExportMissingValuePolicyPayload
    classification_label_policy: _ClassificationLabelPolicyPayload
    dtype_policy: _DtypePolicyPayload

    @field_validator("feature_order_policy")
    @classmethod
    def _validate_feature_order_policy(cls, value: str) -> str:
        if value != FEATURE_ORDER_POLICY_POSITIONAL:
            raise ValueError(
                "preprocessor_state.feature_order_policy must equal "
                f"{FEATURE_ORDER_POLICY_POSITIONAL!r}"
            )
        return value


class _ManifestPayloadV3(_ContractsPayloadModel):
    schema_version: Literal["tab-foundry-export-v3"]
    producer: _ProducerInfoPayload
    task: Literal["classification"]
    model: _ManifestModelPayloadV3
    created_at_utc: StrictStr
    manifest_sha256: StrictStr
    inference: _InferenceConfigPayload
    preprocessor: _ExportPreprocessorStatePayload
    weights: _ExportWeightsPayload

    @field_validator("created_at_utc")
    @classmethod
    def _validate_created_at_utc(cls, value: str) -> str:
        return _validate_created_at_utc(value)

    @field_validator("manifest_sha256")
    @classmethod
    def _validate_manifest_sha256(cls, value: str) -> str:
        return _validate_hex_digest(value)


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
    stage: str | None
    stage_label: str | None
    module_overrides: dict[str, Any] | None
    d_col: int
    d_icl: int
    input_normalization: str
    feature_group_size: int
    many_class_train_mode: str
    max_mixed_radix_digits: int
    norm_type: str
    tfcol_n_heads: int
    tfcol_n_layers: int
    tfcol_n_inducing: int
    tfrow_n_heads: int
    tfrow_n_layers: int
    tfrow_cls_tokens: int
    tfrow_norm: str
    tficl_n_heads: int
    tficl_n_layers: int
    tficl_ff_expansion: int
    many_class_base: int
    head_hidden_dim: int
    use_digit_position_embed: bool
    staged_dropout: float
    pre_encoder_clip: float | None
    sandwich_latents: int
    sandwich_layers: int
    sandwich_heads: int
    sandwich_ff_expansion: int
    sandwich_activation: str
    sandwich_block_norm: str
    sandwich_summary_tokens_per_axis: int
    sandwich_self_attention_per_cross: int
    sandwich_pre_row_attention_layers: int
    sandwich_pre_column_attention_layers: int
    sandwich_pre_column_inducing_tokens: int
    feature_type_conditioning: str
    floating_likelihood: str
    integer_likelihood: str

    @classmethod
    def from_build_spec(
        cls,
        spec: Any,
        *,
        arch: str | None = None,
    ) -> "ExportModelSpec":
        return cls(
            arch=str(spec.arch if arch is None else arch),
            stage=None if spec.stage is None else str(spec.stage),
            stage_label=None if spec.stage_label is None else str(spec.stage_label),
            module_overrides=None if spec.module_overrides is None else dict(spec.module_overrides),
            d_col=int(spec.d_col),
            d_icl=int(spec.d_icl),
            input_normalization=str(spec.input_normalization),
            feature_group_size=int(spec.feature_group_size),
            many_class_train_mode=str(spec.many_class_train_mode),
            max_mixed_radix_digits=int(spec.max_mixed_radix_digits),
            norm_type=str(spec.norm_type),
            tfcol_n_heads=int(spec.tfcol_n_heads),
            tfcol_n_layers=int(spec.tfcol_n_layers),
            tfcol_n_inducing=int(spec.tfcol_n_inducing),
            tfrow_n_heads=int(spec.tfrow_n_heads),
            tfrow_n_layers=int(spec.tfrow_n_layers),
            tfrow_cls_tokens=int(spec.tfrow_cls_tokens),
            tfrow_norm=str(spec.tfrow_norm),
            tficl_n_heads=int(spec.tficl_n_heads),
            tficl_n_layers=int(spec.tficl_n_layers),
            tficl_ff_expansion=int(spec.tficl_ff_expansion),
            many_class_base=int(spec.many_class_base),
            head_hidden_dim=int(spec.head_hidden_dim),
            use_digit_position_embed=bool(spec.use_digit_position_embed),
            staged_dropout=float(spec.staged_dropout),
            pre_encoder_clip=None
            if spec.pre_encoder_clip is None
            else float(spec.pre_encoder_clip),
            sandwich_latents=int(spec.sandwich_latents),
            sandwich_layers=int(spec.sandwich_layers),
            sandwich_heads=int(spec.sandwich_heads),
            sandwich_ff_expansion=int(spec.sandwich_ff_expansion),
            sandwich_activation=str(spec.sandwich_activation),
            sandwich_block_norm=str(spec.sandwich_block_norm),
            sandwich_summary_tokens_per_axis=int(spec.sandwich_summary_tokens_per_axis),
            sandwich_self_attention_per_cross=int(spec.sandwich_self_attention_per_cross),
            sandwich_pre_row_attention_layers=int(spec.sandwich_pre_row_attention_layers),
            sandwich_pre_column_attention_layers=int(spec.sandwich_pre_column_attention_layers),
            sandwich_pre_column_inducing_tokens=int(spec.sandwich_pre_column_inducing_tokens),
            feature_type_conditioning=str(spec.feature_type_conditioning),
            floating_likelihood=str(spec.floating_likelihood),
            integer_likelihood=str(spec.integer_likelihood),
        )

    def to_build_spec(self, task: str) -> Any:
        from tab_foundry.model.spec import model_build_spec_from_mappings

        return model_build_spec_from_mappings(
            task=task,
            primary={
                "arch": self.arch,
                "stage": self.stage,
                "stage_label": self.stage_label,
                "module_overrides": self.module_overrides,
                "d_col": self.d_col,
                "d_icl": self.d_icl,
                "input_normalization": self.input_normalization,
                "feature_group_size": self.feature_group_size,
                "many_class_train_mode": self.many_class_train_mode,
                "max_mixed_radix_digits": self.max_mixed_radix_digits,
                "norm_type": self.norm_type,
                "tfcol_n_heads": self.tfcol_n_heads,
                "tfcol_n_layers": self.tfcol_n_layers,
                "tfcol_n_inducing": self.tfcol_n_inducing,
                "tfrow_n_heads": self.tfrow_n_heads,
                "tfrow_n_layers": self.tfrow_n_layers,
                "tfrow_cls_tokens": self.tfrow_cls_tokens,
                "tfrow_norm": self.tfrow_norm,
                "tficl_n_heads": self.tficl_n_heads,
                "tficl_n_layers": self.tficl_n_layers,
                "tficl_ff_expansion": self.tficl_ff_expansion,
                "many_class_base": self.many_class_base,
                "head_hidden_dim": self.head_hidden_dim,
                "use_digit_position_embed": self.use_digit_position_embed,
                "staged_dropout": self.staged_dropout,
                "pre_encoder_clip": self.pre_encoder_clip,
                "sandwich_latents": self.sandwich_latents,
                "sandwich_layers": self.sandwich_layers,
                "sandwich_heads": self.sandwich_heads,
                "sandwich_ff_expansion": self.sandwich_ff_expansion,
                "sandwich_activation": self.sandwich_activation,
                "sandwich_block_norm": self.sandwich_block_norm,
                "sandwich_summary_tokens_per_axis": self.sandwich_summary_tokens_per_axis,
                "sandwich_self_attention_per_cross": self.sandwich_self_attention_per_cross,
                "sandwich_pre_row_attention_layers": self.sandwich_pre_row_attention_layers,
                "sandwich_pre_column_attention_layers": self.sandwich_pre_column_attention_layers,
                "sandwich_pre_column_inducing_tokens": self.sandwich_pre_column_inducing_tokens,
                "feature_type_conditioning": self.feature_type_conditioning,
                "floating_likelihood": self.floating_likelihood,
                "integer_likelihood": self.integer_likelihood,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        if self.arch == STAGED_MODEL_ARCH:
            if self.stage is None:
                payload.pop("stage", None)
            if self.stage_label is None:
                payload.pop("stage_label", None)
            if self.module_overrides is None:
                payload.pop("module_overrides", None)
            if self.pre_encoder_clip is None:
                payload.pop("pre_encoder_clip", None)
            if self.staged_dropout is None:
                payload.pop("staged_dropout", None)
            for field_name in (
                "sandwich_latents",
                "sandwich_layers",
                "sandwich_heads",
                "sandwich_ff_expansion",
                "sandwich_activation",
                "sandwich_block_norm",
                "sandwich_summary_tokens_per_axis",
                "sandwich_self_attention_per_cross",
                "sandwich_pre_row_attention_layers",
                "sandwich_pre_column_attention_layers",
                "sandwich_pre_column_inducing_tokens",
                "feature_type_conditioning",
                "floating_likelihood",
                "integer_likelihood",
            ):
                payload.pop(field_name, None)
            return payload
        for field_name in ("stage", "stage_label", "module_overrides", "staged_dropout"):
            payload.pop(field_name, None)
        if self.arch != SANDWICH_MODEL_ARCH:
            payload.pop("pre_encoder_clip", None)
            for field_name in (
                "sandwich_latents",
                "sandwich_layers",
                "sandwich_heads",
                "sandwich_ff_expansion",
                "sandwich_activation",
                "sandwich_block_norm",
                "sandwich_summary_tokens_per_axis",
                "sandwich_self_attention_per_cross",
                "sandwich_pre_row_attention_layers",
                "sandwich_pre_column_attention_layers",
                "sandwich_pre_column_inducing_tokens",
                "feature_type_conditioning",
                "floating_likelihood",
                "integer_likelihood",
            ):
                payload.pop(field_name, None)
            return payload
        if self.stage is None:
            payload.pop("stage", None)
        if self.pre_encoder_clip is None:
            payload.pop("pre_encoder_clip", None)
        return payload


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
    model_stage: str | None
    group_shifts: list[int]
    feature_group_size: int
    many_class_threshold: int
    many_class_inference_mode: str
    quantile_levels: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        if self.model_stage is None:
            payload.pop("model_stage", None)
        if self.quantile_levels is None:
            payload.pop("quantile_levels", None)
        return payload


@dataclass(slots=True)
class ExportMissingValuePolicy:
    strategy: str
    all_nan_fill: float
    impute_missing: bool = True

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
    preprocessor: ExportPreprocessorState | None = None
    weights: ExportWeights | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "producer": self.producer.to_dict(),
            "task": self.task,
            "model": self.model.to_dict(),
            "created_at_utc": self.created_at_utc,
        }
        if self.inference is None or self.preprocessor is None or self.weights is None:
            raise RuntimeError("v3 manifest requires inference, preprocessor, and weights")
        payload["inference"] = self.inference.to_dict()
        payload["preprocessor"] = self.preprocessor.to_dict()
        payload["weights"] = self.weights.to_dict()
        if self.manifest_sha256 is not None:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


@dataclass(slots=True)
class ValidatedBundle:
    manifest: ExportManifest
    inference_config: InferenceConfig
    preprocessor_state: ExportPreprocessorState
