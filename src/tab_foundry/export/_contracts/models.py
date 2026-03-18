"""Public data structures and strict payload models for export contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictBool, StrictInt, StrictStr, field_validator


SCHEMA_VERSION_V2 = "tab-foundry-export-v2"
SCHEMA_VERSION_V3 = "tab-foundry-export-v3"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION_V2, SCHEMA_VERSION_V3)
SUPPORTED_TASKS = ("classification", "regression")
SUPPORTED_MANY_CLASS_INFERENCE_MODES = ("full_probs",)
EXPECTED_GROUP_SHIFTS = [0, 1, 3]
EXPECTED_MANY_CLASS_THRESHOLD = 10
EXPECTED_V2_FEATURE_ORDER_POLICY = "lexicographic_f_columns"
EXPECTED_MISSING_VALUE_ALL_NAN_FILL = 0.0


class _ContractsPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _ManifestModelPayloadV2(_ContractsPayloadModel):
    arch: StrictStr
    stage: StrictStr | None = None
    d_col: StrictInt
    d_icl: StrictInt
    input_normalization: StrictStr | None = None
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


class _ManifestModelPayloadV3(_ManifestModelPayloadV2):
    input_normalization: StrictStr


class _InferenceConfigPayload(_ContractsPayloadModel):
    task: StrictStr
    model_arch: StrictStr
    model_stage: StrictStr | None = None
    group_shifts: list[StrictInt]
    feature_group_size: StrictInt
    many_class_threshold: StrictInt
    many_class_inference_mode: StrictStr
    quantile_levels: list[FiniteFloat] | None = None


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
        )

    def to_build_spec(self, task: str) -> Any:
        from tab_foundry.model.spec import model_build_spec_from_mappings

        return model_build_spec_from_mappings(
            task=task,
            primary={
                "arch": self.arch,
                "stage": self.stage,
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
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        if self.stage is None:
            payload.pop("stage", None)
        return payload


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
