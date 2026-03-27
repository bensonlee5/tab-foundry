"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES


SUPPORTED_MODEL_TASKS = ("classification",)
SIMPLE_MODEL_ARCH: Final = "tabfoundry_simple"
STAGED_MODEL_ARCH: Final = "tabfoundry_staged"
SANDWICH_MODEL_ARCH: Final = "tabfoundry_sandwich"
SUPPORTED_MODEL_ARCHES = (SIMPLE_MODEL_ARCH, STAGED_MODEL_ARCH, SANDWICH_MODEL_ARCH)
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3
MAX_MODEL_STAGED_DROPOUT = 0.5
MIN_MODEL_MANY_CLASS_BASE = 2
_LINEAR_WEIGHT_TENSOR_RANK = 2
_LEGACY_SANDWICH_FIELDS = ("sandwich_row_latents", "sandwich_col_latents")


class ModelStage(StrEnum):
    """Public stage ladder for the staged research family."""

    NANO_EXACT = "nano_exact"
    LABEL_TOKEN = "label_token"
    SHARED_NORM = "shared_norm"
    PRENORM_BLOCK = "prenorm_block"
    SMALL_CLASS_HEAD = "small_class_head"
    TEST_SELF = "test_self"
    GROUPED_TOKENS = "grouped_tokens"
    ROW_CLS_POOL = "row_cls_pool"
    COLUMN_SET = "column_set"
    QASS_CONTEXT = "qass_context"
    MANY_CLASS = "many_class"


SUPPORTED_MODEL_STAGES = tuple(stage.value for stage in ModelStage)


def _coerce_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
    raise ValueError(f"{context} must be boolean-compatible, got {value!r}")


def _normalize_optional_label(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _normalize_jsonable_mapping(value: Any, *, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping or null, got {value!r}")
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        if isinstance(raw_value, Mapping):
            normalized[key] = _normalize_jsonable_mapping(raw_value, context=f"{context}.{key}")
            continue
        if isinstance(raw_value, list):
            normalized[key] = [
                _normalize_jsonable_mapping(item, context=f"{context}.{key}[{idx}]")
                if isinstance(item, Mapping)
                else item
                for idx, item in enumerate(raw_value)
            ]
            continue
        normalized[key] = raw_value
    return normalized


def _normalize_model_stage(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_MODEL_STAGES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_MODEL_STAGES} or null, got {value!r}"
        )
    return normalized


def resolve_model_stage(*, arch: str, stage: Any) -> str | None:
    """Normalize arch/stage pairs and enforce public compatibility rules."""

    normalized_stage = _normalize_model_stage(stage, context="model.stage")
    if arch == STAGED_MODEL_ARCH:
        return normalized_stage or ModelStage.NANO_EXACT.value
    if normalized_stage is not None:
        raise ValueError(
            "model.stage is only supported when model.arch='tabfoundry_staged'; "
            f"got arch={arch!r}, stage={normalized_stage!r}"
        )
    return None


def _reject_legacy_sandwich_fields(
    *,
    arch: str,
    primary_map: Mapping[str, Any],
    fallback_map: Mapping[str, Any],
) -> None:
    if arch != SANDWICH_MODEL_ARCH:
        return
    legacy_keys = sorted(
        {
            key
            for key in _LEGACY_SANDWICH_FIELDS
            if primary_map.get(key) is not None or fallback_map.get(key) is not None
        }
    )
    if not legacy_keys:
        return
    legacy_fields = ", ".join(f"model.{key}" for key in legacy_keys)
    raise ValueError(
        "tabfoundry_sandwich no longer supports "
        f"{legacy_fields}; use model.sandwich_latents instead."
    )


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _SimpleModelParams(_SpecModel):
    d_col: int = Field(default=128, gt=0)
    d_icl: int = Field(default=512, gt=0)
    tfcol_n_heads: int = Field(default=8, gt=0)
    tfcol_n_layers: int = Field(default=3, gt=0)
    tfcol_n_inducing: int = Field(default=128, gt=0)
    tfrow_n_heads: int = Field(default=8, gt=0)
    tfrow_n_layers: int = Field(default=3, gt=0)
    tfrow_cls_tokens: int = Field(default=4, gt=0)
    tfrow_norm: str = "layernorm"
    tficl_n_heads: int = Field(default=8, gt=0)
    tficl_n_layers: int = Field(default=12, gt=0)
    tficl_ff_expansion: int = Field(default=2, gt=0)
    head_hidden_dim: int = Field(default=1024, gt=0)
    use_digit_position_embed: bool = True

    @field_validator("tfrow_norm", mode="before")
    @classmethod
    def _validate_tfrow_norm(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_NORM_TYPES:
            raise ValueError(f"tfrow_norm must be one of {SUPPORTED_NORM_TYPES}, got {value!r}")
        return normalized

    @field_validator("use_digit_position_embed", mode="before")
    @classmethod
    def _coerce_use_digit_position_embed(cls, value: Any) -> bool:
        return _coerce_bool(value, context="use_digit_position_embed")


class _StagedModelParams(_SimpleModelParams):
    stage: str | None = ModelStage.NANO_EXACT.value
    stage_label: str | None = None
    module_overrides: dict[str, Any] | None = None
    staged_dropout: float = Field(default=0.0, ge=0.0, le=MAX_MODEL_STAGED_DROPOUT)
    pre_encoder_clip: float | None = Field(default=None, gt=0.0)

    @field_validator("stage", mode="before")
    @classmethod
    def _resolve_stage(cls, value: Any) -> str | None:
        return resolve_model_stage(arch=STAGED_MODEL_ARCH, stage=value)

    @field_validator("stage_label", mode="before")
    @classmethod
    def _normalize_stage_label(cls, value: Any) -> str | None:
        return _normalize_optional_label(value, context="model.stage_label")

    @field_validator("module_overrides", mode="before")
    @classmethod
    def _normalize_module_overrides(cls, value: Any) -> dict[str, Any] | None:
        return _normalize_jsonable_mapping(value, context="model.module_overrides")

    @field_validator("pre_encoder_clip", mode="before")
    @classmethod
    def _validate_pre_encoder_clip(cls, value: Any) -> float | None:
        if value is None:
            return None
        clip = float(value)
        if clip <= 0.0:
            raise ValueError("pre_encoder_clip must be > 0")
        return clip


class _SandwichModelParams(_SpecModel):
    d_col: int = Field(default=128, gt=0)
    d_icl: int = Field(default=60, gt=0)
    head_hidden_dim: int = Field(default=96, gt=0)
    pre_encoder_clip: float | None = Field(default=None, gt=0.0)
    sandwich_latents: int = Field(default=24, gt=0)
    sandwich_layers: int = Field(default=2, gt=0)
    sandwich_heads: int = Field(default=4, gt=0)
    sandwich_ff_expansion: int = Field(default=2, gt=0)
    sandwich_summary_tokens_per_axis: int = Field(default=4, gt=0)
    sandwich_self_attention_per_cross: int = Field(default=4, ge=0)
    sandwich_pre_row_attention_layers: int = Field(default=1, ge=0)
    sandwich_pre_column_attention_layers: int = Field(default=1, ge=0)
    sandwich_pre_column_inducing_tokens: int = Field(default=16, gt=0)

    @field_validator("pre_encoder_clip", mode="before")
    @classmethod
    def _validate_pre_encoder_clip(cls, value: Any) -> float | None:
        if value is None:
            return None
        clip = float(value)
        if clip <= 0.0:
            raise ValueError("pre_encoder_clip must be > 0")
        return clip


class _BaseModelBuildSpecPayload(_SpecModel):
    task: str
    arch: str
    input_normalization: str = "none"
    feature_group_size: int = Field(default=1, gt=0)
    many_class_train_mode: str = "path_nll"
    max_mixed_radix_digits: int = Field(default=64, gt=0)
    norm_type: str = "layernorm"
    many_class_base: int = Field(default=10, ge=MIN_MODEL_MANY_CLASS_BASE)

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_MODEL_TASKS:
            raise ValueError(f"Unsupported task: {normalized!r}")
        return normalized

    @field_validator("input_normalization", mode="before")
    @classmethod
    def _validate_input_normalization(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_INPUT_NORMALIZATION_MODES:
            raise ValueError(
                "input_normalization must be "
                f"{SUPPORTED_INPUT_NORMALIZATION_MODES}, got {normalized!r}"
            )
        return normalized

    @field_validator("norm_type", mode="before")
    @classmethod
    def _validate_norm_type(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_NORM_TYPES:
            raise ValueError(f"norm_type must be one of {SUPPORTED_NORM_TYPES}, got {value!r}")
        return normalized

    @field_validator("many_class_train_mode", mode="before")
    @classmethod
    def _validate_many_class_train_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
            raise ValueError(
                "many_class_train_mode must be "
                f"{SUPPORTED_MANY_CLASS_TRAIN_MODES}, got {normalized!r}"
            )
        return normalized

    def _common_flat_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"params"})
        return {str(key): value for key, value in payload.items()}


class _SimpleModelBuildSpecPayload(_BaseModelBuildSpecPayload):
    arch: Literal["tabfoundry_simple"] = SIMPLE_MODEL_ARCH
    params: _SimpleModelParams = Field(default_factory=_SimpleModelParams)


class _StagedModelBuildSpecPayload(_BaseModelBuildSpecPayload):
    arch: Literal["tabfoundry_staged"] = STAGED_MODEL_ARCH
    params: _StagedModelParams = Field(default_factory=_StagedModelParams)


class _SandwichModelBuildSpecPayload(_BaseModelBuildSpecPayload):
    arch: Literal["tabfoundry_sandwich"] = SANDWICH_MODEL_ARCH
    params: _SandwichModelParams = Field(default_factory=_SandwichModelParams)


_ModelBuildSpecPayload = Annotated[
    _SimpleModelBuildSpecPayload | _StagedModelBuildSpecPayload | _SandwichModelBuildSpecPayload,
    Field(discriminator="arch"),
]
_MODEL_BUILD_SPEC_PAYLOAD_ADAPTER: TypeAdapter[_ModelBuildSpecPayload] = TypeAdapter(
    _ModelBuildSpecPayload
)


DEFAULT_MODEL_ARCH = cast(str, _SandwichModelBuildSpecPayload.model_fields["arch"].default)
DEFAULT_MODEL_STAGE: str | None = None
DEFAULT_MODEL_STAGE_LABEL: str | None = None
DEFAULT_MODEL_MODULE_OVERRIDES: dict[str, Any] | None = None
DEFAULT_MODEL_D_COL = cast(int, _SimpleModelParams.model_fields["d_col"].default)
DEFAULT_MODEL_D_ICL = cast(int, _SimpleModelParams.model_fields["d_icl"].default)
DEFAULT_MODEL_INPUT_NORMALIZATION = cast(
    str,
    _BaseModelBuildSpecPayload.model_fields["input_normalization"].default,
)
DEFAULT_MODEL_FEATURE_GROUP_SIZE = cast(
    int,
    _BaseModelBuildSpecPayload.model_fields["feature_group_size"].default,
)
DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE = cast(
    str,
    _BaseModelBuildSpecPayload.model_fields["many_class_train_mode"].default,
)
DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS = cast(
    int,
    _BaseModelBuildSpecPayload.model_fields["max_mixed_radix_digits"].default,
)
DEFAULT_MODEL_NORM_TYPE = cast(str, _BaseModelBuildSpecPayload.model_fields["norm_type"].default)
DEFAULT_MODEL_TFCOL_N_HEADS = cast(int, _SimpleModelParams.model_fields["tfcol_n_heads"].default)
DEFAULT_MODEL_TFCOL_N_LAYERS = cast(int, _SimpleModelParams.model_fields["tfcol_n_layers"].default)
DEFAULT_MODEL_TFCOL_N_INDUCING = cast(
    int,
    _SimpleModelParams.model_fields["tfcol_n_inducing"].default,
)
DEFAULT_MODEL_TFROW_N_HEADS = cast(int, _SimpleModelParams.model_fields["tfrow_n_heads"].default)
DEFAULT_MODEL_TFROW_N_LAYERS = cast(int, _SimpleModelParams.model_fields["tfrow_n_layers"].default)
DEFAULT_MODEL_TFROW_CLS_TOKENS = cast(
    int,
    _SimpleModelParams.model_fields["tfrow_cls_tokens"].default,
)
DEFAULT_MODEL_TFROW_NORM = cast(str, _SimpleModelParams.model_fields["tfrow_norm"].default)
DEFAULT_MODEL_TFICL_N_HEADS = cast(int, _SimpleModelParams.model_fields["tficl_n_heads"].default)
DEFAULT_MODEL_TFICL_N_LAYERS = cast(int, _SimpleModelParams.model_fields["tficl_n_layers"].default)
DEFAULT_MODEL_TFICL_FF_EXPANSION = cast(
    int,
    _SimpleModelParams.model_fields["tficl_ff_expansion"].default,
)
DEFAULT_MODEL_MANY_CLASS_BASE = cast(
    int,
    _BaseModelBuildSpecPayload.model_fields["many_class_base"].default,
)
DEFAULT_MODEL_HEAD_HIDDEN_DIM = cast(
    int,
    _SimpleModelParams.model_fields["head_hidden_dim"].default,
)
DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED = cast(
    bool,
    _SimpleModelParams.model_fields["use_digit_position_embed"].default,
)
DEFAULT_MODEL_STAGED_DROPOUT = cast(
    float,
    _StagedModelParams.model_fields["staged_dropout"].default,
)
DEFAULT_MODEL_PRE_ENCODER_CLIP: float | None = cast(
    float | None,
    _SandwichModelParams.model_fields["pre_encoder_clip"].default,
)
DEFAULT_SANDWICH_MODEL_D_ICL = cast(
    int,
    _SandwichModelParams.model_fields["d_icl"].default,
)
DEFAULT_SANDWICH_MODEL_HEAD_HIDDEN_DIM = cast(
    int,
    _SandwichModelParams.model_fields["head_hidden_dim"].default,
)
DEFAULT_MODEL_SANDWICH_LATENTS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_latents"].default,
)
DEFAULT_MODEL_SANDWICH_LAYERS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_layers"].default,
)
DEFAULT_MODEL_SANDWICH_HEADS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_heads"].default,
)
DEFAULT_MODEL_SANDWICH_FF_EXPANSION = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_ff_expansion"].default,
)
DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_summary_tokens_per_axis"].default,
)
DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_self_attention_per_cross"].default,
)
DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_pre_row_attention_layers"].default,
)
DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_pre_column_attention_layers"].default,
)
DEFAULT_MODEL_SANDWICH_PRE_COLUMN_INDUCING_TOKENS = cast(
    int,
    _SandwichModelParams.model_fields["sandwich_pre_column_inducing_tokens"].default,
)


_FLAT_COMPAT_DEFAULTS: dict[str, Any] = {
    "stage": DEFAULT_MODEL_STAGE,
    "stage_label": DEFAULT_MODEL_STAGE_LABEL,
    "module_overrides": DEFAULT_MODEL_MODULE_OVERRIDES,
    "d_col": DEFAULT_MODEL_D_COL,
    "d_icl": DEFAULT_MODEL_D_ICL,
    "input_normalization": DEFAULT_MODEL_INPUT_NORMALIZATION,
    "feature_group_size": DEFAULT_MODEL_FEATURE_GROUP_SIZE,
    "many_class_train_mode": DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE,
    "max_mixed_radix_digits": DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS,
    "norm_type": DEFAULT_MODEL_NORM_TYPE,
    "tfcol_n_heads": DEFAULT_MODEL_TFCOL_N_HEADS,
    "tfcol_n_layers": DEFAULT_MODEL_TFCOL_N_LAYERS,
    "tfcol_n_inducing": DEFAULT_MODEL_TFCOL_N_INDUCING,
    "tfrow_n_heads": DEFAULT_MODEL_TFROW_N_HEADS,
    "tfrow_n_layers": DEFAULT_MODEL_TFROW_N_LAYERS,
    "tfrow_cls_tokens": DEFAULT_MODEL_TFROW_CLS_TOKENS,
    "tfrow_norm": DEFAULT_MODEL_TFROW_NORM,
    "tficl_n_heads": DEFAULT_MODEL_TFICL_N_HEADS,
    "tficl_n_layers": DEFAULT_MODEL_TFICL_N_LAYERS,
    "tficl_ff_expansion": DEFAULT_MODEL_TFICL_FF_EXPANSION,
    "many_class_base": DEFAULT_MODEL_MANY_CLASS_BASE,
    "head_hidden_dim": DEFAULT_MODEL_HEAD_HIDDEN_DIM,
    "use_digit_position_embed": DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED,
    "staged_dropout": DEFAULT_MODEL_STAGED_DROPOUT,
    "pre_encoder_clip": DEFAULT_MODEL_PRE_ENCODER_CLIP,
    "sandwich_latents": DEFAULT_MODEL_SANDWICH_LATENTS,
    "sandwich_layers": DEFAULT_MODEL_SANDWICH_LAYERS,
    "sandwich_heads": DEFAULT_MODEL_SANDWICH_HEADS,
    "sandwich_ff_expansion": DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    "sandwich_summary_tokens_per_axis": DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS,
    "sandwich_self_attention_per_cross": DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS,
    "sandwich_pre_row_attention_layers": DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS,
    "sandwich_pre_column_attention_layers": DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS,
    "sandwich_pre_column_inducing_tokens": DEFAULT_MODEL_SANDWICH_PRE_COLUMN_INDUCING_TOKENS,
}


_COMMON_MODEL_KEYS = (
    "task",
    "arch",
    "input_normalization",
    "feature_group_size",
    "many_class_train_mode",
    "max_mixed_radix_digits",
    "norm_type",
    "many_class_base",
)
_SIMPLE_PARAM_KEYS = (
    "d_col",
    "d_icl",
    "tfcol_n_heads",
    "tfcol_n_layers",
    "tfcol_n_inducing",
    "tfrow_n_heads",
    "tfrow_n_layers",
    "tfrow_cls_tokens",
    "tfrow_norm",
    "tficl_n_heads",
    "tficl_n_layers",
    "tficl_ff_expansion",
    "head_hidden_dim",
    "use_digit_position_embed",
)
_STAGED_PARAM_KEYS = _SIMPLE_PARAM_KEYS + (
    "stage",
    "stage_label",
    "module_overrides",
    "staged_dropout",
    "pre_encoder_clip",
)
_SANDWICH_PARAM_KEYS = (
    "d_col",
    "d_icl",
    "head_hidden_dim",
    "pre_encoder_clip",
    "sandwich_latents",
    "sandwich_layers",
    "sandwich_heads",
    "sandwich_ff_expansion",
    "sandwich_summary_tokens_per_axis",
    "sandwich_self_attention_per_cross",
    "sandwich_pre_row_attention_layers",
    "sandwich_pre_column_attention_layers",
    "sandwich_pre_column_inducing_tokens",
)
_STAGED_COMPAT_KEYS = ("stage", "stage_label", "module_overrides")


def _resolved_arch(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MODEL_ARCHES:
        raise ValueError(f"Unsupported model arch: {normalized!r}")
    return normalized


def _with_staged_arch_compat(mapping: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in mapping.items()}
    if normalized.get("arch") is None and any(normalized.get(key) is not None for key in _STAGED_COMPAT_KEYS):
        normalized["arch"] = STAGED_MODEL_ARCH
    return normalized


def _payload_mapping_from_flat_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _with_staged_arch_compat(mapping)
    task = str(mapping.get("task", "classification")).strip().lower()
    arch = _resolved_arch(mapping.get("arch", DEFAULT_MODEL_ARCH))
    if arch != STAGED_MODEL_ARCH and mapping.get("stage") is not None:
        _ = resolve_model_stage(arch=arch, stage=mapping.get("stage"))
    if arch != STAGED_MODEL_ARCH and mapping.get("stage_label") is not None:
        raise ValueError(
            "model.stage_label is only supported when model.arch='tabfoundry_staged'; "
            f"got arch={arch!r}"
        )
    if arch != STAGED_MODEL_ARCH and mapping.get("module_overrides") is not None:
        raise ValueError(
            "model.module_overrides is only supported when model.arch='tabfoundry_staged'; "
            f"got arch={arch!r}"
        )
    payload: dict[str, Any] = {"task": task, "arch": arch}
    for key in _COMMON_MODEL_KEYS[2:]:
        if mapping.get(key) is not None:
            payload[key] = mapping[key]
    if arch == SIMPLE_MODEL_ARCH:
        params = {key: mapping[key] for key in _SIMPLE_PARAM_KEYS if mapping.get(key) is not None}
    elif arch == STAGED_MODEL_ARCH:
        params = {key: mapping[key] for key in _STAGED_PARAM_KEYS if mapping.get(key) is not None}
    else:
        params = {key: mapping[key] for key in _SANDWICH_PARAM_KEYS if mapping.get(key) is not None}
    if params:
        payload["params"] = params
    return payload


def _validate_payload(payload: Any) -> _ModelBuildSpecPayload:
    if isinstance(
        payload,
        (_SimpleModelBuildSpecPayload, _StagedModelBuildSpecPayload, _SandwichModelBuildSpecPayload),
    ):
        return payload
    candidate = payload
    if isinstance(candidate, ModelBuildSpec):
        return candidate.payload
    if isinstance(candidate, Mapping):
        candidate = _payload_mapping_from_flat_mapping(candidate)
    try:
        return _MODEL_BUILD_SPEC_PAYLOAD_ADAPTER.validate_python(candidate)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _flat_dict_from_payload(payload: _ModelBuildSpecPayload) -> dict[str, Any]:
    flat = dict(_FLAT_COMPAT_DEFAULTS)
    flat.update(payload._common_flat_dict())
    flat.update(payload.params.model_dump(exclude_none=False))
    flat["arch"] = payload.arch
    flat["task"] = payload.task
    if payload.arch != STAGED_MODEL_ARCH:
        flat["stage"] = None
        flat["stage_label"] = None
        flat["module_overrides"] = None
        if payload.arch != SANDWICH_MODEL_ARCH:
            flat["pre_encoder_clip"] = DEFAULT_MODEL_PRE_ENCODER_CLIP
    return flat


class ModelBuildSpec:
    """Canonical model-construction settings shared across train/eval/export/load."""

    __slots__ = ("payload", "_flat")

    payload: _ModelBuildSpecPayload
    _flat: dict[str, Any]

    def __init__(self, **data: Any) -> None:
        raw_payload: Any
        if len(data) == 1 and "payload" in data:
            raw_payload = data["payload"]
        else:
            raw_payload = data
        payload = _validate_payload(raw_payload)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "_flat", _flat_dict_from_payload(payload))

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - defensive immutability
        raise AttributeError(f"{self.__class__.__name__} is immutable")

    def __getattr__(self, name: str) -> Any:
        flat = object.__getattribute__(self, "_flat")
        if name in flat:
            return flat[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"ModelBuildSpec({self._flat!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModelBuildSpec) and self._flat == other._flat

    def to_dict(self) -> dict[str, Any]:
        return dict(self._flat)


def model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a canonical model spec from a primary mapping with optional fallback."""

    primary_map = _with_staged_arch_compat(primary) if primary is not None else {}
    fallback_map = _with_staged_arch_compat(fallback) if fallback is not None else {}

    def _pick(name: str, default: Any) -> Any:
        if name in primary_map and primary_map[name] is not None:
            return primary_map[name]
        if name in fallback_map and fallback_map[name] is not None:
            return fallback_map[name]
        return default

    arch_value = _resolved_arch(_pick("arch", DEFAULT_MODEL_ARCH))
    _reject_legacy_sandwich_fields(
        arch=arch_value,
        primary_map=primary_map,
        fallback_map=fallback_map,
    )

    def _arch_default(name: str, default: Any, *, sandwich_default: Any | None = None) -> Any:
        if name in primary_map and primary_map[name] is not None:
            return primary_map[name]
        if name in fallback_map and fallback_map[name] is not None:
            return fallback_map[name]
        if arch_value == SANDWICH_MODEL_ARCH and sandwich_default is not None:
            return sandwich_default
        return default

    return ModelBuildSpec(
        task=str(task).strip().lower(),
        arch=arch_value,
        stage=_pick("stage", DEFAULT_MODEL_STAGE),
        stage_label=_pick("stage_label", DEFAULT_MODEL_STAGE_LABEL),
        module_overrides=_pick("module_overrides", DEFAULT_MODEL_MODULE_OVERRIDES),
        d_col=int(_pick("d_col", DEFAULT_MODEL_D_COL)),
        d_icl=int(
            _arch_default(
                "d_icl",
                DEFAULT_MODEL_D_ICL,
                sandwich_default=DEFAULT_SANDWICH_MODEL_D_ICL,
            )
        ),
        input_normalization=str(_pick("input_normalization", DEFAULT_MODEL_INPUT_NORMALIZATION)),
        feature_group_size=int(_pick("feature_group_size", DEFAULT_MODEL_FEATURE_GROUP_SIZE)),
        many_class_train_mode=str(
            _pick("many_class_train_mode", DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE)
        ),
        max_mixed_radix_digits=int(
            _pick("max_mixed_radix_digits", DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS)
        ),
        norm_type=str(_pick("norm_type", DEFAULT_MODEL_NORM_TYPE)),
        tfcol_n_heads=int(_pick("tfcol_n_heads", DEFAULT_MODEL_TFCOL_N_HEADS)),
        tfcol_n_layers=int(_pick("tfcol_n_layers", DEFAULT_MODEL_TFCOL_N_LAYERS)),
        tfcol_n_inducing=int(_pick("tfcol_n_inducing", DEFAULT_MODEL_TFCOL_N_INDUCING)),
        tfrow_n_heads=int(_pick("tfrow_n_heads", DEFAULT_MODEL_TFROW_N_HEADS)),
        tfrow_n_layers=int(_pick("tfrow_n_layers", DEFAULT_MODEL_TFROW_N_LAYERS)),
        tfrow_cls_tokens=int(_pick("tfrow_cls_tokens", DEFAULT_MODEL_TFROW_CLS_TOKENS)),
        tfrow_norm=str(_pick("tfrow_norm", DEFAULT_MODEL_TFROW_NORM)),
        tficl_n_heads=int(_pick("tficl_n_heads", DEFAULT_MODEL_TFICL_N_HEADS)),
        tficl_n_layers=int(_pick("tficl_n_layers", DEFAULT_MODEL_TFICL_N_LAYERS)),
        tficl_ff_expansion=int(
            _pick("tficl_ff_expansion", DEFAULT_MODEL_TFICL_FF_EXPANSION)
        ),
        many_class_base=int(_pick("many_class_base", DEFAULT_MODEL_MANY_CLASS_BASE)),
        head_hidden_dim=int(
            _arch_default(
                "head_hidden_dim",
                DEFAULT_MODEL_HEAD_HIDDEN_DIM,
                sandwich_default=DEFAULT_SANDWICH_MODEL_HEAD_HIDDEN_DIM,
            )
        ),
        use_digit_position_embed=_coerce_bool(
            _pick("use_digit_position_embed", DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED),
            context="use_digit_position_embed",
        ),
        staged_dropout=float(_pick("staged_dropout", DEFAULT_MODEL_STAGED_DROPOUT)),
        pre_encoder_clip=_pick("pre_encoder_clip", DEFAULT_MODEL_PRE_ENCODER_CLIP),
        sandwich_latents=int(_pick("sandwich_latents", DEFAULT_MODEL_SANDWICH_LATENTS)),
        sandwich_layers=int(_pick("sandwich_layers", DEFAULT_MODEL_SANDWICH_LAYERS)),
        sandwich_heads=int(_pick("sandwich_heads", DEFAULT_MODEL_SANDWICH_HEADS)),
        sandwich_ff_expansion=int(
            _pick("sandwich_ff_expansion", DEFAULT_MODEL_SANDWICH_FF_EXPANSION)
        ),
        sandwich_summary_tokens_per_axis=int(
            _pick(
                "sandwich_summary_tokens_per_axis",
                DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS,
            )
        ),
        sandwich_self_attention_per_cross=int(
            _pick(
                "sandwich_self_attention_per_cross",
                DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS,
            )
        ),
        sandwich_pre_row_attention_layers=int(
            _pick(
                "sandwich_pre_row_attention_layers",
                DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS,
            )
        ),
        sandwich_pre_column_attention_layers=int(
            _pick(
                "sandwich_pre_column_attention_layers",
                DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS,
            )
        ),
        sandwich_pre_column_inducing_tokens=int(
            _pick(
                "sandwich_pre_column_inducing_tokens",
                DEFAULT_MODEL_SANDWICH_PRE_COLUMN_INDUCING_TOKENS,
            )
        ),
    )


def _feature_group_size_from_state_dict(
    state_dict: Mapping[str, Any] | None,
) -> int | None:
    if state_dict is None:
        return None
    raw_weight = state_dict.get(_GROUP_LINEAR_WEIGHT_KEY)
    shape = getattr(raw_weight, "shape", None)
    if shape is None or len(shape) != _LINEAR_WEIGHT_TENSOR_RANK:
        return None
    try:
        in_features = int(shape[1])
    except (IndexError, TypeError, ValueError):
        return None
    if in_features <= 0 or in_features % _GROUP_SHIFT_COUNT != 0:
        return None
    return in_features // _GROUP_SHIFT_COUNT


def _validate_checkpoint_feature_group_size(
    *,
    spec: ModelBuildSpec,
    state_dict: Mapping[str, Any] | None,
    feature_group_size_is_configured: bool,
) -> None:
    checkpoint_feature_group_size = _feature_group_size_from_state_dict(state_dict)
    if checkpoint_feature_group_size is None:
        return
    if checkpoint_feature_group_size == spec.feature_group_size:
        return

    if feature_group_size_is_configured:
        raise ValueError(
            "Resolved feature_group_size="
            f"{spec.feature_group_size} is incompatible with checkpoint weights "
            f"implying feature_group_size={checkpoint_feature_group_size}; "
            "load the checkpoint with an explicit feature_group_size override that matches "
            "the weights or regenerate the checkpoint with an explicit feature_group_size in "
            "its saved config."
        )

    raise ValueError(
        "Checkpoint config omitted feature_group_size, which now defaults to 1, but "
        f"checkpoint weights imply feature_group_size={checkpoint_feature_group_size}; "
        "regenerate the checkpoint with an explicit feature_group_size or load it with an "
        "explicit feature_group_size override."
    )


def checkpoint_model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a checkpoint-backed model spec and validate weight compatibility."""

    primary_map = _with_staged_arch_compat(primary) if primary is not None else {}
    fallback_map = _with_staged_arch_compat(fallback) if fallback is not None else {}
    for source_name, mapping in (("primary", primary_map), ("fallback", fallback_map)):
        raw_arch = mapping.get("arch")
        if raw_arch is None:
            continue
        normalized_arch = str(raw_arch).strip().lower()
        if normalized_arch == "tabfoundry":
            raise ValueError(
                "Legacy model.arch='tabfoundry' is no longer supported; "
                "rebuild or export this checkpoint with model.arch='tabfoundry_staged' "
                "or 'tabfoundry_simple'."
            )
        if normalized_arch not in SUPPORTED_MODEL_ARCHES:
            raise ValueError(f"Unsupported model arch in {source_name} mapping: {raw_arch!r}")
    if _GROUP_LINEAR_WEIGHT_KEY in (state_dict or {}):
        raise ValueError(
            "Legacy tabfoundry checkpoints are no longer supported; "
            "this checkpoint contains grouped-token weights under "
            f"{_GROUP_LINEAR_WEIGHT_KEY!r}. Rebuild it on tabfoundry_staged or "
            "tabfoundry_simple before loading."
        )
    feature_group_size_is_configured = (
        primary_map.get("feature_group_size") is not None
        or fallback_map.get("feature_group_size") is not None
    )
    spec = model_build_spec_from_mappings(
        task=task,
        primary=primary_map,
        fallback=fallback,
    )
    _validate_checkpoint_feature_group_size(
        spec=spec,
        state_dict=state_dict,
        feature_group_size_is_configured=feature_group_size_is_configured,
    )
    return spec
