"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import math
from typing import Annotated, Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES


SUPPORTED_MODEL_TASKS = ("classification",)
SIMPLE_MODEL_ARCH: Final = "tabfoundry_simple"
STAGED_MODEL_ARCH: Final = "tabfoundry_staged"
SANDWICH_MODEL_ARCH: Final = "tabfoundry_sandwich"
ROUTED_SANDWICH_MODEL_ARCH: Final = "routed_sandwich"
GRID_SANDWICH_MODEL_ARCH: Final = "grid_sandwich"
SUPPORTED_MODEL_ARCHES = (
    SIMPLE_MODEL_ARCH,
    STAGED_MODEL_ARCH,
    SANDWICH_MODEL_ARCH,
    ROUTED_SANDWICH_MODEL_ARCH,
    GRID_SANDWICH_MODEL_ARCH,
)
SANDWICH_FAMILY_MODEL_ARCHES = (
    SANDWICH_MODEL_ARCH,
    ROUTED_SANDWICH_MODEL_ARCH,
    GRID_SANDWICH_MODEL_ARCH,
)
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
SUPPORTED_FEATURE_TYPE_CONDITIONING = ("film", "additive_embedding")
SUPPORTED_FLOATING_LIKELIHOODS = ("single_gaussian",)
SUPPORTED_INTEGER_LIKELIHOODS = ("hybrid_mixture", "discrete")
SUPPORTED_SANDWICH_ACTIVATIONS = ("gelu", "rational")
SUPPORTED_SANDWICH_BLOCK_NORMS = ("layernorm", "none")
SUPPORTED_ROUTED_RESIDUAL_MODES = ("dynamic_hyper",)
SUPPORTED_ROUTED_RESIDUAL_SCALES = ("deepnorm",)
SUPPORTED_GRID_RESIDUAL_MODES = ("prenorm", "hyper_connection_lite")
SUPPORTED_GRID_ATTENTION_MODES = ("standard", "differential")
SUPPORTED_GRID_FFN_MODES = ("gelu", "swiglu", "geglu")
SUPPORTED_GRID_MOE_SCOPES = ("none", "grid_core_ffn")
DEFAULT_MODEL_ARCH: Final = SANDWICH_MODEL_ARCH
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3
MAX_MODEL_STAGED_DROPOUT = 0.5
MIN_MODEL_MANY_CLASS_BASE = 2
_LINEAR_WEIGHT_TENSOR_RANK = 2
_LEGACY_SANDWICH_FIELDS = ("sandwich_row_latents", "sandwich_col_latents")
_LEGACY_SANDWICH_FEATURE_TYPE_EMBEDDING_KEY = "feature_type_embedding.weight"
_ROUTED_SANDWICH_DEAD_FIELDS = ("sandwich_summary_tokens_per_axis",)
_GRID_SANDWICH_DEAD_FIELDS = (
    "sandwich_latents",
    "sandwich_self_attention_per_cross",
    "sandwich_summary_tokens_per_axis",
)


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


def resolve_model_stage(*, arch: str, stage: Any) -> str | None:
    """Normalize arch/stage pairs and enforce public compatibility rules."""

    if stage is None:
        normalized_stage = None
    else:
        normalized_stage = str(stage).strip().lower()
        if not normalized_stage:
            normalized_stage = None
        elif normalized_stage not in SUPPORTED_MODEL_STAGES:
            raise ValueError(
                f"model.stage must be one of {SUPPORTED_MODEL_STAGES} or null, got {stage!r}"
            )
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


def _arch_specific_sandwich_dead_fields(*, arch: str) -> tuple[str, ...]:
    if arch == ROUTED_SANDWICH_MODEL_ARCH:
        return _ROUTED_SANDWICH_DEAD_FIELDS
    if arch == GRID_SANDWICH_MODEL_ARCH:
        return _GRID_SANDWICH_DEAD_FIELDS
    return ()


def _invalid_arch_specific_sandwich_fields(
    *,
    arch: str,
    mapping: Mapping[str, Any],
) -> list[str]:
    invalid_fields: list[str] = []
    for field_name in _arch_specific_sandwich_dead_fields(arch=arch):
        field_value = mapping.get(field_name)
        field_default = SANDWICH_DEFAULTS[field_name]
        if field_value is not None and int(field_value) != int(field_default):
            invalid_fields.append(f"model.{field_name}")
    return invalid_fields


def _raise_arch_specific_sandwich_field_error(
    *,
    arch: str,
    invalid_fields: list[str],
) -> None:
    if not invalid_fields:
        return
    if arch == ROUTED_SANDWICH_MODEL_ARCH:
        raise ValueError(
            "routed_sandwich does not support model.sandwich_summary_tokens_per_axis; "
            "use model.routed_row_summary_tokens and model.routed_column_summary_tokens."
        )
    invalid_fields_text = ", ".join(invalid_fields)
    raise ValueError(
        "grid_sandwich does not support "
        f"{invalid_fields_text}; it keeps an explicit grid core without latent or summary-stream knobs."
    )


def _mapping_explicitly_targets_arch(*, mapping: Mapping[str, Any], arch: str) -> bool:
    raw_arch = mapping.get("arch")
    if raw_arch is None:
        return False
    return str(raw_arch).strip().lower() == arch


def _reject_layered_arch_specific_sandwich_fields(
    *,
    arch: str,
    primary_map: Mapping[str, Any],
    fallback_map: Mapping[str, Any],
) -> None:
    if arch not in {ROUTED_SANDWICH_MODEL_ARCH, GRID_SANDWICH_MODEL_ARCH}:
        return
    _raise_arch_specific_sandwich_field_error(
        arch=arch,
        invalid_fields=_invalid_arch_specific_sandwich_fields(
            arch=arch,
            mapping=primary_map,
        ),
    )
    if not _mapping_explicitly_targets_arch(mapping=fallback_map, arch=arch):
        return
    _raise_arch_specific_sandwich_field_error(
        arch=arch,
        invalid_fields=_invalid_arch_specific_sandwich_fields(
            arch=arch,
            mapping=fallback_map,
        ),
    )


def _sanitize_arch_specific_sandwich_fields(
    *,
    arch: str,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    sanitized = {str(key): value for key, value in mapping.items()}
    for field_name in _arch_specific_sandwich_dead_fields(arch=arch):
        sanitized.pop(field_name, None)
    return sanitized


# ---------------------------------------------------------------------------
# Pydantic param models
# ---------------------------------------------------------------------------


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
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"use_digit_position_embed must be boolean-compatible, got {value!r}")


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
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

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
    sandwich_activation: str = "gelu"
    sandwich_block_norm: str = "layernorm"
    sandwich_summary_tokens_per_axis: int = Field(default=4, gt=0)
    sandwich_self_attention_per_cross: int = Field(default=4, ge=0)
    sandwich_pre_row_attention_layers: int = Field(default=1, ge=0)
    sandwich_pre_column_attention_layers: int = Field(default=1, ge=0)
    sandwich_pre_column_inducing_tokens: int = Field(default=16, gt=0)
    sandwich_packed_attention: bool = False
    feature_type_conditioning: str = "film"
    floating_likelihood: str = "single_gaussian"
    integer_likelihood: str = "hybrid_mixture"

    @field_validator("pre_encoder_clip", mode="before")
    @classmethod
    def _validate_pre_encoder_clip(cls, value: Any) -> float | None:
        if value is None:
            return None
        clip = float(value)
        if clip <= 0.0:
            raise ValueError("pre_encoder_clip must be > 0")
        return clip

    @field_validator("sandwich_packed_attention", mode="before")
    @classmethod
    def _coerce_sandwich_packed_attention(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"sandwich_packed_attention must be boolean-compatible, got {value!r}")

    @field_validator("feature_type_conditioning", mode="before")
    @classmethod
    def _validate_feature_type_conditioning(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_FEATURE_TYPE_CONDITIONING:
            raise ValueError(
                "feature_type_conditioning must be one of "
                f"{SUPPORTED_FEATURE_TYPE_CONDITIONING}, got {value!r}"
            )
        return normalized

    @field_validator("sandwich_activation", mode="before")
    @classmethod
    def _validate_sandwich_activation(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_SANDWICH_ACTIVATIONS:
            raise ValueError(
                "sandwich_activation must be one of "
                f"{SUPPORTED_SANDWICH_ACTIVATIONS}, got {value!r}"
            )
        return normalized

    @field_validator("sandwich_block_norm", mode="before")
    @classmethod
    def _validate_sandwich_block_norm(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_SANDWICH_BLOCK_NORMS:
            raise ValueError(
                "sandwich_block_norm must be one of "
                f"{SUPPORTED_SANDWICH_BLOCK_NORMS}, got {value!r}"
            )
        return normalized

    @field_validator("floating_likelihood", mode="before")
    @classmethod
    def _validate_floating_likelihood(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_FLOATING_LIKELIHOODS:
            raise ValueError(
                "floating_likelihood must be one of "
                f"{SUPPORTED_FLOATING_LIKELIHOODS}, got {value!r}"
            )
        return normalized

    @field_validator("integer_likelihood", mode="before")
    @classmethod
    def _validate_integer_likelihood(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_INTEGER_LIKELIHOODS:
            raise ValueError(
                "integer_likelihood must be one of "
                f"{SUPPORTED_INTEGER_LIKELIHOODS}, got {value!r}"
            )
        return normalized


class _RoutedSandwichModelParams(_SandwichModelParams):
    routed_residual_mode: str = "dynamic_hyper"
    routed_residual_streams: int = Field(default=2, gt=1)
    routed_residual_scale: str = "deepnorm"
    routed_row_summary_tokens: int = Field(default=4, gt=0)
    routed_column_summary_tokens: int = Field(default=2, gt=0)
    routed_evidence_tokens: int = Field(default=16, gt=0)
    routed_direct_cell_bypass: bool = False

    @field_validator("routed_residual_mode", mode="before")
    @classmethod
    def _validate_routed_residual_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_ROUTED_RESIDUAL_MODES:
            raise ValueError(
                "routed_residual_mode must be one of "
                f"{SUPPORTED_ROUTED_RESIDUAL_MODES}, got {value!r}"
            )
        return normalized

    @field_validator("routed_residual_scale", mode="before")
    @classmethod
    def _validate_routed_residual_scale(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_ROUTED_RESIDUAL_SCALES:
            raise ValueError(
                "routed_residual_scale must be one of "
                f"{SUPPORTED_ROUTED_RESIDUAL_SCALES}, got {value!r}"
            )
        return normalized

    @field_validator("routed_direct_cell_bypass", mode="before")
    @classmethod
    def _coerce_routed_direct_cell_bypass(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"routed_direct_cell_bypass must be boolean-compatible, got {value!r}")


class _GridSandwichModelParams(_SandwichModelParams):
    grid_residual_mode: str = "prenorm"
    grid_attention_mode: str = "standard"
    grid_ffn_mode: str = "gelu"
    grid_recurrence_steps: int | None = Field(default=None, gt=0)
    grid_recurrence_unique_layers: int | None = Field(default=None, gt=0)
    classification_logit_softcap: float | None = Field(default=None, gt=0.0)
    attention_qk_norm: bool = False
    grid_moe_scope: str = "none"
    grid_moe_num_experts: int = Field(default=1, gt=0)
    grid_moe_top_k: int = Field(default=1, gt=0)
    grid_moe_router_init_std: float = Field(default=0.01, gt=0.0)
    grid_moe_normalize_top_k: bool = False
    grid_moe_shared_expert: bool = False
    grid_moe_shared_expert_scale: float = Field(default=1.0, ge=0.0)
    grid_moe_router_temperature: float = Field(default=1.0, gt=0.0)

    @field_validator("grid_residual_mode", mode="before")
    @classmethod
    def _validate_grid_residual_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_GRID_RESIDUAL_MODES:
            raise ValueError(
                "grid_residual_mode must be one of "
                f"{SUPPORTED_GRID_RESIDUAL_MODES}, got {value!r}"
            )
        return normalized

    @field_validator("grid_attention_mode", mode="before")
    @classmethod
    def _validate_grid_attention_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_GRID_ATTENTION_MODES:
            raise ValueError(
                "grid_attention_mode must be one of "
                f"{SUPPORTED_GRID_ATTENTION_MODES}, got {value!r}"
            )
        return normalized

    @field_validator("grid_ffn_mode", mode="before")
    @classmethod
    def _validate_grid_ffn_mode(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_GRID_FFN_MODES:
            raise ValueError(
                "grid_ffn_mode must be one of "
                f"{SUPPORTED_GRID_FFN_MODES}, got {value!r}"
            )
        return normalized

    @field_validator("grid_moe_scope", mode="before")
    @classmethod
    def _validate_grid_moe_scope(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_GRID_MOE_SCOPES:
            raise ValueError(
                "grid_moe_scope must be one of "
                f"{SUPPORTED_GRID_MOE_SCOPES}, got {value!r}"
            )
        return normalized

    @field_validator("attention_qk_norm", mode="before")
    @classmethod
    def _coerce_attention_qk_norm(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"attention_qk_norm must be boolean-compatible, got {value!r}")

    @field_validator("grid_moe_router_init_std", mode="before")
    @classmethod
    def _validate_grid_moe_router_init_std(cls, value: Any) -> float:
        std = float(value)
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("grid_moe_router_init_std must be a finite float > 0")
        return std

    @field_validator("grid_moe_normalize_top_k", mode="before")
    @classmethod
    def _coerce_grid_moe_normalize_top_k(cls, value: Any) -> bool:
        return cls._coerce_grid_bool(value, field_name="grid_moe_normalize_top_k")

    @field_validator("grid_moe_shared_expert", mode="before")
    @classmethod
    def _coerce_grid_moe_shared_expert(cls, value: Any) -> bool:
        return cls._coerce_grid_bool(value, field_name="grid_moe_shared_expert")

    @classmethod
    def _coerce_grid_bool(cls, value: Any, *, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"{field_name} must be boolean-compatible, got {value!r}")

    @field_validator("grid_moe_shared_expert_scale", mode="before")
    @classmethod
    def _validate_grid_moe_shared_expert_scale(cls, value: Any) -> float:
        scale = float(value)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("grid_moe_shared_expert_scale must be a finite float >= 0")
        return scale

    @field_validator("grid_moe_router_temperature", mode="before")
    @classmethod
    def _validate_grid_moe_router_temperature(cls, value: Any) -> float:
        temperature = float(value)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("grid_moe_router_temperature must be a finite float > 0")
        return temperature

    @field_validator("classification_logit_softcap", mode="before")
    @classmethod
    def _validate_classification_logit_softcap(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
        cap = float(value)
        if cap <= 0.0:
            raise ValueError("classification_logit_softcap must be null or > 0")
        return cap

    @field_validator("grid_recurrence_steps", mode="before")
    @classmethod
    def _validate_grid_recurrence_steps(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
        steps = int(value)
        if steps <= 0:
            raise ValueError("grid_recurrence_steps must be null or a positive integer")
        return steps

    @field_validator("grid_recurrence_unique_layers", mode="before")
    @classmethod
    def _validate_grid_recurrence_unique_layers(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
        layers = int(value)
        if layers <= 0:
            raise ValueError("grid_recurrence_unique_layers must be null or a positive integer")
        return layers

    @model_validator(mode="after")
    def _validate_grid_recurrence_layer_cycle(self) -> "_GridSandwichModelParams":
        if self.grid_recurrence_unique_layers is None:
            pass
        elif self.grid_recurrence_steps is None:
            raise ValueError(
                "grid_recurrence_unique_layers requires grid_recurrence_steps to be set"
            )
        elif int(self.grid_recurrence_unique_layers) > int(self.grid_recurrence_steps):
            raise ValueError(
                "grid_recurrence_unique_layers must be less than or equal to "
                "grid_recurrence_steps"
            )
        if self.grid_moe_scope != "none" and int(self.grid_moe_num_experts) <= 1:
            raise ValueError("grid_moe_num_experts must be > 1 when grid_moe_scope is enabled")
        if int(self.grid_moe_top_k) > int(self.grid_moe_num_experts):
            raise ValueError("grid_moe_top_k must be <= grid_moe_num_experts")
        return self


# ---------------------------------------------------------------------------
# Payload models (discriminated union over arch)
# ---------------------------------------------------------------------------


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


class _RoutedSandwichModelBuildSpecPayload(_BaseModelBuildSpecPayload):
    arch: Literal["routed_sandwich"] = ROUTED_SANDWICH_MODEL_ARCH
    params: _RoutedSandwichModelParams = Field(default_factory=_RoutedSandwichModelParams)


class _GridSandwichModelBuildSpecPayload(_BaseModelBuildSpecPayload):
    arch: Literal["grid_sandwich"] = GRID_SANDWICH_MODEL_ARCH
    params: _GridSandwichModelParams = Field(default_factory=_GridSandwichModelParams)


_ModelBuildSpecPayload = Annotated[
    _SimpleModelBuildSpecPayload
    | _StagedModelBuildSpecPayload
    | _SandwichModelBuildSpecPayload
    | _RoutedSandwichModelBuildSpecPayload
    | _GridSandwichModelBuildSpecPayload,
    Field(discriminator="arch"),
]
_MODEL_BUILD_SPEC_PAYLOAD_ADAPTER: TypeAdapter[_ModelBuildSpecPayload] = TypeAdapter(
    _ModelBuildSpecPayload
)


# ---------------------------------------------------------------------------
# Derived defaults and key sets (from Pydantic models, not hardcoded)
# ---------------------------------------------------------------------------


def _build_flat_defaults() -> dict[str, Any]:
    """Compute the superset of all param defaults across all architectures."""
    defaults: dict[str, Any] = {}
    for name, info in _BaseModelBuildSpecPayload.model_fields.items():
        if name not in ("task", "arch", "params"):
            defaults[name] = info.default
    for name, info in _StagedModelParams.model_fields.items():
        defaults[name] = info.default
    for name, info in _SandwichModelParams.model_fields.items():
        if name not in defaults:
            defaults[name] = info.default
    for name, info in _RoutedSandwichModelParams.model_fields.items():
        if name not in defaults:
            defaults[name] = info.default
    for name, info in _GridSandwichModelParams.model_fields.items():
        if name not in defaults:
            defaults[name] = info.default
    return defaults


FLAT_DEFAULTS: dict[str, Any] = _build_flat_defaults()


def _build_sandwich_defaults() -> dict[str, Any]:
    """Compute defaults specific to the sandwich architecture."""
    defaults: dict[str, Any] = {}
    for name, info in _BaseModelBuildSpecPayload.model_fields.items():
        if name not in ("task", "arch", "params"):
            defaults[name] = info.default
    for name, info in _SandwichModelParams.model_fields.items():
        defaults[name] = info.default
    return defaults


SANDWICH_DEFAULTS: dict[str, Any] = _build_sandwich_defaults()
ROUTED_SANDWICH_DEFAULTS: dict[str, Any] = {
    **SANDWICH_DEFAULTS,
    **{
        name: info.default for name, info in _RoutedSandwichModelParams.model_fields.items()
    },
}
GRID_SANDWICH_DEFAULTS: dict[str, Any] = {
    **SANDWICH_DEFAULTS,
    **{
        name: info.default for name, info in _GridSandwichModelParams.model_fields.items()
    },
}
_COMMON_PAYLOAD_NAMES = frozenset(_BaseModelBuildSpecPayload.model_fields) - {"task", "arch", "params"}
_SIMPLE_PARAM_NAMES = frozenset(_SimpleModelParams.model_fields)
_STAGED_PARAM_NAMES = frozenset(_StagedModelParams.model_fields)
_SANDWICH_PARAM_NAMES = frozenset(_SandwichModelParams.model_fields)
_ROUTED_SANDWICH_PARAM_NAMES = frozenset(_RoutedSandwichModelParams.model_fields)
_GRID_SANDWICH_PARAM_NAMES = frozenset(_GridSandwichModelParams.model_fields)
_STAGED_COMPAT_KEYS = ("stage", "stage_label", "module_overrides")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolved_arch(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MODEL_ARCHES:
        raise ValueError(f"Unsupported model arch: {normalized!r}")
    return normalized


def _with_staged_arch_compat(mapping: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in mapping.items()}
    if normalized.get("arch") is None and any(
        normalized.get(key) is not None for key in _STAGED_COMPAT_KEYS
    ):
        normalized["arch"] = STAGED_MODEL_ARCH
    return normalized


def _payload_mapping_from_flat_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _with_staged_arch_compat(mapping)
    task = str(mapping.get("task", "classification")).strip().lower()
    arch = _resolved_arch(mapping.get("arch", DEFAULT_MODEL_ARCH))
    _reject_legacy_sandwich_fields(
        arch=arch,
        primary_map=mapping,
        fallback_map={},
    )
    mapping = _sanitize_arch_specific_sandwich_fields(arch=arch, mapping=mapping)
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
    for key in _COMMON_PAYLOAD_NAMES:
        if mapping.get(key) is not None:
            payload[key] = mapping[key]
    if arch == SIMPLE_MODEL_ARCH:
        param_names = _SIMPLE_PARAM_NAMES
    elif arch == STAGED_MODEL_ARCH:
        param_names = _STAGED_PARAM_NAMES
    elif arch == ROUTED_SANDWICH_MODEL_ARCH:
        param_names = _ROUTED_SANDWICH_PARAM_NAMES
    elif arch == GRID_SANDWICH_MODEL_ARCH:
        param_names = _GRID_SANDWICH_PARAM_NAMES
    else:
        param_names = _SANDWICH_PARAM_NAMES
    params = {key: mapping[key] for key in param_names if mapping.get(key) is not None}
    if params:
        payload["params"] = params
    return payload


def _validate_payload(payload: Any) -> _ModelBuildSpecPayload:
    if isinstance(
        payload,
        (
            _SimpleModelBuildSpecPayload,
            _StagedModelBuildSpecPayload,
            _SandwichModelBuildSpecPayload,
            _RoutedSandwichModelBuildSpecPayload,
            _GridSandwichModelBuildSpecPayload,
        ),
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
    flat = dict(FLAT_DEFAULTS)
    flat.update(payload._common_flat_dict())
    flat.update(payload.params.model_dump(exclude_none=False))
    flat["arch"] = payload.arch
    flat["task"] = payload.task
    if payload.arch != STAGED_MODEL_ARCH:
        flat["stage"] = None
        flat["stage_label"] = None
        flat["module_overrides"] = None
    return flat


def _serialized_dict_from_payload(payload: _ModelBuildSpecPayload) -> dict[str, Any]:
    serialized = payload._common_flat_dict()
    serialized.update(payload.params.model_dump(exclude_none=False))
    serialized["arch"] = payload.arch
    serialized["task"] = payload.task
    for field_name in _arch_specific_sandwich_dead_fields(arch=payload.arch):
        serialized.pop(field_name, None)
    return serialized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        return _serialized_dict_from_payload(self.payload)


def model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a canonical model spec from a primary mapping with optional fallback."""

    primary_map = _with_staged_arch_compat(primary) if primary is not None else {}
    fallback_map = _with_staged_arch_compat(fallback) if fallback is not None else {}

    arch_value = _resolved_arch(
        primary_map.get("arch") or fallback_map.get("arch") or DEFAULT_MODEL_ARCH
    )
    _reject_legacy_sandwich_fields(
        arch=arch_value,
        primary_map=primary_map,
        fallback_map=fallback_map,
    )
    if fallback_map:
        _reject_layered_arch_specific_sandwich_fields(
            arch=arch_value,
            primary_map=primary_map,
            fallback_map=fallback_map,
        )

    merged: dict[str, Any] = {}
    for key, value in fallback_map.items():
        if value is not None:
            merged[key] = value
    for key, value in primary_map.items():
        if value is not None:
            merged[key] = value
    merged["task"] = str(task).strip().lower()
    merged["arch"] = arch_value

    return ModelBuildSpec(**merged)


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
    arch_value = _resolved_arch(
        primary_map.get("arch") or fallback_map.get("arch") or DEFAULT_MODEL_ARCH
    )
    if (
        arch_value == SANDWICH_MODEL_ARCH
        and primary_map.get("feature_type_conditioning") is None
        and fallback_map.get("feature_type_conditioning") is None
    ):
        primary_map = dict(primary_map)
        primary_map["feature_type_conditioning"] = (
            "additive_embedding"
            if _LEGACY_SANDWICH_FEATURE_TYPE_EMBEDDING_KEY in (state_dict or {})
            else "film"
        )
    spec = model_build_spec_from_mappings(
        task=task,
        primary=primary_map,
        fallback=fallback_map,
    )
    _validate_checkpoint_feature_group_size(
        spec=spec,
        state_dict=state_dict,
        feature_group_size_is_configured=feature_group_size_is_configured,
    )
    return spec
