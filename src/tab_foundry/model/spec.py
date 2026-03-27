"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES


SUPPORTED_MODEL_TASKS = ("classification",)
STAGED_MODEL_ARCH = "tabfoundry_staged"
SANDWICH_MODEL_ARCH = "tabfoundry_sandwich"
SUPPORTED_MODEL_ARCHES = ("tabfoundry_simple", STAGED_MODEL_ARCH, SANDWICH_MODEL_ARCH)
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3
DEFAULT_MODEL_ARCH = STAGED_MODEL_ARCH
DEFAULT_MODEL_STAGE: str | None = None
DEFAULT_MODEL_STAGE_LABEL: str | None = None
DEFAULT_MODEL_MODULE_OVERRIDES: dict[str, Any] | None = None
DEFAULT_MODEL_D_COL = 128
DEFAULT_MODEL_D_ICL = 512
DEFAULT_MODEL_INPUT_NORMALIZATION = "none"
DEFAULT_MODEL_FEATURE_GROUP_SIZE = 1
DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE = "path_nll"
DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS = 64
DEFAULT_MODEL_NORM_TYPE = "layernorm"
DEFAULT_MODEL_TFCOL_N_HEADS = 8
DEFAULT_MODEL_TFCOL_N_LAYERS = 3
DEFAULT_MODEL_TFCOL_N_INDUCING = 128
DEFAULT_MODEL_TFROW_N_HEADS = 8
DEFAULT_MODEL_TFROW_N_LAYERS = 3
DEFAULT_MODEL_TFROW_CLS_TOKENS = 4
DEFAULT_MODEL_TFROW_NORM = "layernorm"
DEFAULT_MODEL_TFICL_N_HEADS = 8
DEFAULT_MODEL_TFICL_N_LAYERS = 12
DEFAULT_MODEL_TFICL_FF_EXPANSION = 2
DEFAULT_MODEL_MANY_CLASS_BASE = 10
DEFAULT_MODEL_HEAD_HIDDEN_DIM = 1024
DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED = True
DEFAULT_MODEL_STAGED_DROPOUT = 0.0
DEFAULT_MODEL_PRE_ENCODER_CLIP: float | None = None
DEFAULT_SANDWICH_MODEL_D_ICL = 60
DEFAULT_SANDWICH_MODEL_HEAD_HIDDEN_DIM = 96
DEFAULT_MODEL_SANDWICH_LATENTS = 24
DEFAULT_MODEL_SANDWICH_LAYERS = 2
DEFAULT_MODEL_SANDWICH_HEADS = 4
DEFAULT_MODEL_SANDWICH_FF_EXPANSION = 2
DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS = 4
DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS = 4
DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS = 1
DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS = 1
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
            normalized[key] = _normalize_jsonable_mapping(
                raw_value,
                context=f"{context}.{key}",
            )
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


@dataclass(slots=True, frozen=True)
class ModelBuildSpec:
    """Canonical model-construction settings shared across train/eval/export/load."""

    task: str
    arch: str = DEFAULT_MODEL_ARCH
    stage: str | None = DEFAULT_MODEL_STAGE
    stage_label: str | None = DEFAULT_MODEL_STAGE_LABEL
    module_overrides: dict[str, Any] | None = DEFAULT_MODEL_MODULE_OVERRIDES
    d_col: int = DEFAULT_MODEL_D_COL
    d_icl: int = DEFAULT_MODEL_D_ICL
    input_normalization: str = DEFAULT_MODEL_INPUT_NORMALIZATION
    feature_group_size: int = DEFAULT_MODEL_FEATURE_GROUP_SIZE
    many_class_train_mode: str = DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE
    max_mixed_radix_digits: int = DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS
    norm_type: str = DEFAULT_MODEL_NORM_TYPE
    tfcol_n_heads: int = DEFAULT_MODEL_TFCOL_N_HEADS
    tfcol_n_layers: int = DEFAULT_MODEL_TFCOL_N_LAYERS
    tfcol_n_inducing: int = DEFAULT_MODEL_TFCOL_N_INDUCING
    tfrow_n_heads: int = DEFAULT_MODEL_TFROW_N_HEADS
    tfrow_n_layers: int = DEFAULT_MODEL_TFROW_N_LAYERS
    tfrow_cls_tokens: int = DEFAULT_MODEL_TFROW_CLS_TOKENS
    tfrow_norm: str = DEFAULT_MODEL_TFROW_NORM
    tficl_n_heads: int = DEFAULT_MODEL_TFICL_N_HEADS
    tficl_n_layers: int = DEFAULT_MODEL_TFICL_N_LAYERS
    tficl_ff_expansion: int = DEFAULT_MODEL_TFICL_FF_EXPANSION
    many_class_base: int = DEFAULT_MODEL_MANY_CLASS_BASE
    head_hidden_dim: int = DEFAULT_MODEL_HEAD_HIDDEN_DIM
    use_digit_position_embed: bool = DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED
    staged_dropout: float = DEFAULT_MODEL_STAGED_DROPOUT
    pre_encoder_clip: float | None = DEFAULT_MODEL_PRE_ENCODER_CLIP
    sandwich_latents: int = DEFAULT_MODEL_SANDWICH_LATENTS
    sandwich_layers: int = DEFAULT_MODEL_SANDWICH_LAYERS
    sandwich_heads: int = DEFAULT_MODEL_SANDWICH_HEADS
    sandwich_ff_expansion: int = DEFAULT_MODEL_SANDWICH_FF_EXPANSION
    sandwich_summary_tokens_per_axis: int = DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS
    sandwich_self_attention_per_cross: int = DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS
    sandwich_pre_row_attention_layers: int = DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS
    sandwich_pre_column_attention_layers: int = DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS

    def __post_init__(self) -> None:
        task = str(self.task).strip().lower()
        if task not in SUPPORTED_MODEL_TASKS:
            raise ValueError(f"Unsupported task: {task!r}")
        object.__setattr__(self, "task", task)

        arch = str(self.arch).strip().lower()
        if arch not in SUPPORTED_MODEL_ARCHES:
            raise ValueError(f"Unsupported model arch: {arch!r}")
        object.__setattr__(self, "arch", arch)
        object.__setattr__(self, "stage", resolve_model_stage(arch=arch, stage=self.stage))
        object.__setattr__(
            self,
            "stage_label",
            _normalize_optional_label(self.stage_label, context="model.stage_label"),
        )
        object.__setattr__(
            self,
            "module_overrides",
            _normalize_jsonable_mapping(
                self.module_overrides,
                context="model.module_overrides",
            ),
        )
        if arch != STAGED_MODEL_ARCH:
            if self.stage_label is not None:
                raise ValueError(
                    "model.stage_label is only supported when model.arch='tabfoundry_staged'; "
                    f"got arch={arch!r}"
                )
            if self.module_overrides is not None:
                raise ValueError(
                    "model.module_overrides is only supported when model.arch='tabfoundry_staged'; "
                    f"got arch={arch!r}"
                )

        input_normalization = str(self.input_normalization).strip().lower()
        if input_normalization not in SUPPORTED_INPUT_NORMALIZATION_MODES:
            raise ValueError(
                "input_normalization must be "
                f"{SUPPORTED_INPUT_NORMALIZATION_MODES}, got {input_normalization!r}"
            )
        object.__setattr__(self, "input_normalization", input_normalization)

        norm_type = str(self.norm_type).strip().lower()
        if norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {SUPPORTED_NORM_TYPES}, got {norm_type!r}"
            )
        object.__setattr__(self, "norm_type", norm_type)

        tfrow_norm = str(self.tfrow_norm).strip().lower()
        if tfrow_norm not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"tfrow_norm must be one of {SUPPORTED_NORM_TYPES}, got {tfrow_norm!r}"
            )
        object.__setattr__(self, "tfrow_norm", tfrow_norm)

        many_class_train_mode = str(self.many_class_train_mode).strip().lower()
        if many_class_train_mode not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
            raise ValueError(
                "many_class_train_mode must be "
                f"{SUPPORTED_MANY_CLASS_TRAIN_MODES}, got {many_class_train_mode!r}"
            )
        object.__setattr__(self, "many_class_train_mode", many_class_train_mode)

        for field_name in (
            "d_col",
            "d_icl",
            "feature_group_size",
            "max_mixed_radix_digits",
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
            "sandwich_latents",
            "sandwich_layers",
            "sandwich_heads",
            "sandwich_ff_expansion",
            "sandwich_summary_tokens_per_axis",
        ):
            value = int(getattr(self, field_name))
            object.__setattr__(self, field_name, value)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        for field_name in (
            "sandwich_self_attention_per_cross",
            "sandwich_pre_row_attention_layers",
            "sandwich_pre_column_attention_layers",
        ):
            value = int(getattr(self, field_name))
            object.__setattr__(self, field_name, value)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if self.many_class_base < MIN_MODEL_MANY_CLASS_BASE:
            raise ValueError(
                f"many_class_base must be >= {MIN_MODEL_MANY_CLASS_BASE}, got {self.many_class_base}"
            )

        use_digit_position_embed = _coerce_bool(
            self.use_digit_position_embed,
            context="use_digit_position_embed",
        )
        object.__setattr__(self, "use_digit_position_embed", use_digit_position_embed)

        staged_dropout = float(self.staged_dropout)
        if not 0.0 <= staged_dropout <= MAX_MODEL_STAGED_DROPOUT:
            raise ValueError(
                f"staged_dropout must be in [0.0, {MAX_MODEL_STAGED_DROPOUT}], "
                f"got {staged_dropout}"
            )
        object.__setattr__(self, "staged_dropout", staged_dropout)

        pre_encoder_clip = self.pre_encoder_clip
        if pre_encoder_clip is not None:
            pre_encoder_clip = float(pre_encoder_clip)
            if pre_encoder_clip <= 0:
                raise ValueError(f"pre_encoder_clip must be > 0 when set, got {pre_encoder_clip}")
            object.__setattr__(self, "pre_encoder_clip", pre_encoder_clip)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a canonical model spec from a primary mapping with optional fallback."""

    primary_map = primary if primary is not None else {}
    fallback_map = fallback if fallback is not None else {}

    def _pick(name: str, default: Any) -> Any:
        if name in primary_map and primary_map[name] is not None:
            return primary_map[name]
        if name in fallback_map and fallback_map[name] is not None:
            return fallback_map[name]
        return default

    arch_value = str(_pick("arch", DEFAULT_MODEL_ARCH)).strip().lower()
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

    primary_map = dict(primary) if primary is not None else {}
    fallback_map = fallback if fallback is not None else {}
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
