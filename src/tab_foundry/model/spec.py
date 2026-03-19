"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES


SUPPORTED_MODEL_TASKS = ("classification",)
STAGED_MODEL_ARCH = "tabfoundry_staged"
SUPPORTED_MODEL_ARCHES = ("tabfoundry_simple", STAGED_MODEL_ARCH)
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3


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


@dataclass(slots=True, frozen=True)
class ModelBuildSpec:
    """Canonical model-construction settings shared across train/eval/export/load."""

    task: str
    arch: str = STAGED_MODEL_ARCH
    stage: str | None = None
    stage_label: str | None = None
    module_overrides: dict[str, Any] | None = None
    d_col: int = 128
    d_icl: int = 512
    input_normalization: str = "none"
    feature_group_size: int = 1
    many_class_train_mode: str = "path_nll"
    max_mixed_radix_digits: int = 64
    norm_type: str = "layernorm"
    tfcol_n_heads: int = 8
    tfcol_n_layers: int = 3
    tfcol_n_inducing: int = 128
    tfrow_n_heads: int = 8
    tfrow_n_layers: int = 3
    tfrow_cls_tokens: int = 4
    tfrow_norm: str = "layernorm"
    tficl_n_heads: int = 8
    tficl_n_layers: int = 12
    tficl_ff_expansion: int = 2
    many_class_base: int = 10
    head_hidden_dim: int = 1024
    use_digit_position_embed: bool = True
    staged_dropout: float = 0.0
    pre_encoder_clip: float | None = None

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
        ):
            value = int(getattr(self, field_name))
            object.__setattr__(self, field_name, value)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if self.many_class_base <= 1:
            raise ValueError(f"many_class_base must be >= 2, got {self.many_class_base}")

        use_digit_position_embed = _coerce_bool(
            self.use_digit_position_embed,
            context="use_digit_position_embed",
        )
        object.__setattr__(self, "use_digit_position_embed", use_digit_position_embed)

        staged_dropout = float(self.staged_dropout)
        if not 0.0 <= staged_dropout <= 0.5:
            raise ValueError(f"staged_dropout must be in [0.0, 0.5], got {staged_dropout}")
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

    return ModelBuildSpec(
        task=str(task).strip().lower(),
        arch=str(_pick("arch", STAGED_MODEL_ARCH)),
        stage=_pick("stage", None),
        stage_label=_pick("stage_label", None),
        module_overrides=_pick("module_overrides", None),
        d_col=int(_pick("d_col", 128)),
        d_icl=int(_pick("d_icl", 512)),
        input_normalization=str(_pick("input_normalization", "none")),
        feature_group_size=int(_pick("feature_group_size", 1)),
        many_class_train_mode=str(_pick("many_class_train_mode", "path_nll")),
        max_mixed_radix_digits=int(_pick("max_mixed_radix_digits", 64)),
        norm_type=str(_pick("norm_type", "layernorm")),
        tfcol_n_heads=int(_pick("tfcol_n_heads", 8)),
        tfcol_n_layers=int(_pick("tfcol_n_layers", 3)),
        tfcol_n_inducing=int(_pick("tfcol_n_inducing", 128)),
        tfrow_n_heads=int(_pick("tfrow_n_heads", 8)),
        tfrow_n_layers=int(_pick("tfrow_n_layers", 3)),
        tfrow_cls_tokens=int(_pick("tfrow_cls_tokens", 4)),
        tfrow_norm=str(_pick("tfrow_norm", "layernorm")),
        tficl_n_heads=int(_pick("tficl_n_heads", 8)),
        tficl_n_layers=int(_pick("tficl_n_layers", 12)),
        tficl_ff_expansion=int(_pick("tficl_ff_expansion", 2)),
        many_class_base=int(_pick("many_class_base", 10)),
        head_hidden_dim=int(_pick("head_hidden_dim", 1024)),
        use_digit_position_embed=_coerce_bool(
            _pick("use_digit_position_embed", True),
            context="use_digit_position_embed",
        ),
        staged_dropout=float(_pick("staged_dropout", 0.0)),
        pre_encoder_clip=_pick("pre_encoder_clip", None),
    )


def _feature_group_size_from_state_dict(
    state_dict: Mapping[str, Any] | None,
) -> int | None:
    if state_dict is None:
        return None
    raw_weight = state_dict.get(_GROUP_LINEAR_WEIGHT_KEY)
    shape = getattr(raw_weight, "shape", None)
    if shape is None or len(shape) != 2:
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
