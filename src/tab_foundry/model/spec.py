"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.missingness import (
    normalize_missingness_mode,
)
from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES


SUPPORTED_MODEL_TASKS = ("classification", "regression")
STAGED_MODEL_ARCH = "tabfoundry_staged"
SUPPORTED_MODEL_ARCHES = ("tabfoundry", "tabfoundry_simple", STAGED_MODEL_ARCH)
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
_NULLABLE_MODEL_FIELDS = frozenset({"stage", "stage_label", "module_overrides"})
_REQUIRED_CHECKPOINT_MODEL_FIELDS = (
    "arch",
    "stage",
    "stage_label",
    "module_overrides",
    "d_col",
    "d_icl",
    "input_normalization",
    "missingness_mode",
    "feature_group_size",
    "many_class_train_mode",
    "max_mixed_radix_digits",
    "norm_type",
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
    "many_class_base",
    "head_hidden_dim",
    "use_digit_position_embed",
)
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3
_SIMPLE_FEATURE_ENCODER_WEIGHT_KEY = "feature_encoder.linear_layer.weight"
_SHARED_FEATURE_ENCODER_WEIGHT_KEY = "feature_encoder.linear.weight"
_NAN_EMBEDDING_PARAMETER_KEYS = (
    "feature_encoder.nan_embedding",
    "feature_encoder.nan_embedding_token",
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
    arch: str = "tabfoundry"
    stage: str | None = None
    stage_label: str | None = None
    module_overrides: dict[str, Any] | None = None
    d_col: int = 128
    d_icl: int = 512
    input_normalization: str = "none"
    missingness_mode: str = "none"
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
        object.__setattr__(
            self,
            "missingness_mode",
            normalize_missingness_mode(
                self.missingness_mode,
                context="missingness_mode",
            ),
        )

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

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a canonical model spec from one primary mapping."""

    primary_map = primary if primary is not None else {}

    def _pick(name: str, default: Any) -> Any:
        if name in primary_map and primary_map[name] is not None:
            return primary_map[name]
        return default

    return ModelBuildSpec(
        task=str(task).strip().lower(),
        arch=str(_pick("arch", "tabfoundry")),
        stage=_pick("stage", None),
        stage_label=_pick("stage_label", None),
        module_overrides=_pick("module_overrides", None),
        d_col=int(_pick("d_col", 128)),
        d_icl=int(_pick("d_icl", 512)),
        input_normalization=str(_pick("input_normalization", "none")),
        missingness_mode=str(_pick("missingness_mode", "none")),
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
    )


def _feature_group_size_from_state_dict(
    state_dict: Mapping[str, Any] | None,
    *,
    missingness_mode: str,
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
    mode_multiplier = 2 if missingness_mode == "feature_mask" else 1
    divisor = _GROUP_SHIFT_COUNT * mode_multiplier
    if in_features <= 0 or in_features % divisor != 0:
        return None
    return in_features // divisor


def _state_dict_has_nan_embedding(state_dict: Mapping[str, Any] | None) -> bool:
    if state_dict is None:
        return False
    return any(key in state_dict for key in _NAN_EMBEDDING_PARAMETER_KEYS)


def _merge_explicit_model_overrides(
    *,
    primary: Mapping[str, Any] | None,
    explicit_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(primary) if primary is not None else {}
    if explicit_overrides is None:
        return merged
    for key, value in explicit_overrides.items():
        merged[str(key)] = value
    return merged


def _require_checkpoint_model_fields(model_cfg: Mapping[str, Any]) -> None:
    missing = sorted(
        field_name
        for field_name in _REQUIRED_CHECKPOINT_MODEL_FIELDS
        if field_name not in model_cfg
        or (model_cfg[field_name] is None and field_name not in _NULLABLE_MODEL_FIELDS)
    )
    if not missing:
        return
    missing_rendered = ", ".join(missing)
    raise ValueError(
        "Checkpoint config.model is missing required reconstruction fields: "
        f"{missing_rendered}. Regenerate the checkpoint with explicit model metadata "
        "or provide explicit checkpoint model overrides for every missing field."
    )


def _raise_on_ambiguous_checkpoint_layout(
    *,
    model_cfg: Mapping[str, Any],
    state_dict: Mapping[str, Any] | None,
) -> None:
    arch = str(model_cfg.get("arch", "")).strip().lower()
    feature_group_size_missing = "feature_group_size" not in model_cfg or model_cfg["feature_group_size"] is None
    missingness_mode_missing = "missingness_mode" not in model_cfg or model_cfg["missingness_mode"] is None
    if arch != "tabfoundry" or state_dict is None or (not feature_group_size_missing and not missingness_mode_missing):
        return
    candidates = [
        (mode, feature_group_size)
        for mode in ("none", "feature_mask")
        if (
            feature_group_size := _feature_group_size_from_state_dict(
                state_dict,
                missingness_mode=mode,
            )
        )
        is not None
    ]
    if len(candidates) < 2:
        return
    rendered = ", ".join(
        f"(missingness_mode={mode!r}, feature_group_size={feature_group_size})"
        for mode, feature_group_size in candidates
    )
    raise ValueError(
        "Checkpoint weights are ambiguous across multiple tabfoundry layouts: "
        f"{rendered}. Persist explicit model.missingness_mode and "
        "model.feature_group_size metadata or provide explicit checkpoint "
        "model overrides."
    )


def _missingness_mode_from_state_dict(
    *,
    spec: ModelBuildSpec,
    state_dict: Mapping[str, Any] | None,
) -> str | None:
    if state_dict is None:
        return None
    if spec.arch == "tabfoundry":
        # Shifted-grouped tabfoundry weights are validated by the resolved
        # feature_group_size + missingness_mode pair, and shapes such as
        # in_features=6 are ambiguous across legacy grouped-token layouts.
        return None

    if spec.arch == "tabfoundry_simple":
        raw_weight = state_dict.get(_SIMPLE_FEATURE_ENCODER_WEIGHT_KEY)
        shape = getattr(raw_weight, "shape", None)
        if shape is None or len(shape) != 2:
            return None
        try:
            in_features = int(shape[1])
        except (IndexError, TypeError, ValueError):
            return None
        if in_features == 2:
            return "feature_mask"
        if in_features == 1 and _state_dict_has_nan_embedding(state_dict):
            return "explicit_token"
        if in_features == 1:
            return "none"
        return None

    if spec.arch == STAGED_MODEL_ARCH:
        from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface

        surface = resolve_staged_surface(spec)
        if surface.feature_encoder == "nano":
            raw_weight = state_dict.get(_SIMPLE_FEATURE_ENCODER_WEIGHT_KEY)
            shape = getattr(raw_weight, "shape", None)
            if shape is None or len(shape) != 2:
                return None
            try:
                in_features = int(shape[1])
            except (IndexError, TypeError, ValueError):
                return None
            if in_features == 2:
                return "feature_mask"
            if in_features == 1 and _state_dict_has_nan_embedding(state_dict):
                return "explicit_token"
            if in_features == 1:
                return "none"
            return None

        raw_weight = state_dict.get(_SHARED_FEATURE_ENCODER_WEIGHT_KEY)
        shape = getattr(raw_weight, "shape", None)
        if shape is None or len(shape) != 2:
            return None
        try:
            in_features = int(shape[1])
        except (IndexError, TypeError, ValueError):
            return None
        base_token_dim = 1 if surface.tokenizer == "scalar_per_feature" else 3
        if in_features == base_token_dim * 2:
            return "feature_mask"
        if in_features == base_token_dim and _state_dict_has_nan_embedding(state_dict):
            return "explicit_token"
        if in_features == base_token_dim:
            return "none"
        return None

    return None


def _validate_checkpoint_feature_group_size(
    *,
    spec: ModelBuildSpec,
    state_dict: Mapping[str, Any] | None,
) -> None:
    checkpoint_feature_group_size = _feature_group_size_from_state_dict(
        state_dict,
        missingness_mode=spec.missingness_mode,
    )
    if checkpoint_feature_group_size is None:
        return
    if checkpoint_feature_group_size == spec.feature_group_size:
        return

    raise ValueError(
        "Resolved feature_group_size="
        f"{spec.feature_group_size} is incompatible with checkpoint weights "
        f"implying feature_group_size={checkpoint_feature_group_size}; "
        "update the persisted checkpoint metadata or provide an explicit "
        "checkpoint model override that matches the weights."
    )


def _validate_checkpoint_missingness_mode(
    *,
    spec: ModelBuildSpec,
    state_dict: Mapping[str, Any] | None,
) -> None:
    checkpoint_missingness_mode = _missingness_mode_from_state_dict(
        spec=spec,
        state_dict=state_dict,
    )
    if checkpoint_missingness_mode is None or checkpoint_missingness_mode == spec.missingness_mode:
        return
    raise ValueError(
        "Resolved missingness_mode="
        f"{spec.missingness_mode!r} is incompatible with checkpoint weights "
        f"implying missingness_mode={checkpoint_missingness_mode!r}; "
        "update the persisted checkpoint metadata or provide an explicit "
        "checkpoint model override that matches the weights."
    )


def checkpoint_model_build_spec_from_mappings(
    *,
    task: str,
    primary: Mapping[str, Any] | None = None,
    explicit_overrides: Mapping[str, Any] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> ModelBuildSpec:
    """Resolve a checkpoint-backed model spec and validate weight compatibility."""

    effective_model_cfg = _merge_explicit_model_overrides(
        primary=primary,
        explicit_overrides=explicit_overrides,
    )
    _raise_on_ambiguous_checkpoint_layout(
        model_cfg=effective_model_cfg,
        state_dict=state_dict,
    )
    _require_checkpoint_model_fields(effective_model_cfg)
    if (
        str(effective_model_cfg["arch"]).strip().lower() == STAGED_MODEL_ARCH
        and effective_model_cfg["stage"] is None
    ):
        raise ValueError(
            "Checkpoint config.model.stage must be explicitly set when "
            "model.arch='tabfoundry_staged'."
        )
    spec = model_build_spec_from_mappings(
        task=task,
        primary=effective_model_cfg,
    )
    _validate_checkpoint_missingness_mode(
        spec=spec,
        state_dict=state_dict,
    )
    _validate_checkpoint_feature_group_size(
        spec=spec,
        state_dict=state_dict,
    )
    return spec
