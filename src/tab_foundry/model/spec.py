"""Canonical model build spec and config resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SUPPORTED_MODEL_TASKS = ("classification", "regression")
SUPPORTED_MANY_CLASS_TRAIN_MODES = ("path_nll", "full_probs")
_GROUP_LINEAR_WEIGHT_KEY = "group_linear.weight"
_GROUP_SHIFT_COUNT = 3


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


@dataclass(slots=True, frozen=True)
class ModelBuildSpec:
    """Canonical model-construction settings shared across train/eval/export/load."""

    task: str
    d_col: int = 128
    d_icl: int = 512
    input_normalization: str = "none"
    feature_group_size: int = 1
    many_class_train_mode: str = "path_nll"
    max_mixed_radix_digits: int = 64
    tfcol_n_heads: int = 8
    tfcol_n_layers: int = 3
    tfcol_n_inducing: int = 128
    tfrow_n_heads: int = 8
    tfrow_n_layers: int = 3
    tfrow_cls_tokens: int = 4
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

        input_normalization = str(self.input_normalization).strip().lower()
        object.__setattr__(self, "input_normalization", input_normalization)

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
        d_col=int(_pick("d_col", 128)),
        d_icl=int(_pick("d_icl", 512)),
        input_normalization=str(_pick("input_normalization", "none")),
        feature_group_size=int(_pick("feature_group_size", 1)),
        many_class_train_mode=str(_pick("many_class_train_mode", "path_nll")),
        max_mixed_radix_digits=int(_pick("max_mixed_radix_digits", 64)),
        tfcol_n_heads=int(_pick("tfcol_n_heads", 8)),
        tfcol_n_layers=int(_pick("tfcol_n_layers", 3)),
        tfcol_n_inducing=int(_pick("tfcol_n_inducing", 128)),
        tfrow_n_heads=int(_pick("tfrow_n_heads", 8)),
        tfrow_n_layers=int(_pick("tfrow_n_layers", 3)),
        tfrow_cls_tokens=int(_pick("tfrow_cls_tokens", 4)),
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
