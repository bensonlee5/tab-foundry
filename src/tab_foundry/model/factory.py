"""Model factory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from torch import nn

from .tabiclv2 import TabICLv2Classifier, TabICLv2Regressor


@dataclass(slots=True, frozen=True)
class ModelBuildSpec:
    """Canonical model-construction settings shared across train/eval/export/load."""

    task: str
    d_col: int = 128
    d_icl: int = 512
    input_normalization: str = "none"
    feature_group_size: int = 32
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

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


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
        feature_group_size=int(_pick("feature_group_size", 32)),
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


def build_model_from_spec(spec: ModelBuildSpec) -> nn.Module:
    """Instantiate model from a canonical model spec."""

    return build_model(**spec.to_dict())


def build_model(
    task: str,
    *,
    d_col: int = 128,
    d_icl: int = 512,
    input_normalization: str = "none",
    feature_group_size: int = 32,
    many_class_train_mode: str = "path_nll",
    max_mixed_radix_digits: int = 64,
    tfcol_n_heads: int = 8,
    tfcol_n_layers: int = 3,
    tfcol_n_inducing: int = 128,
    tfrow_n_heads: int = 8,
    tfrow_n_layers: int = 3,
    tfrow_cls_tokens: int = 4,
    tficl_n_heads: int = 8,
    tficl_n_layers: int = 12,
    tficl_ff_expansion: int = 2,
    many_class_base: int = 10,
    head_hidden_dim: int = 1024,
    use_digit_position_embed: bool = True,
) -> nn.Module:
    """Instantiate model for task."""

    if task == "classification":
        return TabICLv2Classifier(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )
    if task == "regression":
        return TabICLv2Regressor(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            head_hidden_dim=head_hidden_dim,
        )
    raise ValueError(f"Unsupported task: {task}")
