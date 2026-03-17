"""Model factory."""

from __future__ import annotations

from torch import nn

from .architectures.tabfoundry import TabFoundryClassifier, TabFoundryRegressor
from .architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from .architectures.tabfoundry_staged import TabFoundryStagedClassifier
from .spec import ModelBuildSpec


def build_model_from_spec(spec: ModelBuildSpec) -> nn.Module:
    """Instantiate model from a canonical model spec."""

    return build_model(**spec.to_dict())


def build_model(
    task: str,
    *,
    arch: str = "tabfoundry",
    stage: str | None = None,
    stage_label: str | None = None,
    module_overrides: dict[str, object] | None = None,
    d_col: int = 128,
    d_icl: int = 512,
    input_normalization: str = "none",
    feature_group_size: int = 1,
    many_class_train_mode: str = "path_nll",
    max_mixed_radix_digits: int = 64,
    norm_type: str = "layernorm",
    tfcol_n_heads: int = 8,
    tfcol_n_layers: int = 3,
    tfcol_n_inducing: int = 128,
    tfrow_n_heads: int = 8,
    tfrow_n_layers: int = 3,
    tfrow_cls_tokens: int = 4,
    tfrow_norm: str = "layernorm",
    tficl_n_heads: int = 8,
    tficl_n_layers: int = 12,
    tficl_ff_expansion: int = 2,
    many_class_base: int = 10,
    head_hidden_dim: int = 1024,
    use_digit_position_embed: bool = True,
    staged_dropout: float = 0.0,
    pre_encoder_clip: float | None = None,
) -> nn.Module:
    """Instantiate model for task."""

    normalized_arch = str(arch).strip().lower()
    if normalized_arch == "tabfoundry" and task == "classification":
        if stage is not None or stage_label is not None or module_overrides is not None:
            raise ValueError("tabfoundry does not support staged model surface fields")
        return TabFoundryClassifier(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )
    if normalized_arch == "tabfoundry" and task == "regression":
        if stage is not None or stage_label is not None or module_overrides is not None:
            raise ValueError("tabfoundry does not support staged model surface fields")
        return TabFoundryRegressor(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            head_hidden_dim=head_hidden_dim,
        )
    if normalized_arch == "tabfoundry_simple":
        if stage is not None or stage_label is not None or module_overrides is not None:
            raise ValueError("tabfoundry_simple does not support staged model surface fields")
        if task != "classification":
            raise ValueError(
                "tabfoundry_simple only supports task='classification' in phase 1; "
                f"got {task!r}"
            )
        return TabFoundrySimpleClassifier(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )
    if normalized_arch == "tabfoundry_staged":
        if task != "classification":
            raise ValueError(
                "tabfoundry_staged only supports task='classification' in this branch; "
                f"got {task!r}"
            )
        return TabFoundryStagedClassifier(
            stage=stage,
            stage_label=stage_label,
            module_overrides=module_overrides,
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
            staged_dropout=staged_dropout,
            pre_encoder_clip=pre_encoder_clip,
        )
    raise ValueError(f"Unsupported model arch: {arch!r}")
