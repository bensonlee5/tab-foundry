"""Model factory."""

from __future__ import annotations

from torch import nn

from .architectures.tabfoundry import TabFoundryClassifier, TabFoundryRegressor
from .spec import ModelBuildSpec


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
        return TabFoundryClassifier(
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
        return TabFoundryRegressor(
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
