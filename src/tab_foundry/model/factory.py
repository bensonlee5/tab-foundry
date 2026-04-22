"""Model factory."""

from __future__ import annotations

from torch import nn

from .architectures.grid_sandwich import GridSandwichClassifier
from .architectures.routed_sandwich import RoutedSandwichClassifier
from .architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from .architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from .architectures.tabfoundry_staged import TabFoundryStagedClassifier
from .spec import (
    GRID_SANDWICH_MODEL_ARCH,
    ModelBuildSpec,
    ROUTED_SANDWICH_MODEL_ARCH,
    SANDWICH_MODEL_ARCH,
    STAGED_MODEL_ARCH,
)


def build_model_from_spec(spec: ModelBuildSpec) -> nn.Module:
    """Instantiate a model from one canonical model spec."""

    normalized_task = str(spec.task).strip().lower()
    if normalized_task != "classification":
        raise ValueError(
            "Only task='classification' is currently supported; "
            f"got {spec.task!r}. Regression will be rebuilt on tabfoundry_staged later."
        )

    normalized_arch = str(spec.arch).strip().lower()
    if normalized_arch == "tabfoundry":
        raise ValueError(
            "Legacy model arch 'tabfoundry' is no longer supported; "
            "use 'tabfoundry_staged', 'tabfoundry_simple', 'tabfoundry_sandwich', "
            "'routed_sandwich', or 'grid_sandwich'."
        )

    if normalized_arch == "tabfoundry_simple":
        return TabFoundrySimpleClassifier(
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

    if normalized_arch == STAGED_MODEL_ARCH:
        return TabFoundryStagedClassifier(
            stage=spec.stage,
            stage_label=spec.stage_label,
            module_overrides=spec.module_overrides,
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
            staged_dropout=float(spec.staged_dropout),
            pre_encoder_clip=spec.pre_encoder_clip,
        )

    if normalized_arch == SANDWICH_MODEL_ARCH:
        return TabFoundrySandwichClassifier(
            d_icl=int(spec.d_icl),
            input_normalization=str(spec.input_normalization),
            many_class_base=int(spec.many_class_base),
            norm_type=str(spec.norm_type),
            head_hidden_dim=int(spec.head_hidden_dim),
            pre_encoder_clip=spec.pre_encoder_clip,
            sandwich_latents=int(spec.sandwich_latents),
            sandwich_layers=int(spec.sandwich_layers),
            sandwich_heads=int(spec.sandwich_heads),
            sandwich_ff_expansion=int(spec.sandwich_ff_expansion),
            sandwich_activation=str(spec.sandwich_activation),
            sandwich_block_norm=str(spec.sandwich_block_norm),
            sandwich_summary_tokens_per_axis=int(spec.sandwich_summary_tokens_per_axis),
            sandwich_self_attention_per_cross=int(spec.sandwich_self_attention_per_cross),
            sandwich_pre_row_attention_layers=int(spec.sandwich_pre_row_attention_layers),
            sandwich_pre_column_attention_layers=int(spec.sandwich_pre_column_attention_layers),
            sandwich_pre_column_inducing_tokens=int(spec.sandwich_pre_column_inducing_tokens),
            sandwich_packed_attention=bool(spec.sandwich_packed_attention),
            feature_type_conditioning=str(spec.feature_type_conditioning),
            floating_likelihood=str(spec.floating_likelihood),
            integer_likelihood=str(spec.integer_likelihood),
        )

    if normalized_arch == ROUTED_SANDWICH_MODEL_ARCH:
        return RoutedSandwichClassifier(
            d_icl=int(spec.d_icl),
            input_normalization=str(spec.input_normalization),
            many_class_base=int(spec.many_class_base),
            norm_type=str(spec.norm_type),
            head_hidden_dim=int(spec.head_hidden_dim),
            pre_encoder_clip=spec.pre_encoder_clip,
            sandwich_latents=int(spec.sandwich_latents),
            sandwich_layers=int(spec.sandwich_layers),
            sandwich_heads=int(spec.sandwich_heads),
            sandwich_ff_expansion=int(spec.sandwich_ff_expansion),
            sandwich_activation=str(spec.sandwich_activation),
            sandwich_block_norm=str(spec.sandwich_block_norm),
            sandwich_self_attention_per_cross=int(spec.sandwich_self_attention_per_cross),
            sandwich_pre_row_attention_layers=int(spec.sandwich_pre_row_attention_layers),
            sandwich_pre_column_attention_layers=int(spec.sandwich_pre_column_attention_layers),
            sandwich_pre_column_inducing_tokens=int(spec.sandwich_pre_column_inducing_tokens),
            sandwich_packed_attention=bool(spec.sandwich_packed_attention),
            feature_type_conditioning=str(spec.feature_type_conditioning),
            routed_residual_mode=str(spec.routed_residual_mode),
            routed_residual_streams=int(spec.routed_residual_streams),
            routed_residual_scale=str(spec.routed_residual_scale),
            routed_row_summary_tokens=int(spec.routed_row_summary_tokens),
            routed_column_summary_tokens=int(spec.routed_column_summary_tokens),
            routed_evidence_tokens=int(spec.routed_evidence_tokens),
            routed_direct_cell_bypass=bool(spec.routed_direct_cell_bypass),
            floating_likelihood=str(spec.floating_likelihood),
            integer_likelihood=str(spec.integer_likelihood),
        )

    if normalized_arch == GRID_SANDWICH_MODEL_ARCH:
        return GridSandwichClassifier(
            d_icl=int(spec.d_icl),
            input_normalization=str(spec.input_normalization),
            many_class_base=int(spec.many_class_base),
            norm_type=str(spec.norm_type),
            head_hidden_dim=int(spec.head_hidden_dim),
            pre_encoder_clip=spec.pre_encoder_clip,
            sandwich_layers=int(spec.sandwich_layers),
            sandwich_heads=int(spec.sandwich_heads),
            sandwich_ff_expansion=int(spec.sandwich_ff_expansion),
            sandwich_activation=str(spec.sandwich_activation),
            sandwich_block_norm=str(spec.sandwich_block_norm),
            sandwich_pre_row_attention_layers=int(spec.sandwich_pre_row_attention_layers),
            sandwich_pre_column_attention_layers=int(spec.sandwich_pre_column_attention_layers),
            sandwich_pre_column_inducing_tokens=int(spec.sandwich_pre_column_inducing_tokens),
            sandwich_packed_attention=bool(spec.sandwich_packed_attention),
            feature_type_conditioning=str(spec.feature_type_conditioning),
            grid_residual_mode=str(spec.grid_residual_mode),
            grid_attention_mode=str(spec.grid_attention_mode),
            grid_ffn_mode=str(spec.grid_ffn_mode),
            grid_recurrence_steps=spec.grid_recurrence_steps,
            grid_recurrence_unique_layers=spec.grid_recurrence_unique_layers,
        )

    raise ValueError(f"Unsupported model arch: {spec.arch!r}")
