"""Model factory."""

from __future__ import annotations

from torch import nn

from .architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from .architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from .architectures.tabfoundry_staged import TabFoundryStagedClassifier
from .spec import (
    DEFAULT_MODEL_ARCH,
    DEFAULT_MODEL_D_COL,
    DEFAULT_MODEL_D_ICL,
    DEFAULT_MODEL_FEATURE_GROUP_SIZE,
    DEFAULT_MODEL_HEAD_HIDDEN_DIM,
    DEFAULT_MODEL_INPUT_NORMALIZATION,
    DEFAULT_MODEL_MANY_CLASS_BASE,
    DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE,
    DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS,
    DEFAULT_MODEL_MODULE_OVERRIDES,
    DEFAULT_MODEL_NORM_TYPE,
    DEFAULT_MODEL_PRE_ENCODER_CLIP,
    DEFAULT_MODEL_STAGE,
    DEFAULT_MODEL_STAGE_LABEL,
    DEFAULT_MODEL_STAGED_DROPOUT,
    DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    DEFAULT_MODEL_SANDWICH_HEADS,
    DEFAULT_MODEL_SANDWICH_LAYERS,
    DEFAULT_MODEL_SANDWICH_LATENTS,
    DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS,
    DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS,
    DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS,
    DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS,
    DEFAULT_MODEL_TFCOL_N_HEADS,
    DEFAULT_MODEL_TFCOL_N_INDUCING,
    DEFAULT_MODEL_TFCOL_N_LAYERS,
    DEFAULT_MODEL_TFICL_FF_EXPANSION,
    DEFAULT_MODEL_TFICL_N_HEADS,
    DEFAULT_MODEL_TFICL_N_LAYERS,
    DEFAULT_MODEL_TFROW_CLS_TOKENS,
    DEFAULT_MODEL_TFROW_N_HEADS,
    DEFAULT_MODEL_TFROW_N_LAYERS,
    DEFAULT_MODEL_TFROW_NORM,
    DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED,
    DEFAULT_SANDWICH_MODEL_D_ICL,
    DEFAULT_SANDWICH_MODEL_HEAD_HIDDEN_DIM,
    ModelBuildSpec,
    SANDWICH_MODEL_ARCH,
    STAGED_MODEL_ARCH,
)


def build_model_from_spec(spec: ModelBuildSpec) -> nn.Module:
    """Instantiate model from a canonical model spec."""

    return build_model(**spec.to_dict())


def build_model(
    task: str,
    *,
    arch: str = DEFAULT_MODEL_ARCH,
    stage: str | None = DEFAULT_MODEL_STAGE,
    stage_label: str | None = DEFAULT_MODEL_STAGE_LABEL,
    module_overrides: dict[str, object] | None = DEFAULT_MODEL_MODULE_OVERRIDES,
    d_col: int = DEFAULT_MODEL_D_COL,
    d_icl: int = DEFAULT_MODEL_D_ICL,
    input_normalization: str = DEFAULT_MODEL_INPUT_NORMALIZATION,
    feature_group_size: int = DEFAULT_MODEL_FEATURE_GROUP_SIZE,
    many_class_train_mode: str = DEFAULT_MODEL_MANY_CLASS_TRAIN_MODE,
    max_mixed_radix_digits: int = DEFAULT_MODEL_MAX_MIXED_RADIX_DIGITS,
    norm_type: str = DEFAULT_MODEL_NORM_TYPE,
    tfcol_n_heads: int = DEFAULT_MODEL_TFCOL_N_HEADS,
    tfcol_n_layers: int = DEFAULT_MODEL_TFCOL_N_LAYERS,
    tfcol_n_inducing: int = DEFAULT_MODEL_TFCOL_N_INDUCING,
    tfrow_n_heads: int = DEFAULT_MODEL_TFROW_N_HEADS,
    tfrow_n_layers: int = DEFAULT_MODEL_TFROW_N_LAYERS,
    tfrow_cls_tokens: int = DEFAULT_MODEL_TFROW_CLS_TOKENS,
    tfrow_norm: str = DEFAULT_MODEL_TFROW_NORM,
    tficl_n_heads: int = DEFAULT_MODEL_TFICL_N_HEADS,
    tficl_n_layers: int = DEFAULT_MODEL_TFICL_N_LAYERS,
    tficl_ff_expansion: int = DEFAULT_MODEL_TFICL_FF_EXPANSION,
    many_class_base: int = DEFAULT_MODEL_MANY_CLASS_BASE,
    head_hidden_dim: int = DEFAULT_MODEL_HEAD_HIDDEN_DIM,
    use_digit_position_embed: bool = DEFAULT_MODEL_USE_DIGIT_POSITION_EMBED,
    staged_dropout: float = DEFAULT_MODEL_STAGED_DROPOUT,
    pre_encoder_clip: float | None = DEFAULT_MODEL_PRE_ENCODER_CLIP,
    sandwich_latents: int = DEFAULT_MODEL_SANDWICH_LATENTS,
    sandwich_layers: int = DEFAULT_MODEL_SANDWICH_LAYERS,
    sandwich_heads: int = DEFAULT_MODEL_SANDWICH_HEADS,
    sandwich_ff_expansion: int = DEFAULT_MODEL_SANDWICH_FF_EXPANSION,
    sandwich_summary_tokens_per_axis: int = DEFAULT_MODEL_SANDWICH_SUMMARY_TOKENS_PER_AXIS,
    sandwich_self_attention_per_cross: int = DEFAULT_MODEL_SANDWICH_SELF_ATTENTION_PER_CROSS,
    sandwich_pre_row_attention_layers: int = DEFAULT_MODEL_SANDWICH_PRE_ROW_ATTENTION_LAYERS,
    sandwich_pre_column_attention_layers: int = DEFAULT_MODEL_SANDWICH_PRE_COLUMN_ATTENTION_LAYERS,
) -> nn.Module:
    """Instantiate model for task."""

    normalized_task = str(task).strip().lower()
    if normalized_task != "classification":
        raise ValueError(
            "Only task='classification' is currently supported; "
            f"got {task!r}. Regression will be rebuilt on tabfoundry_staged later."
        )
    normalized_arch = str(arch).strip().lower()
    if normalized_arch == "tabfoundry":
        raise ValueError(
            "Legacy model arch 'tabfoundry' is no longer supported; "
            "use 'tabfoundry_staged', 'tabfoundry_simple', or 'tabfoundry_sandwich'."
        )
    if normalized_arch == "tabfoundry_simple":
        if stage is not None or stage_label is not None or module_overrides is not None:
            raise ValueError("tabfoundry_simple does not support staged model surface fields")
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
    if normalized_arch == STAGED_MODEL_ARCH:
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
    if normalized_arch == SANDWICH_MODEL_ARCH:
        if stage is not None or stage_label is not None or module_overrides is not None:
            raise ValueError("tabfoundry_sandwich does not support staged model surface fields")
        resolved_d_icl = (
            DEFAULT_SANDWICH_MODEL_D_ICL
            if d_icl == DEFAULT_MODEL_D_ICL
            else d_icl
        )
        resolved_head_hidden_dim = (
            DEFAULT_SANDWICH_MODEL_HEAD_HIDDEN_DIM
            if head_hidden_dim == DEFAULT_MODEL_HEAD_HIDDEN_DIM
            else head_hidden_dim
        )
        return TabFoundrySandwichClassifier(
            d_icl=resolved_d_icl,
            input_normalization=input_normalization,
            many_class_base=many_class_base,
            norm_type=norm_type,
            head_hidden_dim=resolved_head_hidden_dim,
            pre_encoder_clip=pre_encoder_clip,
            sandwich_latents=sandwich_latents,
            sandwich_layers=sandwich_layers,
            sandwich_heads=sandwich_heads,
            sandwich_ff_expansion=sandwich_ff_expansion,
            sandwich_summary_tokens_per_axis=sandwich_summary_tokens_per_axis,
            sandwich_self_attention_per_cross=sandwich_self_attention_per_cross,
            sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
            sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
        )
    raise ValueError(f"Unsupported model arch: {arch!r}")
