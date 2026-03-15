"""Component builders for staged tabfoundry resolved surfaces."""

from __future__ import annotations

from torch import nn

from .resolved import ResolvedStageSurface
from .subsystems import (
    DirectClassifierHead,
    IdentityColumnEncoder,
    LabelTokenTargetConditioner,
    MeanPaddedLinearTargetConditioner,
    NanoBinaryHead,
    NanoFeatureEncoder,
    NanoPostNormBlock,
    PreNormCellBlock,
    RowCLSPool,
    ScalarPerFeatureTokenizer,
    SequenceContextEncoder,
    SetColumnEncoder,
    SharedLinearFeatureEncoder,
    ShiftedGroupedTokenizer,
    TargetColumnPool,
)


def build_tokenizer(surface: ResolvedStageSurface) -> nn.Module:
    if surface.tokenizer == "scalar_per_feature":
        return ScalarPerFeatureTokenizer()
    if surface.tokenizer == "shifted_grouped":
        return ShiftedGroupedTokenizer()
    raise RuntimeError(f"Unsupported tokenizer variant: {surface.tokenizer!r}")


def build_feature_encoder(
    surface: ResolvedStageSurface,
    *,
    tokenizer: nn.Module,
    d_icl: int,
) -> nn.Module:
    if surface.feature_encoder == "nano":
        return NanoFeatureEncoder(d_icl)
    token_dim = int(getattr(tokenizer, "token_dim"))
    return SharedLinearFeatureEncoder(token_dim=token_dim, embedding_size=d_icl)


def build_target_conditioner(
    surface: ResolvedStageSurface,
    *,
    d_icl: int,
    many_class_base: int,
) -> nn.Module:
    if surface.target_conditioner == "mean_padded_linear":
        return MeanPaddedLinearTargetConditioner(d_icl)
    if surface.target_conditioner == "label_token":
        return LabelTokenTargetConditioner(many_class_base, d_icl)
    raise RuntimeError(f"Unsupported target conditioner variant: {surface.target_conditioner!r}")


def build_table_block(surface: ResolvedStageSurface, *, d_icl: int) -> nn.Module:
    if surface.table_block.style == "nano_postnorm":
        return NanoPostNormBlock(
            embedding_size=d_icl,
            nhead=surface.table_block.n_heads,
            mlp_hidden_size=surface.table_block.mlp_hidden_dim,
            norm_type=surface.table_block.norm_type,
        )
    if surface.table_block.style == "prenorm":
        return PreNormCellBlock(
            embedding_size=d_icl,
            nhead=surface.table_block.n_heads,
            mlp_hidden_size=surface.table_block.mlp_hidden_dim,
            allow_test_self_attention=surface.table_block.allow_test_self_attention,
            norm_type=surface.table_block.norm_type,
        )
    raise RuntimeError(f"Unsupported table block style: {surface.table_block.style!r}")


def build_column_encoder(surface: ResolvedStageSurface, *, d_icl: int) -> nn.Module:
    if surface.column_encoder == "none":
        return IdentityColumnEncoder()
    if surface.column_encoder == "tfcol":
        return SetColumnEncoder(
            embedding_size=d_icl,
            n_heads=int(surface.column_encoder_config.n_heads or 0),
            n_layers=int(surface.column_encoder_config.n_layers or 0),
            n_inducing=int(surface.column_encoder_config.n_inducing or 0),
            norm_type=str(surface.column_encoder_config.norm_type or "layernorm"),
        )
    raise RuntimeError(f"Unsupported column encoder variant: {surface.column_encoder!r}")


def build_row_pool(surface: ResolvedStageSurface, *, d_icl: int) -> nn.Module:
    if surface.row_pool == "target_column":
        return TargetColumnPool()
    if surface.row_pool == "row_cls":
        return RowCLSPool(
            embedding_size=d_icl,
            n_heads=int(surface.row_pool_config.n_heads or 0),
            n_layers=int(surface.row_pool_config.n_layers or 0),
            cls_tokens=int(surface.row_pool_config.cls_tokens or 0),
            norm_type=str(surface.row_pool_config.norm_type or "layernorm"),
        )
    raise RuntimeError(f"Unsupported row pool variant: {surface.row_pool!r}")


def build_context_encoder(surface: ResolvedStageSurface, *, d_icl: int) -> SequenceContextEncoder | None:
    if surface.context_encoder == "none":
        return None
    if surface.context_encoder in {"plain", "qass"}:
        return SequenceContextEncoder(
            embedding_size=d_icl,
            n_heads=int(surface.context_encoder_config.n_heads or 0),
            n_layers=int(surface.context_encoder_config.n_layers or 0),
            ff_expansion=int(surface.context_encoder_config.ff_expansion or 0),
            use_qass=bool(surface.context_encoder_config.use_qass),
            allow_test_self_attention=bool(
                surface.context_encoder_config.allow_test_self_attention
            ),
            norm_type=str(surface.context_encoder_config.norm_type or "layernorm"),
        )
    raise RuntimeError(f"Unsupported context encoder variant: {surface.context_encoder!r}")


def build_context_label_embed(
    surface: ResolvedStageSurface,
    *,
    d_icl: int,
    many_class_base: int,
) -> nn.Embedding | None:
    if surface.context_encoder != "none" or surface.head == "many_class":
        return nn.Embedding(many_class_base, d_icl)
    return None


def build_digit_position_embed(
    surface: ResolvedStageSurface,
    *,
    d_icl: int,
    max_mixed_radix_digits: int,
    use_digit_position_embed: bool,
) -> nn.Embedding | None:
    if surface.head == "many_class" and use_digit_position_embed:
        return nn.Embedding(max_mixed_radix_digits, d_icl)
    return None


def build_direct_head(
    surface: ResolvedStageSurface,
    *,
    d_icl: int,
    head_hidden_dim: int,
    many_class_base: int,
) -> nn.Module:
    if surface.head == "binary_direct":
        return NanoBinaryHead(d_icl, head_hidden_dim)
    return DirectClassifierHead(d_icl, head_hidden_dim, many_class_base)
