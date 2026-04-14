"""Feature preparation helpers for the sandwich architecture family."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from tab_foundry.feature_types import (
    feature_type_ids_from_resolved as _shared_feature_type_ids_from_resolved,
    feature_type_ids_from_task_metadata as _shared_feature_type_ids_from_task_metadata,
    normalize_feature_types,
)
from tab_foundry.model.components.non_finite import clip_finite_values

from .blocks import _InducedSetAttentionBlock, _SelfAttentionBlock
from .states import SandwichFeatureState, SandwichRawInputState


_TRAIN_ROLE_ID = 0
_TEST_ROLE_ID = 1


def build_raw_input_state(
    *,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor | None,
    train_test_split_index: int,
    num_classes: int,
    feature_type_ids: torch.Tensor,
) -> SandwichRawInputState:
    """Build the canonical raw-input state for the sandwich family."""

    return SandwichRawInputState(
        x_all=x_all,
        y_train=y_train,
        y_test=y_test,
        train_test_split_index=train_test_split_index,
        num_classes=num_classes,
        feature_type_ids=feature_type_ids,
    )


def fourier_positions(
    *,
    num_positions: int,
    embedding_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build Fourier features for one positional axis."""

    positions = torch.arange(num_positions, device=device, dtype=torch.float32).unsqueeze(1)
    div_terms = torch.exp(
        torch.arange(0, embedding_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / float(embedding_size))
    )
    encoding = torch.zeros((num_positions, embedding_size), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_terms)
    odd_width = encoding[:, 1::2].shape[1]
    if odd_width > 0:
        encoding[:, 1::2] = torch.cos(positions * div_terms[:odd_width])
    return encoding.to(dtype=dtype).unsqueeze(0)


def normalize_required_feature_types(
    feature_types: Any,
    *,
    expected_count: int,
    context: str,
) -> list[str]:
    """Resolve one explicit feature-type list for the sandwich family."""

    if feature_types is None:
        raise ValueError(f"{context} is required for tabfoundry_sandwich")
    return normalize_feature_types(
        feature_types,
        expected_count=expected_count,
        context=context,
    )


def feature_type_ids_from_resolved(
    resolved_types_by_task: list[list[str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map normalized feature-type strings into vocabulary ids."""

    return _shared_feature_type_ids_from_resolved(
        resolved_types_by_task,
        device=device,
    )


def feature_type_ids_from_forward_batched(
    feature_types: list[str] | list[list[str]] | None,
    *,
    batch_size: int,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve explicit feature types for forward_batched entrypoints."""

    if feature_types is None:
        raise ValueError("tabfoundry_sandwich forward_batched() requires explicit feature_types")
    if not feature_types or isinstance(feature_types[0], str):
        if batch_size != 1:
            raise ValueError(
                "tabfoundry_sandwich forward_batched() requires one feature_types list per task "
                f"when batch_size={batch_size}"
            )
        resolved = [
            normalize_required_feature_types(
                feature_types,
                expected_count=num_features,
                context="forward_batched.feature_types",
            )
        ]
        return feature_type_ids_from_resolved(resolved, device=device)
    if not isinstance(feature_types, list) or len(feature_types) != batch_size:
        raise ValueError(
            "tabfoundry_sandwich forward_batched() requires one feature_types list per task "
            f"when batch_size={batch_size}, got {type(feature_types).__name__}"
        )
    resolved_types_by_task = [
        normalize_required_feature_types(
            value,
            expected_count=num_features,
            context=f"forward_batched.feature_types[{index}]",
        )
        for index, value in enumerate(feature_types)
    ]
    return feature_type_ids_from_resolved(
        resolved_types_by_task,
        device=device,
    )


def feature_type_ids_from_metadata(
    metadata: dict[str, Any],
    *,
    batch_size: int,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve feature types from task metadata."""

    return _shared_feature_type_ids_from_task_metadata(
        metadata,
        batch_size=batch_size,
        num_features=num_features,
        device=device,
    )


def role_ids(
    *,
    batch_size: int,
    num_rows: int,
    num_train_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Build train/test role ids for one batched row sequence."""

    ids = torch.full(
        (batch_size, num_rows),
        _TRAIN_ROLE_ID,
        device=device,
        dtype=torch.int64,
    )
    ids[:, num_train_rows:] = _TEST_ROLE_ID
    return ids


def row_feature_self_attention(
    model: Any,
    block: _SelfAttentionBlock,
    feature_cells: torch.Tensor,
) -> torch.Tensor:
    """Apply one row-wise self-attention block over feature cells."""

    batch_size, num_rows, num_features, embedding_size = (
        int(feature_cells.shape[0]),
        int(feature_cells.shape[1]),
        int(feature_cells.shape[2]),
        int(feature_cells.shape[3]),
    )
    row_major = feature_cells.reshape(batch_size * num_rows, num_features, embedding_size)
    mixed = model._self_block(block, row_major)
    return mixed.reshape(batch_size, num_rows, num_features, embedding_size)


def column_row_isab(
    model: Any,
    block: _InducedSetAttentionBlock,
    feature_cells: torch.Tensor,
) -> torch.Tensor:
    """Apply one induced-set block over row-major column slices."""

    batch_size, num_rows, num_features, embedding_size = (
        int(feature_cells.shape[0]),
        int(feature_cells.shape[1]),
        int(feature_cells.shape[2]),
        int(feature_cells.shape[3]),
    )
    column_major = feature_cells.transpose(1, 2).contiguous()
    column_major = column_major.reshape(batch_size * num_features, num_rows, embedding_size)
    inducing = block.inducing_seed.expand(batch_size * num_features, -1, -1).to(
        device=column_major.device,
        dtype=column_major.dtype,
    )
    inducing = model._cross_block(block.rows_to_inducing, inducing, column_major)
    inducing = model._self_block(block.inducing_self, inducing)
    mixed = model._cross_block(block.rows_from_inducing, column_major, inducing)
    mixed = mixed.reshape(batch_size, num_features, num_rows, embedding_size)
    return mixed.transpose(1, 2).contiguous()


def pre_perceiver_cell_mixer(model: Any, feature_cells: torch.Tensor) -> torch.Tensor:
    """Run the optional pre-Perceiver row and column mixing stack."""

    mixed_cells = feature_cells
    for index, block in enumerate(model.pre_row_attention_blocks):
        block = cast(_SelfAttentionBlock, block)
        mixed_cells = row_feature_self_attention(model, block, mixed_cells)
        model.trace_activation(f"post_pre_row_attention_{index}", mixed_cells)
    for index, block in enumerate(model.pre_column_attention_blocks):
        block = cast(_InducedSetAttentionBlock, block)
        mixed_cells = column_row_isab(model, block, mixed_cells)
        model.trace_activation(f"post_pre_column_attention_{index}", mixed_cells)
    model.trace_activation("post_pre_perceiver_cells", mixed_cells)
    return mixed_cells


def build_feature_state(
    model: Any,
    raw_state: SandwichRawInputState,
    *,
    apply_input_normalization: bool = True,
) -> SandwichFeatureState:
    """Encode raw inputs into the sandwich feature-cell state."""

    return SandwichFeatureState(
        raw_state=raw_state,
        feature_cells=feature_cells(
            model,
            raw_state.x_all,
            train_test_split_index=raw_state.train_test_split_index,
            feature_type_ids=raw_state.feature_type_ids,
            apply_input_normalization=apply_input_normalization,
        ),
    )


def feature_cells(
    model: Any,
    x_all: torch.Tensor,
    *,
    train_test_split_index: int,
    feature_type_ids: torch.Tensor,
    apply_input_normalization: bool = True,
) -> torch.Tensor:
    """Encode raw inputs into the sandwich feature-cell tensor."""

    encoder_inputs = (
        model._normalize_x_all(x_all, train_test_split_index=train_test_split_index)
        if apply_input_normalization
        else x_all.to(torch.float32)
    )
    if model.pre_encoder_clip is not None:
        encoder_inputs = clip_finite_values(
            encoder_inputs,
            clip_value=float(model.pre_encoder_clip),
        )
    tokenized_x, _ = model.tokenizer(encoder_inputs)
    current_feature_cells = model.feature_encoder(tokenized_x)
    model.trace_activation("post_feature_encoder", current_feature_cells)
    if model.feature_type_film is not None:
        current_feature_cells = model.feature_type_film(
            current_feature_cells,
            feature_type_ids=feature_type_ids,
        )
    elif model.feature_type_embedding is not None:
        feature_type_embed = model.feature_type_embedding(feature_type_ids).unsqueeze(1)
        feature_type_embed = feature_type_embed.to(dtype=current_feature_cells.dtype)
        current_feature_cells = current_feature_cells + feature_type_embed
    else:  # pragma: no cover - defensive invariant
        raise RuntimeError("tabfoundry_sandwich requires one feature-type conditioning path")
    row_pos = model._fourier_positions(
        num_positions=int(current_feature_cells.shape[1]),
        embedding_size=int(current_feature_cells.shape[3]),
        device=current_feature_cells.device,
        dtype=current_feature_cells.dtype,
    ).unsqueeze(2)
    col_pos = model._fourier_positions(
        num_positions=int(current_feature_cells.shape[2]),
        embedding_size=int(current_feature_cells.shape[3]),
        device=current_feature_cells.device,
        dtype=current_feature_cells.dtype,
    ).unsqueeze(1)
    current_feature_cells = current_feature_cells + row_pos + col_pos
    model.trace_activation("post_cell_encoding", current_feature_cells)
    return pre_perceiver_cell_mixer(model, current_feature_cells)
