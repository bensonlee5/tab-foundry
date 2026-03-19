"""Shared forward-path helpers for the staged classifier."""

from __future__ import annotations

from typing import Any, cast

import torch

from tab_foundry.input_normalization import InputNormalizationMode, normalize_train_test_tensors
from tab_foundry.model.components.non_finite import clip_finite_values
from tab_foundry.types import TaskBatch

from .states import CellTableState, HeadOutputState, RawInputState, RowState


def prepare_task_inputs(batch: TaskBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train_test_split_index = int(batch.x_train.shape[0])
    if train_test_split_index <= 0:
        raise RuntimeError("tabfoundry_staged requires at least one training row")
    x_all = torch.cat([batch.x_train, batch.x_test], dim=0).to(torch.float32).unsqueeze(0)
    y_train = batch.y_train.to(torch.int64).unsqueeze(0)
    y_test = batch.y_test.to(torch.int64).unsqueeze(0)
    return x_all, y_train, y_test, train_test_split_index


def validate_batched_inputs(
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    train_test_split_index: int,
) -> None:
    if x_all.ndim != 3:
        raise ValueError(f"x_all must have shape [B, R, C], got {tuple(x_all.shape)}")
    if y_train.ndim != 2:
        raise ValueError(f"y_train must have shape [B, R_train], got {tuple(y_train.shape)}")
    if int(x_all.shape[0]) != int(y_train.shape[0]):
        raise ValueError("x_all and y_train must have matching batch dimensions")
    if train_test_split_index <= 0 or train_test_split_index >= int(x_all.shape[1]):
        raise ValueError(
            "train_test_split_index must satisfy 0 < split < num_rows, got "
            f"split={train_test_split_index}, num_rows={x_all.shape[1]}"
        )
    if int(y_train.shape[1]) != train_test_split_index:
        raise ValueError("y_train length must match train_test_split_index")


def normalize_x_all(model: Any, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
    if model.surface.normalization_mode != "shared":
        return x_all
    x_train = x_all[:, :train_test_split_index, :]
    x_test = x_all[:, train_test_split_index:, :]
    train_norm, test_norm = normalize_train_test_tensors(
        x_train,
        x_test,
        mode=cast(InputNormalizationMode, model.input_normalization),
        preserve_non_finite=model.surface.tokenizer == "scalar_per_feature_nan_mask",
    )
    return torch.cat([train_norm, test_norm], dim=1)


def build_raw_input_state(
    model: Any,
    *,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor | None,
    train_test_split_index: int,
    num_classes: int,
) -> RawInputState:
    validate_batched_inputs(x_all, y_train, train_test_split_index)
    normalized = normalize_x_all(model, x_all, train_test_split_index=train_test_split_index)
    if model.pre_encoder_clip is not None:
        if model.surface.tokenizer == "scalar_per_feature_nan_mask":
            normalized = clip_finite_values(
                normalized,
                clip_value=float(model.pre_encoder_clip),
            )
        else:
            normalized = normalized.clamp(-model.pre_encoder_clip, model.pre_encoder_clip)
    return RawInputState(
        x_all=normalized,
        y_train=y_train,
        y_test=y_test,
        train_test_split_index=train_test_split_index,
        num_classes=num_classes,
    )


def feature_cells(model: Any, raw_state: RawInputState) -> torch.Tensor:
    tokenized_x, _token_padding_mask = model.tokenizer(raw_state.x_all)
    if model.surface.feature_encoder == "nano":
        feature_cells = model.feature_encoder(raw_state.x_all, raw_state.train_test_split_index)
    else:
        feature_cells = model.feature_encoder(tokenized_x)
    model.trace_activation("post_feature_encoder", feature_cells)
    return feature_cells


def build_table_tokens_from_raw(model: Any, raw_state: RawInputState) -> torch.Tensor:
    encoded_feature_cells = feature_cells(model, raw_state)
    target_cells = model.target_conditioner(
        raw_state.y_train,
        num_rows=int(raw_state.x_all.shape[1]),
    )
    model.trace_activation("post_target_conditioner", target_cells)
    return torch.cat([encoded_feature_cells, target_cells], dim=2)


def build_table_tokens_batched(
    model: Any,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    *,
    train_test_split_index: int,
) -> torch.Tensor:
    raw_state = build_raw_input_state(
        model,
        x_all=x_all,
        y_train=y_train,
        y_test=None,
        train_test_split_index=train_test_split_index,
        num_classes=max(2, int(y_train.max().item()) + 1),
    )
    return build_table_tokens_from_raw(model, raw_state)


def encode_table_batched(
    model: Any,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    *,
    train_test_split_index: int,
) -> torch.Tensor:
    cells = build_table_tokens_batched(
        model,
        x_all,
        y_train,
        train_test_split_index=train_test_split_index,
    )
    if model.post_encoder_norm is not None:
        cells = model.post_encoder_norm(cells)
    model.trace_activation("pre_transformer", cells)
    for index, block in enumerate(model.transformer_blocks):
        cells = block(cells, train_test_split_index=train_test_split_index)
        model.trace_activation(f"post_transformer_block_{index}", cells)
    if model.post_stack_norm is not None:
        cells = model.post_stack_norm(cells)
    return cells


def build_table_tokens(model: Any, batch: TaskBatch) -> tuple[torch.Tensor, int]:
    x_all, y_train, _y_test, train_test_split_index = prepare_task_inputs(batch)
    table = build_table_tokens_batched(
        model,
        x_all,
        y_train,
        train_test_split_index=train_test_split_index,
    )
    return table.squeeze(0), train_test_split_index


def encode_table(model: Any, batch: TaskBatch) -> tuple[torch.Tensor, int]:
    x_all, y_train, _y_test, train_test_split_index = prepare_task_inputs(batch)
    encoded = encode_table_batched(
        model,
        x_all,
        y_train,
        train_test_split_index=train_test_split_index,
    )
    return encoded.squeeze(0), train_test_split_index


def encode_to_cell_state(model: Any, raw_state: RawInputState) -> CellTableState:
    cells = build_table_tokens_from_raw(model, raw_state)
    if model.post_encoder_norm is not None:
        cells = model.post_encoder_norm(cells)
    model.trace_activation("pre_transformer", cells)
    for index, block in enumerate(model.transformer_blocks):
        cells = block(cells, train_test_split_index=raw_state.train_test_split_index)
        model.trace_activation(f"post_transformer_block_{index}", cells)
    if model.post_stack_norm is not None:
        cells = model.post_stack_norm(cells)
    return CellTableState(
        cells=cells,
        train_test_split_index=raw_state.train_test_split_index,
        num_classes=raw_state.num_classes,
    )


def pool_rows(model: Any, cell_state: CellTableState) -> RowState:
    encoded_cells = model.column_encoder(cell_state.cells)
    model.trace_activation("post_column_encoder", encoded_cells)
    rows = model.row_pool(encoded_cells, token_padding_mask=None)
    model.trace_activation("post_row_pool", rows)
    return RowState(
        rows=rows,
        train_test_split_index=cell_state.train_test_split_index,
        num_classes=cell_state.num_classes,
    )


def condition_rows(
    model: Any,
    row_state: RowState,
    *,
    train_target_embeddings: torch.Tensor | None = None,
) -> HeadOutputState:
    rows = row_state.rows
    if model.context_encoder is not None:
        if train_target_embeddings is None:
            raise RuntimeError("train_target_embeddings must be provided for context encoding")
        rows = model.context_encoder(
            rows,
            train_target_embeddings=train_target_embeddings,
            train_test_split_index=row_state.train_test_split_index,
        )
        model.trace_activation("post_context_encoder", rows)
    return HeadOutputState(
        rows=rows,
        train_test_split_index=row_state.train_test_split_index,
        num_classes=row_state.num_classes,
    )


def context_train_embeddings(model: Any, y_train: torch.Tensor) -> torch.Tensor:
    assert model.context_label_embed is not None
    return model.context_label_embed(y_train.clamp(max=model.many_class_base - 1))
