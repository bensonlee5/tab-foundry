"""Classification forward helpers for the sandwich architecture family."""

from __future__ import annotations

from typing import Any, cast

import torch

from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows

from .blocks import _CrossAttentionBlock, _PerceiverStage, _SelfAttentionBlock
from . import feature_flow as _feature_flow
from .states import SandwichClassificationState, SandwichFeatureState


_MIN_CLASS_COUNT = 2
_ROW_SUMMARY_TOKEN_ID = 0
_COLUMN_SUMMARY_TOKEN_ID = 1
_CELL_TOKEN_ID = 2


def summary_query_attention(
    model: Any,
    block: _CrossAttentionBlock,
    *,
    query: torch.Tensor,
    key_value: torch.Tensor,
    outer_count: int,
) -> torch.Tensor:
    """Apply one shared summary-query attention over a batched 4D tensor."""

    batch_size, _, inner_count, embedding_size = (
        int(key_value.shape[0]),
        int(key_value.shape[1]),
        int(key_value.shape[2]),
        int(key_value.shape[3]),
    )
    query_count = int(query.shape[1])
    flat_kv = key_value.reshape(batch_size * outer_count, inner_count, embedding_size)
    flat_query = query.expand(batch_size * outer_count, -1, -1).to(
        device=key_value.device,
        dtype=key_value.dtype,
    )
    summaries = model._cross_block(block, flat_query, flat_kv)
    return summaries.reshape(batch_size, outer_count, query_count, embedding_size)


def pool_test_rows(
    model: Any,
    *,
    latent_readout_rows: torch.Tensor,
    cell_readout_rows: torch.Tensor,
) -> torch.Tensor:
    """Pool the test-row readout facets with the learned pool query."""

    batch_size, num_test_rows, facet_count, embedding_size = (
        int(cell_readout_rows.shape[0]),
        int(cell_readout_rows.shape[1]),
        int(cell_readout_rows.shape[2]),
        int(cell_readout_rows.shape[3]),
    )
    pool_query = latent_readout_rows.mean(dim=2, keepdim=True)
    pool_query = pool_query + model.test_row_pool_query.view(1, 1, 1, model.d_icl).to(
        device=cell_readout_rows.device,
        dtype=cell_readout_rows.dtype,
    )
    flat_query = pool_query.reshape(batch_size * num_test_rows, 1, embedding_size)
    flat_kv = cell_readout_rows.reshape(batch_size * num_test_rows, facet_count, embedding_size)
    pooled = model._cross_block(model.test_row_pool, flat_query, flat_kv)
    return pooled.reshape(batch_size, num_test_rows, embedding_size)


def row_summary_bytes(model: Any, feature_cells: torch.Tensor) -> torch.Tensor:
    """Build the raw row-summary bytes before label/role conditioning."""

    summaries = summary_query_attention(
        model,
        model.row_summary_builder,
        query=model.row_summary_query,
        key_value=feature_cells,
        outer_count=int(feature_cells.shape[1]),
    )
    model.trace_activation("post_row_summary", summaries)
    return summaries


def _flatten_summary_grid(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.reshape(
        int(tokens.shape[0]),
        int(tokens.shape[1]) * int(tokens.shape[2]),
        int(tokens.shape[3]),
    )


def column_summary_bytes(model: Any, feature_cells: torch.Tensor) -> torch.Tensor:
    """Build the raw column-summary bytes before token-type tagging."""

    column_major = feature_cells.transpose(1, 2).contiguous()
    summaries = summary_query_attention(
        model,
        model.column_summary_builder,
        query=model.column_summary_query,
        key_value=column_major,
        outer_count=int(column_major.shape[1]),
    )
    model.trace_activation("post_column_summary", summaries)
    return summaries


def _column_summary_tokens_from_cells(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    trace_name: str,
) -> torch.Tensor:
    column_summaries = column_summary_bytes(model, feature_cells)
    token_type = model.token_type_embedding.weight[_COLUMN_SUMMARY_TOKEN_ID].to(
        dtype=column_summaries.dtype
    )
    tokens = column_summaries + token_type.view(1, 1, 1, -1)
    flattened_tokens = _flatten_summary_grid(tokens)
    model.trace_activation(trace_name, flattened_tokens)
    return flattened_tokens


def row_summary_tokens(
    model: Any,
    *,
    feature_cells: torch.Tensor,
    y_train: torch.Tensor,
) -> torch.Tensor:
    """Build flattened row-summary tokens with label and role conditioning."""

    row_summaries = row_summary_bytes(model, feature_cells)
    num_rows = int(feature_cells.shape[1])
    conditioned = (
        model.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(dtype=row_summaries.dtype)
    )
    row_pos = model._fourier_positions(
        num_positions=num_rows,
        embedding_size=int(row_summaries.shape[3]),
        device=row_summaries.device,
        dtype=row_summaries.dtype,
    )
    current_role_ids = _feature_flow.role_ids(
        batch_size=int(row_summaries.shape[0]),
        num_rows=num_rows,
        num_train_rows=int(y_train.shape[1]),
        device=row_summaries.device,
    )
    role_embed = model.y_role_embedding(current_role_ids).to(dtype=row_summaries.dtype)
    token_type = model.token_type_embedding.weight[_ROW_SUMMARY_TOKEN_ID].to(
        dtype=row_summaries.dtype
    )
    tokens = (
        row_summaries
        + conditioned.unsqueeze(2)
        + row_pos.unsqueeze(2)
        + role_embed.unsqueeze(2)
        + token_type.view(1, 1, 1, -1)
    )
    flattened_tokens = _flatten_summary_grid(tokens)
    model.trace_activation("post_row_summary_tokens", flattened_tokens)
    return flattened_tokens


def split_role_conditioned_column_summary_tokens(
    model: Any,
    *,
    conditioned_feature_cells: torch.Tensor,
    train_test_split_index: int,
) -> torch.Tensor:
    """Build separate train/test column summaries from conditioned full-cell tokens."""

    train_column_tokens = _column_summary_tokens_from_cells(
        model,
        conditioned_feature_cells[:, :train_test_split_index, :, :],
        trace_name="post_train_column_summary_tokens",
    )
    test_column_tokens = _column_summary_tokens_from_cells(
        model,
        conditioned_feature_cells[:, train_test_split_index:, :, :],
        trace_name="post_test_column_summary_tokens",
    )
    column_tokens = torch.cat([train_column_tokens, test_column_tokens], dim=1)
    model.trace_activation("post_split_column_summary_tokens", column_tokens)
    return column_tokens


def column_summary_tokens(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    conditioned_feature_cells: torch.Tensor | None = None,
    train_test_split_index: int | None = None,
) -> torch.Tensor:
    """Build flattened column-summary tokens."""

    if model.sandwich_column_summary_mode == "split_role_conditioned":
        if conditioned_feature_cells is None or train_test_split_index is None:
            raise RuntimeError(
                "split_role_conditioned column summaries require conditioned cells and "
                "train_test_split_index"
            )
        return split_role_conditioned_column_summary_tokens(
            model,
            conditioned_feature_cells=conditioned_feature_cells,
            train_test_split_index=train_test_split_index,
        )
    return _column_summary_tokens_from_cells(
        model,
        feature_cells,
        trace_name="post_column_summary_tokens",
    )


def full_cell_token_grid(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    y_train: torch.Tensor,
) -> torch.Tensor:
    """Build the conditioned full-cell token grid before flattening."""

    batch_size, num_rows, _num_features, _embedding_size = (
        int(feature_cells.shape[0]),
        int(feature_cells.shape[1]),
        int(feature_cells.shape[2]),
        int(feature_cells.shape[3]),
    )
    conditioned = (
        model.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(dtype=feature_cells.dtype)
    )
    current_role_ids = _feature_flow.role_ids(
        batch_size=batch_size,
        num_rows=num_rows,
        num_train_rows=int(y_train.shape[1]),
        device=feature_cells.device,
    )
    role_embed = model.y_role_embedding(current_role_ids).to(dtype=feature_cells.dtype)
    token_type = model.token_type_embedding.weight[_CELL_TOKEN_ID].to(dtype=feature_cells.dtype)
    full_cell_tokens = (
        feature_cells
        + conditioned.unsqueeze(2)
        + role_embed.unsqueeze(2)
        + token_type.view(1, 1, 1, -1)
    )
    model.trace_activation("post_full_cell_tokens", full_cell_tokens)
    return full_cell_tokens


def full_cell_tokens(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    y_train: torch.Tensor,
) -> torch.Tensor:
    """Build the flattened full-cell stream used by the hybrid sandwich readout."""

    batch_size, num_rows, num_features, _embedding_size = (
        int(feature_cells.shape[0]),
        int(feature_cells.shape[1]),
        int(feature_cells.shape[2]),
        int(feature_cells.shape[3]),
    )
    current_full_cell_token_grid = full_cell_token_grid(
        model,
        feature_cells,
        y_train=y_train,
    )
    full_cell_stream = current_full_cell_token_grid.reshape(
        batch_size,
        num_rows * num_features,
        model.d_icl,
    )
    model.trace_activation("post_full_cell_stream", full_cell_stream)
    return full_cell_stream


def summary_input_tokens(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    y_train: torch.Tensor,
    conditioned_feature_cells: torch.Tensor | None = None,
    train_test_split_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the concatenated summary input stream plus row-token grid."""

    row_tokens = row_summary_tokens(model, feature_cells=feature_cells, y_train=y_train)
    column_tokens = column_summary_tokens(
        model,
        feature_cells,
        conditioned_feature_cells=conditioned_feature_cells,
        train_test_split_index=train_test_split_index,
    )
    summary_tokens = torch.cat([row_tokens, column_tokens], dim=1)
    model.trace_activation("post_summary_input", summary_tokens)
    return summary_tokens, row_tokens


def train_row_tokens_for_class_memory(
    model: Any,
    raw_state: Any,
) -> torch.Tensor:
    """Build train-only row tokens for the optional class-memory path."""

    train_rows = int(raw_state.train_test_split_index)
    train_feature_cells = _feature_flow.feature_cells(
        model,
        raw_state.x_all[:, :train_rows, :],
        train_test_split_index=train_rows,
        feature_type_ids=raw_state.feature_type_ids,
    )
    train_row_tokens = row_summary_tokens(
        model,
        feature_cells=train_feature_cells,
        y_train=raw_state.y_train,
    )
    model.trace_activation("post_class_memory_train_row_tokens", train_row_tokens)
    return train_row_tokens


def build_class_memory_slots(
    model: Any,
    *,
    train_row_tokens: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build per-task class memory slots from train-only row summary tokens."""

    if model.class_memory_query is None or model.class_memory_builder is None:
        raise RuntimeError("class memory blocks are required when sandwich_use_class_memory=true")
    class_queries = model.class_memory_query[:, :num_classes, :].to(
        device=train_row_tokens.device,
        dtype=train_row_tokens.dtype,
    )
    class_queries = class_queries.expand(int(train_row_tokens.shape[0]), -1, -1)
    class_memory = model._cross_block(model.class_memory_builder, class_queries, train_row_tokens)
    model.trace_activation("post_class_memory_slots", class_memory)
    return class_memory


def build_classification_state(
    model: Any,
    feature_state: SandwichFeatureState,
) -> SandwichClassificationState:
    """Build the pre-Perceiver classification state for one sandwich batch."""

    feature_cells = feature_state.feature_cells
    raw_state = feature_state.raw_state
    y_train = raw_state.y_train
    current_full_cell_token_grid = full_cell_token_grid(model, feature_cells, y_train=y_train)
    current_full_cell_stream = current_full_cell_token_grid.reshape(
        int(current_full_cell_token_grid.shape[0]),
        int(current_full_cell_token_grid.shape[1]) * int(current_full_cell_token_grid.shape[2]),
        int(current_full_cell_token_grid.shape[3]),
    )
    model.trace_activation("post_full_cell_stream", current_full_cell_stream)
    current_summary_input, current_row_tokens = summary_input_tokens(
        model,
        feature_cells,
        y_train=y_train,
        conditioned_feature_cells=current_full_cell_token_grid,
        train_test_split_index=int(raw_state.train_test_split_index),
    )
    current_train_row_tokens_for_class_memory = (
        train_row_tokens_for_class_memory(model, raw_state)
        if model.sandwich_use_class_memory
        else None
    )
    return SandwichClassificationState(
        feature_state=feature_state,
        full_cell_stream=current_full_cell_stream,
        summary_input=current_summary_input,
        row_tokens=current_row_tokens,
        train_row_tokens_for_class_memory=current_train_row_tokens_for_class_memory,
    )


def validate_num_classes(model: Any, num_classes: int) -> None:
    """Validate that the direct multiclass head can represent the target task."""

    if num_classes < _MIN_CLASS_COUNT:
        raise RuntimeError(
            f"tabfoundry_sandwich requires at least {_MIN_CLASS_COUNT} classes, got {num_classes}"
        )
    if num_classes > model.many_class_base:
        raise RuntimeError(
            "tabfoundry_sandwich uses a direct multiclass head and requires "
            f"num_classes <= many_class_base={model.many_class_base}, got {num_classes}"
        )


def forward_logits(model: Any, classification_state: SandwichClassificationState) -> torch.Tensor:
    """Run the sandwich classification path from prepared classification state."""

    raw_state = classification_state.feature_state.raw_state
    feature_cells = classification_state.feature_state.feature_cells
    initial_input = torch.cat(
        [classification_state.full_cell_stream, classification_state.summary_input],
        dim=1,
    )
    model.trace_activation("post_perceiver_input", initial_input)
    latents = model.latent_seed.expand(int(raw_state.x_all.shape[0]), -1, -1)
    final_stage_index = len(model.perceiver_stages) - 1
    for index, stage in enumerate(model.perceiver_stages):
        stage = cast(_PerceiverStage, stage)
        uses_full_cell_refresh = (
            model.sandwich_last_stage_full_cell_refresh and index == final_stage_index
        )
        key_value = (
            initial_input
            if index == 0 or uses_full_cell_refresh
            else classification_state.summary_input
        )
        latents = model._cross_block(stage.input_read, latents, key_value)
        model.trace_activation(f"post_stage_{index}_cross", latents)
        for self_index, self_block in enumerate(stage.self_blocks):
            self_block = cast(_SelfAttentionBlock, self_block)
            latents = model._self_block(self_block, latents)
            model.trace_activation(f"post_stage_{index}_self_{self_index}", latents)
        model.trace_activation(f"post_stage_{index}_self", latents)
    batch_size = int(raw_state.x_all.shape[0])
    num_rows = int(feature_cells.shape[1])
    num_test_rows = num_rows - raw_state.train_test_split_index
    row_token_grid = classification_state.row_tokens.reshape(
        batch_size,
        num_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )
    test_queries = row_token_grid[:, raw_state.train_test_split_index :, :, :].reshape(
        batch_size,
        num_test_rows * model.summary_tokens_per_axis,
        model.d_icl,
    )
    test_rows = model._cross_block(model.latent_readout, test_queries, latents)
    model.trace_activation("post_latent_readout", test_rows)
    latent_readout_rows = test_rows.reshape(
        batch_size,
        num_test_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )
    test_rows = model._cross_block(
        model.cell_readout, test_rows, classification_state.full_cell_stream
    )
    model.trace_activation("post_test_readout", test_rows)
    cell_readout_rows = test_rows.reshape(
        batch_size,
        num_test_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )
    pooled_test_rows = pool_test_rows(
        model,
        latent_readout_rows=latent_readout_rows,
        cell_readout_rows=cell_readout_rows,
    )
    model.trace_activation("post_test_row_pool", pooled_test_rows)
    if model.sandwich_use_class_memory:
        if classification_state.train_row_tokens_for_class_memory is None:
            raise RuntimeError(
                "class memory requires train_row_tokens_for_class_memory in the classification state"
            )
        if model.class_memory_readout is None:
            raise RuntimeError(
                "class memory readout block is required when sandwich_use_class_memory=true"
            )
        class_memory = build_class_memory_slots(
            model,
            train_row_tokens=classification_state.train_row_tokens_for_class_memory,
            num_classes=int(raw_state.num_classes),
        )
        pooled_test_rows = model._cross_block(
            model.class_memory_readout, pooled_test_rows, class_memory
        )
        model.trace_activation("post_class_memory_readout", pooled_test_rows)
    return model.direct_head(pooled_test_rows)


def build_classification_output(
    *,
    logits: torch.Tensor,
    num_classes: int,
) -> ClassificationOutput:
    """Build the public classification output for the sandwich family."""

    return ClassificationOutput(
        logits=flatten_classification_output_rows(logits),
        num_classes=num_classes,
        class_probs=None,
    )
