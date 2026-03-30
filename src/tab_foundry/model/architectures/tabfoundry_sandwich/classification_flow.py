"""Classification forward helpers for the sandwich architecture family."""

from __future__ import annotations

from typing import Any, cast

import torch

from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows

from .blocks import _CrossAttentionBlock, _PerceiverStage, _SelfAttentionBlock
from .feature_flow import fourier_positions, role_ids
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


def row_summary_tokens(
    model: Any,
    *,
    feature_cells: torch.Tensor,
    y_train: torch.Tensor,
) -> torch.Tensor:
    """Build flattened row-summary tokens with label and role conditioning."""

    row_summaries = row_summary_bytes(model, feature_cells)
    num_rows = int(feature_cells.shape[1])
    conditioned = model.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
        dtype=row_summaries.dtype
    )
    row_pos = fourier_positions(
        num_positions=num_rows,
        embedding_size=int(row_summaries.shape[3]),
        device=row_summaries.device,
        dtype=row_summaries.dtype,
    )
    current_role_ids = role_ids(
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
    flattened_tokens = tokens.reshape(
        int(tokens.shape[0]),
        num_rows * model.summary_tokens_per_axis,
        int(tokens.shape[3]),
    )
    model.trace_activation("post_row_summary_tokens", flattened_tokens)
    return flattened_tokens


def column_summary_tokens(model: Any, feature_cells: torch.Tensor) -> torch.Tensor:
    """Build flattened column-summary tokens."""

    column_summaries = column_summary_bytes(model, feature_cells)
    token_type = model.token_type_embedding.weight[_COLUMN_SUMMARY_TOKEN_ID].to(
        dtype=column_summaries.dtype
    )
    tokens = column_summaries + token_type.view(1, 1, 1, -1)
    flattened_tokens = tokens.reshape(
        int(tokens.shape[0]),
        int(tokens.shape[1]) * model.summary_tokens_per_axis,
        int(tokens.shape[3]),
    )
    model.trace_activation("post_column_summary_tokens", flattened_tokens)
    return flattened_tokens


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
    conditioned = model.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
        dtype=feature_cells.dtype
    )
    current_role_ids = role_ids(
        batch_size=batch_size,
        num_rows=num_rows,
        num_train_rows=int(y_train.shape[1]),
        device=feature_cells.device,
    )
    role_embed = model.y_role_embedding(current_role_ids).to(dtype=feature_cells.dtype)
    token_type = model.token_type_embedding.weight[_CELL_TOKEN_ID].to(dtype=feature_cells.dtype)
    full_cell_token_grid = (
        feature_cells
        + conditioned.unsqueeze(2)
        + role_embed.unsqueeze(2)
        + token_type.view(1, 1, 1, -1)
    )
    model.trace_activation("post_full_cell_tokens", full_cell_token_grid)
    full_cell_stream = full_cell_token_grid.reshape(batch_size, num_rows * num_features, model.d_icl)
    model.trace_activation("post_full_cell_stream", full_cell_stream)
    return full_cell_stream


def summary_input_tokens(
    model: Any,
    feature_cells: torch.Tensor,
    *,
    y_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the concatenated summary input stream plus row-token grid."""

    row_tokens = row_summary_tokens(model, feature_cells=feature_cells, y_train=y_train)
    column_tokens = column_summary_tokens(model, feature_cells)
    summary_tokens = torch.cat([row_tokens, column_tokens], dim=1)
    model.trace_activation("post_summary_input", summary_tokens)
    return summary_tokens, row_tokens


def build_classification_state(
    model: Any,
    feature_state: SandwichFeatureState,
) -> SandwichClassificationState:
    """Build the pre-Perceiver classification state for one sandwich batch."""

    feature_cells = feature_state.feature_cells
    y_train = feature_state.raw_state.y_train
    current_full_cell_stream = full_cell_tokens(model, feature_cells, y_train=y_train)
    current_summary_input, current_row_tokens = summary_input_tokens(
        model,
        feature_cells,
        y_train=y_train,
    )
    return SandwichClassificationState(
        feature_state=feature_state,
        full_cell_stream=current_full_cell_stream,
        summary_input=current_summary_input,
        row_tokens=current_row_tokens,
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
    for index, stage in enumerate(model.perceiver_stages):
        stage = cast(_PerceiverStage, stage)
        key_value = initial_input if index == 0 else classification_state.summary_input
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
    test_rows = model._cross_block(model.cell_readout, test_rows, classification_state.full_cell_stream)
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
