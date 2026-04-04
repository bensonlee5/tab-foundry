from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from tab_foundry.model.architectures.tabfoundry_sandwich import blocks as sandwich_blocks
from tab_foundry.model.architectures.tabfoundry_sandwich import cell_likelihood as sandwich_cell_likelihood
from tab_foundry.model.architectures.tabfoundry_sandwich import classification_flow as sandwich_classification_flow
from tab_foundry.model.architectures.tabfoundry_sandwich import feature_flow as sandwich_feature_flow
from tab_foundry.model.architectures.tabfoundry_sandwich import model as sandwich_model
from tab_foundry.model.components.rational import RationalActivation
from tab_foundry.model.outputs import (
    CellLikelihoodOutput,
    validate_cell_likelihood_output_contract,
)
from tab_foundry.task_batching import collate_task_batch
from tab_foundry.types import TaskBatch


_DEFAULT_FEATURE_TYPES = ["floating", "integer", "bool", "string_binary"]
_SECONDARY_FEATURE_TYPES = ["integer", "floating", "unknown", "bool"]


def _batch(
    *,
    num_classes: int = 3,
    include_feature_types: bool = True,
    feature_types: list[str] | None = None,
) -> TaskBatch:
    metadata = {"source": "unit_test"}
    if include_feature_types:
        metadata["feature_types"] = list(feature_types or _DEFAULT_FEATURE_TYPES)
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, float("nan"), 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 2], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata=metadata,
        num_classes=num_classes,
    )


def _finite_batch(
    *,
    num_classes: int = 3,
    feature_types: list[str] | None = None,
) -> TaskBatch:
    metadata = {
        "source": "unit_test",
        "feature_types": list(feature_types or _DEFAULT_FEATURE_TYPES),
    }
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 0.5, 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 2], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata=metadata,
        num_classes=num_classes,
    )


def _batched_inputs() -> tuple[torch.Tensor, torch.Tensor, int]:
    x_all = torch.tensor(
        [
            [
                [1.0, 2.0, float("nan"), 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0, -1.0],
                [0.5, 0.5, 0.5, 0.5],
                [3.0, -1.0, 2.0, 4.0],
            ],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=torch.int64,
    )
    return x_all, y_train, 3


def _batched_feature_types() -> list[list[str]]:
    return [list(_DEFAULT_FEATURE_TYPES), list(_SECONDARY_FEATURE_TYPES)]


def _model(
    *,
    many_class_base: int = 4,
    sandwich_ff_expansion: int = 2,
    sandwich_activation: str = "gelu",
    sandwich_block_norm: str = "layernorm",
    sandwich_summary_tokens_per_axis: int = 4,
    sandwich_self_attention_per_cross: int = 4,
    feature_type_conditioning: str = "film",
    input_normalization: str = "train_zscore_clip",
) -> TabFoundrySandwichClassifier:
    return TabFoundrySandwichClassifier(
        d_icl=32,
        input_normalization=input_normalization,
        many_class_base=many_class_base,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=sandwich_ff_expansion,
        sandwich_activation=sandwich_activation,
        sandwich_block_norm=sandwich_block_norm,
        sandwich_summary_tokens_per_axis=sandwich_summary_tokens_per_axis,
        sandwich_self_attention_per_cross=sandwich_self_attention_per_cross,
        sandwich_pre_row_attention_layers=1,
        sandwich_pre_column_attention_layers=1,
        sandwich_pre_column_inducing_tokens=8,
        feature_type_conditioning=feature_type_conditioning,
    )


def _raw_state(
    model: TabFoundrySandwichClassifier,
    batch: TaskBatch,
):
    x_all, y_train, y_test, train_test_split_index = model._prepare_task_inputs(batch)
    feature_type_ids = sandwich_feature_flow.feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    return sandwich_feature_flow.build_raw_input_state(
        x_all=x_all,
        y_train=y_train,
        y_test=y_test,
        train_test_split_index=train_test_split_index,
        num_classes=model._task_num_classes(batch),
        feature_type_ids=feature_type_ids,
    )


def _cross_block_state_dict_keys(
    prefix: str,
    *,
    activation: str = "gelu",
    block_norm: str = "layernorm",
) -> list[str]:
    keys = [
        f"{prefix}.attn.in_proj_bias",
        f"{prefix}.attn.in_proj_weight",
        f"{prefix}.attn.out_proj.bias",
        f"{prefix}.attn.out_proj.weight",
        f"{prefix}.ff.0.bias",
        f"{prefix}.ff.0.weight",
    ]
    if activation == "rational":
        keys.extend([f"{prefix}.ff.1.denominator", f"{prefix}.ff.1.numerator"])
    keys.extend([f"{prefix}.ff.2.bias", f"{prefix}.ff.2.weight"])
    if block_norm == "layernorm":
        keys.extend(
            [
                f"{prefix}.ff_norm.bias",
                f"{prefix}.ff_norm.weight",
                f"{prefix}.kv_norm.bias",
                f"{prefix}.kv_norm.weight",
                f"{prefix}.query_norm.bias",
                f"{prefix}.query_norm.weight",
            ]
        )
    return keys


def _self_block_state_dict_keys(
    prefix: str,
    *,
    activation: str = "gelu",
    block_norm: str = "layernorm",
) -> list[str]:
    keys = [
        f"{prefix}.attn.in_proj_bias",
        f"{prefix}.attn.in_proj_weight",
        f"{prefix}.attn.out_proj.bias",
        f"{prefix}.attn.out_proj.weight",
        f"{prefix}.ff.0.bias",
        f"{prefix}.ff.0.weight",
    ]
    if block_norm == "layernorm":
        keys.extend([f"{prefix}.attn_norm.bias", f"{prefix}.attn_norm.weight"])
    if activation == "rational":
        keys.extend([f"{prefix}.ff.1.denominator", f"{prefix}.ff.1.numerator"])
    keys.extend([f"{prefix}.ff.2.bias", f"{prefix}.ff.2.weight"])
    if block_norm == "layernorm":
        keys.extend([f"{prefix}.ff_norm.bias", f"{prefix}.ff_norm.weight"])
    return keys


def _expected_sandwich_state_dict_keys(
    *,
    activation: str = "gelu",
    block_norm: str = "layernorm",
) -> list[str]:
    expected = [
        "cell_bos",
        "column_summary_query",
        "direct_head.net.0.bias",
        "direct_head.net.0.weight",
        "direct_head.net.2.bias",
        "direct_head.net.2.weight",
        "discrete_oov.bias",
        "discrete_oov.weight",
        "discrete_query.bias",
        "discrete_query.weight",
        "feature_encoder.linear.weight",
        "feature_type_film.params.weight",
        "gaussian_head.0.bias",
        "gaussian_head.0.weight",
        "gaussian_head.2.bias",
        "gaussian_head.2.weight",
        "integer_gate.0.bias",
        "integer_gate.0.weight",
        "integer_gate.2.bias",
        "integer_gate.2.weight",
        "latent_seed",
        "pre_column_attention_blocks.0.inducing_seed",
        "row_summary_query",
        "test_row_pool_query",
        "token_type_embedding.weight",
        "y_conditioner.embedding.weight",
        "y_conditioner.test_token",
        "y_role_embedding.weight",
    ]
    for index in range(2):
        expected.extend(
            _self_block_state_dict_keys(
                f"cell_decoder_blocks.{index}",
                activation=activation,
                block_norm=block_norm,
            )
        )
        expected.extend(
            _cross_block_state_dict_keys(
                f"perceiver_stages.{index}.input_read",
                activation=activation,
                block_norm=block_norm,
            )
        )
        for self_index in range(4):
            expected.extend(
                _self_block_state_dict_keys(
                    f"perceiver_stages.{index}.self_blocks.{self_index}",
                    activation=activation,
                    block_norm=block_norm,
                )
            )
    expected.extend(
        _cross_block_state_dict_keys("cell_readout", activation=activation, block_norm=block_norm)
    )
    expected.extend(
        _cross_block_state_dict_keys(
            "column_summary_builder",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _cross_block_state_dict_keys(
            "latent_readout",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _self_block_state_dict_keys(
            "pre_column_attention_blocks.0.inducing_self",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _cross_block_state_dict_keys(
            "pre_column_attention_blocks.0.rows_from_inducing",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _cross_block_state_dict_keys(
            "pre_column_attention_blocks.0.rows_to_inducing",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _self_block_state_dict_keys(
            "pre_row_attention_blocks.0",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _cross_block_state_dict_keys(
            "row_summary_builder",
            activation=activation,
            block_norm=block_norm,
        )
    )
    expected.extend(
        _cross_block_state_dict_keys("test_row_pool", activation=activation, block_norm=block_norm)
    )
    return sorted(expected)


def test_tabfoundry_sandwich_forward_shapes() -> None:
    output = _model()(_batch())

    assert output.logits is not None
    assert output.class_probs is None
    assert output.num_classes == 3
    assert tuple(output.logits.shape) == (2, 4)


def test_tabfoundry_sandwich_forward_batched_shapes() -> None:
    model = _model()
    x_all, y_train, train_test_split_index = _batched_inputs()

    logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=train_test_split_index,
        feature_types=_batched_feature_types(),
    )

    assert tuple(logits.shape) == (2, 2, 4)
    assert torch.isfinite(logits).all()


def test_tabfoundry_sandwich_accepts_zero_self_attention_per_cross() -> None:
    model = _model(sandwich_self_attention_per_cross=0)
    output = model(_batch())

    assert tuple(output.logits.shape) == (2, 4)
    assert torch.isfinite(output.logits).all()


def test_tabfoundry_sandwich_uses_hybrid_full_cell_and_summary_inputs() -> None:
    model = _model()
    raw_state = _raw_state(model, _batch())
    feature_state = sandwich_feature_flow.build_feature_state(model, raw_state)
    classification_state = sandwich_classification_flow.build_classification_state(
        model,
        feature_state,
    )
    initial_input = torch.cat(
        [classification_state.full_cell_stream, classification_state.summary_input],
        dim=1,
    )

    assert tuple(classification_state.full_cell_stream.shape) == (1, 20, 32)
    assert tuple(classification_state.summary_input.shape) == (1, 36, 32)
    assert tuple(initial_input.shape) == (1, 56, 32)
    assert tuple(classification_state.row_tokens.shape) == (1, 20, 32)
    assert tuple(model.latent_seed.shape) == (1, 12, 32)
    assert tuple(raw_state.y_train.shape) == (1, 3)
    assert torch.allclose(
        classification_state.summary_input[:, :20, :],
        classification_state.row_tokens,
    )


def test_tabfoundry_sandwich_fuses_label_query_state_into_row_summaries() -> None:
    model = _model()
    raw_state = _raw_state(model, _batch())
    feature_state = sandwich_feature_flow.build_feature_state(model, raw_state)

    raw_row_summaries = sandwich_classification_flow.row_summary_bytes(
        model,
        feature_state.feature_cells,
    )
    row_tokens = sandwich_classification_flow.row_summary_tokens(
        model,
        feature_cells=feature_state.feature_cells,
        y_train=raw_state.y_train,
    )

    assert tuple(raw_row_summaries.shape) == (1, 5, 4, 32)
    assert tuple(row_tokens.shape) == (1, 20, 32)
    assert not torch.allclose(raw_row_summaries.reshape(1, 20, 32), row_tokens)


def test_tabfoundry_sandwich_broadcasts_row_conditioning_into_full_cell_stream() -> None:
    model = _model()
    raw_state = _raw_state(model, _batch())
    feature_state = sandwich_feature_flow.build_feature_state(model, raw_state)

    full_cell_stream = sandwich_classification_flow.full_cell_tokens(
        model,
        feature_state.feature_cells,
        y_train=raw_state.y_train,
    )
    full_cell_token_grid = full_cell_stream.reshape_as(feature_state.feature_cells)
    conditioning_delta = full_cell_token_grid - feature_state.feature_cells

    assert tuple(full_cell_stream.shape) == (1, 20, 32)
    torch.testing.assert_close(conditioning_delta[:, 0, 0, :], conditioning_delta[:, 0, 1, :])
    torch.testing.assert_close(conditioning_delta[:, 4, 0, :], conditioning_delta[:, 4, 3, :])
    assert not torch.allclose(conditioning_delta[:, 0, 0, :], conditioning_delta[:, 4, 0, :])


def test_tabfoundry_sandwich_pools_test_readout_facets_with_learned_query() -> None:
    model = _model()
    x_all, y_train, train_test_split_index = _batched_inputs()
    feature_type_ids = sandwich_feature_flow.feature_type_ids_from_metadata(
        {
            "task_members": [
                {"feature_types": feature_types} for feature_types in _batched_feature_types()
            ]
        },
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    raw_state = sandwich_feature_flow.build_raw_input_state(
        x_all=x_all,
        y_train=y_train,
        y_test=None,
        train_test_split_index=train_test_split_index,
        num_classes=3,
        feature_type_ids=feature_type_ids,
    )
    feature_state = sandwich_feature_flow.build_feature_state(model, raw_state)
    classification_state = sandwich_classification_flow.build_classification_state(
        model,
        feature_state,
    )
    initial_input = torch.cat(
        [classification_state.full_cell_stream, classification_state.summary_input],
        dim=1,
    )
    latents = model.latent_seed.expand(int(x_all.shape[0]), -1, -1)
    for index, stage in enumerate(model.perceiver_stages):
        key_value = initial_input if index == 0 else classification_state.summary_input
        latents = model._cross_block(stage.input_read, latents, key_value)
        for self_block in stage.self_blocks:
            latents = model._self_block(self_block, latents)
    batch_size = int(x_all.shape[0])
    num_rows = int(feature_state.feature_cells.shape[1])
    num_test_rows = num_rows - train_test_split_index
    row_token_grid = classification_state.row_tokens.reshape(
        batch_size,
        num_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )
    test_queries = row_token_grid[:, train_test_split_index:, :, :].reshape(
        batch_size,
        num_test_rows * model.summary_tokens_per_axis,
        model.d_icl,
    )
    latent_readout = model._cross_block(model.latent_readout, test_queries, latents)
    cell_readout = model._cross_block(
        model.cell_readout,
        latent_readout,
        classification_state.full_cell_stream,
    )
    latent_readout_rows = latent_readout.reshape(
        batch_size,
        num_test_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )
    cell_readout_rows = cell_readout.reshape(
        batch_size,
        num_test_rows,
        model.summary_tokens_per_axis,
        model.d_icl,
    )

    pooled = model._pool_test_rows(
        latent_readout_rows=latent_readout_rows,
        cell_readout_rows=cell_readout_rows,
    )

    assert tuple(pooled.shape) == (2, 2, 32)
    assert not torch.allclose(pooled, cell_readout_rows.mean(dim=2))


def test_tabfoundry_sandwich_pre_column_isab_uses_inducing_bottleneck() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=4,
        sandwich_pre_row_attention_layers=0,
        sandwich_pre_column_attention_layers=1,
        sandwich_pre_column_inducing_tokens=3,
    )
    raw_state = _raw_state(model, _batch())
    isab_block = model.pre_column_attention_blocks[0]
    events: list[tuple[str, tuple[int, ...], tuple[int, ...] | None]] = []
    original_cross_block = model._cross_block
    original_self_block = model._self_block

    def _recording_cross_block(block, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if block is isab_block.rows_to_inducing:
            events.append(("rows_to_inducing", tuple(query.shape), tuple(key_value.shape)))
        elif block is isab_block.rows_from_inducing:
            events.append(("rows_from_inducing", tuple(query.shape), tuple(key_value.shape)))
        return original_cross_block(block, query, key_value)

    def _recording_self_block(block, hidden: torch.Tensor) -> torch.Tensor:
        if block is isab_block.inducing_self:
            events.append(("inducing_self", tuple(hidden.shape), None))
        return original_self_block(block, hidden)

    model._cross_block = _recording_cross_block  # type: ignore[method-assign]
    model._self_block = _recording_self_block  # type: ignore[method-assign]

    feature_cells = sandwich_feature_flow.feature_cells(
        model,
        raw_state.x_all,
        train_test_split_index=raw_state.train_test_split_index,
        feature_type_ids=raw_state.feature_type_ids,
    )

    assert tuple(feature_cells.shape) == (1, 5, 4, 32)
    assert events == [
        ("rows_to_inducing", (4, 3, 32), (4, 5, 32)),
        ("inducing_self", (4, 3, 32), None),
        ("rows_from_inducing", (4, 5, 32), (4, 3, 32)),
    ]


def test_tabfoundry_sandwich_initializes_latent_seed_with_truncated_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], float, float, float, float]] = []
    original_init = sandwich_model._init_truncated_normal_

    def _recording_init(
        tensor: torch.Tensor,
        *,
        mean: float,
        std: float,
        a: float,
        b: float,
    ) -> torch.Tensor:
        calls.append((tuple(tensor.shape), mean, std, a, b))
        return original_init(tensor, mean=mean, std=std, a=a, b=b)

    monkeypatch.setattr(sandwich_model, "_init_truncated_normal_", _recording_init)
    monkeypatch.setattr(sandwich_blocks, "_init_truncated_normal_", _recording_init)

    model = _model()

    assert calls == [
        ((1, 8, 32), 0.0, 0.02, -2.0, 2.0),
        ((1, 12, 32), 0.0, 0.02, -2.0, 2.0),
    ]
    assert tuple(model.latent_seed.shape) == (1, 12, 32)
    assert torch.isfinite(model.latent_seed).all()


def test_tabfoundry_sandwich_preserves_representative_state_dict_keys_and_strict_load() -> None:
    reference = _model()
    checkpointed = _model()

    expected_keys = _expected_sandwich_state_dict_keys()
    observed_keys = sorted(reference.state_dict().keys())

    assert observed_keys == expected_keys
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    assert sorted(checkpointed.state_dict().keys()) == expected_keys


def test_tabfoundry_sandwich_rational_norm_free_wiring_and_state_dict_keys() -> None:
    model = _model(sandwich_activation="rational", sandwich_block_norm="none")

    assert isinstance(model.pre_row_attention_blocks[0].ff[1], RationalActivation)
    assert isinstance(model.row_summary_builder.query_norm, torch.nn.Identity)
    assert isinstance(model.pre_column_attention_blocks[0].inducing_self.attn_norm, torch.nn.Identity)
    assert sorted(model.state_dict().keys()) == _expected_sandwich_state_dict_keys(
        activation="rational",
        block_norm="none",
    )


def test_tabfoundry_sandwich_rational_norm_free_forward_shapes() -> None:
    output = _model(sandwich_activation="rational", sandwich_block_norm="none")(_batch())

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert torch.isfinite(output.logits).all()


def test_tabfoundry_sandwich_runs_repeated_cross_then_self_stages() -> None:
    model = _model()
    order: list[str] = []
    original_cross_block = model._cross_block
    original_self_block = model._self_block
    stage_reads = {
        id(stage.input_read): f"cross_{index}" for index, stage in enumerate(model.perceiver_stages)
    }
    stage_latent_blocks = {
        id(latent_block): f"self_{index}_{self_index}"
        for index, stage in enumerate(model.perceiver_stages)
        for self_index, latent_block in enumerate(stage.self_blocks)
    }

    def _recording_cross_block(block, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        stage_name = stage_reads.get(id(block))
        if stage_name is not None:
            order.append(stage_name)
        elif block is model.latent_readout:
            order.append("latent_readout")
        elif block is model.cell_readout:
            order.append("cell_readout")
        elif block is model.test_row_pool:
            order.append("test_row_pool")
        return original_cross_block(block, query, key_value)

    def _recording_self_block(block, hidden: torch.Tensor) -> torch.Tensor:
        stage_name = stage_latent_blocks.get(id(block))
        if stage_name is not None:
            order.append(stage_name)
        return original_self_block(block, hidden)

    model._cross_block = _recording_cross_block  # type: ignore[method-assign]
    model._self_block = _recording_self_block  # type: ignore[method-assign]

    _ = model(_batch())

    assert order == [
        "cross_0",
        "self_0_0",
        "self_0_1",
        "self_0_2",
        "self_0_3",
        "cross_1",
        "self_1_0",
        "self_1_1",
        "self_1_2",
        "self_1_3",
        "latent_readout",
        "cell_readout",
        "test_row_pool",
    ]


def test_tabfoundry_sandwich_uses_full_cell_context_only_on_stage_zero() -> None:
    model = _model()
    observed_context_lengths: list[int] = []
    original_cross_block = model._cross_block
    stage_reads = {
        id(stage.input_read): index for index, stage in enumerate(model.perceiver_stages)
    }

    def _recording_cross_block(block, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        stage_index = stage_reads.get(id(block))
        if stage_index is not None:
            observed_context_lengths.append(int(key_value.shape[1]))
        return original_cross_block(block, query, key_value)

    model._cross_block = _recording_cross_block  # type: ignore[method-assign]

    _ = model(_batch())

    assert observed_context_lengths[:2] == [56, 36]


def test_tabfoundry_sandwich_exposes_activation_trace_hooks() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=4,
        sandwich_pre_row_attention_layers=1,
        sandwich_pre_column_attention_layers=1,
    )
    model.enable_activation_trace()

    _ = model(_batch())
    trace = model.flush_activation_trace_stats()

    assert trace is not None
    assert "post_feature_encoder" in trace
    assert "post_perceiver_input" in trace
    assert "post_pre_row_attention_0" in trace
    assert "post_pre_column_attention_0" in trace
    assert "post_pre_perceiver_cells" in trace
    assert "post_full_cell_stream" in trace
    assert "post_stage_0_cross" in trace
    assert "post_stage_0_self_0" in trace
    assert "post_stage_0_self" in trace
    assert "post_latent_readout" in trace
    assert "post_test_readout" in trace
    assert "post_test_row_pool" in trace


def test_tabfoundry_sandwich_rational_norm_free_exposes_activation_trace_hooks() -> None:
    model = _model(sandwich_activation="rational", sandwich_block_norm="none")
    model.enable_activation_trace()

    _ = model(_batch())
    trace = model.flush_activation_trace_stats()

    assert trace is not None
    assert "post_pre_row_attention_0" in trace
    assert "post_pre_column_attention_0" in trace
    assert "post_stage_0_cross" in trace
    assert "post_test_row_pool" in trace


def test_tabfoundry_sandwich_layers_count_matches_repeated_stages() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=3,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=4,
        sandwich_pre_row_attention_layers=1,
        sandwich_pre_column_attention_layers=1,
    )

    assert len(model.perceiver_stages) == 3
    assert all(len(stage.self_blocks) == 4 for stage in model.perceiver_stages)


def test_tabfoundry_sandwich_allows_configurable_self_attention_per_cross() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=2,
        sandwich_pre_row_attention_layers=1,
        sandwich_pre_column_attention_layers=1,
    )

    assert all(len(stage.self_blocks) == 2 for stage in model.perceiver_stages)


def test_tabfoundry_sandwich_allows_configurable_pre_perceiver_attention_layers() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=4,
        sandwich_pre_row_attention_layers=2,
        sandwich_pre_column_attention_layers=1,
        sandwich_pre_column_inducing_tokens=3,
    )

    assert len(model.pre_row_attention_blocks) == 2
    assert len(model.pre_column_attention_blocks) == 1
    assert model.pre_column_inducing_tokens == 3


def test_tabfoundry_sandwich_requires_feature_types_for_forward() -> None:
    with pytest.raises(RuntimeError, match="feature_types"):
        _ = _model()(_batch(include_feature_types=False))


def test_tabfoundry_sandwich_requires_feature_types_for_forward_batched() -> None:
    model = _model()
    x_all, y_train, train_test_split_index = _batched_inputs()

    with pytest.raises(ValueError, match="feature_types"):
        _ = model.forward_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
            feature_types=None,
        )


def test_tabfoundry_sandwich_forward_batched_matches_forward_with_feature_types() -> None:
    model = _model()
    batch = _batch()
    x_all, y_train, _y_test, train_test_split_index = model._prepare_task_inputs(batch)

    output = model(batch)
    logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=train_test_split_index,
        feature_types=list(_DEFAULT_FEATURE_TYPES),
    )

    assert output.logits is not None
    assert torch.allclose(logits.squeeze(0), output.logits, atol=1.0e-6, rtol=1.0e-5)


def test_tabfoundry_sandwich_supports_task_batched_feature_type_metadata() -> None:
    model = _model()
    batch = collate_task_batch(
        [
            _batch(feature_types=_DEFAULT_FEATURE_TYPES),
            _batch(feature_types=_SECONDARY_FEATURE_TYPES),
        ],
        requested_task_batch_size=2,
    )

    output = model(batch)

    assert output.logits is not None
    assert tuple(output.logits.shape) == (4, 4)


def test_tabfoundry_sandwich_rejects_missing_task_member_feature_types() -> None:
    model = _model()
    batch = collate_task_batch(
        [
            _batch(feature_types=_DEFAULT_FEATURE_TYPES),
            _batch(include_feature_types=False),
        ],
        requested_task_batch_size=2,
    )

    with pytest.raises(RuntimeError, match="task_members\\[1\\]\\.feature_types"):
        _ = model(batch)


def test_tabfoundry_sandwich_rejects_true_many_class_batches() -> None:
    model = _model(many_class_base=3)

    with pytest.raises(RuntimeError, match="direct multiclass head"):
        _ = model(_batch(num_classes=5))


def test_tabfoundry_sandwich_accepts_five_class_batches_on_evolved_surface() -> None:
    model = _model(
        many_class_base=10,
        sandwich_summary_tokens_per_axis=3,
        feature_type_conditioning="film",
    )

    output = model(_batch(num_classes=5))

    assert output.logits is not None
    assert output.num_classes == 5
    assert tuple(output.logits.shape) == (2, 10)


def test_tabfoundry_sandwich_accepts_ten_class_batches_on_evolved_surface() -> None:
    model = _model(
        many_class_base=10,
        sandwich_summary_tokens_per_axis=3,
        feature_type_conditioning="film",
    )

    output = model(_batch(num_classes=10))

    assert output.logits is not None
    assert output.num_classes == 10
    assert tuple(output.logits.shape) == (2, 10)


def test_tabfoundry_sandwich_feature_type_film_changes_encoded_cells() -> None:
    model = _model(feature_type_conditioning="film")
    assert model.feature_type_film is not None
    with torch.no_grad():
        params = model.feature_type_film.params.weight
        params.zero_()
        params[0, 0] = 0.25
        params[1, 1] = -0.5
        params[2, 32 + 2] = 0.75
        params[3, 32 + 3] = -1.0
    batch = _batch()
    raw_state = _raw_state(model, batch)
    default_ids = sandwich_feature_flow.feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(raw_state.x_all.shape[0]),
        num_features=int(raw_state.x_all.shape[2]),
        device=raw_state.x_all.device,
    )
    secondary_ids = sandwich_feature_flow.feature_type_ids_from_metadata(
        {"feature_types": list(_SECONDARY_FEATURE_TYPES)},
        batch_size=int(raw_state.x_all.shape[0]),
        num_features=int(raw_state.x_all.shape[2]),
        device=raw_state.x_all.device,
    )

    default_cells = sandwich_feature_flow.feature_cells(
        model,
        raw_state.x_all,
        train_test_split_index=raw_state.train_test_split_index,
        feature_type_ids=default_ids,
    )
    secondary_cells = sandwich_feature_flow.feature_cells(
        model,
        raw_state.x_all,
        train_test_split_index=raw_state.train_test_split_index,
        feature_type_ids=secondary_ids,
    )

    assert not torch.allclose(default_cells, secondary_cells)


def test_tabfoundry_sandwich_forward_cell_likelihood_emits_typed_payloads() -> None:
    model = _model()
    raw_state = _raw_state(model, _batch())
    feature_state = sandwich_feature_flow.build_feature_state(model, raw_state)

    output = sandwich_cell_likelihood.forward_cell_likelihood(model, feature_state)

    assert validate_cell_likelihood_output_contract(
        output,
        expected_shape=(1, 5, 4),
        context="sandwich unit test",
    ) == (1, 5, 4)
    assert output.bpc is not None
    assert output.bpf is not None
    assert output.floating_predictions is not None
    assert output.integer_predictions is not None
    assert output.categorical_predictions is not None
    assert torch.isfinite(output.bpc)
    assert torch.isfinite(output.bpf)
    assert int(torch.isnan(output.per_cell_bits).sum().item()) == 1
    assert output.aux_metrics is not None
    assert 0.0 <= output.aux_metrics["acc"] <= 1.0
    assert output.aux_metrics["bpc_cell_count"] == pytest.approx(19.0)
    assert output.aux_metrics["bpf_feature_count"] == pytest.approx(4.0)
    assert output.aux_metrics["excluded_non_finite_cell_count"] == pytest.approx(1.0)
    assert len(output.floating_predictions) == 1
    assert len(output.integer_predictions) == 1
    assert len(output.categorical_predictions) == 2
    integer_prediction = output.integer_predictions[0]
    assert integer_prediction.feature_index == 1
    assert tuple(integer_prediction.discrete_logits.shape) == (5, 4)
    assert tuple(integer_prediction.support_values.shape) == (3,)


def test_tabfoundry_sandwich_forward_cell_likelihood_rejects_all_non_finite_targets() -> None:
    model = _model()
    batch = TaskBatch(
        x_train=torch.full((3, 2), float("nan"), dtype=torch.float32),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.full((2, 2), float("nan"), dtype=torch.float32),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata={"source": "unit_test", "feature_types": ["floating", "floating"]},
        num_classes=2,
    )

    with pytest.raises(RuntimeError, match="at least one finite target cell"):
        _ = model.forward_cell_likelihood(batch)


def test_tabfoundry_sandwich_forward_cell_likelihood_rejects_num_classes_above_many_class_base() -> None:
    model = _model(many_class_base=2)

    with pytest.raises(RuntimeError, match="num_classes <= many_class_base=2"):
        _ = model.forward_cell_likelihood(_batch(num_classes=3))


def test_tabfoundry_sandwich_forward_cell_likelihood_infers_num_classes_from_test_labels() -> None:
    model = _model()
    batch = TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 0.5, 4.0],
                [2.0, 1.0, 3.0, 0.0],
                [0.5, -1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 0, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [1.5, 2.5, 0.0, -1.0],
                [0.0, -0.5, 1.5, 2.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 1], dtype=torch.int64),
        metadata={"source": "unit_test", "feature_types": list(_DEFAULT_FEATURE_TYPES)},
        num_classes=None,
    )

    output = model.forward_cell_likelihood(batch)

    assert output.bpc is not None
    assert torch.isfinite(output.bpc)


def test_tabfoundry_sandwich_cell_bpc_honors_input_normalization() -> None:
    base_model = _model(input_normalization="none")
    normalized_model = _model(input_normalization="train_zscore_clip")
    normalized_model.load_state_dict(base_model.state_dict())
    batch = _finite_batch()

    base_output = base_model.forward_cell_likelihood(batch)
    normalized_output = normalized_model.forward_cell_likelihood(batch)

    assert base_output.bpc is not None
    assert normalized_output.bpc is not None
    assert not torch.allclose(base_output.per_cell_bits, normalized_output.per_cell_bits)
    assert float(base_output.bpc.detach().item()) != pytest.approx(
        float(normalized_output.bpc.detach().item())
    )


def test_tabfoundry_sandwich_forward_dispatches_to_cell_bpc_surface() -> None:
    model = _model()
    model.set_loss_surface("cell_bpc")

    output = model(_batch())

    assert isinstance(output, CellLikelihoodOutput)
