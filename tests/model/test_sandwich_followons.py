from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.grid_sandwich import GridSandwichClassifier
from tab_foundry.model.architectures.routed_sandwich import RoutedSandwichClassifier
from tab_foundry.types import TaskBatch


_FEATURE_TYPES = ["floating", "integer", "bool"]


def _task_batch(*, num_classes: int = 2) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 1.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [7.0, 8.0, 1.0],
                [9.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([1, 0], dtype=torch.int64),
        metadata={"source": "unit_test", "feature_types": list(_FEATURE_TYPES)},
        num_classes=num_classes,
    )


def _batched_inputs() -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    batch = _task_batch()
    x_all = torch.cat([batch.x_train, batch.x_test], dim=0).unsqueeze(0)
    y_train = batch.y_train.unsqueeze(0)
    return x_all, y_train, int(batch.x_train.shape[0]), list(_FEATURE_TYPES)


def _routed_model(*, routed_direct_cell_bypass: bool = False) -> RoutedSandwichClassifier:
    return RoutedSandwichClassifier(
        d_icl=32,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=8,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        routed_row_summary_tokens=2,
        routed_column_summary_tokens=1,
        routed_evidence_tokens=4,
        routed_direct_cell_bypass=routed_direct_cell_bypass,
    )


def _routed_classification_state(
    model: RoutedSandwichClassifier,
    batch: TaskBatch,
):
    num_classes = model._task_num_classes(batch)
    x_all, y_train, y_test, train_test_split_index = model._prepare_task_inputs(batch)
    feature_type_ids = model._feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    raw_state = model._build_raw_input_state(
        x_all=x_all,
        y_train=y_train,
        y_test=y_test,
        train_test_split_index=train_test_split_index,
        num_classes=num_classes,
        feature_type_ids=feature_type_ids,
    )
    feature_state = model._build_feature_state(raw_state)
    return model._build_classification_state(feature_state)


def _grid_model(
    *,
    sandwich_pre_row_attention_layers: int = 1,
    sandwich_pre_column_attention_layers: int = 1,
    grid_residual_mode: str = "prenorm",
    grid_attention_mode: str = "standard",
    grid_ffn_mode: str = "gelu",
    grid_recurrence_steps: int | None = None,
) -> GridSandwichClassifier:
    return GridSandwichClassifier(
        d_icl=32,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
        sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
        sandwich_pre_column_inducing_tokens=8,
        grid_residual_mode=grid_residual_mode,
        grid_attention_mode=grid_attention_mode,
        grid_ffn_mode=grid_ffn_mode,
        grid_recurrence_steps=grid_recurrence_steps,
    )


def test_routed_sandwich_forward_and_forward_batched_shapes_match() -> None:
    model = _routed_model()
    batch = _task_batch()
    x_all, y_train, split_index, feature_types = _batched_inputs()

    output = model(batch)
    batched_logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=split_index,
        feature_types=feature_types,
    )

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert output.num_classes == 2
    assert tuple(batched_logits.shape) == (1, 2, 4)
    assert torch.allclose(output.logits, batched_logits.squeeze(0), atol=1e-5, rtol=1e-5)
    assert model.test_query_seed.shape == (1, 1, 2, 32)
    assert model.deepnorm_residual_depth == 11
    assert model.deepnorm_alpha == pytest.approx((2.0 * 11.0) ** 0.25)
    assert model.deepnorm_beta == pytest.approx((8.0 * 11.0) ** (-0.25))


def test_routed_sandwich_requires_feature_types_for_forward_batched() -> None:
    model = _routed_model()
    x_all, y_train, split_index, _feature_types = _batched_inputs()

    with pytest.raises(ValueError, match="feature_types"):
        _ = model.forward_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=split_index,
            feature_types=None,  # type: ignore[arg-type]
        )


def test_routed_sandwich_rejects_non_classification_loss_surface() -> None:
    model = _routed_model()

    with pytest.raises(ValueError, match="classification"):
        model.set_loss_surface("cell_bpc")


@pytest.mark.parametrize("routed_direct_cell_bypass", [False, True])
def test_routed_sandwich_stage_inputs_follow_direct_cell_bypass(
    monkeypatch: pytest.MonkeyPatch,
    routed_direct_cell_bypass: bool,
) -> None:
    model = _routed_model(routed_direct_cell_bypass=routed_direct_cell_bypass)
    batch = _task_batch()
    classification_state = _routed_classification_state(model, batch)
    stage0_block = model.perceiver_stages[0].input_read
    stage1_block = model.perceiver_stages[1].input_read
    recorded_inputs: dict[str, torch.Tensor] = {}
    original_routed_cross_block = model._routed_cross_block

    def _recording_routed_cross_block(
        block,
        query_streams: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        if block is stage0_block and "stage0" not in recorded_inputs:
            recorded_inputs["stage0"] = key_value.detach().clone()
        if block is stage1_block and "stage1" not in recorded_inputs:
            recorded_inputs["stage1"] = key_value.detach().clone()
        return original_routed_cross_block(block, query_streams, key_value)

    monkeypatch.setattr(model, "_routed_cross_block", _recording_routed_cross_block)

    _ = model(batch)

    expected_stage0 = classification_state.stage0_input
    assert torch.allclose(recorded_inputs["stage0"], expected_stage0)
    assert torch.allclose(recorded_inputs["stage1"], classification_state.context_bank)
    if routed_direct_cell_bypass:
        assert int(recorded_inputs["stage0"].shape[1]) == (
            int(classification_state.full_cell_stream.shape[1])
            + int(classification_state.context_bank.shape[1])
        )
    else:
        assert int(recorded_inputs["stage0"].shape[1]) == int(classification_state.context_bank.shape[1])


@pytest.mark.parametrize(
    ("routed_direct_cell_bypass", "expect_same_source"),
    [(False, True), (True, False)],
)
def test_routed_sandwich_forward_passes_expected_pool_sources(
    monkeypatch: pytest.MonkeyPatch,
    routed_direct_cell_bypass: bool,
    expect_same_source: bool,
) -> None:
    model = _routed_model(routed_direct_cell_bypass=routed_direct_cell_bypass)
    batch = _task_batch()
    recorded_sources: dict[str, int] = {}

    def _recording_pool(
        *,
        latent_query_streams: torch.Tensor,
        value_streams: torch.Tensor,
        num_test_rows: int,
    ) -> torch.Tensor:
        recorded_sources["latent_id"] = id(latent_query_streams)
        recorded_sources["value_id"] = id(value_streams)
        recorded_sources["num_test_rows"] = num_test_rows
        return torch.zeros(
            (int(latent_query_streams.shape[0]), num_test_rows, model.d_icl),
            device=latent_query_streams.device,
            dtype=latent_query_streams.dtype,
        )

    monkeypatch.setattr(model, "_pool_test_rows", _recording_pool)

    output = model(batch)

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert recorded_sources["num_test_rows"] == 2
    assert (recorded_sources["latent_id"] == recorded_sources["value_id"]) is expect_same_source


def test_routed_sandwich_pool_test_rows_uses_latent_query_and_value_sources_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _routed_model(routed_direct_cell_bypass=True)
    num_test_rows = 2
    latent_query_streams = torch.randn(
        1,
        num_test_rows * model.routed_row_summary_tokens,
        model.routed_residual_streams,
        model.d_icl,
    )
    value_streams = torch.randn_like(latent_query_streams)
    query_primary = torch.full(
        (1, num_test_rows * model.routed_row_summary_tokens, model.d_icl),
        3.0,
    )
    value_primary = torch.full(
        (1, num_test_rows * model.routed_row_summary_tokens, model.d_icl),
        7.0,
    )
    recorded_inputs: dict[str, torch.Tensor] = {}

    def _fake_width_mix(streams: torch.Tensor) -> torch.Tensor:
        if streams is latent_query_streams:
            return query_primary
        if streams is value_streams:
            return value_primary
        raise AssertionError("unexpected routed stream input")

    def _fake_cross_block(
        block,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        assert block is model.test_row_pool
        recorded_inputs["query"] = query.detach().clone()
        recorded_inputs["key_value"] = key_value.detach().clone()
        return torch.zeros_like(query)

    monkeypatch.setattr(model.latent_memory_router, "width_mix", _fake_width_mix)
    monkeypatch.setattr(model, "_cross_block", _fake_cross_block)

    pooled = model._pool_test_rows(
        latent_query_streams=latent_query_streams,
        value_streams=value_streams,
        num_test_rows=num_test_rows,
    )

    expected_query = query_primary.reshape(
        1,
        num_test_rows,
        model.routed_row_summary_tokens,
        model.d_icl,
    ).mean(dim=2, keepdim=True)
    expected_query = expected_query + model.test_row_pool_query.view(1, 1, 1, model.d_icl)
    expected_query = expected_query.reshape(1 * num_test_rows, 1, model.d_icl)
    expected_key_value = value_primary.reshape(
        1 * num_test_rows,
        model.routed_row_summary_tokens,
        model.d_icl,
    )

    assert tuple(pooled.shape) == (1, num_test_rows, model.d_icl)
    assert torch.allclose(recorded_inputs["query"], expected_query)
    assert torch.allclose(recorded_inputs["key_value"], expected_key_value)


def test_grid_sandwich_forward_and_forward_batched_shapes_match() -> None:
    model = _grid_model()
    batch = _task_batch()
    x_all, y_train, split_index, feature_types = _batched_inputs()

    output = model(batch)
    batched_logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=split_index,
        feature_types=feature_types,
    )

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert output.num_classes == 2
    assert tuple(batched_logits.shape) == (1, 2, 4)
    assert torch.allclose(output.logits, batched_logits.squeeze(0), atol=1e-5, rtol=1e-5)
    assert len(model.grid_layers) == 2


@pytest.mark.parametrize(
    "model_kwargs",
    (
        {"grid_residual_mode": "hyper_connection_lite"},
        {"grid_attention_mode": "differential"},
        {"grid_ffn_mode": "swiglu"},
        {"grid_recurrence_steps": 3},
    ),
)
def test_grid_sandwich_experimental_modes_preserve_forward_shapes(
    model_kwargs: dict[str, object],
) -> None:
    model = _grid_model(**model_kwargs)
    batch = _task_batch(num_classes=3)
    x_all, y_train, split_index, feature_types = _batched_inputs()

    output = model(batch)
    batched_logits = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=split_index,
        feature_types=feature_types,
    )

    assert output.logits is not None
    assert tuple(output.logits.shape) == (2, 4)
    assert tuple(batched_logits.shape) == (1, 2, 4)


def test_grid_sandwich_hyper_connection_uses_two_streams_and_collapses_to_cells() -> None:
    model = _grid_model(grid_residual_mode="hyper_connection_lite")
    batch = _task_batch()
    x_all, y_train, _y_test, split_index = model._prepare_task_inputs(batch)
    feature_type_ids = model._feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    feature_cells = model._feature_cells(
        x_all,
        train_test_split_index=split_index,
        feature_type_ids=feature_type_ids,
    )
    streams = model._initialize_grid_streams(feature_cells)

    assert tuple(streams.shape) == (1, 5, 3, 2, 32)
    assert torch.allclose(streams.mean(dim=3), feature_cells)
    assert model.grid_layers[0].row_router is not None
    assert model.grid_layers[0].column_router is not None


def test_grid_sandwich_recurrent_core_shares_one_grid_layer() -> None:
    model = _grid_model(grid_recurrence_steps=8)

    assert model.grid_recurrence_steps == 8
    assert model.grid_core_iterations == 8
    assert len(model.grid_layers) == 1


def test_grid_sandwich_grid_core_intervention_none_preserves_logits() -> None:
    torch.manual_seed(7)
    baseline = _grid_model().eval()
    intervened = _grid_model().eval()
    intervened.load_state_dict(baseline.state_dict())
    intervened.set_grid_core_intervention(mode="none")
    batch = _task_batch()

    with torch.no_grad():
        baseline_logits = baseline(batch).logits
        intervened_logits = intervened(batch).logits

    assert baseline_logits is not None
    assert intervened_logits is not None
    assert torch.allclose(baseline_logits, intervened_logits)


def test_grid_sandwich_grid_core_intervention_ablate_and_repeat_call_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _grid_model().eval()
    batch = _task_batch()
    calls: list[tuple[str, int]] = []

    def _record_row(block, hidden: torch.Tensor) -> torch.Tensor:
        calls.append(("row", id(block)))
        return hidden

    def _record_column(block, hidden: torch.Tensor) -> torch.Tensor:
        calls.append(("column", id(block)))
        return hidden

    monkeypatch.setattr(model, "_row_feature_self_attention", _record_row)
    monkeypatch.setattr(model, "_column_row_isab", _record_column)

    with torch.no_grad():
        model.set_grid_core_intervention(mode="ablate_chunk", start_layer=0, end_layer=0)
        _ = model(batch)

    layer1 = model.grid_layers[1]
    assert calls == [
        ("row", id(layer1.row_mixer)),
        ("column", id(layer1.column_mixer)),
    ]

    calls.clear()
    with torch.no_grad():
        model.set_grid_core_intervention(
            mode="repeat_chunk",
            start_layer=0,
            end_layer=1,
            repeat_count=2,
        )
        _ = model(batch)

    layer0 = model.grid_layers[0]
    assert calls == [
        ("row", id(layer0.row_mixer)),
        ("column", id(layer0.column_mixer)),
        ("row", id(layer1.row_mixer)),
        ("column", id(layer1.column_mixer)),
        ("row", id(layer0.row_mixer)),
        ("column", id(layer0.column_mixer)),
        ("row", id(layer1.row_mixer)),
        ("column", id(layer1.column_mixer)),
    ]


def test_grid_sandwich_grid_core_intervention_rejects_recurrent_checkpoint() -> None:
    model = _grid_model(grid_recurrence_steps=3)

    with pytest.raises(ValueError, match="distinct grid mixer layers"):
        model.set_grid_core_intervention(
            mode="repeat_chunk",
            start_layer=0,
            end_layer=0,
            repeat_count=2,
        )


def test_grid_sandwich_differential_attention_initializes_lambdas() -> None:
    model = _grid_model(grid_attention_mode="differential")
    lambda_values = [
        parameter.detach()
        for name, parameter in model.named_parameters()
        if name.endswith("lambda_scale")
    ]

    assert lambda_values
    assert all(torch.allclose(value, torch.tensor(0.1)) for value in lambda_values)


def test_grid_sandwich_forward_does_not_depend_on_test_labels() -> None:
    model = _grid_model().eval()
    batch = _task_batch()
    changed_test_labels = TaskBatch(
        x_train=batch.x_train,
        y_train=batch.y_train,
        x_test=batch.x_test,
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata=batch.metadata,
        num_classes=batch.num_classes,
    )

    with torch.no_grad():
        logits = model(batch).logits
        changed_logits = model(changed_test_labels).logits

    assert logits is not None
    assert changed_logits is not None
    assert torch.allclose(logits, changed_logits)


def test_grid_sandwich_honors_pre_perceiver_mixer_depths() -> None:
    baseline = _grid_model(
        sandwich_pre_row_attention_layers=0,
        sandwich_pre_column_attention_layers=0,
    )
    mixed = _grid_model(
        sandwich_pre_row_attention_layers=2,
        sandwich_pre_column_attention_layers=1,
    )

    baseline_params = sum(int(parameter.numel()) for parameter in baseline.parameters())
    mixed_params = sum(int(parameter.numel()) for parameter in mixed.parameters())

    assert len(baseline.pre_row_attention_blocks) == 0
    assert len(baseline.pre_column_attention_blocks) == 0
    assert len(mixed.pre_row_attention_blocks) == 2
    assert len(mixed.pre_column_attention_blocks) == 1
    assert mixed_params > baseline_params


def test_grid_sandwich_rejects_non_classification_loss_surface() -> None:
    model = _grid_model()

    with pytest.raises(ValueError, match="classification"):
        model.set_loss_surface("cell_bpc")
