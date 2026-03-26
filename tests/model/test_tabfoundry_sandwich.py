from __future__ import annotations

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from tab_foundry.model.architectures.tabfoundry_sandwich import model as sandwich_model
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


def _model() -> TabFoundrySandwichClassifier:
    return TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )


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


def test_tabfoundry_sandwich_uses_r_plus_c_perceiver_input_tokens() -> None:
    model = _model()
    batch = _batch()
    x_all, y_train, _y_test, train_test_split_index = model._prepare_task_inputs(batch)
    feature_type_ids = model._feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    feature_cells = model._feature_cells(
        x_all,
        train_test_split_index=train_test_split_index,
        feature_type_ids=feature_type_ids,
    )
    repeated_input, row_tokens = model._perceiver_input_tokens(feature_cells, y_train=y_train)

    assert tuple(repeated_input.shape) == (1, 9, 32)
    assert tuple(row_tokens.shape) == (1, 5, 32)
    assert tuple(model.latent_seed.shape) == (1, 12, 32)
    assert tuple(y_train.shape) == (1, 3)
    assert torch.allclose(repeated_input[:, :5, :], row_tokens)


def test_tabfoundry_sandwich_fuses_label_query_state_into_row_summaries() -> None:
    model = _model()
    batch = _batch()
    x_all, y_train, _y_test, train_test_split_index = model._prepare_task_inputs(batch)
    feature_type_ids = model._feature_type_ids_from_metadata(
        batch.metadata,
        batch_size=int(x_all.shape[0]),
        num_features=int(x_all.shape[2]),
        device=x_all.device,
    )
    feature_cells = model._feature_cells(
        x_all,
        train_test_split_index=train_test_split_index,
        feature_type_ids=feature_type_ids,
    )

    raw_row_summaries = model._row_summary_bytes(feature_cells)
    row_tokens = model._row_summary_tokens(feature_cells=feature_cells, y_train=y_train)

    assert tuple(raw_row_summaries.shape) == tuple(row_tokens.shape)
    assert not torch.allclose(raw_row_summaries, row_tokens)


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

    model = _model()

    assert calls == [((1, 12, 32), 0.0, 0.02, -2.0, 2.0)]
    assert tuple(model.latent_seed.shape) == (1, 12, 32)
    assert torch.isfinite(model.latent_seed).all()


def test_tabfoundry_sandwich_runs_repeated_cross_then_self_stages() -> None:
    model = _model()
    order: list[str] = []
    original_cross_block = model._cross_block
    original_self_block = model._self_block
    stage_reads = {
        id(stage.input_read): f"cross_{index}"
        for index, stage in enumerate(model.perceiver_stages)
    }
    stage_latent_blocks = {
        id(stage.latent_block): f"self_{index}"
        for index, stage in enumerate(model.perceiver_stages)
    }

    def _recording_cross_block(block, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        stage_name = stage_reads.get(id(block))
        if stage_name is not None:
            order.append(stage_name)
        elif block is model.test_readout:
            order.append("readout")
        return original_cross_block(block, query, key_value)

    def _recording_self_block(block, hidden: torch.Tensor) -> torch.Tensor:
        stage_name = stage_latent_blocks.get(id(block))
        if stage_name is not None:
            order.append(stage_name)
        return original_self_block(block, hidden)

    model._cross_block = _recording_cross_block  # type: ignore[method-assign]
    model._self_block = _recording_self_block  # type: ignore[method-assign]

    _ = model(_batch())

    assert order == ["cross_0", "self_0", "cross_1", "self_1", "readout"]


def test_tabfoundry_sandwich_exposes_activation_trace_hooks() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )
    model.enable_activation_trace()

    _ = model(_batch())
    trace = model.flush_activation_trace_stats()

    assert trace is not None
    assert "post_feature_encoder" in trace
    assert "post_perceiver_input" in trace
    assert "post_stage_0_cross" in trace
    assert "post_stage_0_self" in trace
    assert "post_test_readout" in trace


def test_tabfoundry_sandwich_layers_count_matches_repeated_stages() -> None:
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=4,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=3,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )

    assert len(model.perceiver_stages) == 3


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
    model = TabFoundrySandwichClassifier(
        d_icl=32,
        many_class_base=3,
        head_hidden_dim=64,
        sandwich_latents=12,
        sandwich_layers=1,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
    )

    with pytest.raises(RuntimeError, match="small-class only"):
        _ = model(_batch(num_classes=5))
