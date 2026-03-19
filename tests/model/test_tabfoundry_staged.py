from __future__ import annotations

import math

import pytest
import torch

from tab_foundry.input_normalization import normalize_train_test_tensors
from tab_foundry.model.architectures.tabfoundry_staged.model import TabFoundryStagedClassifier
from tab_foundry.model.architectures.tabfoundry_staged.recipes import STAGE_RECIPE_REGISTRY
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    staged_surface_uses_internal_benchmark_normalization,
)
from tab_foundry.model.architectures.tabfoundry_staged.subsystems import (
    PreNormCellBlock,
    ScalarPerFeatureMissingnessTokenizer,
)
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.model.spec import ModelBuildSpec, ModelStage
from tab_foundry.types import TaskBatch


def _batch(*, num_classes: int = 2) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [0.5, 1.5, 2.5],
                [3.5, 4.5, 5.5],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata={},
        num_classes=num_classes,
    )


def _staged(stage: str, **overrides: object) -> TabFoundryStagedClassifier:
    kwargs = {
        "stage": stage,
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
    }
    kwargs.update(overrides)
    return TabFoundryStagedClassifier(**kwargs)


def _simple(**overrides: object) -> TabFoundrySimpleClassifier:
    kwargs = {
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
    }
    kwargs.update(overrides)
    return TabFoundrySimpleClassifier(**kwargs)


def test_stage_recipe_registry_covers_public_stage_enum() -> None:
    assert set(STAGE_RECIPE_REGISTRY) == set(ModelStage)


@pytest.mark.parametrize(
    ("stage", "normalization_mode"),
    [
        ("nano_exact", "internal"),
        ("label_token", "internal"),
        ("shared_norm", "shared"),
        ("prenorm_block", "shared"),
        ("small_class_head", "shared"),
        ("test_self", "shared"),
        ("grouped_tokens", "shared"),
        ("row_cls_pool", "shared"),
        ("column_set", "shared"),
        ("qass_context", "shared"),
        ("many_class", "shared"),
    ],
)
def test_stage_recipe_contracts_are_exposed_on_model(stage: str, normalization_mode: str) -> None:
    model = _staged(stage, many_class_base=10)
    assert model.stage == stage
    assert model.benchmark_profile == stage
    assert model.recipe.constraints.normalization_mode == normalization_mode


def test_many_class_stage_rejects_invalid_many_class_train_mode() -> None:
    with pytest.raises(ValueError, match="many_class_train_mode"):
        _ = _staged(
            "many_class",
            many_class_base=4,
            input_normalization="none",
            many_class_train_mode="full_prob",
        )


def test_shared_normalization_stage_rejects_invalid_input_normalization() -> None:
    with pytest.raises(ValueError, match="input_normalization"):
        _ = _staged(
            "small_class_head",
            many_class_base=4,
            input_normalization="bogus",
        )


def test_nano_exact_stage_matches_simple_feature_encoder() -> None:
    simple = _simple(d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged = _staged("nano_exact", d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged.feature_encoder.load_state_dict(simple.feature_encoder.state_dict(), strict=True)

    x_all = torch.randn(2, 8, 3, dtype=torch.float32)
    observed = staged.feature_encoder(x_all.clone(), 5)
    expected = simple.feature_encoder(x_all.clone(), 5)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_nano_exact_stage_matches_simple_target_conditioner() -> None:
    simple = _simple(d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged = _staged("nano_exact", d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged.target_conditioner.encoder.load_state_dict(simple.target_encoder.state_dict(), strict=True)

    y_train = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    observed = staged.target_conditioner(y_train.clone(), num_rows=8)
    expected = simple.target_encoder(y_train.clone(), 8)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_nano_exact_stage_matches_simple_transformer_block() -> None:
    simple = _simple(d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged = _staged("nano_exact", d_icl=32, tficl_n_heads=4, tficl_n_layers=1, head_hidden_dim=64)
    staged.transformer_blocks[0].block.load_state_dict(
        simple.transformer_blocks[0].state_dict(),
        strict=True,
    )

    src = torch.randn(2, 8, 4, 32, dtype=torch.float32)
    observed = staged.transformer_blocks[0](src.clone(), train_test_split_index=5)
    expected = simple.transformer_blocks[0](src.clone(), train_test_split_index=5)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_nano_exact_stage_matches_simple_forward_batched_logits() -> None:
    simple = _simple(d_icl=32, tficl_n_heads=4, tficl_n_layers=2, head_hidden_dim=64)
    staged = _staged("nano_exact", d_icl=32, tficl_n_heads=4, tficl_n_layers=2, head_hidden_dim=64)
    staged.feature_encoder.load_state_dict(simple.feature_encoder.state_dict(), strict=True)
    staged.target_conditioner.encoder.load_state_dict(simple.target_encoder.state_dict(), strict=True)
    staged.direct_head.decoder.load_state_dict(simple.decoder.state_dict(), strict=True)
    for staged_block, simple_block in zip(staged.transformer_blocks, simple.transformer_blocks, strict=True):
        staged_block.block.load_state_dict(simple_block.state_dict(), strict=True)

    x_all = torch.randn(2, 8, 3, dtype=torch.float32)
    y_train = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    observed = staged.forward_batched(
        x_all=x_all.clone(),
        y_train=y_train.clone(),
        train_test_split_index=5,
    )
    expected = simple.forward_batched(
        x_all=x_all.clone(),
        y_train=y_train.clone(),
        train_test_split_index=5,
    )
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_binary_only_stages_reject_multiclass() -> None:
    with pytest.raises(RuntimeError, match="binary-only"):
        _ = _staged("nano_exact")(_batch(num_classes=3))


def test_row_cls_pool_stage_supports_rmsnorm() -> None:
    model = _staged(
        "row_cls_pool",
        tfrow_n_heads=2,
        tfrow_n_layers=1,
        tfrow_cls_tokens=2,
        tfrow_norm="rmsnorm",
    )
    model.eval()

    with torch.no_grad():
        out = model(_batch())

    assert model.module_hyperparameters["row_pool"]["norm_type"] == "rmsnorm"
    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)


def test_simple_classifier_supports_global_rmsnorm() -> None:
    model = _simple(
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
        norm_type="rmsnorm",
    )

    out = model(_batch())

    assert model.norm_type == "rmsnorm"
    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)


def test_staged_surface_supports_global_rmsnorm_across_active_modules() -> None:
    model = _staged(
        "nano_exact",
        stage_label="delta_global_rmsnorm",
        module_overrides={"column_encoder": "tfcol", "context_encoder": "qass"},
        norm_type="rmsnorm",
        tfrow_norm="rmsnorm",
        tfcol_n_heads=2,
        tfcol_n_layers=1,
        tfcol_n_inducing=8,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=64,
    )

    out = model(_batch())

    assert model.module_hyperparameters["table_block"]["norm_type"] == "rmsnorm"
    assert model.module_hyperparameters["column_encoder"]["norm_type"] == "rmsnorm"
    assert model.module_hyperparameters["context_encoder"]["norm_type"] == "rmsnorm"
    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)


def test_small_class_stage_rejects_more_classes_than_many_class_base() -> None:
    model = _staged("small_class_head", many_class_base=4, input_normalization="none")
    with pytest.raises(RuntimeError, match="many_class_base"):
        _ = model(_batch(num_classes=5))


def test_many_class_stage_emits_probabilities_in_eval_mode() -> None:
    model = _staged("many_class", many_class_base=4, input_normalization="none")
    model.eval()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 12, (8,)),
        metadata={},
        num_classes=12,
    )

    out = model(batch)

    assert out.logits is None
    assert out.class_probs is not None
    assert out.class_probs.shape == (8, 12)
    assert torch.allclose(out.class_probs.sum(dim=-1), torch.ones(8), atol=1.0e-5)


def test_many_class_stage_emits_path_terms_in_train_mode() -> None:
    model = _staged("many_class", many_class_base=4, input_normalization="none")
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 12, (8,)),
        metadata={},
        num_classes=12,
    )

    out = model(batch)

    assert out.class_probs is None
    assert out.path_logits is not None
    assert out.path_targets is not None
    assert out.path_sample_counts is not None
    assert len(out.path_logits) == len(out.path_targets) == len(out.path_sample_counts)


def test_many_class_stage_accepts_full_probs_in_train_mode() -> None:
    model = _staged(
        "many_class",
        many_class_base=4,
        input_normalization="none",
        many_class_train_mode="full_probs",
    )
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 12, (8,)),
        metadata={},
        num_classes=12,
    )

    out = model(batch)

    assert out.logits is None
    assert out.class_probs is not None
    assert out.path_logits is None
    assert out.path_targets is None
    assert out.path_sample_counts is None


def test_many_class_stage_rejects_tensor_batched_internal_path() -> None:
    model = _staged("many_class", many_class_base=4, input_normalization="none")
    x_all = torch.randn(2, 8, 12)
    y_train = torch.randint(0, 6, (2, 5), dtype=torch.int64)
    y_test = torch.randint(0, 12, (2, 3), dtype=torch.int64)
    raw_state = model._build_raw_input_state(
        x_all=x_all,
        y_train=y_train,
        y_test=y_test,
        train_test_split_index=5,
        num_classes=12,
    )

    with pytest.raises(RuntimeError, match="staged many_class currently requires a single task"):
        _ = model._forward_many_class(raw_state)


def test_module_overrides_surface_stage_label_and_row_pool_hyperparameters() -> None:
    model = _staged(
        "nano_exact",
        stage_label="delta_row_cls_pool",
        module_overrides={"row_pool": "row_cls"},
        tfrow_n_heads=2,
        tfrow_n_layers=1,
        tfrow_cls_tokens=2,
    )

    assert model.stage == "nano_exact"
    assert model.stage_label == "delta_row_cls_pool"
    assert model.benchmark_profile == "delta_row_cls_pool"
    assert model.surface.row_pool == "row_cls"
    assert model.module_hyperparameters["row_pool"]["n_heads"] == 2
    assert model.module_hyperparameters["row_pool"]["n_layers"] == 1
    assert model.module_hyperparameters["row_pool"]["cls_tokens"] == 2


@pytest.mark.parametrize("post_encoder_norm", ["none", "layernorm", "rmsnorm"])
def test_module_overrides_surface_post_encoder_norm(post_encoder_norm: str) -> None:
    module_overrides: dict[str, object] = {"feature_encoder": "shared"}
    if post_encoder_norm != "none":
        module_overrides["post_encoder_norm"] = post_encoder_norm
    model = _staged(
        "nano_exact",
        stage_label=f"delta_post_encoder_norm_{post_encoder_norm}",
        module_overrides=module_overrides,
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
    )

    out = model(_batch())

    assert model.module_selection["post_encoder_norm"] == post_encoder_norm
    assert model.module_hyperparameters["post_encoder_norm"]["name"] == post_encoder_norm
    expected_norm = None if post_encoder_norm == "none" else post_encoder_norm
    assert model.module_hyperparameters["post_encoder_norm"]["norm_type"] == expected_norm
    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)


@pytest.mark.parametrize("post_stack_norm", ["none", "layernorm", "rmsnorm"])
def test_module_overrides_surface_post_stack_norm(post_stack_norm: str) -> None:
    module_overrides: dict[str, object] = {"feature_encoder": "shared"}
    if post_stack_norm != "none":
        module_overrides["post_stack_norm"] = post_stack_norm
    model = _staged(
        "nano_exact",
        stage_label=f"delta_post_stack_norm_{post_stack_norm}",
        module_overrides=module_overrides,
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
    )

    out = model(_batch())

    assert model.module_selection["post_stack_norm"] == post_stack_norm
    assert model.module_hyperparameters["post_stack_norm"]["name"] == post_stack_norm
    expected_norm = None if post_stack_norm == "none" else post_stack_norm
    assert model.module_hyperparameters["post_stack_norm"]["norm_type"] == expected_norm
    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)


def test_module_overrides_surface_prenorm_depth_scaled_residual_gain() -> None:
    model = _staged(
        "nano_exact",
        stage_label="delta_depth_scaled_prenorm",
        module_overrides={
            "table_block_style": "prenorm",
            "table_block_residual_scale": "depth_scaled",
        },
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=4,
        head_hidden_dim=64,
    )

    assert model.module_selection["table_block_residual_scale"] == "depth_scaled"
    expected_gain = pytest.approx((3.0 * 4.0) ** -0.5)
    assert model.module_hyperparameters["table_block"]["residual_scale"] == "depth_scaled"
    assert model.module_hyperparameters["table_block"]["residual_branch_gain"] == expected_gain
    assert all(
        isinstance(block, PreNormCellBlock) and block.residual_branch_gain == expected_gain
        for block in model.transformer_blocks
    )


def test_activation_trace_records_expected_points_and_resets() -> None:
    model = _staged(
        "nano_exact",
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=2,
        head_hidden_dim=64,
    )
    x_all = torch.randn(2, 8, 3, dtype=torch.float32)
    y_train = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    model.enable_activation_trace()
    _ = model.forward_batched(
        x_all=x_all,
        y_train=y_train,
        train_test_split_index=5,
    )
    snapshot = model.flush_activation_trace()

    assert snapshot is not None
    assert set(snapshot) == {
        "post_feature_encoder",
        "post_target_conditioner",
        "pre_transformer",
        "post_transformer_block_0",
        "post_transformer_block_1",
        "post_column_encoder",
        "post_row_pool",
    }
    assert all(value >= 0.0 for value in snapshot.values())
    assert model.flush_activation_trace() == {}
    model.disable_activation_trace()
    assert model.flush_activation_trace() is None


def test_activation_trace_aggregates_repeated_names_using_element_weighted_rms() -> None:
    model = _staged(
        "nano_exact",
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
    )

    model.enable_activation_trace()
    model.trace_activation("constant", torch.full((1, 2, 3), 2.0, dtype=torch.float32))
    model.trace_activation("constant", torch.full((2, 5, 7), 2.0, dtype=torch.float32))
    model.trace_activation("constant", torch.full((1, 1, 1), 4.0, dtype=torch.float32))
    snapshot = model.flush_activation_trace()

    assert snapshot is not None
    expected = math.sqrt(((6 * 4.0) + (70 * 4.0) + (1 * 16.0)) / 77.0)
    assert snapshot["constant"] == pytest.approx(expected)


def test_many_class_activation_trace_aggregates_recursive_context_calls() -> None:
    model = _staged("many_class", many_class_base=4, input_normalization="none")
    model.train()
    batch = TaskBatch(
        x_train=torch.randn(24, 12),
        y_train=torch.randint(0, 6, (24,)),
        x_test=torch.randn(8, 12),
        y_test=torch.randint(0, 12, (8,)),
        metadata={},
        num_classes=12,
    )
    expected_sum_sq = 0.0
    expected_count = 0

    def _capture_context_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal expected_sum_sq, expected_count
        trace_tensor = output.detach().to(torch.float32)
        expected_sum_sq += float(trace_tensor.square().sum().item())
        expected_count += int(trace_tensor.numel())

    assert model.context_encoder is not None
    handle = model.context_encoder.register_forward_hook(_capture_context_output)
    try:
        model.enable_activation_trace()
        _ = model(batch)
        snapshot = model.flush_activation_trace()
    finally:
        handle.remove()

    assert snapshot is not None
    assert "post_context_encoder" in snapshot
    assert expected_count > 0
    assert snapshot["post_context_encoder"] == pytest.approx(
        math.sqrt(expected_sum_sq / float(expected_count))
    )


def test_shared_normalization_matches_stacked_2d_behavior_for_batched_forward_inputs() -> None:
    model = _staged(
        "small_class_head",
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
    )
    x_all = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0],
            ],
            [
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0],
            ],
        ],
        dtype=torch.float32,
    )
    x_train = x_all[:, :5, :]
    x_test = x_all[:, 5:, :]
    expected_train_parts: list[torch.Tensor] = []
    expected_test_parts: list[torch.Tensor] = []
    for batch_idx in range(int(x_all.shape[0])):
        train_norm, test_norm = normalize_train_test_tensors(
            x_train[batch_idx],
            x_test[batch_idx],
            mode=model.input_normalization,
        )
        expected_train_parts.append(train_norm)
        expected_test_parts.append(test_norm)

    observed = model._normalize_x_all(x_all, train_test_split_index=5)
    expected = torch.cat(
        [
            torch.stack(expected_train_parts, dim=0),
            torch.stack(expected_test_parts, dim=0),
        ],
        dim=1,
    )

    torch.testing.assert_close(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_stage_labels_do_not_trigger_legacy_constraint_enforcement() -> None:
    model = _staged(
        "nano_exact",
        stage_label="anchor_model",
        input_normalization="none",
        tfrow_n_heads=2,
    )

    assert model.surface.feature_encoder == "nano"
    assert model.surface.normalization_mode == "internal"
    assert staged_surface_uses_internal_benchmark_normalization(model.model_spec) is True


def test_module_overrides_reject_ineffective_tokenizer_change_under_nano_encoder() -> None:
    with pytest.raises(ValueError, match="tokenizer is ineffective"):
        _ = _staged(
            "nano_exact",
            stage_label="delta_shifted_grouped_tokenizer",
            module_overrides={"tokenizer": "shifted_grouped"},
        )


def test_module_overrides_reject_many_class_head_without_context_encoder() -> None:
    with pytest.raises(ValueError, match="requires a non-'none' context_encoder"):
        _ = _staged(
            "nano_exact",
            stage_label="delta_many_class_head",
            module_overrides={"head": "many_class"},
        )


def test_module_overrides_allow_many_class_head_with_context_encoder() -> None:
    model = _staged(
        "nano_exact",
        stage_label="delta_many_class_head",
        input_normalization="none",
        many_class_base=4,
        module_overrides={"head": "many_class", "context_encoder": "qass"},
    )

    assert model.surface.head == "many_class"
    assert model.surface.context_encoder == "qass"


def test_prenorm_missingness_tokenizer_produces_finite_logits() -> None:
    model = _staged(
        "prenorm_block",
        d_icl=32,
        tficl_n_heads=4,
        tficl_n_layers=1,
        head_hidden_dim=64,
        module_overrides={
            "feature_encoder": "shared",
            "post_encoder_norm": "layernorm",
            "table_block_style": "prenorm",
            "tokenizer": "scalar_per_feature_nan_mask",
        },
    )
    model.eval()
    batch = TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [4.0, 5.0, float("nan")],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [0.5, float("nan"), 2.5],
                [3.5, 4.5, 5.5],
            ],
            dtype=torch.float32,
        ),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata={},
        num_classes=2,
    )

    with torch.no_grad():
        out = model(batch)

    assert out.logits is not None
    assert tuple(out.logits.shape) == (2, 2)
    assert torch.isfinite(out.logits).all()


def test_missingness_tokenizer_emits_distinct_non_finite_flags() -> None:
    tokenizer = ScalarPerFeatureMissingnessTokenizer()
    tokenized, token_padding_mask = tokenizer(
        torch.tensor(
            [[[1.5, float("nan"), float("inf"), float("-inf")]]],
            dtype=torch.float32,
        )
    )

    assert token_padding_mask is None
    assert tokenizer.token_dim == 4
    assert torch.allclose(tokenized[0, 0, 0], torch.tensor([1.5, 0.0, 0.0, 0.0]))
    assert torch.allclose(tokenized[0, 0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(tokenized[0, 0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(tokenized[0, 0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_pre_encoder_clip_preserves_non_finite_categories_for_missingness_tokenizer() -> None:
    model = _staged(
        "prenorm_block",
        input_normalization="none",
        pre_encoder_clip=5.0,
        module_overrides={
            "feature_encoder": "shared",
            "post_encoder_norm": "layernorm",
            "table_block_style": "prenorm",
            "tokenizer": "scalar_per_feature_nan_mask",
        },
    )
    x_all = torch.tensor(
        [
            [
                [100.0, float("nan"), float("inf"), float("-inf")],
                [4.0, 5.0, 6.0, 7.0],
                [50.0, 8.0, 9.0, 10.0],
                [3.5, 4.5, float("inf"), float("-inf")],
            ],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor([[0, 1]], dtype=torch.int64)
    raw_state = model._build_raw_input_state(
        x_all=x_all,
        y_train=y_train,
        y_test=torch.tensor([[0, 1]], dtype=torch.int64),
        train_test_split_index=2,
        num_classes=2,
    )
    tokenized_x, _ = model.tokenizer(raw_state.x_all)

    assert raw_state.x_all[0, 0, 0].item() == pytest.approx(5.0)
    assert torch.isnan(raw_state.x_all[0, 0, 1])
    assert torch.isposinf(raw_state.x_all[0, 0, 2])
    assert torch.isneginf(raw_state.x_all[0, 0, 3])
    assert torch.allclose(tokenized_x[0, 0, 0], torch.tensor([5.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(tokenized_x[0, 0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(tokenized_x[0, 0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(tokenized_x[0, 0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_missingness_tokenizer_uses_internal_benchmark_normalization() -> None:
    spec = ModelBuildSpec(
        task="classification",
        arch="tabfoundry_staged",
        stage="shared_norm",
        input_normalization="train_zscore_clip",
        module_overrides={"tokenizer": "scalar_per_feature_nan_mask"},
    )

    assert staged_surface_uses_internal_benchmark_normalization(spec) is True
