from __future__ import annotations

from tab_foundry.model.inspection import (
    model_surface_payload,
    parameter_counts_from_model_spec,
    synthetic_forward_batch,
)
from tab_foundry.model.spec import model_build_spec_from_mappings


def _staged_spec(*, stage: str, stage_label: str) -> object:
    return model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_staged",
            "stage": stage,
            "stage_label": stage_label,
            "d_icl": 32,
            "many_class_base": 4,
            "tficl_n_heads": 4,
            "tficl_n_layers": 1,
            "head_hidden_dim": 64,
            "tfrow_n_heads": 2,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
            "tfcol_n_heads": 2,
            "tfcol_n_layers": 1,
            "tfcol_n_inducing": 8,
        },
    )


def _sandwich_spec() -> object:
    return model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_sandwich",
            "d_icl": 32,
            "many_class_base": 4,
            "head_hidden_dim": 64,
            "sandwich_latents": 12,
            "sandwich_layers": 2,
            "sandwich_heads": 4,
            "sandwich_ff_expansion": 2,
            "sandwich_summary_tokens_per_axis": 4,
            "sandwich_self_attention_per_cross": 4,
        },
    )


def test_synthetic_forward_batch_binary_surface_returns_logits() -> None:
    batch = synthetic_forward_batch(_staged_spec(stage="row_cls_pool", stage_label="row_cls_pool_test"))

    assert batch.expected_output_kind == "logits"
    assert batch.expected_num_classes == 4
    assert tuple(batch.task_batch.x_train.shape) == (3, 4)
    assert tuple(batch.task_batch.x_test.shape) == (2, 4)
    assert batch.train_test_split_index == 3


def test_synthetic_forward_batch_many_class_surface_returns_class_probs() -> None:
    batch = synthetic_forward_batch(_staged_spec(stage="many_class", stage_label="many_class_test"))

    assert batch.expected_output_kind == "class_probs"
    assert batch.expected_num_classes == 5
    assert tuple(batch.x_all.shape) == (1, 5, 4)
    assert tuple(batch.y_train_batched.shape) == (1, 3)


def test_parameter_counts_and_surface_payload_include_staged_metadata() -> None:
    spec = _staged_spec(stage="row_cls_pool", stage_label="row_cls_pool_test")

    counts = parameter_counts_from_model_spec(spec)
    payload = model_surface_payload(spec)

    assert counts["total_params"] > 0
    assert counts["trainable_params"] > 0
    assert payload["stage_label"] == "row_cls_pool_test"
    assert payload["benchmark_profile"] == "row_cls_pool_test"
    assert payload["module_selection"]["row_pool"] == "row_cls"
    assert "table_block" in payload["module_hyperparameters"]


def test_parameter_counts_and_surface_payload_include_sandwich_metadata() -> None:
    spec = _sandwich_spec()

    counts = parameter_counts_from_model_spec(spec)
    payload = model_surface_payload(spec)
    batch = synthetic_forward_batch(spec)

    assert counts["total_params"] > 0
    assert counts["trainable_params"] > 0
    assert payload["arch"] == "tabfoundry_sandwich"
    assert payload["architecture"] == {
        "initial_input_tokens": "full_cell_plus_row_col_summary_stream",
        "initial_input_token_count": "R_times_C_plus_K_times_(R_plus_C)",
        "repeated_input_tokens": "row_col_summary_stream",
        "repeated_input_token_count": "K_times_(R_plus_C)",
        "summary_tokens_per_axis": 4,
        "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
        "pre_row_attention_layers": 1,
        "pre_column_attention_layers": 1,
        "pre_column_inducing_tokens": 16,
        "label_injection": "fused_into_row_summaries_and_feature_cells",
        "summary_builder": "summary_query_attention",
        "position_encoding": "shared_fourier_row_col",
        "feature_type_encoding": "film",
        "floating_likelihood": "single_gaussian",
        "integer_likelihood": "hybrid_mixture",
        "sandwich_activation": "gelu",
        "sandwich_block_norm": "layernorm",
        "latent_core": "stage0_full_cell_plus_summary_then_summary_repeated_cross_self_stages",
        "layer_semantics": "stage0_hybrid_then_summary_repeated_stages",
        "readout": "latent_then_full_cell_cross_attention_then_latent_conditioned_query_pool",
        "latents": 12,
        "layers": 2,
        "heads": 4,
        "ff_expansion": 2,
        "self_attention_per_cross": 4,
    }
    assert batch.expected_output_kind == "logits"
    assert batch.expected_num_classes == 4
