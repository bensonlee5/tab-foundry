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
