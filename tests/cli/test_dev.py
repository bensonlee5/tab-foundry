from __future__ import annotations

from tab_foundry.cli.dev import forward_check, resolve_config_payload


_SMALL_STAGED_OVERRIDES = [
    "experiment=cls_smoke",
    "logging.use_wandb=false",
    "model.stage=row_cls_pool",
    "model.stage_label=row_cls_pool_dev_test",
    "model.d_icl=32",
    "model.many_class_base=4",
    "model.tficl_n_heads=4",
    "model.tficl_n_layers=1",
    "model.head_hidden_dim=64",
    "model.tfrow_n_heads=2",
    "model.tfrow_n_layers=1",
    "model.tfrow_cls_tokens=2",
    "model.tfcol_n_heads=2",
    "model.tfcol_n_layers=1",
    "model.tfcol_n_inducing=8",
]


def test_resolve_config_payload_reports_resolved_surfaces() -> None:
    payload = resolve_config_payload(_SMALL_STAGED_OVERRIDES)

    assert payload["experiment"] == "cls_smoke"
    assert payload["task"] == "classification"
    assert payload["model"]["stage_label"] == "row_cls_pool_dev_test"
    assert payload["model"]["module_selection"]["row_pool"] == "row_cls"
    assert payload["training"]["surface_label"] == "training_default"
    assert payload["model"]["parameter_counts"]["total_params"] > 0


def test_forward_check_passes_for_direct_head_surface() -> None:
    payload = forward_check(
        _SMALL_STAGED_OVERRIDES,
        requested_device="cpu",
        seed=7,
    )

    assert payload["output_kind"] == "logits"
    assert payload["surface_label"] == "row_cls_pool_dev_test"
    assert payload["output_shape"][0] == 2
    assert payload["batched_output"] is not None


def test_forward_check_passes_for_many_class_surface() -> None:
    payload = forward_check(
        [
            "experiment=cls_smoke",
            "logging.use_wandb=false",
            "model.stage=many_class",
            "model.stage_label=many_class_dev_test",
            "model.d_icl=32",
            "model.many_class_base=4",
            "model.tficl_n_heads=4",
            "model.tficl_n_layers=1",
            "model.head_hidden_dim=64",
            "model.tfrow_n_heads=2",
            "model.tfrow_n_layers=1",
            "model.tfrow_cls_tokens=2",
            "model.tfcol_n_heads=2",
            "model.tfcol_n_layers=1",
            "model.tfcol_n_inducing=8",
        ],
        requested_device="cpu",
        seed=11,
    )

    assert payload["output_kind"] == "class_probs"
    assert payload["surface_label"] == "many_class_dev_test"
    assert payload["output_shape"] == [2, 5]
    assert payload["batched_output"] is None
