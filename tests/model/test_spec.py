from __future__ import annotations

import pytest
import torch

from tab_foundry.model.factory import build_model
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)


def _checkpoint_model_cfg(*, task: str = "classification", **overrides: object) -> dict[str, object]:
    return model_build_spec_from_mappings(task=task, primary=overrides).to_dict()


def test_model_build_spec_defaults_feature_group_size_to_one() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={},
    )

    assert spec.arch == "tabfoundry"
    assert spec.missingness_mode == "none"
    assert spec.feature_group_size == 1


def test_build_model_defaults_feature_group_size_to_one() -> None:
    cls_model = build_model(task="classification")
    reg_model = build_model(task="regression")

    assert getattr(cls_model, "feature_group_size") == 1
    assert getattr(reg_model, "feature_group_size") == 1
    assert getattr(cls_model, "missingness_mode") == "none"
    assert getattr(reg_model, "missingness_mode") == "none"


def test_build_model_supports_tabfoundry_simple_classification() -> None:
    model = build_model(
        task="classification",
        arch="tabfoundry_simple",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=4,
        tficl_n_layers=3,
        head_hidden_dim=192,
    )

    assert isinstance(model, TabFoundrySimpleClassifier)


def test_build_model_rejects_tabfoundry_simple_regression() -> None:
    with pytest.raises(ValueError, match="classification"):
        _ = build_model(task="regression", arch="tabfoundry_simple")


def test_model_build_spec_rejects_invalid_missingness_mode() -> None:
    with pytest.raises(ValueError, match="missingness_mode"):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"missingness_mode": "bogus"},
        )


def test_checkpoint_build_spec_requires_explicit_reconstruction_fields() -> None:
    with pytest.raises(ValueError, match="missing required reconstruction fields"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={},
            state_dict={"group_linear.weight": torch.zeros((128, 3))},
        )


def test_checkpoint_build_spec_rejects_ambiguous_tabfoundry_layout_without_explicit_metadata() -> None:
    primary = _checkpoint_model_cfg()
    primary.pop("feature_group_size")
    primary.pop("missingness_mode")

    with pytest.raises(ValueError, match="ambiguous across multiple tabfoundry layouts"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary=primary,
            state_dict={"group_linear.weight": torch.zeros((128, 6))},
        )


def test_checkpoint_build_spec_accepts_explicit_override_for_simple_missingness() -> None:
    primary = _checkpoint_model_cfg(arch="tabfoundry_simple")
    primary.pop("missingness_mode")

    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary=primary,
        explicit_overrides={"missingness_mode": "explicit_token"},
        state_dict={
            "feature_encoder.linear_layer.weight": torch.zeros((96, 1)),
            "feature_encoder.nan_embedding": torch.zeros((96,)),
        },
    )

    assert spec.arch == "tabfoundry_simple"
    assert spec.missingness_mode == "explicit_token"


def test_checkpoint_build_spec_rejects_mismatched_configured_missingness_mode() -> None:
    with pytest.raises(ValueError, match="missingness_mode"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary=_checkpoint_model_cfg(
                arch="tabfoundry_simple",
                missingness_mode="none",
            ),
            state_dict={
                "feature_encoder.linear_layer.weight": torch.zeros((96, 1)),
                "feature_encoder.nan_embedding": torch.zeros((96,)),
            },
        )


def test_checkpoint_build_spec_rejects_legacy_grouped_weights_without_override() -> None:
    primary = _checkpoint_model_cfg(missingness_mode="none")
    primary.pop("feature_group_size")

    with pytest.raises(ValueError, match="ambiguous across multiple tabfoundry layouts"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary=primary,
            state_dict={"group_linear.weight": torch.zeros((128, 96))},
        )


def test_checkpoint_build_spec_supports_explicit_nondefault_feature_group_size() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary=_checkpoint_model_cfg(
            feature_group_size=32,
            missingness_mode="none",
        ),
        state_dict={"group_linear.weight": torch.zeros((128, 96))},
    )

    assert spec.feature_group_size == 32


def test_staged_model_defaults_stage_to_nano_exact() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_staged"},
    )

    assert spec.arch == "tabfoundry_staged"
    assert spec.stage == "nano_exact"


def test_staged_model_spec_accepts_stage_label_and_module_overrides() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "delta_row_cls_pool",
            "module_overrides": {"row_pool": "row_cls"},
        },
    )

    assert spec.stage == "nano_exact"
    assert spec.stage_label == "delta_row_cls_pool"
    assert spec.module_overrides == {"row_pool": "row_cls"}


def test_non_staged_arch_rejects_stage() -> None:
    with pytest.raises(ValueError, match="model.stage"):
        _ = ModelBuildSpec(task="classification", arch="tabfoundry", stage="nano_exact")


def test_non_staged_arch_rejects_stage_surface_fields() -> None:
    with pytest.raises(ValueError, match="stage_label"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry",
            stage_label="delta_label_token",
        )
    with pytest.raises(ValueError, match="module_overrides"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry",
            module_overrides={"row_pool": "row_cls"},
        )


def test_checkpoint_build_spec_round_trips_model_arch() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary=_checkpoint_model_cfg(arch="tabfoundry_simple"),
        state_dict={},
    )

    assert spec.arch == "tabfoundry_simple"
