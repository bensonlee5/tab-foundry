from __future__ import annotations

import pytest
import torch

from tab_foundry.model.factory import build_model
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.model.spec import (
    STAGED_MODEL_ARCH,
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)


def test_model_build_spec_defaults_feature_group_size_to_one() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={},
    )

    assert spec.arch == STAGED_MODEL_ARCH
    assert spec.feature_group_size == 1


def test_build_model_defaults_feature_group_size_to_one() -> None:
    cls_model = build_model(task="classification")

    assert int(cls_model.model_spec.feature_group_size) == 1


def test_build_model_rejects_regression() -> None:
    with pytest.raises(ValueError, match="classification"):
        _ = build_model(task="regression")


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


def test_build_model_rejects_legacy_tabfoundry_arch() -> None:
    with pytest.raises(ValueError, match="Legacy model arch"):
        _ = build_model(task="classification", arch="tabfoundry")


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
        _ = ModelBuildSpec(task="classification", arch="tabfoundry_simple", stage="nano_exact")


def test_non_staged_arch_rejects_stage_surface_fields() -> None:
    with pytest.raises(ValueError, match="stage_label"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_simple",
            stage_label="delta_label_token",
        )
    with pytest.raises(ValueError, match="module_overrides"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_simple",
            module_overrides={"row_pool": "row_cls"},
        )


def test_checkpoint_build_spec_round_trips_model_arch() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_simple"},
        state_dict={},
    )

    assert spec.arch == "tabfoundry_simple"


def test_model_build_spec_rejects_regression_task() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        _ = ModelBuildSpec(task="regression")


def test_checkpoint_build_spec_rejects_legacy_tabfoundry_arch() -> None:
    with pytest.raises(ValueError, match="model.arch='tabfoundry'"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "tabfoundry"},
            state_dict={},
        )


def test_checkpoint_build_spec_rejects_legacy_tabfoundry_state_dict() -> None:
    with pytest.raises(ValueError, match="Legacy tabfoundry checkpoints"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={},
            state_dict={"group_linear.weight": torch.zeros((128, 96))},
        )
