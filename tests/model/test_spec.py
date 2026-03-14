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


def test_model_build_spec_defaults_feature_group_size_to_one() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={},
    )

    assert spec.arch == "tabfoundry"
    assert spec.feature_group_size == 1


def test_build_model_defaults_feature_group_size_to_one() -> None:
    cls_model = build_model(task="classification")
    reg_model = build_model(task="regression")

    assert getattr(cls_model, "feature_group_size") == 1
    assert getattr(reg_model, "feature_group_size") == 1


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


def test_checkpoint_build_spec_defaults_omitted_feature_group_size_to_one() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={},
        state_dict={"group_linear.weight": torch.zeros((128, 3))},
    )

    assert spec.feature_group_size == 1


def test_checkpoint_build_spec_rejects_legacy_grouped_weights_without_override() -> None:
    with pytest.raises(ValueError, match="omitted feature_group_size"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={},
            state_dict={"group_linear.weight": torch.zeros((128, 96))},
        )


def test_checkpoint_build_spec_supports_explicit_nondefault_feature_group_size() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={"feature_group_size": 32},
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


def test_non_staged_arch_rejects_stage() -> None:
    with pytest.raises(ValueError, match="model.stage"):
        _ = ModelBuildSpec(task="classification", arch="tabfoundry", stage="nano_exact")


def test_checkpoint_build_spec_round_trips_model_arch() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_simple"},
        state_dict={},
    )

    assert spec.arch == "tabfoundry_simple"
