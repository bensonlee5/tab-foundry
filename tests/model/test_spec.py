from __future__ import annotations

import pytest
import torch

from tab_foundry.model.factory import build_model
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)


def test_model_build_spec_defaults_feature_group_size_to_one() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={},
    )

    assert spec.feature_group_size == 1


def test_build_model_defaults_feature_group_size_to_one() -> None:
    cls_model = build_model(task="classification")
    reg_model = build_model(task="regression")

    assert getattr(cls_model, "feature_group_size") == 1
    assert getattr(reg_model, "feature_group_size") == 1


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
