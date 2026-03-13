from __future__ import annotations

from tab_foundry.model.factory import build_model
from tab_foundry.model.spec import model_build_spec_from_mappings


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
