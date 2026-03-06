from __future__ import annotations

import json
from pathlib import Path

from tab_foundry.export.contracts import (
    validate_inference_config_dict,
    validate_manifest_dict,
    validate_preprocessor_state_dict,
)


def _load_fixture(name: str) -> dict[str, object]:
    fixture = Path(__file__).resolve().parent / "fixtures" / name
    with fixture.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def test_manifest_fixture_validates() -> None:
    payload = _load_fixture("manifest_v1.json")
    manifest = validate_manifest_dict(payload)
    assert manifest.schema_version == "tab-foundry-export-v1"
    assert manifest.task == "classification"
    assert manifest.model.tfcol_n_heads == 8
    assert manifest.model.tficl_n_layers == 12
    assert manifest.model.many_class_base == 10
    assert manifest.model.head_hidden_dim == 1024
    assert manifest.model.use_digit_position_embed is True


def test_inference_fixture_validates() -> None:
    payload = _load_fixture("inference_config_classification_v1.json")
    inference_cfg = validate_inference_config_dict(payload)
    assert inference_cfg.group_shifts == [0, 1, 3]
    assert inference_cfg.many_class_threshold == 10
    assert inference_cfg.many_class_inference_mode == "full_probs"


def test_preprocessor_fixture_validates() -> None:
    payload = _load_fixture("preprocessor_state_v1.json")
    state = validate_preprocessor_state_dict(payload)
    assert state.feature_order_policy == "lexicographic_f_columns"
    assert state.missing_value_policy["strategy"] == "train_mean"
    assert state.missing_value_policy["all_nan_fill"] == 0.0
    assert state.classification_label_policy["mapping"] == "train_only_remap"
    assert state.classification_label_policy["unseen_test_label"] == "filter"
    assert state.dtype_policy["features"] == "float32"
