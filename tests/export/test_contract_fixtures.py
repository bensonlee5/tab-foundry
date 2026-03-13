from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.export.contracts import (
    ExportModelSpec,
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
    payload = _load_fixture("manifest_v2.json")
    manifest = validate_manifest_dict(payload)
    assert manifest.schema_version == "tab-foundry-export-v2"
    assert manifest.task == "classification"
    assert manifest.model.feature_group_size == 1
    assert manifest.model.tfcol_n_heads == 8
    assert manifest.model.tficl_n_layers == 12
    assert manifest.model.many_class_base == 10
    assert manifest.model.head_hidden_dim == 1024
    assert manifest.model.use_digit_position_embed is True


def test_manifest_fixture_model_roundtrips_through_canonical_build_spec() -> None:
    payload = _load_fixture("manifest_v2.json")
    manifest = validate_manifest_dict(payload)

    build_spec = manifest.model.to_build_spec(task=manifest.task)
    roundtrip = ExportModelSpec.from_build_spec(build_spec)

    assert roundtrip.to_dict() == manifest.model.to_dict()
    assert build_spec.task == "classification"
    assert build_spec.input_normalization == "none"


def test_manifest_validation_applies_model_defaults_via_canonical_spec() -> None:
    payload = _load_fixture("manifest_v2.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    for key in (
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tficl_n_heads",
        "tficl_n_layers",
        "tficl_ff_expansion",
        "many_class_base",
        "head_hidden_dim",
        "use_digit_position_embed",
    ):
        model_payload.pop(key, None)
    payload["model"] = model_payload

    manifest = validate_manifest_dict(payload)

    assert manifest.model.tfcol_n_heads == 8
    assert manifest.model.tfcol_n_layers == 3
    assert manifest.model.tfcol_n_inducing == 128
    assert manifest.model.tfrow_n_heads == 8
    assert manifest.model.tfrow_n_layers == 3
    assert manifest.model.tfrow_cls_tokens == 4
    assert manifest.model.tficl_n_heads == 8
    assert manifest.model.tficl_n_layers == 12
    assert manifest.model.tficl_ff_expansion == 2
    assert manifest.model.many_class_base == 10
    assert manifest.model.head_hidden_dim == 1024
    assert manifest.model.use_digit_position_embed is True


def test_inference_fixture_validates() -> None:
    payload = _load_fixture("inference_config_classification_v2.json")
    inference_cfg = validate_inference_config_dict(payload)
    assert inference_cfg.model_arch == "tabfoundry"
    assert inference_cfg.feature_group_size == 1
    assert inference_cfg.group_shifts == [0, 1, 3]
    assert inference_cfg.many_class_threshold == 10
    assert inference_cfg.many_class_inference_mode == "full_probs"


def test_preprocessor_fixture_validates() -> None:
    payload = _load_fixture("preprocessor_state_v2.json")
    state = validate_preprocessor_state_dict(payload)
    assert state.feature_order_policy == "lexicographic_f_columns"
    assert state.missing_value_policy["strategy"] == "train_mean"
    assert state.missing_value_policy["all_nan_fill"] == 0.0
    assert state.classification_label_policy["mapping"] == "train_only_remap"
    assert state.classification_label_policy["unseen_test_label"] == "filter"
    assert state.dtype_policy["features"] == "float32"


def test_manifest_validation_rejects_old_model_arch() -> None:
    payload = _load_fixture("manifest_v2.json")
    model_payload = dict(payload["model"])
    model_payload["arch"] = "tabiclv2"
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="Unsupported model arch"):
        validate_manifest_dict(payload)


def test_inference_validation_rejects_old_model_arch() -> None:
    payload = _load_fixture("inference_config_classification_v2.json")
    payload["model_arch"] = "tabiclv2"

    with pytest.raises(ValueError, match="Unsupported inference model_arch"):
        validate_inference_config_dict(payload)
