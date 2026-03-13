from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.export.contracts import (
    ExportModelSpec,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    FittedPreprocessorState,
    LegacyPreprocessorState,
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


def test_manifest_v2_fixture_validates() -> None:
    payload = _load_fixture("manifest_v2.json")
    manifest = validate_manifest_dict(payload)
    assert manifest.schema_version == SCHEMA_VERSION_V2
    assert manifest.task == "classification"
    assert manifest.model.feature_group_size == 1
    assert manifest.model.input_normalization == "none"


def test_manifest_v3_fixture_validates_and_roundtrips_input_normalization() -> None:
    payload = _load_fixture("manifest_v3.json")
    manifest = validate_manifest_dict(payload)
    assert manifest.schema_version == SCHEMA_VERSION_V3
    assert manifest.model.input_normalization == "train_zscore"

    build_spec = manifest.model.to_build_spec(task=manifest.task)
    roundtrip = ExportModelSpec.from_build_spec(build_spec)

    assert roundtrip.to_dict() == manifest.model.to_dict()
    assert build_spec.input_normalization == "train_zscore"


def test_manifest_v2_validation_applies_model_defaults_via_canonical_spec() -> None:
    payload = _load_fixture("manifest_v2.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    for key in (
        "input_normalization",
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

    assert manifest.model.input_normalization == "none"
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


def test_manifest_v3_validation_requires_input_normalization() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    model_payload.pop("input_normalization", None)
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="manifest.model keys mismatch"):
        validate_manifest_dict(payload)


def test_inference_fixtures_validate() -> None:
    v2_payload = _load_fixture("inference_config_classification_v2.json")
    v3_payload = _load_fixture("inference_config_classification_v3.json")

    v2_cfg = validate_inference_config_dict(v2_payload)
    v3_cfg = validate_inference_config_dict(v3_payload)

    assert v2_cfg.model_arch == "tabfoundry"
    assert v2_cfg.feature_group_size == 1
    assert v3_cfg.group_shifts == [0, 1, 3]
    assert v3_cfg.many_class_inference_mode == "full_probs"


def test_preprocessor_fixtures_validate_for_v2_and_v3() -> None:
    v2_payload = _load_fixture("preprocessor_state_v2.json")
    v3_cls_payload = _load_fixture("preprocessor_state_classification_v3.json")
    v3_reg_payload = _load_fixture("preprocessor_state_regression_v3.json")

    v2_state = validate_preprocessor_state_dict(v2_payload, schema_version=SCHEMA_VERSION_V2)
    v3_cls_state = validate_preprocessor_state_dict(
        v3_cls_payload,
        schema_version=SCHEMA_VERSION_V3,
        task="classification",
    )
    v3_reg_state = validate_preprocessor_state_dict(
        v3_reg_payload,
        schema_version=SCHEMA_VERSION_V3,
        task="regression",
    )

    assert isinstance(v2_state, LegacyPreprocessorState)
    assert v2_state.feature_order_policy == "lexicographic_f_columns"
    assert v2_state.classification_label_policy["unseen_test_label"] == "filter"

    assert isinstance(v3_cls_state, FittedPreprocessorState)
    assert v3_cls_state.feature_ids == [0, 1, 2]
    assert v3_cls_state.missing_value_policy.fill_values == [1.5, 2.5, 0.0]
    assert v3_cls_state.classification_label_policy is not None
    assert v3_cls_state.classification_label_policy.label_values == [3, 7]

    assert isinstance(v3_reg_state, FittedPreprocessorState)
    assert v3_reg_state.classification_label_policy is None
    assert v3_reg_state.feature_ids == [0, 1]


def test_manifest_validation_rejects_old_model_arch() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_payload = dict(payload["model"])
    model_payload["arch"] = "tabiclv2"
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="Unsupported model arch"):
        validate_manifest_dict(payload)


def test_inference_validation_rejects_old_model_arch() -> None:
    payload = _load_fixture("inference_config_classification_v3.json")
    payload["model_arch"] = "tabiclv2"

    with pytest.raises(ValueError, match="Unsupported inference model_arch"):
        validate_inference_config_dict(payload)
