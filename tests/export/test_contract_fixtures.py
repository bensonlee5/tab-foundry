from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.export.contracts import (
    ExportModelSpec,
    ExportPreprocessorState,
    LegacyPreprocessorState,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
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
    assert manifest.model.missingness_mode == "none"
    assert manifest.model.norm_type == "layernorm"
    assert manifest.model.tfrow_norm == "layernorm"


def test_manifest_v3_fixture_validates_and_roundtrips_embedded_sections() -> None:
    payload = _load_fixture("manifest_v3.json")
    manifest = validate_manifest_dict(payload)
    assert manifest.schema_version == SCHEMA_VERSION_V3
    assert manifest.model.input_normalization == "train_zscore"
    assert manifest.model.missingness_mode == "none"
    assert manifest.model.norm_type == "layernorm"
    assert manifest.model.tfrow_norm == "layernorm"
    assert manifest.manifest_sha256 is not None
    assert manifest.inference is not None
    assert manifest.preprocessor is not None
    assert manifest.weights is not None
    assert manifest.inference.group_shifts == [0, 1, 3]
    assert manifest.inference.missingness_mode == "none"
    assert manifest.inference.many_class_inference_mode == "full_probs"
    assert isinstance(manifest.preprocessor, ExportPreprocessorState)
    assert manifest.preprocessor.impute_missing is True
    assert manifest.preprocessor.classification_label_policy is not None
    assert manifest.preprocessor.classification_label_policy.mapping == "train_only_remap"
    assert manifest.weights.file == "weights.safetensors"

    build_spec = manifest.model.to_build_spec(task=manifest.task)
    roundtrip = ExportModelSpec.from_build_spec(build_spec)

    assert roundtrip.to_dict() == manifest.model.to_dict()
    assert build_spec.input_normalization == "train_zscore"
    assert build_spec.norm_type == "layernorm"
    assert build_spec.tfrow_norm == "layernorm"


def test_manifest_v2_validation_requires_explicit_model_reconstruction_fields() -> None:
    payload = _load_fixture("manifest_v2.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    for key in (
        "input_normalization",
        "norm_type",
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tfrow_norm",
        "tficl_n_heads",
        "tficl_n_layers",
        "tficl_ff_expansion",
        "many_class_base",
        "head_hidden_dim",
        "use_digit_position_embed",
    ):
        model_payload.pop(key, None)
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="manifest.model keys mismatch"):
        validate_manifest_dict(payload)


def test_manifest_v3_validation_requires_input_normalization() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    model_payload.pop("input_normalization", None)
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="manifest.model keys mismatch"):
        validate_manifest_dict(payload)


def test_manifest_v3_validation_requires_manifest_sha256() -> None:
    payload = _load_fixture("manifest_v3.json")
    payload.pop("manifest_sha256", None)

    with pytest.raises(ValueError, match="manifest.manifest_sha256"):
        validate_manifest_dict(payload)


def test_manifest_v3_validation_rejects_malformed_manifest_sha256() -> None:
    payload = _load_fixture("manifest_v3.json")
    payload["manifest_sha256"] = "bad"

    with pytest.raises(ValueError, match="manifest.manifest_sha256"):
        validate_manifest_dict(payload)


def test_manifest_v3_validation_rejects_invalid_input_normalization() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_raw = payload["model"]
    assert isinstance(model_raw, dict)
    model_payload = dict(model_raw)
    model_payload["input_normalization"] = "bogus"
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="input_normalization"):
        validate_manifest_dict(payload)


def test_v2_section_fixtures_validate() -> None:
    v2_inference_payload = _load_fixture("inference_config_classification_v2.json")
    v2_preproc_payload = _load_fixture("preprocessor_state_v2.json")

    v2_cfg = validate_inference_config_dict(v2_inference_payload)
    v2_state = validate_preprocessor_state_dict(v2_preproc_payload, schema_version=SCHEMA_VERSION_V2)

    assert v2_cfg.model_arch == "tabfoundry"
    assert v2_cfg.missingness_mode == "none"
    assert v2_cfg.feature_group_size == 1
    assert isinstance(v2_state, LegacyPreprocessorState)
    assert v2_state.impute_missing is True
    assert v2_state.feature_order_policy == "lexicographic_f_columns"
    assert v2_state.classification_label_policy["unseen_test_label"] == "filter"


def test_v3_section_validation_supports_classification_and_regression_policies() -> None:
    manifest_payload = _load_fixture("manifest_v3.json")
    cls_inference = validate_inference_config_dict(manifest_payload["inference"])
    cls_preprocessor = validate_preprocessor_state_dict(
        manifest_payload["preprocessor"],
        schema_version=SCHEMA_VERSION_V3,
        task="classification",
    )
    reg_preprocessor = validate_preprocessor_state_dict(
        {
            "feature_order_policy": "positional_feature_ids",
            "missing_value_policy": {
                "strategy": "train_mean",
                "all_nan_fill": 0.0,
            },
            "impute_missing": True,
            "classification_label_policy": None,
            "dtype_policy": {
                "features": "float32",
                "classification_labels": "int64",
                "regression_targets": "float32",
            },
        },
        schema_version=SCHEMA_VERSION_V3,
        task="regression",
    )

    assert cls_inference.many_class_inference_mode == "full_probs"
    assert cls_inference.missingness_mode == "none"
    assert isinstance(cls_preprocessor, ExportPreprocessorState)
    assert cls_preprocessor.impute_missing is True
    assert cls_preprocessor.classification_label_policy is not None
    assert cls_preprocessor.classification_label_policy.unseen_test_label == "filter"
    assert isinstance(reg_preprocessor, ExportPreprocessorState)
    assert reg_preprocessor.classification_label_policy is None


def test_manifest_validation_rejects_old_model_arch() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_payload = dict(payload["model"])
    model_payload["arch"] = "tabiclv2"
    payload["model"] = model_payload

    with pytest.raises(ValueError, match="Unsupported model arch"):
        validate_manifest_dict(payload)


def test_v3_manifest_validation_rejects_old_inference_model_arch() -> None:
    payload = _load_fixture("manifest_v3.json")
    inference_payload = dict(payload["inference"])
    inference_payload["model_arch"] = "tabiclv2"
    payload["inference"] = inference_payload

    with pytest.raises(ValueError, match="Unsupported inference model_arch"):
        validate_manifest_dict(payload)


def test_manifest_validation_accepts_additive_staged_model_stage_fields() -> None:
    payload = _load_fixture("manifest_v3.json")
    model_payload = dict(payload["model"])
    model_payload["arch"] = "tabfoundry_staged"
    model_payload["stage"] = "nano_exact"
    model_payload["input_normalization"] = "train_zscore_clip"
    model_payload["many_class_base"] = 2
    payload["model"] = model_payload

    inference_payload = dict(payload["inference"])
    inference_payload["model_arch"] = "tabfoundry_staged"
    inference_payload["model_stage"] = "nano_exact"
    payload["inference"] = inference_payload

    manifest = validate_manifest_dict(payload)

    assert manifest.model.arch == "tabfoundry_staged"
    assert manifest.model.stage == "nano_exact"
    assert manifest.inference is not None
    assert manifest.inference.model_stage == "nano_exact"
