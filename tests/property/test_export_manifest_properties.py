from __future__ import annotations

import copy
import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from tab_foundry.export.contracts import (
    canonicalize_v3_manifest_payload,
    compute_v3_manifest_sha256,
    ExportPreprocessorState,
    SCHEMA_VERSION_V3,
    validate_preprocessor_state_dict,
)


def _load_manifest_v3_fixture() -> dict[str, object]:
    fixture = Path(__file__).resolve().parents[1] / "export" / "fixtures" / "manifest_v3.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _reordered_mapping(payload: dict[str, object], order: tuple[str, ...]) -> dict[str, object]:
    return {key: payload[key] for key in order}


@st.composite
def _manifest_ordering_case(draw: st.DrawFn) -> tuple[dict[str, object], dict[str, object]]:
    payload = _load_manifest_v3_fixture()
    top_level_keys = tuple(payload.keys())
    model_keys = tuple(payload["model"].keys())  # type: ignore[index]
    inference_keys = tuple(payload["inference"].keys())  # type: ignore[index]
    preproc_keys = tuple(payload["preprocessor"].keys())  # type: ignore[index]
    payload_variant = copy.deepcopy(payload)
    payload_variant = _reordered_mapping(payload_variant, draw(st.permutations(top_level_keys)))
    payload_variant["model"] = _reordered_mapping(
        copy.deepcopy(payload["model"]),  # type: ignore[arg-type]
        draw(st.permutations(model_keys)),
    )
    payload_variant["inference"] = _reordered_mapping(
        copy.deepcopy(payload["inference"]),  # type: ignore[arg-type]
        draw(st.permutations(inference_keys)),
    )
    payload_variant["preprocessor"] = _reordered_mapping(
        copy.deepcopy(payload["preprocessor"]),  # type: ignore[arg-type]
        draw(st.permutations(preproc_keys)),
    )
    return payload, payload_variant


@settings(deadline=None, max_examples=30)
@given(case=_manifest_ordering_case())
def test_canonicalize_v3_manifest_payload_is_stable_across_dict_order(
    case: tuple[dict[str, object], dict[str, object]],
) -> None:
    original, reordered = case

    assert canonicalize_v3_manifest_payload(original) == canonicalize_v3_manifest_payload(reordered)
    assert compute_v3_manifest_sha256(original) == compute_v3_manifest_sha256(reordered)


@settings(deadline=None, max_examples=20)
@given(manifest_sha256=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64))
def test_manifest_sha_ignores_only_manifest_sha256_field(manifest_sha256: str) -> None:
    payload = _load_manifest_v3_fixture()
    original_sha = compute_v3_manifest_sha256(payload)
    payload["manifest_sha256"] = manifest_sha256

    assert compute_v3_manifest_sha256(payload) == original_sha


@settings(deadline=None, max_examples=25)
@given(
    mutation=st.sampled_from(
        [
            ("producer", "version", "9.9.9"),
            ("model", "d_icl", 1024),
            ("model", "input_normalization", "train_zscore_clip"),
            ("inference", "feature_group_size", 2),
            ("weights", "file", "other_weights.safetensors"),
            ("created_at_utc", None, "2026-03-14T00:00:00Z"),
        ]
    )
)
def test_manifest_sha_changes_when_semantic_fields_change(
    mutation: tuple[str, str | None, object],
) -> None:
    payload = _load_manifest_v3_fixture()
    original_sha = compute_v3_manifest_sha256(payload)
    section, key, value = mutation
    if key is None:
        payload[section] = value
    else:
        nested = payload[section]
        assert isinstance(nested, dict)
        nested[key] = value

    assert compute_v3_manifest_sha256(payload) != original_sha


@settings(deadline=None, max_examples=25)
@given(
    mutation=st.sampled_from(
        [
            ("missing_top", "producer"),
            ("missing_model", "arch"),
            ("extra_inference", "extra_key"),
            ("extra_preprocessor", "extra_key"),
        ]
    )
)
def test_manifest_validators_reject_missing_or_extra_section_keys(
    mutation: tuple[str, str],
) -> None:
    from tab_foundry.export.contracts import validate_manifest_dict

    payload = _load_manifest_v3_fixture()
    kind, key = mutation
    if kind == "missing_top":
        payload.pop(key, None)
    elif kind == "missing_model":
        model = payload["model"]
        assert isinstance(model, dict)
        model.pop(key, None)
    elif kind == "extra_inference":
        inference = payload["inference"]
        assert isinstance(inference, dict)
        inference[key] = "boom"
    elif kind == "extra_preprocessor":
        preprocessor = payload["preprocessor"]
        assert isinstance(preprocessor, dict)
        preprocessor[key] = "boom"

    with pytest.raises(ValueError, match="keys mismatch"):
        validate_manifest_dict(payload)


def test_validate_preprocessor_state_dict_roundtrips_valid_v3_payloads() -> None:
    payload: dict[str, object] = {
        "feature_order_policy": "positional_feature_ids",
        "missing_value_policy": {
            "strategy": "train_mean",
            "all_nan_fill": 0.0,
            "impute_missing": True,
        },
        "classification_label_policy": {
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }

    validated = validate_preprocessor_state_dict(
        payload,
        schema_version=SCHEMA_VERSION_V3,
        task="classification",
    )

    assert isinstance(validated, ExportPreprocessorState)
    assert validated.to_dict() == payload


def test_validate_preprocessor_state_dict_defaults_missing_impute_missing_to_true() -> None:
    payload: dict[str, object] = {
        "feature_order_policy": "positional_feature_ids",
        "missing_value_policy": {
            "strategy": "train_mean",
            "all_nan_fill": 0.0,
        },
        "classification_label_policy": {
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }

    validated = validate_preprocessor_state_dict(
        payload,
        schema_version=SCHEMA_VERSION_V3,
        task="classification",
    )

    assert isinstance(validated, ExportPreprocessorState)
    assert validated.missing_value_policy.impute_missing is True


@pytest.mark.parametrize("all_nan_fill", [float("nan"), float("inf"), float("-inf")])
def test_validate_preprocessor_state_dict_rejects_nonfinite_all_nan_fill(
    all_nan_fill: float,
) -> None:
    payload: dict[str, object] = {
        "feature_order_policy": "positional_feature_ids",
        "missing_value_policy": {
            "strategy": "train_mean",
            "all_nan_fill": all_nan_fill,
            "impute_missing": True,
        },
        "classification_label_policy": {
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }

    with pytest.raises(ValueError, match="all_nan_fill must be finite"):
        validate_preprocessor_state_dict(
            payload,
            schema_version=SCHEMA_VERSION_V3,
            task="classification",
        )


@settings(deadline=None, max_examples=25)
@given(task=st.sampled_from(("regression", "REGRESSION")))
def test_validate_preprocessor_state_dict_rejects_unsupported_task(task: str) -> None:
    payload: dict[str, object] = {
        "feature_order_policy": "positional_feature_ids",
        "missing_value_policy": {
            "strategy": "train_mean",
            "all_nan_fill": 0.0,
            "impute_missing": True,
        },
        "classification_label_policy": {
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }

    with pytest.raises(ValueError, match="Unsupported preprocessor_state task"):
        validate_preprocessor_state_dict(
            payload,
            schema_version=SCHEMA_VERSION_V3,
            task=task,
        )


def test_validate_preprocessor_state_dict_rejects_missing_classification_policy() -> None:
    payload: dict[str, object] = {
        "feature_order_policy": "positional_feature_ids",
        "missing_value_policy": {
            "strategy": "train_mean",
            "all_nan_fill": 0.0,
            "impute_missing": True,
        },
        "classification_label_policy": None,
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }

    with pytest.raises(ValueError, match="classification_label_policy"):
        validate_preprocessor_state_dict(
            payload,
            schema_version=SCHEMA_VERSION_V3,
            task="classification",
        )
