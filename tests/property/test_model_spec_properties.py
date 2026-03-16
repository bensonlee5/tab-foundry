from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import pytest
import torch

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.spec import (
    _coerce_bool,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
    resolve_model_stage,
    STAGED_MODEL_ARCH,
    SUPPORTED_MODEL_ARCHES,
    SUPPORTED_MODEL_TASKS,
)


def _case_variants(token: str) -> st.SearchStrategy[str]:
    return st.sampled_from([token, token.upper(), token.title(), f" {token.upper()} "])


def _checkpoint_model_cfg(*, task: str, **overrides: object) -> dict[str, object]:
    return model_build_spec_from_mappings(task=task, primary=overrides).to_dict()


def _simple_state_dict(missingness_mode: str) -> dict[str, torch.Tensor]:
    if missingness_mode == "feature_mask":
        return {"feature_encoder.linear_layer.weight": torch.zeros((96, 2))}
    if missingness_mode == "explicit_token":
        return {
            "feature_encoder.linear_layer.weight": torch.zeros((96, 1)),
            "feature_encoder.nan_embedding": torch.zeros((96,)),
        }
    return {"feature_encoder.linear_layer.weight": torch.zeros((96, 1))}


@given(stage=st.one_of(st.none(), st.just(""), _case_variants("nano_exact")))
def test_resolve_model_stage_defaults_to_nano_exact_for_staged_arch(stage: str | None) -> None:
    assert resolve_model_stage(arch=STAGED_MODEL_ARCH, stage=stage) == "nano_exact"


@given(
    arch=st.sampled_from([arch for arch in SUPPORTED_MODEL_ARCHES if arch != STAGED_MODEL_ARCH]),
    stage=st.sampled_from(["nano_exact", "row_cls_pool", "qass_context"]),
)
def test_resolve_model_stage_rejects_stage_for_non_staged_arches(arch: str, stage: str) -> None:
    with pytest.raises(ValueError, match="model.stage is only supported"):
        _ = resolve_model_stage(arch=arch, stage=stage)


@given(
    value=st.sampled_from(
        [
            True,
            False,
            0,
            1,
            "0",
            "1",
            "true",
            "false",
            "TRUE",
            "FALSE",
            " yes ",
            " no ",
            "on",
            "off",
        ]
    )
)
def test_coerce_bool_accepts_supported_boolean_tokens(value: bool | int | str) -> None:
    expected = value
    if isinstance(value, str):
        expected = value.strip().lower() in {"1", "true", "yes", "on"}
    assert _coerce_bool(value, context="value") is bool(expected)


@given(
    value=st.one_of(
        st.integers().filter(lambda item: item not in {0, 1}),
        st.floats(allow_nan=False, allow_infinity=False).filter(lambda item: item not in {0.0, 1.0}),
        st.text(min_size=1).filter(
            lambda item: item.strip().lower() not in {"0", "1", "true", "false", "yes", "no", "on", "off"}
        ),
    )
)
def test_coerce_bool_rejects_non_boolean_compatible_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be boolean-compatible"):
        _ = _coerce_bool(value, context="value")


@given(
    task=st.sampled_from(SUPPORTED_MODEL_TASKS),
    primary_arch=st.sampled_from(SUPPORTED_MODEL_ARCHES),
    primary_input_normalization=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES),
    explicit_feature_group_size=st.integers(min_value=1, max_value=64),
)
def test_model_build_spec_primary_mapping_normalizes_case_and_uses_explicit_values(
    task: str,
    primary_arch: str,
    primary_input_normalization: str,
    explicit_feature_group_size: int,
) -> None:
    spec = model_build_spec_from_mappings(
        task=task.upper(),
        primary={
            "arch": primary_arch.upper(),
            "input_normalization": primary_input_normalization.upper(),
            "feature_group_size": explicit_feature_group_size,
        },
    )

    assert spec.task == task
    assert spec.arch == primary_arch
    assert spec.input_normalization == primary_input_normalization
    assert spec.feature_group_size == explicit_feature_group_size


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS))
def test_model_build_spec_none_in_primary_uses_internal_defaults(task: str) -> None:
    spec = model_build_spec_from_mappings(
        task=task,
        primary={
            "arch": None,
            "input_normalization": None,
            "feature_group_size": None,
        },
    )

    assert spec.arch == "tabfoundry"
    assert spec.input_normalization == "none"
    assert spec.feature_group_size == 1


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS))
def test_checkpoint_build_spec_requires_explicit_reconstruction_metadata(task: str) -> None:
    with pytest.raises(ValueError, match="missing required reconstruction fields"):
        _ = checkpoint_model_build_spec_from_mappings(
            task=task,
            primary={},
            state_dict={"group_linear.weight": torch.zeros((128, 3))},
        )


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS), feature_group_size=st.integers(min_value=2, max_value=64))
def test_checkpoint_build_spec_preserves_explicit_nondefault_feature_group_size(
    task: str,
    feature_group_size: int,
) -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=_checkpoint_model_cfg(
            task=task,
            feature_group_size=feature_group_size,
            missingness_mode="none",
        ),
        state_dict={"group_linear.weight": torch.zeros((128, feature_group_size * 3))},
    )

    assert spec.feature_group_size == feature_group_size


@given(
    task=st.sampled_from(SUPPORTED_MODEL_TASKS),
    feature_group_size=st.integers(min_value=3, max_value=63).filter(lambda value: value % 2 == 1),
)
def test_checkpoint_build_spec_rejects_missing_feature_group_size_metadata(
    task: str,
    feature_group_size: int,
) -> None:
    primary = _checkpoint_model_cfg(task=task, missingness_mode="none")
    primary.pop("feature_group_size")

    with pytest.raises(ValueError, match="missing required reconstruction fields: feature_group_size"):
        _ = checkpoint_model_build_spec_from_mappings(
            task=task,
            primary=primary,
            state_dict={"group_linear.weight": torch.zeros((128, feature_group_size * 3))},
        )


@given(
    missingness_mode=st.sampled_from(["none", "feature_mask", "explicit_token"]),
)
def test_checkpoint_build_spec_accepts_explicit_simple_missingness_overrides(
    missingness_mode: str,
) -> None:
    primary = _checkpoint_model_cfg(task="classification", arch="tabfoundry_simple")
    primary.pop("missingness_mode")

    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary=primary,
        explicit_overrides={"missingness_mode": missingness_mode},
        state_dict=_simple_state_dict(missingness_mode),
    )

    assert spec.arch == "tabfoundry_simple"
    assert spec.missingness_mode == missingness_mode
