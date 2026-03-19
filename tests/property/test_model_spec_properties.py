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
    fallback_arch=st.sampled_from(SUPPORTED_MODEL_ARCHES),
    primary_input_normalization=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES),
    fallback_input_normalization=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES),
    fallback_feature_group_size=st.integers(min_value=1, max_value=64),
)
def test_model_build_spec_primary_mapping_takes_precedence_and_normalizes_case(
    task: str,
    primary_arch: str,
    fallback_arch: str,
    primary_input_normalization: str,
    fallback_input_normalization: str,
    fallback_feature_group_size: int,
) -> None:
    spec = model_build_spec_from_mappings(
        task=task.upper(),
        primary={
            "arch": primary_arch.upper(),
            "input_normalization": primary_input_normalization.upper(),
            "feature_group_size": None,
        },
        fallback={
            "arch": fallback_arch,
            "input_normalization": fallback_input_normalization,
            "feature_group_size": fallback_feature_group_size,
        },
    )

    assert spec.task == task
    assert spec.arch == primary_arch
    assert spec.input_normalization == primary_input_normalization
    assert spec.feature_group_size == fallback_feature_group_size


@given(
    task=st.sampled_from(SUPPORTED_MODEL_TASKS),
    fallback_arch=st.sampled_from(SUPPORTED_MODEL_ARCHES),
    fallback_input_normalization=st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES),
    fallback_feature_group_size=st.integers(min_value=1, max_value=64),
)
def test_model_build_spec_none_in_primary_does_not_erase_fallback(
    task: str,
    fallback_arch: str,
    fallback_input_normalization: str,
    fallback_feature_group_size: int,
) -> None:
    spec = model_build_spec_from_mappings(
        task=task,
        primary={
            "arch": None,
            "input_normalization": None,
            "feature_group_size": None,
        },
        fallback={
            "arch": fallback_arch,
            "input_normalization": fallback_input_normalization,
            "feature_group_size": fallback_feature_group_size,
        },
    )

    assert spec.arch == fallback_arch
    assert spec.input_normalization == fallback_input_normalization
    assert spec.feature_group_size == fallback_feature_group_size


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS))
def test_checkpoint_build_spec_defaults_feature_group_size_without_legacy_weights(task: str) -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task=task,
        primary={},
        state_dict={},
    )

    assert spec.feature_group_size == 1


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS), feature_group_size=st.integers(min_value=2, max_value=64))
def test_checkpoint_build_spec_preserves_explicit_nondefault_feature_group_size_without_legacy_weights(
    task: str,
    feature_group_size: int,
) -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task=task,
        primary={"feature_group_size": feature_group_size},
        state_dict={},
    )

    assert spec.feature_group_size == feature_group_size


@given(task=st.sampled_from(SUPPORTED_MODEL_TASKS), feature_group_size=st.integers(min_value=2, max_value=64))
def test_checkpoint_build_spec_rejects_legacy_tabfoundry_state_dict(
    task: str,
    feature_group_size: int,
) -> None:
    with pytest.raises(ValueError, match="Legacy tabfoundry checkpoints are no longer supported"):
        _ = checkpoint_model_build_spec_from_mappings(
            task=task,
            primary={"feature_group_size": feature_group_size},
            state_dict={"group_linear.weight": torch.zeros((128, feature_group_size * 3))},
        )
