from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tab_foundry.export.contracts import ExportModelSpec, InferenceConfig, validate_inference_config_dict
from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.missingness import SUPPORTED_MISSINGNESS_MODES
from tab_foundry.model.spec import (
    model_build_spec_from_mappings,
    ModelStage,
    STAGED_MODEL_ARCH,
    SUPPORTED_MANY_CLASS_TRAIN_MODES,
    SUPPORTED_MODEL_ARCHES,
    SUPPORTED_MODEL_TASKS,
)


def _case_variants(token: str) -> st.SearchStrategy[str]:
    return st.sampled_from([token, token.upper(), token.title(), f" {token.upper()} "])


@st.composite
def _exportable_model_spec(
    draw: st.DrawFn,
) -> tuple[str, object]:
    task = draw(st.sampled_from(SUPPORTED_MODEL_TASKS))
    arch = draw(st.sampled_from(SUPPORTED_MODEL_ARCHES))
    primary: dict[str, object] = {
        "arch": arch,
        "d_col": draw(st.integers(min_value=1, max_value=256)),
        "d_icl": draw(st.integers(min_value=1, max_value=512)),
        "input_normalization": draw(st.sampled_from(SUPPORTED_INPUT_NORMALIZATION_MODES)),
        "missingness_mode": draw(st.sampled_from(SUPPORTED_MISSINGNESS_MODES)),
        "feature_group_size": draw(st.integers(min_value=1, max_value=16)),
        "many_class_train_mode": draw(st.sampled_from(SUPPORTED_MANY_CLASS_TRAIN_MODES)),
        "max_mixed_radix_digits": draw(st.integers(min_value=1, max_value=128)),
        "tfcol_n_heads": draw(st.integers(min_value=1, max_value=16)),
        "tfcol_n_layers": draw(st.integers(min_value=1, max_value=8)),
        "tfcol_n_inducing": draw(st.integers(min_value=1, max_value=256)),
        "tfrow_n_heads": draw(st.integers(min_value=1, max_value=16)),
        "tfrow_n_layers": draw(st.integers(min_value=1, max_value=8)),
        "tfrow_cls_tokens": draw(st.integers(min_value=1, max_value=8)),
        "tficl_n_heads": draw(st.integers(min_value=1, max_value=16)),
        "tficl_n_layers": draw(st.integers(min_value=1, max_value=16)),
        "tficl_ff_expansion": draw(st.integers(min_value=1, max_value=8)),
        "many_class_base": draw(st.integers(min_value=2, max_value=16)),
        "head_hidden_dim": draw(st.integers(min_value=1, max_value=1024)),
        "use_digit_position_embed": draw(st.booleans()),
    }
    if arch == STAGED_MODEL_ARCH:
        stage = draw(st.one_of(st.none(), st.sampled_from([stage.value for stage in ModelStage])))
        if stage is not None:
            primary["stage"] = stage
    spec = model_build_spec_from_mappings(task=task, primary=primary)
    return task, spec


@st.composite
def _valid_inference_payload(draw: st.DrawFn) -> tuple[dict[str, object], str | None]:
    task = draw(st.sampled_from(SUPPORTED_MODEL_TASKS))
    arch = draw(st.sampled_from(SUPPORTED_MODEL_ARCHES))
    expected_stage: str | None = None
    payload: dict[str, object] = {
        "task": task,
        "model_arch": arch,
        "missingness_mode": draw(st.sampled_from(SUPPORTED_MISSINGNESS_MODES)),
        "group_shifts": [0, 1, 3],
        "feature_group_size": draw(st.integers(min_value=1, max_value=16)),
        "many_class_threshold": 10,
        "many_class_inference_mode": "full_probs",
    }
    if arch == STAGED_MODEL_ARCH:
        stage_token = draw(_case_variants("nano_exact"))
        expected_stage = "nano_exact"
        payload["model_stage"] = stage_token
    if task == "regression":
        payload["quantile_levels"] = [float(index) / 1_000.0 for index in range(1, 1_000)]
    return payload, expected_stage


@settings(deadline=None, max_examples=35)
@given(case=_exportable_model_spec())
def test_export_model_spec_roundtrips_supported_build_spec_fields(
    case: tuple[str, object],
) -> None:
    task, build_spec = case

    export_spec = ExportModelSpec.from_build_spec(build_spec)
    roundtrip_spec = export_spec.to_build_spec(task=task)

    assert ExportModelSpec.from_build_spec(roundtrip_spec).to_dict() == export_spec.to_dict()


@settings(deadline=None, max_examples=35)
@given(case=_valid_inference_payload())
def test_validate_inference_config_dict_normalizes_supported_payloads(
    case: tuple[dict[str, object], str | None],
) -> None:
    payload, expected_stage = case

    validated = validate_inference_config_dict(payload)
    rendered = validated.to_dict()

    assert validated.task == payload["task"]
    assert validated.model_arch == payload["model_arch"]
    assert validated.missingness_mode == payload["missingness_mode"]
    assert validated.feature_group_size == payload["feature_group_size"]
    assert validated.many_class_threshold == 10
    assert validated.many_class_inference_mode == "full_probs"
    if expected_stage is not None:
        assert validated.model_stage == expected_stage
        assert rendered["model_stage"] == expected_stage
    else:
        assert validated.model_stage is None
        assert "model_stage" not in rendered
    if payload["task"] == "regression":
        assert validated.quantile_levels is not None
        assert len(validated.quantile_levels) == 999
        assert rendered["quantile_levels"] == validated.quantile_levels
    else:
        assert validated.quantile_levels is None
        assert "quantile_levels" not in rendered


@settings(deadline=None, max_examples=35)
@given(
    arch=st.sampled_from(SUPPORTED_MODEL_ARCHES),
    feature_group_size=st.integers(min_value=1, max_value=16),
    many_class_threshold=st.integers(min_value=1, max_value=32),
    quantile_levels=st.one_of(
        st.none(),
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=1,
            max_size=5,
        ),
    ),
)
def test_inference_config_to_dict_drops_only_none_fields(
    arch: str,
    feature_group_size: int,
    many_class_threshold: int,
    quantile_levels: list[float] | None,
) -> None:
    model_stage = "nano_exact" if arch == STAGED_MODEL_ARCH else None
    config = InferenceConfig(
        task="classification",
        model_arch=arch,
        model_stage=model_stage,
        missingness_mode="none",
        group_shifts=[0, 1, 3],
        feature_group_size=feature_group_size,
        many_class_threshold=many_class_threshold,
        many_class_inference_mode="full_probs",
        quantile_levels=quantile_levels,
    )

    payload = config.to_dict()

    if model_stage is None:
        assert "model_stage" not in payload
    else:
        assert payload["model_stage"] == "nano_exact"
    if quantile_levels is None:
        assert "quantile_levels" not in payload
    else:
        assert payload["quantile_levels"] == quantile_levels
