from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st
from omegaconf import OmegaConf
import pytest

from tab_foundry.training.runtime import resolve_grad_accum_steps
from tab_foundry.training.schedule import build_stage_configs, StageConfig, warmup_steps_for_stage


_NAME_TEXT = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=24)


@given(
    steps=st.integers(min_value=1, max_value=20_000),
    warmup_ratio=st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
)
def test_warmup_steps_for_stage_stays_within_valid_bounds(steps: int, warmup_ratio: float) -> None:
    warmup_steps = warmup_steps_for_stage(
        StageConfig(
            name="stage",
            steps=steps,
            lr_max=1.0e-3,
            warmup_ratio=warmup_ratio,
        )
    )

    assert 0 <= warmup_steps <= steps - 1
    if steps <= 1 or warmup_ratio <= 0.0:
        assert warmup_steps == 0
    else:
        assert warmup_steps >= 1


@given(steps=st.integers(min_value=1, max_value=20_000))
def test_warmup_steps_for_stage_is_zero_for_zero_warmup_ratio(steps: int) -> None:
    warmup_steps = warmup_steps_for_stage(
        StageConfig(
            name="stage",
            steps=steps,
            lr_max=1.0e-3,
            warmup_ratio=0.0,
        )
    )

    assert warmup_steps == 0


@given(
    stages=st.lists(
        st.fixed_dictionaries(
            {
                "name": _NAME_TEXT,
                "steps": st.integers(min_value=1, max_value=10_000),
                "lr_max": st.floats(min_value=1.0e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
                "lr_schedule": st.sampled_from(["cosine", "linear", "COSINE", "LINEAR"]),
                "warmup_ratio": st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
            }
        ),
        min_size=1,
        max_size=6,
    )
)
def test_build_stage_configs_accepts_valid_positive_steps_and_valid_warmup_ratios(
    stages: list[dict[str, object]],
) -> None:
    resolved = build_stage_configs(stages)

    assert len(resolved) == len(stages)
    for raw, stage in zip(stages, resolved, strict=True):
        assert stage.name == raw["name"]
        assert stage.steps == raw["steps"]
        assert stage.lr_max == pytest.approx(float(raw["lr_max"]))
        assert stage.lr_schedule == str(raw["lr_schedule"]).strip().lower()
        assert stage.warmup_ratio == pytest.approx(float(raw["warmup_ratio"]))


@given(steps=st.integers(max_value=0))
def test_build_stage_configs_rejects_non_positive_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="stage steps must be >= 1"):
        _ = build_stage_configs([{"name": "stage", "steps": steps, "lr_max": 1.0e-3}])


@given(
    warmup_ratio=st.one_of(
        st.floats(max_value=-1.0e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
def test_build_stage_configs_rejects_out_of_range_warmup_ratios(warmup_ratio: float) -> None:
    with pytest.raises(ValueError, match="stage warmup_ratio must be in \\[0, 1\\)"):
        _ = build_stage_configs(
            [{"name": "stage", "steps": 10, "lr_max": 1.0e-3, "warmup_ratio": warmup_ratio}]
        )


@given(
    cfg_steps=st.integers(min_value=1, max_value=1_000),
    override_steps=st.integers(min_value=1, max_value=1_000),
)
def test_resolve_grad_accum_steps_prefers_override_and_never_returns_below_one(
    cfg_steps: int,
    override_steps: int,
) -> None:
    cfg = OmegaConf.create({"grad_accum_steps": cfg_steps})

    assert resolve_grad_accum_steps(cfg) == cfg_steps
    assert resolve_grad_accum_steps(cfg, override=override_steps) == override_steps
    assert resolve_grad_accum_steps(cfg) >= 1
    assert resolve_grad_accum_steps(cfg, override=override_steps) >= 1


@given(
    cfg_steps=st.integers(min_value=1, max_value=1_000),
    invalid_override=st.integers(max_value=0),
)
def test_resolve_grad_accum_steps_rejects_invalid_overrides(
    cfg_steps: int,
    invalid_override: int,
) -> None:
    cfg = OmegaConf.create({"grad_accum_steps": cfg_steps})

    with pytest.raises(ValueError, match="runtime.grad_accum_steps must be >= 1"):
        _ = resolve_grad_accum_steps(cfg, override=invalid_override)
