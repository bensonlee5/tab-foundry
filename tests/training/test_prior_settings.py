from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tab_foundry.training.prior.settings import PriorMissingnessConfig, PriorRuntimeConfig


def test_prior_runtime_config_coerces_positive_ints_and_bool_flags() -> None:
    runtime_cfg = OmegaConf.create(
        {
            "max_steps": "8",
            "eval_every": 2.9,
            "checkpoint_every": "3",
            "trace_activations": "yes",
            "output_dir": "ignored-by-runtime-parser",
        }
    )

    resolved = PriorRuntimeConfig.from_runtime_cfg(runtime_cfg)

    assert resolved.max_steps == 8
    assert resolved.eval_every == 2
    assert resolved.checkpoint_every == 3
    assert resolved.trace_activations is True


@pytest.mark.parametrize(
    ("runtime_cfg", "message"),
    [
        (
            {"max_steps": 0, "eval_every": 1, "checkpoint_every": 1},
            "runtime.max_steps must be >= 1",
        ),
        (
            {"max_steps": 1, "eval_every": 1, "checkpoint_every": 1, "trace_activations": "maybe"},
            "runtime.trace_activations must be boolean-compatible",
        ),
    ],
)
def test_prior_runtime_config_rejects_invalid_values(
    runtime_cfg: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = PriorRuntimeConfig.from_runtime_cfg(OmegaConf.create(runtime_cfg))


def test_prior_missingness_config_defaults_max_rate_to_min_rate() -> None:
    overrides_cfg = OmegaConf.create(
        {
            "prior_missingness": {
                "enabled": "true",
                "min_rate": "0.25",
            }
        }
    )

    resolved = PriorMissingnessConfig.from_training_overrides(overrides_cfg)

    assert resolved is not None
    assert resolved.enabled is True
    assert resolved.min_rate == pytest.approx(0.25)
    assert resolved.max_rate == pytest.approx(0.25)
    runtime_dict = resolved.to_runtime_dict()
    assert runtime_dict["enabled"] is True
    assert runtime_dict["min_rate"] == pytest.approx(0.25)
    assert runtime_dict["max_rate"] == pytest.approx(0.25)


def test_prior_missingness_config_returns_none_when_disabled() -> None:
    overrides_cfg = OmegaConf.create(
        {
            "prior_missingness": {
                "enabled": 0,
                "min_rate": 0.25,
                "max_rate": 0.5,
            }
        }
    )

    assert PriorMissingnessConfig.from_training_overrides(overrides_cfg) is None


@pytest.mark.parametrize(
    ("overrides_cfg", "message"),
    [
        (
            {"prior_missingness": {"enabled": True, "min_rate": 0.7, "max_rate": 0.2}},
            "training.overrides.prior_missingness.min_rate must be <= max_rate",
        ),
        (
            {"prior_missingness": {"enabled": True, "min_rate": 1.2}},
            "training.overrides.prior_missingness.min_rate must be in \\[0, 1\\]",
        ),
        (
            {"prior_missingness": ["not", "a", "mapping"]},
            "training.overrides.prior_missingness must resolve to a mapping",
        ),
    ],
)
def test_prior_missingness_config_rejects_invalid_values(
    overrides_cfg: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = PriorMissingnessConfig.from_training_overrides(OmegaConf.create(overrides_cfg))
