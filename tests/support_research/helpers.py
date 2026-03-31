from __future__ import annotations

from typing import Any, Mapping


def assert_training_surface_semantics(
    payload: Mapping[str, Any],
    *,
    training_experiment: str,
    surface_role: str,
    training_config_profile: str | None = None,
    comparison_policy: str | None = None,
    external_benchmarks: list[str] | None = None,
) -> None:
    assert payload["training_experiment"] == training_experiment
    assert payload["training_config_profile"] == (
        training_experiment if training_config_profile is None else training_config_profile
    )
    assert payload["surface_role"] == surface_role
    if comparison_policy is not None:
        assert payload["comparison_policy"] == comparison_policy
    if external_benchmarks is not None:
        assert payload["external_benchmarks"] == external_benchmarks
