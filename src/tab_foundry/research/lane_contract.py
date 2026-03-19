"""Shared lane and sweep-surface contract helpers."""

from __future__ import annotations

from typing import Any, Mapping


PFN_CONTROL_LANE = "pfn_control"
HYBRID_DIAGNOSTIC_LANE = "hybrid_diagnostic"
ARCHITECTURE_SCREEN_LANE = "architecture_screen"
CUSTOM_LANE = "custom"

PFN_CONTROL_LANE_LABEL = "tabfoundry_simple plus tabfoundry_staged stage=nano_exact"
HYBRID_DIAGNOSTIC_LANE_LABEL = (
    "tabfoundry_staged hybrid diagnostic surfaces built from nano_exact plus bounded overrides"
)
PFN_CONTROL_SURFACES = frozenset(
    {
        "cls_benchmark_linear_simple",
        "cls_benchmark_linear_simple_prior",
    }
)
ARCHITECTURE_SCREEN_SURFACE = "cls_benchmark_staged"
HYBRID_DIAGNOSTIC_SURFACE = "cls_benchmark_staged_prior"

DEFAULT_TRAINING_EXPERIMENT = HYBRID_DIAGNOSTIC_SURFACE
DEFAULT_TRAINING_CONFIG_PROFILE = HYBRID_DIAGNOSTIC_SURFACE


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def resolve_training_experiment(sweep_meta: Mapping[str, Any]) -> str:
    explicit = _non_empty_string(sweep_meta.get("training_experiment"))
    if explicit is not None:
        return explicit
    return DEFAULT_TRAINING_EXPERIMENT


def resolve_training_config_profile(sweep_meta: Mapping[str, Any]) -> str:
    explicit = _non_empty_string(sweep_meta.get("training_config_profile"))
    if explicit is not None:
        return explicit
    return resolve_training_experiment(sweep_meta)


def resolve_surface_role(sweep_meta: Mapping[str, Any]) -> str:
    explicit = _non_empty_string(sweep_meta.get("surface_role"))
    if explicit is not None:
        return explicit
    training_experiment = resolve_training_experiment(sweep_meta)
    if training_experiment in PFN_CONTROL_SURFACES:
        return PFN_CONTROL_LANE
    if training_experiment == HYBRID_DIAGNOSTIC_SURFACE:
        return HYBRID_DIAGNOSTIC_LANE
    if training_experiment == ARCHITECTURE_SCREEN_SURFACE:
        return ARCHITECTURE_SCREEN_LANE
    return CUSTOM_LANE
