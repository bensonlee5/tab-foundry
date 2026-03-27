"""Shared lane and sweep-surface contract helpers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


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
CORPUS_SCREEN_SURFACE = "cls_benchmark_staged_corpus"
HYBRID_DIAGNOSTIC_SURFACE = "cls_benchmark_staged_prior"
LEGACY_ARCHITECTURE_SCREEN_SURFACE = "cls_benchmark_staged"
ARCHITECTURE_SCREEN_SURFACE = CORPUS_SCREEN_SURFACE
ARCHITECTURE_SCREEN_SURFACES = frozenset(
    {
        LEGACY_ARCHITECTURE_SCREEN_SURFACE,
        ARCHITECTURE_SCREEN_SURFACE,
    }
)

DEFAULT_TRAINING_EXPERIMENT = CORPUS_SCREEN_SURFACE
DEFAULT_TRAINING_CONFIG_PROFILE = CORPUS_SCREEN_SURFACE
LEGACY_FALLBACK_TRAINING_EXPERIMENT = HYBRID_DIAGNOSTIC_SURFACE
LEGACY_FALLBACK_TRAINING_CONFIG_PROFILE = HYBRID_DIAGNOSTIC_SURFACE


class _StringKeyLookup(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def resolve_training_experiment(sweep_meta: Mapping[str, Any] | _StringKeyLookup) -> str:
    explicit = _non_empty_string(sweep_meta.get("training_experiment"))
    if explicit is not None:
        return explicit
    return LEGACY_FALLBACK_TRAINING_EXPERIMENT


def resolve_training_config_profile(sweep_meta: Mapping[str, Any] | _StringKeyLookup) -> str:
    explicit = _non_empty_string(sweep_meta.get("training_config_profile"))
    if explicit is not None:
        return explicit
    explicit_training_experiment = _non_empty_string(sweep_meta.get("training_experiment"))
    if explicit_training_experiment is not None:
        return explicit_training_experiment
    return LEGACY_FALLBACK_TRAINING_CONFIG_PROFILE


def resolve_surface_role(sweep_meta: Mapping[str, Any] | _StringKeyLookup) -> str:
    explicit = _non_empty_string(sweep_meta.get("surface_role"))
    if explicit is not None:
        return explicit
    training_experiment = resolve_training_experiment(sweep_meta)
    if training_experiment in PFN_CONTROL_SURFACES:
        return PFN_CONTROL_LANE
    if training_experiment == HYBRID_DIAGNOSTIC_SURFACE:
        return HYBRID_DIAGNOSTIC_LANE
    if training_experiment in ARCHITECTURE_SCREEN_SURFACES:
        return ARCHITECTURE_SCREEN_LANE
    return CUSTOM_LANE
