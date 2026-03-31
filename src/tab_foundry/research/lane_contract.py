"""Canonical sweep-semantics contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
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

LEGACY_FALLBACK_TRAINING_EXPERIMENT = HYBRID_DIAGNOSTIC_SURFACE
LEGACY_FALLBACK_TRAINING_CONFIG_PROFILE = HYBRID_DIAGNOSTIC_SURFACE
DEFAULT_COMPARISON_POLICY = "anchor_only"


class _StringKeyLookup(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainingSurfaceContext:
    training_experiment: str
    training_config_profile: str
    surface_role: str

    def to_payload_dict(self) -> dict[str, str]:
        return {
            "training_experiment": self.training_experiment,
            "training_config_profile": self.training_config_profile,
            "surface_role": self.surface_role,
        }


@dataclass(frozen=True, slots=True)
class SweepSemantics:
    training_surface: TrainingSurfaceContext
    comparison_policy: str = DEFAULT_COMPARISON_POLICY

    def to_payload_dict(self) -> dict[str, str]:
        return {
            **self.training_surface.to_payload_dict(),
            "comparison_policy": self.comparison_policy,
        }


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _surface_role_for_training_experiment(training_experiment: str) -> str:
    if training_experiment in PFN_CONTROL_SURFACES:
        return PFN_CONTROL_LANE
    if training_experiment == HYBRID_DIAGNOSTIC_SURFACE:
        return HYBRID_DIAGNOSTIC_LANE
    if training_experiment in ARCHITECTURE_SCREEN_SURFACES:
        return ARCHITECTURE_SCREEN_LANE
    return CUSTOM_LANE


def resolve_training_surface_context(
    sweep_meta: Mapping[str, Any] | _StringKeyLookup,
) -> TrainingSurfaceContext:
    training_experiment = (
        _non_empty_string(sweep_meta.get("training_experiment"))
        or LEGACY_FALLBACK_TRAINING_EXPERIMENT
    )
    training_config_profile = (
        _non_empty_string(sweep_meta.get("training_config_profile"))
        or _non_empty_string(sweep_meta.get("training_experiment"))
        or LEGACY_FALLBACK_TRAINING_CONFIG_PROFILE
    )
    surface_role = (
        _non_empty_string(sweep_meta.get("surface_role"))
        or _surface_role_for_training_experiment(training_experiment)
    )
    return TrainingSurfaceContext(
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
    )


def resolve_sweep_semantics(
    sweep_meta: Mapping[str, Any] | _StringKeyLookup,
) -> SweepSemantics:
    return SweepSemantics(
        training_surface=resolve_training_surface_context(sweep_meta),
        comparison_policy=(
            _non_empty_string(sweep_meta.get("comparison_policy"))
            or DEFAULT_COMPARISON_POLICY
        ),
    )


def resolve_new_sweep_training_surface(
    *,
    template_sweep: Mapping[str, Any] | _StringKeyLookup | None,
    training_experiment: str | None,
    training_config_profile: str | None,
    surface_role: str | None,
) -> TrainingSurfaceContext:
    explicit_training_experiment = _non_empty_string(training_experiment)
    explicit_training_config_profile = _non_empty_string(training_config_profile)
    explicit_surface_role = _non_empty_string(surface_role)
    if template_sweep is None:
        if (
            explicit_training_experiment is None
            or explicit_training_config_profile is None
            or explicit_surface_role is None
        ):
            raise RuntimeError(
                "create_sweep requires --parent-sweep-id or all of "
                "--training-experiment, --training-config-profile, and --surface-role"
            )
        return TrainingSurfaceContext(
            training_experiment=explicit_training_experiment,
            training_config_profile=explicit_training_config_profile,
            surface_role=explicit_surface_role,
        )

    inherited = resolve_training_surface_context(template_sweep)
    resolved_training_experiment = explicit_training_experiment or inherited.training_experiment
    resolved_training_config_profile = (
        explicit_training_config_profile
        or explicit_training_experiment
        or inherited.training_config_profile
    )
    resolved_surface_role = (
        explicit_surface_role
        or (
            _surface_role_for_training_experiment(explicit_training_experiment)
            if explicit_training_experiment is not None
            else inherited.surface_role
        )
    )
    return TrainingSurfaceContext(
        training_experiment=resolved_training_experiment,
        training_config_profile=resolved_training_config_profile,
        surface_role=resolved_surface_role,
    )
