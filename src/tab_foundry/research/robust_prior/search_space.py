"""Discrete Dagzoo search-space definitions for robust-prior pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ROBUST_PRIOR_SEARCH_SPACE_V1 = "robust_prior_search_space_v1"


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One categorical controller dimension."""

    name: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RobustPriorProposal:
    """One decoded proposal in the discrete Dagzoo search space."""

    feature_count_bucket: str
    class_count_bucket: str
    categorical_ratio_bucket: str
    max_categorical_cardinality_bucket: str
    graph_node_bucket: str
    target_depth_bucket: str
    mechanism_preset: str
    shift_preset: str
    noise_preset: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_count_bucket": self.feature_count_bucket,
            "class_count_bucket": self.class_count_bucket,
            "categorical_ratio_bucket": self.categorical_ratio_bucket,
            "max_categorical_cardinality_bucket": self.max_categorical_cardinality_bucket,
            "graph_node_bucket": self.graph_node_bucket,
            "target_depth_bucket": self.target_depth_bucket,
            "mechanism_preset": self.mechanism_preset,
            "shift_preset": self.shift_preset,
            "noise_preset": self.noise_preset,
        }


_FEATURE_BUCKETS: dict[str, dict[str, int]] = {
    "compact": {"min": 16, "max": 24},
    "balanced": {"min": 24, "max": 40},
    "wide": {"min": 40, "max": 56},
    "wider": {"min": 56, "max": 64},
}
_CLASS_BUCKETS: dict[str, dict[str, int]] = {
    "binaryish": {"min": 2, "max": 3},
    "small": {"min": 3, "max": 5},
    "medium": {"min": 5, "max": 7},
    "many": {"min": 8, "max": 10},
}
_CATEGORICAL_RATIO_BUCKETS: dict[str, dict[str, float]] = {
    "continuous": {"min": 0.0, "max": 0.05},
    "mixed_low": {"min": 0.15, "max": 0.35},
    "mixed_high": {"min": 0.45, "max": 0.70},
    "categorical": {"min": 0.75, "max": 1.0},
}
_CARDINALITY_BUCKETS: dict[str, int] = {
    "card9": 9,
    "card16": 16,
    "card24": 24,
    "card32": 32,
}
_GRAPH_BUCKETS: dict[str, dict[str, int]] = {
    "graph_small": {"min": 4, "max": 12},
    "graph_medium": {"min": 8, "max": 20},
    "graph_large": {"min": 12, "max": 28},
    "graph_max": {"min": 20, "max": 32},
}
_TARGET_DEPTH_BUCKETS: dict[str, dict[str, float]] = {
    "shallow": {"ratio_min": 0.15, "ratio_max": 0.30, "ratio_mid": 0.22},
    "mid": {"ratio_min": 0.35, "ratio_max": 0.55, "ratio_mid": 0.45},
    "deep": {"ratio_min": 0.60, "ratio_max": 0.85, "ratio_mid": 0.72},
}
_MECHANISM_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"function_family_mix": None, "nonlinearity_mass": 0.25},
    "piecewise": {
        "function_family_mix": {"piecewise": 1.25, "linear": 0.75},
        "nonlinearity_mass": 0.55,
    },
    "gp_bias": {
        "function_family_mix": {"gp": 1.5, "linear": 0.75, "quadratic": 0.75},
        "nonlinearity_mass": 0.65,
    },
    "compositional": {
        "function_family_mix": {
            "piecewise": 2.25,
            "product": 1.75,
            "gp": 1.25,
            "tree": 1.25,
            "nn": 1.0,
            "discretization": 1.0,
            "em": 1.0,
            "quadratic": 0.9,
            "linear": 0.75,
        },
        "nonlinearity_mass": 0.85,
    },
}
_SHIFT_PRESETS: dict[str, dict[str, Any]] = {
    "none": {"enabled": False, "graph_scale": 0.0, "variance_scale": 0.0, "mechanism_scale": 0.0},
    "noise_drift": {
        "enabled": True,
        "mode": "noise_drift",
        "graph_scale": 0.0,
        "variance_scale": 0.35,
        "mechanism_scale": 0.0,
    },
    "mixed": {
        "enabled": True,
        "mode": "mixed",
        "graph_scale": 0.35,
        "variance_scale": 0.35,
        "mechanism_scale": 0.0,
    },
    "mechanism_drift": {
        "enabled": True,
        "mode": "mechanism_drift",
        "graph_scale": 0.0,
        "variance_scale": 0.0,
        "mechanism_scale": 0.35,
    },
}
_NOISE_PRESETS: dict[str, dict[str, Any]] = {
    "gaussian": {"family": "gaussian", "base_scale": 1.0},
    "laplace": {"family": "laplace", "base_scale": 1.0},
    "student_t": {"family": "student_t", "base_scale": 1.0, "student_t_df": 6.0},
    "mixture": {
        "family": "mixture",
        "base_scale": 1.0,
        "student_t_df": 6.0,
        "mixture_weights": {"gaussian": 0.50, "laplace": 0.30, "student_t": 0.20},
    },
}


def _coerce_distribution_map(distribution: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.float64)
        for key, value in distribution.items()
    }


class RobustPriorSearchSpace:
    """Discrete proposal surface plus Dagzoo override mapping."""

    def __init__(self, *, search_space_id: str = ROBUST_PRIOR_SEARCH_SPACE_V1) -> None:
        if str(search_space_id).strip() != ROBUST_PRIOR_SEARCH_SPACE_V1:
            raise RuntimeError(f"unknown robust-prior search space: {search_space_id!r}")
        self.search_space_id = ROBUST_PRIOR_SEARCH_SPACE_V1
        self.dimensions = (
            SearchDimension("feature_count_bucket", tuple(_FEATURE_BUCKETS)),
            SearchDimension("class_count_bucket", tuple(_CLASS_BUCKETS)),
            SearchDimension("categorical_ratio_bucket", tuple(_CATEGORICAL_RATIO_BUCKETS)),
            SearchDimension("max_categorical_cardinality_bucket", tuple(_CARDINALITY_BUCKETS)),
            SearchDimension("graph_node_bucket", tuple(_GRAPH_BUCKETS)),
            SearchDimension("target_depth_bucket", tuple(_TARGET_DEPTH_BUCKETS)),
            SearchDimension("mechanism_preset", tuple(_MECHANISM_PRESETS)),
            SearchDimension("shift_preset", tuple(_SHIFT_PRESETS)),
            SearchDimension("noise_preset", tuple(_NOISE_PRESETS)),
        )
        self._dimension_by_name = {dimension.name: dimension for dimension in self.dimensions}

    def dimension_names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    def encode(self, proposal: RobustPriorProposal) -> list[int]:
        encoded: list[int] = []
        proposal_dict = proposal.to_dict()
        for dimension in self.dimensions:
            value = proposal_dict[dimension.name]
            try:
                encoded.append(dimension.values.index(value))
            except ValueError as exc:
                raise RuntimeError(
                    f"proposal value {value!r} is not valid for dimension {dimension.name!r}"
                ) from exc
        return encoded

    def decode(self, encoded: Sequence[int]) -> RobustPriorProposal:
        if len(encoded) != len(self.dimensions):
            raise RuntimeError("encoded proposal length does not match the robust-prior search space")
        values: dict[str, str] = {}
        for index, dimension in zip(encoded, self.dimensions, strict=True):
            if int(index) < 0 or int(index) >= len(dimension.values):
                raise RuntimeError(
                    f"encoded value {index!r} is out of range for dimension {dimension.name!r}"
                )
            values[dimension.name] = dimension.values[int(index)]
        return RobustPriorProposal(**values)

    def sample_uniform(self, rng: np.random.Generator) -> RobustPriorProposal:
        return RobustPriorProposal(
            **{
                dimension.name: str(rng.choice(dimension.values))
                for dimension in self.dimensions
            }
        )

    def sample_from_distribution(
        self,
        distribution: Mapping[str, Sequence[float]],
        *,
        rng: np.random.Generator,
    ) -> RobustPriorProposal:
        normalized_distribution = _coerce_distribution_map(distribution)
        values: dict[str, str] = {}
        for dimension in self.dimensions:
            probabilities = normalized_distribution[dimension.name]
            if probabilities.shape != (len(dimension.values),):
                raise RuntimeError(
                    f"distribution for {dimension.name!r} must have shape {(len(dimension.values),)}"
                )
            clipped = np.clip(probabilities.astype(np.float64), 0.0, None)
            if float(clipped.sum()) <= 0.0:
                raise RuntimeError(f"distribution for {dimension.name!r} must have positive mass")
            clipped /= float(clipped.sum())
            values[dimension.name] = str(rng.choice(dimension.values, p=clipped))
        return RobustPriorProposal(**values)

    def authored_depth_ratio_band(self, proposal: RobustPriorProposal) -> tuple[float, float]:
        band = _TARGET_DEPTH_BUCKETS[proposal.target_depth_bucket]
        return float(band["ratio_min"]), float(band["ratio_max"])

    def nonlinearity_mass(self, proposal: RobustPriorProposal) -> float:
        return float(_MECHANISM_PRESETS[proposal.mechanism_preset]["nonlinearity_mass"])

    def shift_diagnostics(self, proposal: RobustPriorProposal) -> dict[str, float]:
        preset = _SHIFT_PRESETS[proposal.shift_preset]
        return {
            "shift_enabled": 1.0 if bool(preset["enabled"]) else 0.0,
            "shift_graph_scale": float(preset.get("graph_scale", 0.0)),
            "shift_variance_scale": float(preset.get("variance_scale", 0.0)),
            "shift_mechanism_scale": float(preset.get("mechanism_scale", 0.0)),
        }

    def proposal_vector(self, proposal: RobustPriorProposal) -> dict[str, float]:
        feature_bucket = _FEATURE_BUCKETS[proposal.feature_count_bucket]
        class_bucket = _CLASS_BUCKETS[proposal.class_count_bucket]
        categorical_bucket = _CATEGORICAL_RATIO_BUCKETS[proposal.categorical_ratio_bucket]
        graph_bucket = _GRAPH_BUCKETS[proposal.graph_node_bucket]
        return {
            "feature_count_center": 0.5 * float(feature_bucket["min"] + feature_bucket["max"]),
            "class_count_center": 0.5 * float(class_bucket["min"] + class_bucket["max"]),
            "categorical_ratio_center": 0.5
            * float(categorical_bucket["min"] + categorical_bucket["max"]),
            "max_categorical_cardinality": float(
                _CARDINALITY_BUCKETS[proposal.max_categorical_cardinality_bucket]
            ),
            "graph_node_center": 0.5 * float(graph_bucket["min"] + graph_bucket["max"]),
            "mechanism_nonlinearity_mass": self.nonlinearity_mass(proposal),
            **self.shift_diagnostics(proposal),
        }

    def proposal_to_overrides(self, proposal: RobustPriorProposal) -> dict[str, Any]:
        feature_bucket = _FEATURE_BUCKETS[proposal.feature_count_bucket]
        class_bucket = _CLASS_BUCKETS[proposal.class_count_bucket]
        categorical_bucket = _CATEGORICAL_RATIO_BUCKETS[proposal.categorical_ratio_bucket]
        graph_bucket = _GRAPH_BUCKETS[proposal.graph_node_bucket]
        depth_band = _TARGET_DEPTH_BUCKETS[proposal.target_depth_bucket]
        graph_nodes_max = int(graph_bucket["max"])
        depth_min_nodes = max(1, int(round(graph_nodes_max * float(depth_band["ratio_min"]))))
        depth_max_nodes = max(depth_min_nodes, int(round(graph_nodes_max * float(depth_band["ratio_max"]))))
        depth_max_nodes = min(graph_nodes_max, depth_max_nodes)
        mechanism_mix = _MECHANISM_PRESETS[proposal.mechanism_preset]["function_family_mix"]
        shift_preset = _SHIFT_PRESETS[proposal.shift_preset]
        noise_preset = _NOISE_PRESETS[proposal.noise_preset]
        overrides: dict[str, Any] = {
            "dataset": {
                "task": "classification",
                "n_train": 768,
                "n_test": 256,
                "n_features_min": int(feature_bucket["min"]),
                "n_features_max": int(feature_bucket["max"]),
                "n_classes_min": int(class_bucket["min"]),
                "n_classes_max": int(class_bucket["max"]),
                "categorical_ratio_min": float(categorical_bucket["min"]),
                "categorical_ratio_max": float(categorical_bucket["max"]),
                "max_categorical_cardinality": int(
                    _CARDINALITY_BUCKETS[proposal.max_categorical_cardinality_bucket]
                ),
                "missing_rate": 0.0,
                "missing_mechanism": "none",
                "missing_mar_observed_fraction": 0.5,
                "missing_mar_logit_scale": 1.0,
                "missing_mnar_logit_scale": 1.0,
            },
            "graph": {
                "n_nodes_min": int(graph_bucket["min"]),
                "n_nodes_max": int(graph_bucket["max"]),
                "target_depth_nodes_min": int(depth_min_nodes),
                "target_depth_nodes_max": int(depth_max_nodes),
            },
            "noise": dict(noise_preset),
            "shift": dict(shift_preset),
        }
        if mechanism_mix is not None:
            overrides["mechanism"] = {"function_family_mix": dict(mechanism_mix)}
        else:
            overrides["mechanism"] = {"function_family_mix": None}
        return overrides


def robust_prior_search_space_v1() -> RobustPriorSearchSpace:
    """Return the default robust-prior v1 controller surface."""

    return RobustPriorSearchSpace(search_space_id=ROBUST_PRIOR_SEARCH_SPACE_V1)
