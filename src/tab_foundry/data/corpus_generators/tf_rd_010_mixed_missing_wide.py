"""TF-RD-010 mixed-missingness wide exact-shape recipe generator."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping


_ROW_SPECS = {
    128: (96, 32),
    256: (192, 64),
    512: (384, 128),
    1024: (768, 256),
}
_DEFAULT_FEATURE_GRAPH_BANDS = {
    6: (2, 12),
    10: (2, 20),
    14: (4, 28),
    20: (2, 20),
    32: (4, 32),
    64: (4, 64),
    100: (4, 100),
}
_BALANCED_CLASS_COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
_LATENT_TARGET_DERIVATION = "tabiclv2_latent_node"
_DEFAULT_MISSINGNESS_REGIMES = (
    {"mechanism": "mcar", "rate": 0.15},
    {"mechanism": "mar", "rate": 0.15},
    {"mechanism": "mnar", "rate": 0.15},
)
_VALID_MISSINGNESS_MECHANISMS = {"mcar", "mar", "mnar"}
_FEATURE_GRAPH_BAND_WIDTH = 2


def _coerce_feature_graph_bands(raw_value: Any) -> dict[int, tuple[int, int]]:
    if raw_value is None:
        return dict(_DEFAULT_FEATURE_GRAPH_BANDS)
    if not isinstance(raw_value, Mapping):
        raise ValueError("feature_graph_bands must be a mapping from feature count to [min, max]")
    bands: dict[int, tuple[int, int]] = {}
    for key, raw_band in raw_value.items():
        feature_count = int(key)
        if feature_count <= 0:
            raise ValueError(f"feature_graph_bands keys must be positive, got {key!r}")
        if not isinstance(raw_band, (list, tuple)) or len(raw_band) != _FEATURE_GRAPH_BAND_WIDTH:
            raise ValueError(f"feature_graph_bands[{key!r}] must be a two-item sequence")
        graph_min = int(raw_band[0])
        graph_max = int(raw_band[1])
        if graph_min <= 0 or graph_max < graph_min:
            raise ValueError(
                f"feature_graph_bands[{key!r}] must satisfy 0 < min <= max, got {raw_band!r}"
            )
        bands[feature_count] = (graph_min, graph_max)
    if not bands:
        raise ValueError("feature_graph_bands must not be empty")
    return dict(sorted(bands.items()))


def _coerce_missingness_regimes(raw_value: Any) -> tuple[dict[str, Any], ...]:
    regimes_raw = _DEFAULT_MISSINGNESS_REGIMES if raw_value is None else raw_value
    if not isinstance(regimes_raw, (list, tuple)) or not regimes_raw:
        raise ValueError("missingness_regimes must be a non-empty list")
    regimes: list[dict[str, Any]] = []
    for index, item in enumerate(regimes_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"missingness_regimes[{index}] must be a mapping")
        mechanism = str(item.get("mechanism", "")).strip().lower()
        if mechanism not in _VALID_MISSINGNESS_MECHANISMS:
            raise ValueError(
                f"missingness_regimes[{index}].mechanism must be one of "
                f"{sorted(_VALID_MISSINGNESS_MECHANISMS)}, got {mechanism!r}"
            )
        rate = float(item.get("rate", 0.0))
        if not 0.0 < rate < 1.0:
            raise ValueError(f"missingness_regimes[{index}].rate must be in (0, 1), got {rate}")
        regimes.append({"mechanism": mechanism, "rate": rate})
    return tuple(regimes)


def _missingness_payload(regime: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = str(regime["mechanism"])
    payload: dict[str, Any] = {
        "missing_rate": float(regime["rate"]),
        "missing_mechanism": mechanism,
    }
    if mechanism == "mar":
        payload["missing_mar_observed_fraction"] = 0.6
        payload["missing_mar_logit_scale"] = 1.4
    elif mechanism == "mnar":
        payload["missing_mnar_logit_scale"] = 1.4
    return payload


def build_mixed_missing_wide_recipe(
    *,
    recipe_id: str,
    description: str,
    surface_label: str,
    manifest: Mapping[str, Any],
    provenance_labels: Mapping[str, Any],
    inputs: Mapping[str, Any],
    recipe_path: str | None = None,
) -> dict[str, Any]:
    del recipe_id, description, surface_label, manifest, provenance_labels, recipe_path
    base_config_ref = str(inputs.get("base_config_ref", "configs/default.yaml"))
    num_datasets = int(inputs.get("num_datasets", 256))
    filter_max_attempts = int(inputs.get("filter_max_attempts", 512))
    feature_graph_bands = _coerce_feature_graph_bands(inputs.get("feature_graph_bands"))
    missingness_regimes = _coerce_missingness_regimes(inputs.get("missingness_regimes"))
    invocations: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(missingness_regimes, start=1):
        mechanism = str(regime["mechanism"])
        for row_index, row_total in enumerate(sorted(_ROW_SPECS), start=1):
            n_train, n_test = _ROW_SPECS[row_total]
            for feature_count, class_count in product(
                sorted(feature_graph_bands),
                _BALANCED_CLASS_COUNTS,
            ):
                graph_min, graph_max = feature_graph_bands[feature_count]
                invocation = {
                    "invocation_id": f"{mechanism}_r{row_total:04d}_f{feature_count:03d}_c{class_count:02d}",
                    "base_config_ref": base_config_ref,
                    "config_overrides": {
                        "dataset": {
                            "task": "classification",
                            "n_train": n_train,
                            "n_test": n_test,
                            "n_features_min": feature_count,
                            "n_features_max": feature_count,
                            "n_classes_min": class_count,
                            "n_classes_max": class_count,
                            "categorical_ratio_min": 0.0,
                            "categorical_ratio_max": 1.0,
                            "max_categorical_cardinality": 12,
                        },
                        "graph": {
                            "n_nodes_min": graph_min,
                            "n_nodes_max": graph_max,
                        },
                        "filter": {
                            "max_attempts": filter_max_attempts,
                        },
                    },
                    "num_datasets": num_datasets,
                    "seed": int(f"{regime_index}{row_index}{feature_count:03d}{class_count:02d}"),
                    "device": "cpu",
                    "hardware_policy": "none",
                    **_missingness_payload(regime),
                }
                invocations.append(invocation)
    return {
        "invocations": invocations,
        "review_summary": {
            "grid_family": "mixed_missing_exact_shape_wide_v1",
            "config_refs": [base_config_ref],
            "invocation_count": (
                len(_ROW_SPECS)
                * len(feature_graph_bands)
                * len(_BALANCED_CLASS_COUNTS)
                * len(missingness_regimes)
            ),
            "manifest_record_count": len(invocations) * int(num_datasets),
            "row_totals": sorted(_ROW_SPECS),
            "feature_counts": sorted(feature_graph_bands),
            "feature_graph_bands": {
                str(feature_count): list(band)
                for feature_count, band in sorted(feature_graph_bands.items())
            },
            "class_counts": list(_BALANCED_CLASS_COUNTS),
            "num_datasets_per_invocation": int(num_datasets),
            "missingness_regimes": [
                {"mechanism": str(regime["mechanism"]), "rate": float(regime["rate"])}
                for regime in missingness_regimes
            ],
            "target_derivation": _LATENT_TARGET_DERIVATION,
        },
    }


__all__ = ["build_mixed_missing_wide_recipe"]
