"""TF-RD-010 corpus recipe generators."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping


_ROW_SPECS = {
    128: (96, 32),
    256: (192, 64),
    512: (384, 128),
    1024: (768, 256),
}
_FEATURE_GRAPH_BANDS = {
    6: (2, 12),
    10: (2, 20),
    14: (4, 28),
    20: (2, 20),
}
_ALIGNED_FEATURE_GRAPH_BANDS = {
    6: (2, 12),
    10: (2, 20),
    14: (4, 28),
    20: (6, 40),
}
_BALANCED_CLASS_COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
_ALIGNED_CLASS_COUNTS = (2, 3, 5, 7, 10)


def _missingness_payload(mode: str | None, rate: float | None) -> dict[str, Any]:
    if mode is None or rate is None:
        return {}
    payload: dict[str, Any] = {
        "missing_rate": float(rate),
        "missing_mechanism": str(mode),
    }
    if mode == "mar":
        payload["missing_mar_observed_fraction"] = 0.6
        payload["missing_mar_logit_scale"] = 1.4
    elif mode == "mnar":
        payload["missing_mnar_logit_scale"] = 1.4
    return payload


def _balanced_front_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
) -> dict[str, Any]:
    return {
        "grid_family": "balanced_rows_features_classes_v1",
        "config_refs": [base_config_ref],
        "invocation_count": len(_ROW_SPECS) * len(_FEATURE_GRAPH_BANDS) * len(_BALANCED_CLASS_COUNTS),
        "manifest_record_count": len(_ROW_SPECS) * len(_FEATURE_GRAPH_BANDS) * len(_BALANCED_CLASS_COUNTS) * int(
            num_datasets
        ),
        "row_totals": sorted(_ROW_SPECS),
        "feature_counts": sorted(_ALIGNED_FEATURE_GRAPH_BANDS),
        "class_counts": list(_BALANCED_CLASS_COUNTS),
        "num_datasets_per_invocation": int(num_datasets),
        "missingness": (
            None
            if missing_mode is None or missing_rate is None
            else {"mechanism": str(missing_mode), "rate": float(missing_rate)}
        ),
    }


def _aligned_front_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
) -> dict[str, Any]:
    return {
        "grid_family": "aligned_feature_class_grid_v2",
        "config_refs": [base_config_ref],
        "invocation_count": len(_FEATURE_GRAPH_BANDS) * len(_ALIGNED_CLASS_COUNTS) * 3,
        "manifest_record_count": len(_FEATURE_GRAPH_BANDS) * len(_ALIGNED_CLASS_COUNTS) * 3 * int(num_datasets),
        "row_totals": [96],
        "feature_counts": sorted(_FEATURE_GRAPH_BANDS),
        "class_counts": list(_ALIGNED_CLASS_COUNTS),
        "replicate_seeds": [1, 2, 3],
        "num_datasets_per_invocation": int(num_datasets),
        "missingness": (
            None
            if missing_mode is None or missing_rate is None
            else {"mechanism": str(missing_mode), "rate": float(missing_rate)}
        ),
    }


def build_balanced_medium_recipe(
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
    num_datasets = int(inputs.get("num_datasets", 1))
    filter_max_attempts = int(inputs.get("filter_max_attempts", 256))
    missing_mode_raw = inputs.get("missing_mode")
    missing_mode = None if missing_mode_raw is None else str(missing_mode_raw)
    missing_rate_raw = inputs.get("missing_rate")
    missing_rate = None if missing_rate_raw is None else float(missing_rate_raw)
    invocations: list[dict[str, Any]] = []
    for row_index, row_total in enumerate(sorted(_ROW_SPECS), start=1):
        n_train, n_test = _ROW_SPECS[row_total]
        for feature_count, class_count in product(sorted(_FEATURE_GRAPH_BANDS), _BALANCED_CLASS_COUNTS):
            graph_min, graph_max = _FEATURE_GRAPH_BANDS[feature_count]
            invocation = {
                "invocation_id": f"r{row_total:04d}_f{feature_count:02d}_c{class_count:02d}",
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
                "seed": int(f"{row_index}{feature_count:02d}{class_count:02d}"),
                "device": "cpu",
                "hardware_policy": "none",
                **_missingness_payload(missing_mode, missing_rate),
            }
            invocations.append(invocation)
    return {
        "invocations": invocations,
        "review_summary": _balanced_front_review_summary(
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
    }


def build_aligned_control_recipe(
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
    num_datasets = int(inputs.get("num_datasets", 8))
    filter_max_attempts = int(inputs.get("filter_max_attempts", 32))
    missing_mode_raw = inputs.get("missing_mode")
    missing_mode = None if missing_mode_raw is None else str(missing_mode_raw)
    missing_rate_raw = inputs.get("missing_rate")
    missing_rate = None if missing_rate_raw is None else float(missing_rate_raw)
    invocations: list[dict[str, Any]] = []
    for feature_count, class_count in product(sorted(_ALIGNED_FEATURE_GRAPH_BANDS), _ALIGNED_CLASS_COUNTS):
        graph_min, graph_max = _ALIGNED_FEATURE_GRAPH_BANDS[feature_count]
        for seed in (1, 2, 3):
            invocations.append(
                {
                    "invocation_id": f"f{feature_count:02d}_c{class_count:02d}_s{seed:02d}",
                    "base_config_ref": base_config_ref,
                    "config_overrides": {
                        "dataset": {
                            "task": "classification",
                            "n_train": 64,
                            "n_test": 32,
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
                    "seed": seed,
                    "device": "cpu",
                    "hardware_policy": "none",
                    **_missingness_payload(missing_mode, missing_rate),
                }
            )
    return {
        "invocations": invocations,
        "review_summary": _aligned_front_review_summary(
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
    }
