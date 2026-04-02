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
_POSTERIOR_PREDICTIVE_FACTORIZATION = "independent_p_x_complete_and_p_y_given_x_complete"
_LABEL_TARGET_LOG_LOSS_PER_TEST_CELL = "label-target log loss per test cell"
_TARGET_PARENT_PRIOR = "near_max_mixture"
_TARGET_PARENT_MODE = "max"
_TARGET_PARENT_NEAR_MAX_BAND_MIN_FRACTION = 0.75
_TARGET_PARENT_BELOW_SQRT_PROB = 0.05
_TARGET_PARENT_MIDRANGE_PROB = 0.20


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


def _posterior_predictive_review_summary_fields(
    *,
    teacher_conditional_export: bool,
) -> dict[str, Any]:
    if not teacher_conditional_export:
        return {}
    return {
        "posterior_predictive_factorization": _POSTERIOR_PREDICTIVE_FACTORIZATION,
        "teacher_conditional_export": True,
        "metric_definition": _LABEL_TARGET_LOG_LOSS_PER_TEST_CELL,
    }


def _target_parent_review_summary_fields(
    *,
    teacher_conditional_export: bool,
) -> dict[str, Any]:
    if not teacher_conditional_export:
        return {}
    return {
        "target_parent_prior": _TARGET_PARENT_PRIOR,
        "target_parent_mode": _TARGET_PARENT_MODE,
        "target_parent_near_max_band_min_fraction": _TARGET_PARENT_NEAR_MAX_BAND_MIN_FRACTION,
        "target_parent_below_sqrt_prob": _TARGET_PARENT_BELOW_SQRT_PROB,
        "target_parent_midrange_prob": _TARGET_PARENT_MIDRANGE_PROB,
    }


def _balanced_front_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
    teacher_conditional_export: bool,
) -> dict[str, Any]:
    summary = {
        "grid_family": "balanced_rows_features_classes_v1",
        "config_refs": [base_config_ref],
        "invocation_count": len(_ROW_SPECS) * len(_FEATURE_GRAPH_BANDS) * len(_BALANCED_CLASS_COUNTS),
        "manifest_record_count": len(_ROW_SPECS) * len(_FEATURE_GRAPH_BANDS) * len(_BALANCED_CLASS_COUNTS) * int(
            num_datasets
        ),
        "row_totals": sorted(_ROW_SPECS),
        "feature_counts": sorted(_FEATURE_GRAPH_BANDS),
        "class_counts": list(_BALANCED_CLASS_COUNTS),
        "num_datasets_per_invocation": int(num_datasets),
        "missingness": (
            None
            if missing_mode is None or missing_rate is None
            else {"mechanism": str(missing_mode), "rate": float(missing_rate)}
        ),
    }
    summary.update(
        _posterior_predictive_review_summary_fields(
            teacher_conditional_export=teacher_conditional_export,
        )
    )
    summary.update(
        _target_parent_review_summary_fields(
            teacher_conditional_export=teacher_conditional_export,
        )
    )
    return summary


def _aligned_front_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
    teacher_conditional_export: bool,
) -> dict[str, Any]:
    summary = {
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
    summary.update(
        _posterior_predictive_review_summary_fields(
            teacher_conditional_export=teacher_conditional_export,
        )
    )
    summary.update(
        _target_parent_review_summary_fields(
            teacher_conditional_export=teacher_conditional_export,
        )
    )
    return summary


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
    teacher_conditional_export = bool(inputs.get("teacher_conditional_export", False))
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
            teacher_conditional_export=teacher_conditional_export,
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
    teacher_conditional_export = bool(inputs.get("teacher_conditional_export", False))
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
            teacher_conditional_export=teacher_conditional_export,
        ),
    }


def build_factorized_canary_recipe(
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
    num_datasets = int(inputs.get("num_datasets", 32))
    filter_max_attempts = int(inputs.get("filter_max_attempts", 64))
    teacher_conditional_export = bool(inputs.get("teacher_conditional_export", False))
    invocations: list[dict[str, Any]] = []
    for row_index, row_total in enumerate(sorted(_ROW_SPECS), start=1):
        n_train, n_test = _ROW_SPECS[row_total]
        invocations.append(
            {
                "invocation_id": f"r{row_total:04d}_canary",
                "base_config_ref": base_config_ref,
                "config_overrides": {
                    "dataset": {
                        "task": "classification",
                        "n_train": n_train,
                        "n_test": n_test,
                        "n_features_min": 6,
                        "n_features_max": 6,
                        "n_classes_min": 2,
                        "n_classes_max": 2,
                        "categorical_ratio_min": 0.0,
                        "categorical_ratio_max": 0.0,
                        "max_categorical_cardinality": 12,
                    },
                    "graph": {
                        "n_nodes_min": 2,
                        "n_nodes_max": 6,
                    },
                    "filter": {
                        "max_attempts": filter_max_attempts,
                    },
                },
                "num_datasets": num_datasets,
                "seed": int(f"{row_index}0602"),
                "device": "cpu",
                "hardware_policy": "none",
            }
        )
    return {
        "invocations": invocations,
        "review_summary": {
            "grid_family": "factorized_canary_rows_only_v1",
            "config_refs": [base_config_ref],
            "invocation_count": len(_ROW_SPECS),
            "manifest_record_count": len(_ROW_SPECS) * int(num_datasets),
            "row_totals": sorted(_ROW_SPECS),
            "feature_counts": [6],
            "class_counts": [2],
            "num_datasets_per_invocation": int(num_datasets),
            "missingness": None,
            **_posterior_predictive_review_summary_fields(
                teacher_conditional_export=teacher_conditional_export
            ),
            **_target_parent_review_summary_fields(
                teacher_conditional_export=teacher_conditional_export
            ),
        },
    }
