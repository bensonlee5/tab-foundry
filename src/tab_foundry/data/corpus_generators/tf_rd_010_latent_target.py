"""TF-RD-010 latent-target corpus recipe generators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping


@dataclass(frozen=True)
class _FeatureShapeSpec:
    cell_budget_label: str
    row_total: int
    n_train: int
    n_test: int
    feature_count: int
    graph_min: int
    graph_max: int

    @property
    def cell_count(self) -> int:
        return self.row_total * self.feature_count


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
_FEATURE_SHAPE_SPECS = (
    _FeatureShapeSpec("16k_cells", 128, 96, 32, 128, 8, 64),
    _FeatureShapeSpec("16k_cells", 256, 192, 64, 64, 4, 64),
    _FeatureShapeSpec("16k_cells", 512, 384, 128, 32, 4, 32),
    _FeatureShapeSpec("16k_cells", 1024, 768, 256, 16, 4, 16),
    _FeatureShapeSpec("32k_cells", 128, 96, 32, 256, 8, 96),
    _FeatureShapeSpec("32k_cells", 256, 192, 64, 128, 8, 64),
    _FeatureShapeSpec("32k_cells", 512, 384, 128, 64, 4, 64),
    _FeatureShapeSpec("32k_cells", 1024, 768, 256, 32, 4, 32),
    _FeatureShapeSpec("65k_cells", 256, 192, 64, 256, 8, 96),
    _FeatureShapeSpec("65k_cells", 512, 384, 128, 128, 8, 64),
    _FeatureShapeSpec("65k_cells", 1024, 768, 256, 64, 4, 64),
    _FeatureShapeSpec("65k_cells", 2048, 1536, 512, 32, 4, 32),
    _FeatureShapeSpec("131k_cells", 512, 384, 128, 256, 8, 96),
    _FeatureShapeSpec("131k_cells", 1024, 768, 256, 128, 8, 64),
    _FeatureShapeSpec("131k_cells", 2048, 1536, 512, 64, 4, 64),
    _FeatureShapeSpec("131k_cells", 4096, 3072, 1024, 32, 4, 32),
)
_LATENT_TARGET_DERIVATION = "tabiclv2_latent_node"
_BALANCED_CLASS_COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


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


def _latent_target_review_summary_fields() -> dict[str, Any]:
    return {
        "target_derivation": _LATENT_TARGET_DERIVATION,
    }


def _balanced_front_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
    row_specs: Mapping[int, tuple[int, int]] = _ROW_SPECS,
    feature_graph_bands: Mapping[int, tuple[int, int]] = _FEATURE_GRAPH_BANDS,
    grid_family: str = "balanced_rows_features_classes_v1",
) -> dict[str, Any]:
    summary = {
        "grid_family": grid_family,
        "config_refs": [base_config_ref],
        "invocation_count": len(row_specs) * len(feature_graph_bands) * len(_BALANCED_CLASS_COUNTS),
        "manifest_record_count": len(row_specs) * len(feature_graph_bands) * len(_BALANCED_CLASS_COUNTS) * int(
            num_datasets
        ),
        "row_totals": sorted(row_specs),
        "feature_counts": sorted(feature_graph_bands),
        "class_counts": list(_BALANCED_CLASS_COUNTS),
        "num_datasets_per_invocation": int(num_datasets),
        "missingness": (
            None
            if missing_mode is None or missing_rate is None
            else {"mechanism": str(missing_mode), "rate": float(missing_rate)}
        ),
    }
    summary.update(_latent_target_review_summary_fields())
    return summary


def _balanced_front_invocations(
    *,
    row_specs: Mapping[int, tuple[int, int]],
    feature_graph_bands: Mapping[int, tuple[int, int]],
    base_config_ref: str,
    num_datasets: int,
    filter_max_attempts: int,
    missing_mode: str | None,
    missing_rate: float | None,
) -> list[dict[str, Any]]:
    invocations: list[dict[str, Any]] = []
    for row_index, row_total in enumerate(sorted(row_specs), start=1):
        n_train, n_test = row_specs[row_total]
        for feature_count, class_count in product(sorted(feature_graph_bands), _BALANCED_CLASS_COUNTS):
            graph_min, graph_max = feature_graph_bands[feature_count]
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
    return invocations


def _feature_shape_summary_pairs(
    specs: tuple[_FeatureShapeSpec, ...] = _FEATURE_SHAPE_SPECS,
) -> list[dict[str, Any]]:
    return [
        {
            "cell_budget_label": spec.cell_budget_label,
            "row_total": spec.row_total,
            "n_train": spec.n_train,
            "n_test": spec.n_test,
            "feature_count": spec.feature_count,
            "cell_count": spec.cell_count,
            "graph_nodes_min": spec.graph_min,
            "graph_nodes_max": spec.graph_max,
        }
        for spec in specs
    ]


def _feature_shape_review_summary(
    *,
    base_config_ref: str,
    num_datasets: int,
    missing_mode: str | None,
    missing_rate: float | None,
    specs: tuple[_FeatureShapeSpec, ...] = _FEATURE_SHAPE_SPECS,
) -> dict[str, Any]:
    summary = {
        "grid_family": "paired_feature_shape_rows_features_classes_v1",
        "config_refs": [base_config_ref],
        "invocation_count": len(specs) * len(_BALANCED_CLASS_COUNTS),
        "manifest_record_count": len(specs) * len(_BALANCED_CLASS_COUNTS) * int(num_datasets),
        "shape_pair_count": len(specs),
        "shape_pairs": _feature_shape_summary_pairs(specs),
        "row_totals": sorted({spec.row_total for spec in specs}),
        "feature_counts": sorted({spec.feature_count for spec in specs}),
        "cell_budget_labels": list(dict.fromkeys(spec.cell_budget_label for spec in specs)),
        "class_counts": list(_BALANCED_CLASS_COUNTS),
        "num_datasets_per_invocation": int(num_datasets),
        "missingness": (
            None
            if missing_mode is None or missing_rate is None
            else {"mechanism": str(missing_mode), "rate": float(missing_rate)}
        ),
    }
    summary.update(_latent_target_review_summary_fields())
    return summary


def _feature_shape_invocations(
    *,
    specs: tuple[_FeatureShapeSpec, ...],
    base_config_ref: str,
    num_datasets: int,
    filter_max_attempts: int,
    missing_mode: str | None,
    missing_rate: float | None,
) -> list[dict[str, Any]]:
    invocations: list[dict[str, Any]] = []
    for shape_index, spec in enumerate(specs, start=1):
        for class_count in _BALANCED_CLASS_COUNTS:
            invocation = {
                "invocation_id": f"r{spec.row_total:04d}_f{spec.feature_count:03d}_c{class_count:02d}",
                "base_config_ref": base_config_ref,
                "config_overrides": {
                    "dataset": {
                        "task": "classification",
                        "n_train": spec.n_train,
                        "n_test": spec.n_test,
                        "n_features_min": spec.feature_count,
                        "n_features_max": spec.feature_count,
                        "n_classes_min": class_count,
                        "n_classes_max": class_count,
                        "categorical_ratio_min": 0.0,
                        "categorical_ratio_max": 1.0,
                        "max_categorical_cardinality": 32,
                    },
                    "graph": {
                        "n_nodes_min": spec.graph_min,
                        "n_nodes_max": spec.graph_max,
                    },
                    "filter": {
                        "max_attempts": filter_max_attempts,
                    },
                },
                "num_datasets": num_datasets,
                "seed": int(f"{shape_index:02d}{spec.feature_count:03d}{class_count:02d}"),
                "device": "cpu",
                "hardware_policy": "none",
                **_missingness_payload(missing_mode, missing_rate),
            }
            invocations.append(invocation)
    return invocations


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
    return {
        "invocations": _balanced_front_invocations(
            row_specs=_ROW_SPECS,
            feature_graph_bands=_FEATURE_GRAPH_BANDS,
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            filter_max_attempts=filter_max_attempts,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
        "review_summary": _balanced_front_review_summary(
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
    }


def build_balanced_feature_shape_recipe(
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
    base_config_ref = str(
        inputs.get("base_config_ref", "configs/benchmark_cuda_h100_large_shape.yaml")
    )
    num_datasets = int(inputs.get("num_datasets", 512))
    filter_max_attempts = int(inputs.get("filter_max_attempts", 256))
    missing_mode_raw = inputs.get("missing_mode")
    missing_mode = None if missing_mode_raw is None else str(missing_mode_raw)
    missing_rate_raw = inputs.get("missing_rate")
    missing_rate = None if missing_rate_raw is None else float(missing_rate_raw)
    return {
        "invocations": _feature_shape_invocations(
            specs=_FEATURE_SHAPE_SPECS,
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            filter_max_attempts=filter_max_attempts,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
        "review_summary": _feature_shape_review_summary(
            base_config_ref=base_config_ref,
            num_datasets=num_datasets,
            missing_mode=missing_mode,
            missing_rate=missing_rate,
        ),
    }


def build_latent_target_canary_recipe(
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
            "grid_family": "latent_target_canary_rows_only_v2",
            "config_refs": [base_config_ref],
            "invocation_count": len(_ROW_SPECS),
            "manifest_record_count": len(_ROW_SPECS) * int(num_datasets),
            "row_totals": sorted(_ROW_SPECS),
            "feature_counts": [6],
            "class_counts": [2],
            "num_datasets_per_invocation": int(num_datasets),
            "missingness": None,
            **_latent_target_review_summary_fields(),
        },
    }


__all__ = [
    "build_balanced_feature_shape_recipe",
    "build_balanced_medium_recipe",
    "build_latent_target_canary_recipe",
]
