"""TF-RD-013 corpus recipe generators."""

from __future__ import annotations

from typing import Any, Mapping


def build_current_corpus_default_recipe(
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
    config_ref = str(inputs.get("config_ref", "configs/default.yaml"))
    num_datasets = int(inputs.get("num_datasets", 10))
    seed = int(inputs.get("seed", 1))
    device = str(inputs.get("device", "cpu"))
    hardware_policy = str(inputs.get("hardware_policy", "none"))
    return {
        "dagzoo": {
            "config_ref": config_ref,
            "num_datasets": num_datasets,
            "seed": seed,
            "device": device,
            "hardware_policy": hardware_policy,
        },
        "review_summary": {
            "config_refs": [config_ref],
            "invocation_count": 1,
            "manifest_record_count": num_datasets,
            "num_datasets_per_invocation": num_datasets,
        },
    }


def build_shape_aware_size_recipe(
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
    invocation_dataset_counts = {
        str(key): int(value)
        for key, value in dict(inputs.get("invocation_dataset_counts", {})).items()
    }
    invocations = [
        {
            "invocation_id": "benchmark_cpu",
            "config_ref": "configs/benchmark_cpu.yaml",
            "num_datasets": invocation_dataset_counts["benchmark_cpu"],
            "seed": 1,
            "device": "cpu",
            "hardware_policy": "none",
        },
        {
            "invocation_id": "default_medium",
            "config_ref": "configs/default.yaml",
            "num_datasets": invocation_dataset_counts["default_medium"],
            "seed": 1,
            "device": "cpu",
            "hardware_policy": "none",
        },
        {
            "invocation_id": "large_shape",
            "config_ref": "configs/benchmark_cuda_h100_large_shape.yaml",
            "num_datasets": invocation_dataset_counts["large_shape"],
            "seed": 1,
            "device": "cpu",
            "hardware_policy": "none",
        },
    ]
    return {
        "invocations": invocations,
        "review_summary": {
            "config_refs": [
                "configs/benchmark_cpu.yaml",
                "configs/default.yaml",
                "configs/benchmark_cuda_h100_large_shape.yaml",
            ],
            "invocation_count": len(invocations),
            "manifest_record_count": sum(entry["num_datasets"] for entry in invocations),
            "invocation_dataset_counts": invocation_dataset_counts,
        },
    }
