from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.hardware_architecture_registry import (
    REGISTRY_SCHEMA,
    REGISTRY_VERSION,
    default_hardware_architecture_registry_path,
    load_hardware_architecture_baseline_entry,
    load_hardware_architecture_registry,
    normalize_registry_path_value,
    resolve_registry_path_value,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_baseline_entry(baseline_id: str = "tf_rd_009_a100_medium_v1") -> dict[str, object]:
    return {
        "baseline_id": baseline_id,
        "hardware_profile_id": "a100_80gb",
        "gpu_class": "a100",
        "vram_class_gb": 80,
        "track": "system_delta_classification_medium_v1",
        "surface_role": "classification_scaling_law",
        "runtime_profile": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "config_profile": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "sweep_id": "tf_rd_009_width_depth_medium_v1",
        "surface_labels": {
            "model": "tabfoundry_sandwich",
            "data": "tf_rd_010_dagzoo_medium_control",
            "preprocessing": "runtime_default",
            "training": "prior_cosine_warmup",
        },
        "formal_anchor_run_id": "anchor_60x2",
        "baseline_run_id": "baseline_96x2",
        "preferred_run_id": "preferred_96x2",
        "preferred_delta_ref": "delta_tf_rd_009_cls_sandwich_dicl96_v1",
        "preferred_architecture": {
            "arch": "tabfoundry_sandwich",
            "d_icl": 96,
            "head_hidden_dim": 96,
            "tficl_n_heads": 1,
            "tficl_n_layers": 2,
            "architecture": {"latents": 24},
            "build_spec": {"sandwich_layers": 2, "sandwich_heads": 1},
        },
        "objective_metric": "final_log_loss_at_matched_regime_budget",
        "selection_rule": "best_loss_healthy_only",
        "evidence_run_ids": [
            "anchor_60x2",
            "baseline_96x2",
            "upper_128x2",
            "joint_72x1",
            "joint_112x3",
        ],
        "decision": "keep",
        "rationale": "A100 medium classification baseline retained at 96x2 after healthy-only width-depth comparison.",
        "preferred_runtime_summary": {
            "peak_vram_allocated": 9201039872,
            "peak_vram_reserved": 10930356224,
            "throughput_examples_per_second": 18.06,
            "throughput_tokens_per_second": 107589.79,
            "non_train_overhead_seconds": 18.9,
        },
    }


def _write_registry(path: Path, *, baselines: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA,
                "version": REGISTRY_VERSION,
                "baselines": baselines,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_hardware_architecture_registry_default_path_is_repo_tracked() -> None:
    assert default_hardware_architecture_registry_path() == (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "hardware_architecture_baselines_v1.json"
    )


def test_hardware_architecture_registry_path_values_roundtrip_repo_relative_and_absolute(
    tmp_path: Path,
) -> None:
    repo_local = REPO_ROOT / "reference" / "system_delta_catalog.yaml"
    sibling_path = (REPO_ROOT.parent / "nanoTabPFN" / "300k_150x5_2.h5").resolve()
    outside_repo = tmp_path / "outside.json"

    assert normalize_registry_path_value(repo_local) == "reference/system_delta_catalog.yaml"
    assert resolve_registry_path_value("reference/system_delta_catalog.yaml") == repo_local.resolve()
    assert normalize_registry_path_value(sibling_path) == "../nanoTabPFN/300k_150x5_2.h5"
    assert resolve_registry_path_value("../nanoTabPFN/300k_150x5_2.h5") == sibling_path
    assert normalize_registry_path_value(outside_repo) == str(outside_repo.resolve())
    assert resolve_registry_path_value(str(outside_repo.resolve())) == outside_repo.resolve()


def test_hardware_architecture_registry_load_entry_returns_deep_copy(tmp_path: Path) -> None:
    registry_path = tmp_path / "hardware_architecture_baselines_v1.json"
    _write_registry(
        registry_path,
        baselines={"tf_rd_009_a100_medium_v1": _valid_baseline_entry()},
    )

    loaded = load_hardware_architecture_baseline_entry(
        "tf_rd_009_a100_medium_v1",
        registry_path=registry_path,
    )
    loaded["evidence_run_ids"].append("mutated")  # type: ignore[index]
    loaded["preferred_architecture"]["d_icl"] = -1  # type: ignore[index]

    reloaded = load_hardware_architecture_baseline_entry(
        "tf_rd_009_a100_medium_v1",
        registry_path=registry_path,
    )

    assert reloaded["evidence_run_ids"] == [
        "anchor_60x2",
        "baseline_96x2",
        "upper_128x2",
        "joint_72x1",
        "joint_112x3",
    ]
    assert reloaded["preferred_architecture"]["d_icl"] == 96


def test_hardware_architecture_registry_rejects_malformed_payload(tmp_path: Path) -> None:
    registry_path = tmp_path / "hardware_architecture_baselines_v1.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA,
                "version": REGISTRY_VERSION,
                "baselines": {
                    "tf_rd_009_a100_medium_v1": {
                        "baseline_id": "tf_rd_009_a100_medium_v1",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hardware architecture baseline entry"):
        _ = load_hardware_architecture_registry(registry_path)


def test_checked_in_hardware_registry_tracks_grid_architecture_anchor() -> None:
    entry = load_hardware_architecture_baseline_entry(
        "tf_rd_009_a100_80gb_classification_medium_grid_v1"
    )

    assert entry["decision"] == "keep"
    assert entry["hardware_profile_id"] == "a100_80gb"
    assert entry["surface_role"] == "classification_architecture_anchor"
    assert entry["runtime_profile"] == "cls_workstation_grid_sandwich"
    assert entry["config_profile"] == "cls_workstation_grid_sandwich"
    assert entry["preferred_run_id"] == (
        "sd_tf_rd_009_sandwich_followons_medium_metadatafix_20260420_03_grid_pilot_v1"
    )
    assert entry["baseline_run_id"] == (
        "sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1"
    )
    assert entry["preferred_architecture"]["arch"] == "grid_sandwich"
    assert entry["preferred_architecture"]["d_icl"] == 144
    assert entry["preferred_architecture"]["sandwich_layers"] == 4
    assert entry["preferred_architecture"]["sandwich_heads"] == 1
    assert entry["preferred_architecture"]["head_hidden_dim"] == 96
    assert entry["surface_labels"]["model"] == "grid_sandwich"
