from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_catalog, load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROW_ORDER = [
    "delta_architecture_screen_nano_exact_replay",
    "delta_architecture_screen_shared_norm",
    "delta_architecture_screen_prenorm_block",
    "delta_architecture_screen_small_class_head",
    "delta_architecture_screen_test_self",
]
EXPECTED_STAGE_BY_DELTA = {
    "delta_architecture_screen_nano_exact_replay": "nano_exact",
    "delta_architecture_screen_shared_norm": "shared_norm",
    "delta_architecture_screen_prenorm_block": "prenorm_block",
    "delta_architecture_screen_small_class_head": "small_class_head",
    "delta_architecture_screen_test_self": "test_self",
}
EXPECTED_PARENT_BY_DELTA = {
    "delta_architecture_screen_shared_norm": "delta_architecture_screen_nano_exact_replay",
    "delta_architecture_screen_prenorm_block": "delta_architecture_screen_shared_norm",
    "delta_architecture_screen_small_class_head": "delta_architecture_screen_prenorm_block",
    "delta_architecture_screen_test_self": "delta_architecture_screen_small_class_head",
}
EXPECTED_RUN_ID_BY_DELTA = {
    "delta_architecture_screen_nano_exact_replay": (
        "sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1"
    ),
    "delta_architecture_screen_shared_norm": (
        "sd_shared_surface_bridge_v1_02_delta_architecture_screen_shared_norm_v1"
    ),
    "delta_architecture_screen_prenorm_block": (
        "sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1"
    ),
    "delta_architecture_screen_small_class_head": (
        "sd_shared_surface_bridge_v1_04_delta_architecture_screen_small_class_head_v1"
    ),
    "delta_architecture_screen_test_self": (
        "sd_shared_surface_bridge_v1_05_delta_architecture_screen_test_self_v1"
    ),
}
EXPECTED_PARENT_RUN_ID_BY_DELTA = {
    delta_id: EXPECTED_RUN_ID_BY_DELTA[parent_delta_id]
    for delta_id, parent_delta_id in EXPECTED_PARENT_BY_DELTA.items()
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_shared_surface_bridge_catalog_entries_use_stage_native_architecture_screen_payloads() -> None:
    catalog = load_system_delta_catalog(REPO_ROOT / "reference" / "system_delta_catalog.yaml")

    for delta_id, stage in EXPECTED_STAGE_BY_DELTA.items():
        entry = catalog["deltas"][delta_id]
        default_effective_surface = entry["default_effective_surface"]
        model_payload = default_effective_surface["model"]

        assert model_payload["stage"] == stage
        assert model_payload["stage_label"] == delta_id
        assert "module_overrides" not in model_payload
        assert default_effective_surface["data"] == {"surface_label": "anchor_manifest_default"}
        assert default_effective_surface["preprocessing"] == {"surface_label": "runtime_default"}
        assert default_effective_surface["training"] == {
            "surface_label": "training_default",
            "overrides": {},
        }


def test_shared_surface_bridge_v1_uses_architecture_screen_lane_and_stage_native_order() -> None:
    sweep = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "shared_surface_bridge_v1" / "sweep.yaml")
    queue_instance = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / "shared_surface_bridge_v1" / "queue.yaml"
    )
    materialized = load_system_delta_queue(
        sweep_id="shared_surface_bridge_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    assert sweep["training_experiment"] == "cls_benchmark_staged"
    assert sweep["training_config_profile"] == "cls_benchmark_staged"
    assert sweep["surface_role"] == "architecture_screen"
    assert any(
        "stage-native shared-surface bridge" in note for note in sweep["anchor_surface"]["notes"]
    )
    assert any(
        "same-lane replay anchor" in note
        for note in sweep["anchor_surface"]["notes"]
    )
    assert any(
        "sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1" in note
        for note in sweep["anchor_surface"]["notes"]
    )

    assert [row["delta_ref"] for row in queue_instance["rows"]] == EXPECTED_ROW_ORDER
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROW_ORDER
    assert materialized["training_experiment"] == "cls_benchmark_staged"
    assert materialized["training_config_profile"] == "cls_benchmark_staged"
    assert materialized["surface_role"] == "architecture_screen"

    for row in queue_instance["rows"]:
        assert row.get("parent_delta_ref") == EXPECTED_PARENT_BY_DELTA.get(row["delta_ref"])

    for row in materialized["rows"]:
        expected_stage = EXPECTED_STAGE_BY_DELTA[row["delta_id"]]
        assert row["model"]["stage"] == expected_stage
        assert row["model"]["stage_label"] == row["delta_id"]
        assert "module_overrides" not in row["model"]
        assert row.get("parent_delta_ref") == EXPECTED_PARENT_BY_DELTA.get(row["delta_id"])


def test_shared_surface_bridge_v1_matrix_records_the_stage_native_bridge_rows() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "shared_surface_bridge_v1"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "Training experiment: `cls_benchmark_staged`" in matrix
    assert "Training config profile: `cls_benchmark_staged`" in matrix
    assert "Surface role: `architecture_screen`" in matrix
    assert "delta_architecture_screen_nano_exact_replay" in matrix
    assert "delta_architecture_screen_shared_norm" in matrix
    assert "delta_architecture_screen_prenorm_block" in matrix
    assert "delta_architecture_screen_small_class_head" in matrix
    assert "delta_architecture_screen_test_self" in matrix
    assert "Treat the older `01_nano_exact_md_prior_parity_fix_binary_medium_v1` registry run as historical anchor evidence only" in matrix
    assert "sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1" in matrix
    assert "data surface label `anchor_manifest_default`" in matrix
    assert "Benchmark preprocessing surface label `runtime_default`" in matrix
    assert "Training surface label `training_default`" in matrix
    assert "prior_constant_lr" not in matrix
    assert "registry surface label unavailable" not in matrix
    assert "Lock as the grouped-token handoff row and treat later bridge rows as optional follow-ons" in matrix


def test_shared_surface_bridge_v1_registry_records_prenorm_block_as_locked_handoff() -> None:
    registry = json.loads(
        (REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )

    run = registry["runs"]["sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1"]

    assert run["decision"] == "keep"
    assert (
        run["conclusion"]
        == "Locked as the grouped-token handoff row because it delivered the first material bridge gain while preserving the strongest proper-score result among rows 3-5."
    )


def test_shared_surface_bridge_v1_registry_records_sequential_parent_lineage() -> None:
    registry = json.loads(
        (REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )

    for delta_id, parent_run_id in EXPECTED_PARENT_RUN_ID_BY_DELTA.items():
        run = registry["runs"][EXPECTED_RUN_ID_BY_DELTA[delta_id]]

        assert run["lineage"]["parent_run_id"] == parent_run_id
        assert run["comparisons"]["vs_parent"]["reference_run_id"] == parent_run_id
