from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_medium_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v1_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v4"
)
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]
EXPECTED_RUN_IDS = [
    ANCHOR_RUN_ID,
    "sd_tf_rd_010_classification_evolution_medium_v1_02_"
    "delta_data_manifest_root_tf_rd_010_missingness_mcar_v2",
    "sd_tf_rd_010_classification_evolution_medium_v1_03_"
    "delta_data_manifest_root_tf_rd_010_missingness_mar_v2",
    "sd_tf_rd_010_classification_evolution_medium_v1_04_"
    "delta_data_manifest_root_tf_rd_010_missingness_mnar_v2",
]
EXPECTED_CORPUS_REFS = [
    "tf_rd_010_dagzoo_medium_control_v1",
    "tf_rd_010_missingness_mcar_v1",
    "tf_rd_010_missingness_mar_v1",
    "tf_rd_010_missingness_mnar_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_classification_evolution_medium_v1_is_registered() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_021b_sandwich_feature_removal_v1",
        "status": "ready",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_010_classification_evolution_medium_v1_preserves_completed_historical_evidence() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["status"] == "ready"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert sweep["external_benchmarks"] == []

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#202" in note for note in notes)
    assert any("#205" in note for note in notes)
    assert any("#204" in note for note in notes)
    assert any("tab-realdata-hub" in note for note in notes)
    assert any("sandwich_summary_tokens_per_axis=3" in note for note in notes)
    assert any("final_bpc_at_matched_regime_budget" in note for note in notes)
    assert any("min_classes=2" in note for note in notes)
    assert any("max_missing_pct=20.0" in note for note in notes)
    assert any("144" in note and "tasks per front" in note for note in notes)
    assert any("fixed 400-step contract" in note for note in notes)

    anchor_model = sweep["anchor_context"]["model"]
    assert anchor_model["arch"] == "tabfoundry_sandwich"
    assert anchor_model["module_selection"] is None
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_010_dagzoo_medium_control",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["run_id"] for row in rows] == EXPECTED_RUN_IDS
    assert [row["decision"] for row in rows] == ["defer"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["data"]["corpus_ref"] for row in rows] == EXPECTED_CORPUS_REFS
    assert all(row["training"]["surface_label"] == "prior_cosine_warmup" for row in rows)
    assert all(row["training"]["prior_dump_batch_size"] == 64 for row in rows)
    assert all(row["training"]["synthetic_epoch_budget"]["epochs"] == 1 for row in rows)
    assert all(
        row["training"]["synthetic_epoch_budget"]["budget_unit"] == "corpus_manifest_records"
        for row in rows
    )
    assert all(row["training"]["synthetic_epoch_budget"]["prior_dump_batch_size"] == 64 for row in rows)
    assert all(
        row["training"]["synthetic_epoch_budget"]["allow_partial_final_batch"] is True for row in rows
    )
    assert all("max_steps" not in row["training"]["overrides"].get("runtime", {}) for row in rows)
    assert all("steps" not in row["training"]["overrides"]["schedule"]["stages"][0] for row in rows)
    assert all("benchmark_metrics" in row for row in rows)
    assert rows[0]["benchmark_metrics"]["final_bpc"] == 662.9175287914276
    assert rows[1]["benchmark_metrics"]["final_bpc"] == 1541.902657699585
    assert rows[2]["benchmark_metrics"]["final_bpc"] == 1166.6423996162414
    assert rows[3]["benchmark_metrics"]["final_bpc"] == 1666.9548992443088

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert [row["status"] for row in materialized["rows"]] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["run_id"] for row in materialized["rows"]] == EXPECTED_RUN_IDS
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 3 for row in materialized["rows"])
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 3
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["synthetic_epoch_budget"]["resolved_task_count"] == 144
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["synthetic_epoch_budget"]["resolved_max_steps"] == 3
        for row in materialized["rows"]
    )


def test_tf_rd_010_classification_evolution_medium_v1_matrix_records_completed_runs() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `ready`" in matrix
    assert f"Anchor run id: `{ANCHOR_RUN_ID}`" in matrix
    assert "final BPC `662.9175`" in matrix
    assert "final log loss `1.0907`" in matrix
    assert "Canonical rerun registered as" in matrix
    assert "delta final BPC `+878.9851`" in matrix
    assert "Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_v1'" in matrix


def test_tf_rd_010_medium_registry_preserves_the_completed_historical_runs() -> None:
    registry = json.loads(
        (
            REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
        ).read_text(encoding="utf-8")
    )

    runs = registry["runs"]
    for run_id in EXPECTED_RUN_IDS:
        assert run_id in runs
        assert runs[run_id]["sweep"]["sweep_id"] == SWEEP_ID
    assert runs[ANCHOR_RUN_ID]["artifacts"]["run_dir"].endswith("/train")


def test_tf_rd_010_classification_evolution_medium_v1_inspection_resolves_three_step_row() -> None:
    payload = inspect_sweep_row(
        sweep_id=SWEEP_ID,
        order=1,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    resolved_model = payload["target"]["resolved"]["model"]
    resolved_data = payload["target"]["resolved"]["data"]
    resolved_training = payload["target"]["resolved"]["training"]
    assert resolved_model["arch"] == "tabfoundry_sandwich"
    assert resolved_model.get("stage_label") is None
    assert resolved_model["architecture"]["feature_type_encoding"] == "film"
    assert str(resolved_data["corpus_ref"]).startswith("tf_rd_010_dagzoo_medium_control_v1/")
    assert resolved_training["schedule_stages"][0]["steps"] == 3
