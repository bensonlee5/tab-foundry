from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_large_v2"
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]
EXPECTED_CORPUS_REFS = [
    "tf_rd_010_dagzoo_medium_control_curated_v5",
    "tf_rd_010_missingness_mcar_v3",
    "tf_rd_010_missingness_mar_v3",
    "tf_rd_010_missingness_mnar_v3",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_classification_evolution_large_v2_is_registered() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    entry = sweeps[SWEEP_ID]
    assert entry["parent_sweep_id"] == "tf_rd_010_classification_evolution_medium_v4"
    assert entry["status"] == "blocked_on_anchor_selection"
    assert entry["anchor_run_id"] is None
    assert entry["complexity_level"] == "classification_lg"
    assert entry["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_large_v1/manifest.parquet"
    )
    assert entry["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert entry["external_benchmarks"] == []


def test_tf_rd_010_classification_evolution_large_v2_records_the_active_2500_step_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_010_classification_evolution_medium_v4"
    assert sweep["status"] == "blocked_on_anchor_selection"
    assert sweep["anchor_run_id"] is None
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_large_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert sweep["external_benchmarks"] == []

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert "blocked on selecting and trusting the refreshed medium control anchor" in notes[0]
    assert "tf_rd_010_synthetic_adequacy_v3" in notes[0]
    assert any("159984" in note for note in notes)
    assert any("2500" in note for note in notes)
    assert any("tab-realdata-hub" in note for note in notes)
    assert any("sandwich_summary_tokens_per_axis=3" in note for note in notes)
    assert any("min_classes=2" in note for note in notes)
    assert any("max_missing_pct=20.0" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["blocked_on_anchor_selection"] * len(EXPECTED_ROWS)
    assert [row["run_id"] for row in rows] == [None] * len(EXPECTED_ROWS)
    assert [row["decision"] for row in rows] == [None] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["blocked"] * len(EXPECTED_ROWS)
    assert [row["data"]["corpus_ref"] for row in rows] == EXPECTED_CORPUS_REFS
    assert all("159984" in " ".join(row["notes"]) for row in rows)
    assert all("2500" in " ".join(row["notes"]) for row in rows)
    assert "production_control_curated_v5/train" in " ".join(rows[0]["notes"])
    assert "task_batch_size=16" in " ".join(rows[0]["notes"])
    assert "do not retrain the control row for `large_v2`" in " ".join(rows[0]["notes"])
    assert "curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`" in " ".join(rows[1]["notes"])
    assert all(
        "Keep this row blocked until the medium control anchor is benchmarked and promoted" in row["next_action"]
        for row in rows[1:]
    )
    assert "benchmark the same completed pilot control run" in rows[0]["next_action"]
    assert "do not retrain row 1" in rows[0]["next_action"]
    assert all("benchmark_metrics" not in row for row in rows)

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] is None
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 2500 for row in materialized["rows"])
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 2500
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["synthetic_epoch_budget"]["resolved_task_count"] == 159984
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["synthetic_epoch_budget"]["resolved_max_steps"] == 2500
        for row in materialized["rows"]
    )


def test_tf_rd_010_classification_evolution_large_v2_matrix_records_the_pending_successor_path() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `blocked_on_anchor_selection`" in matrix
    assert "Anchor run id: `null`" in matrix
    assert "Wait for the medium control anchor to be benchmarked, registered, and promoted" in matrix
    assert "159984" in matrix
    assert "`2500` optimizer steps" in matrix
    assert "tf_rd_010_classification_evolution_medium_v4" in matrix
    assert "tf_rd_010_dagzoo_medium_control_curated_v5" in matrix
    assert "final_log_loss_at_matched_regime_budget" in matrix
    assert "label-target log loss per test cell" in matrix


def test_tf_rd_010_classification_evolution_large_v2_inspection_resolves_the_2500_step_row() -> None:
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
    assert resolved_data["recipe_id"] == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert str(resolved_data["corpus_ref"]).startswith("tf_rd_010_dagzoo_medium_control_curated_v5")
    assert resolved_training["schedule_stages"][0]["steps"] == 2500
