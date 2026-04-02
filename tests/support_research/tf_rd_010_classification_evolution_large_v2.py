from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_large_v2"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_large_v2_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1"
)
MCAR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_large_v2_02_"
    "delta_data_manifest_root_tf_rd_010_missingness_mcar_v1"
)
MAR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_large_v2_03_"
    "delta_data_manifest_root_tf_rd_010_missingness_mar_v1"
)
MNAR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_large_v2_04_"
    "delta_data_manifest_root_tf_rd_010_missingness_mnar_v1"
)
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
EXPECTED_REUSE_ARTIFACTS = [
    {
        "run_dir": "outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train",
        "training_surface_fingerprint": "1614c767510feacd669b4868fd2dfacbe7332f0b64b9c694c448caca85794d20",
    },
    {
        "run_dir": "outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mcar/sd_tf_rd_010_classification_evolution_medium_v4_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1/train",
        "training_surface_fingerprint": "60f35937e0c9701505f061f4c886e3ee2027c5376fcb492fb32c097b15b73fa7",
    },
    {
        "run_dir": "outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mar/sd_tf_rd_010_classification_evolution_medium_v4_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1/train",
        "training_surface_fingerprint": "71fd6c814bcd7c0a1799c31746551df915b0915a752da53124df3be6f1f128ee",
    },
    {
        "run_dir": "outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mnar/sd_tf_rd_010_classification_evolution_medium_v4_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1/train",
        "training_surface_fingerprint": "5c2fe334e601ae78d310357633456e020bc186c4a8fffcb117ce9b048bd674f9",
    },
]
EXPECTED_ROW_STATUSES = ["completed", "completed", "completed", "completed"]
EXPECTED_RUN_IDS = [ANCHOR_RUN_ID, MCAR_RUN_ID, MAR_RUN_ID, MNAR_RUN_ID]
EXPECTED_DECISIONS = ["defer", "defer", "defer", "defer"]
EXPECTED_INTERPRETATION_STATUSES = ["completed", "completed", "completed", "completed"]
EXPECTED_FINAL_LOG_LOSSES = [
    0.8974410961425622,
    0.9155278224081353,
    0.9418792099275213,
    0.9411754209242043,
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
    assert entry["status"] == "completed"
    assert entry["anchor_run_id"] == ANCHOR_RUN_ID
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
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_large_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert sweep["external_benchmarks"] == []

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert "local-only all-rows benchmark-only pass" in notes[0]
    assert "promote the large-rung anchor" in notes[0]
    assert any("159984" in note for note in notes)
    assert any("2500" in note for note in notes)
    assert any("tab-realdata-hub" in note for note in notes)
    assert any("sandwich_summary_tokens_per_axis=3" in note for note in notes)
    assert any("[363685, 363699, 363707]" in note for note in notes)
    assert any("Rows 2 through 4 are benchmark-only transfer reads" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == EXPECTED_ROW_STATUSES
    assert [row["run_id"] for row in rows] == EXPECTED_RUN_IDS
    assert [row["decision"] for row in rows] == EXPECTED_DECISIONS
    assert [row["interpretation_status"] for row in rows] == EXPECTED_INTERPRETATION_STATUSES
    assert [row["data"]["corpus_ref"] for row in rows] == EXPECTED_CORPUS_REFS
    assert [row["reuse_train_artifact"] for row in rows] == EXPECTED_REUSE_ARTIFACTS
    assert all("159984" in " ".join(row["notes"]) for row in rows)
    assert all("2500" in " ".join(row["notes"]) for row in rows)
    assert "production_control_curated_v5/train" in " ".join(rows[0]["notes"])
    assert "task_batch_size=16" in " ".join(rows[0]["notes"])
    assert "do not retrain the control row for `large_v2`" in " ".join(rows[0]["notes"])
    assert "curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`" in " ".join(rows[1]["notes"])
    assert "promote it as the large-rung anchor" in rows[0]["next_action"]
    assert all("same `--promote-first-executed-row-to-anchor` pass" in row["next_action"] for row in rows[1:])
    assert all(row["benchmark_checkpoint_selection"] == "all" for row in rows)
    assert all(row["benchmark_metrics"]["objective_metric"] == "final_log_loss_at_matched_regime_budget" for row in rows)
    assert [row["benchmark_metrics"]["final_log_loss"] for row in rows] == EXPECTED_FINAL_LOG_LOSSES
    assert any("Canonical rerun registered as" in note for note in rows[0]["notes"])
    assert any("Canonical rerun registered as" in note for note in rows[1]["notes"])
    assert any("Canonical rerun registered as" in note for note in rows[2]["notes"])
    assert any("Canonical rerun registered as" in note for note in rows[3]["notes"])

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert [row["reuse_train_artifact"] for row in materialized["rows"]] == EXPECTED_REUSE_ARTIFACTS
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


def test_tf_rd_010_classification_evolution_large_v2_resolved_queue_captures_completed_runtime_surface() -> None:
    resolved = _load_yaml(
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / SWEEP_ID
        / "resolved_queue.yaml"
    )

    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["sweep_status"] == "completed"
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["status"] for row in resolved["rows"]] == EXPECTED_ROW_STATUSES
    assert [row["run_id"] for row in resolved["rows"]] == EXPECTED_RUN_IDS
    assert [row["decision"] for row in resolved["rows"]] == EXPECTED_DECISIONS

    row = resolved["rows"][0]
    assert row["status"] == "completed"
    assert row["run_id"] == ANCHOR_RUN_ID
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "completed"

    resolved_surface = row["resolved_surface"]
    runtime = resolved_surface["runtime"]
    training = resolved_surface["training"]
    assert runtime["grad_clip"] == 0.0
    assert runtime["grad_accum_steps"] == 4
    assert runtime["max_steps"] == 2500
    assert training["task_batch_size"] == 16
    assert training["loss_surface"] == "classification"
    assert training["optimizer_min_lr"] == 1.0e-5
    assert training["schedule_stages"][0]["steps"] == 2500
    assert training["schedule_stages"][0]["lr_max"] == 1.0e-3
    assert training["schedule_stages"][0]["lr_schedule"] == "linear"
    assert training["schedule_stages"][0]["warmup_ratio"] == 0.10
    assert row["benchmark_metrics"]["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert row["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSSES[0]
    assert resolved["rows"][1]["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSSES[1]
    assert resolved["rows"][2]["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSSES[2]
    assert resolved["rows"][3]["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSSES[3]


def test_tf_rd_010_classification_evolution_large_v2_matrix_records_the_completed_benchmark_only_pass() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `completed`" in matrix
    assert f"Anchor run id: `{ANCHOR_RUN_ID}`" in matrix
    assert "local all-rows benchmark-only pass" in matrix
    assert "--promote-first-executed-row-to-anchor" in matrix
    assert "159984" in matrix
    assert "`2500` optimizer steps" in matrix
    assert "tf_rd_010_classification_evolution_medium_v4" in matrix
    assert "tf_rd_010_dagzoo_medium_control_curated_v5" in matrix
    assert "production_control_curated_v5/train" in matrix
    assert "final_log_loss_at_matched_regime_budget" in matrix
    assert "label-target log loss per test cell" in matrix
    assert MCAR_RUN_ID in matrix
    assert MAR_RUN_ID in matrix
    assert MNAR_RUN_ID in matrix
    assert "final log loss `0.9155`" in matrix
    assert "final log loss `0.9419`" in matrix
    assert "final log loss `0.9412`" in matrix


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
    assert resolved_data["surface_label"] in {
        "tf_rd_010_dagzoo_medium_control_curated_v5",
        "tf_rd_010_dagzoo_medium_control",
    }
    if resolved_data["recipe_id"] is None:
        assert resolved_data["corpus_ref"] is None
        assert "tf_rd_010_synthetic_adequacy_v3/direct_training/manifest.parquet" in str(
            resolved_data["manifest"]["manifest_path"]
        )
        assert resolved_data["manifest"]["characteristics"]["record_count"] == 159984
    else:
        assert resolved_data["recipe_id"] == "tf_rd_010_dagzoo_medium_control_curated_v5"
        assert str(resolved_data["corpus_ref"]).startswith("tf_rd_010_dagzoo_medium_control_curated_v5")
        assert resolved_data["dagzoo_provenance"]["manifest_record_count"] == 159984
    assert resolved_training["schedule_stages"][0]["steps"] == 2500
    assert payload["row"]["status"] == "completed"
    assert payload["row"]["run_id"] == ANCHOR_RUN_ID
    assert payload["row"]["decision"] == "defer"
    assert payload["row"]["reuse_train_artifact"] == EXPECTED_REUSE_ARTIFACTS[0]
    assert payload["row"]["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSSES[0]
