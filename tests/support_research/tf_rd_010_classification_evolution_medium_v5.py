from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_medium_v5"
PARENT_SWEEP_ID = "tf_rd_010_classification_evolution_medium_v4"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v4_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8"
)
RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v5_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v2"
)
EXPECTED_ROW = "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control"
EXPECTED_CORPUS_REF = "tf_rd_010_dagzoo_medium_control_curated_v5"
EXPECTED_FINAL_LOG_LOSS = 0.6849303353676671


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_classification_evolution_medium_v5_is_registered_as_completed_successor() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    entry = sweeps[SWEEP_ID]
    assert entry["parent_sweep_id"] == PARENT_SWEEP_ID
    assert entry["status"] == "completed"
    assert entry["anchor_run_id"] == ANCHOR_RUN_ID
    assert entry["complexity_level"] == "classification_md"
    assert entry["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
    )
    assert entry["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert entry["external_benchmarks"] == []


def test_tf_rd_010_classification_evolution_medium_v5_records_the_sorted_control_followup() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == PARENT_SWEEP_ID
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert "matched sorted-order follow-up" in notes[0]
    assert "medium_v4" in notes[0]
    assert any("159984" in note for note in notes)
    assert any("2500" in note for note in notes)
    assert any("final_log_loss_at_matched_regime_budget" in note for note in notes)
    assert any("carry forward the original `medium_v4` control" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert row["delta_ref"] == EXPECTED_ROW
    assert row["status"] == "completed"
    assert row["run_id"] == RUN_ID
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "completed"
    assert "reuse_train_artifact" not in row
    assert row["data"]["corpus_ref"] == EXPECTED_CORPUS_REF
    assert row["training"]["task_batch_size"] == 16
    assert row["training"]["prior_dump_batch_size"] == 64
    assert row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert row["training"]["overrides"]["optimizer"]["min_lr"] == 1.0e-5
    assert row["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 1.0e-3
    assert row["training"]["overrides"]["schedule"]["stages"][0]["lr_schedule"] == "linear"
    assert row["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.10
    assert row["benchmark_checkpoint_selection"] == "best_and_final"
    assert "negative evidence only" in row["next_action"]
    assert "TF-RD-010 carried comparator" in row["next_action"]
    assert any("trained fresh" in note or "trained fresh" in row["next_action"] for note in row["notes"])
    assert any("do not" in note and "reuse" in note for note in row["notes"])
    assert any("keeps the original `medium_v4` control as the carried comparator" in note for note in row["notes"])
    assert any("no missingness promotion" in note for note in row["notes"])
    assert row["benchmark_metrics"]["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert row["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSS

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in materialized["rows"]] == [EXPECTED_ROW]
    materialized_row = materialized["rows"][0]
    assert materialized_row["status"] == "completed"
    assert materialized_row["data"]["corpus_ref"] == EXPECTED_CORPUS_REF
    assert materialized_row["training"]["task_batch_size"] == 16
    assert materialized_row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert materialized_row["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert materialized_row["training"]["overrides"]["optimizer"]["min_lr"] == 1.0e-5
    assert materialized_row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 2500
    assert materialized_row["training"]["synthetic_epoch_budget"]["resolved_task_count"] == 159984
    assert materialized_row["training"]["synthetic_epoch_budget"]["resolved_max_steps"] == 2500


def test_tf_rd_010_classification_evolution_medium_v5_resolved_queue_and_matrix_capture_the_followup() -> None:
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
    assert [row["status"] for row in resolved["rows"]] == ["completed"]
    assert [row["run_id"] for row in resolved["rows"]] == [RUN_ID]
    assert [row["decision"] for row in resolved["rows"]] == ["defer"]

    resolved_row = resolved["rows"][0]
    assert resolved_row["data"]["corpus_ref"] == EXPECTED_CORPUS_REF
    assert resolved_row["resolved_surface"]["training"]["schedule_stages"][0]["steps"] == 2500
    assert resolved_row["resolved_surface"]["training"]["task_batch_size"] == 16
    assert resolved_row["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSS

    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")
    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `completed`" in matrix
    assert f"Anchor run id: `{ANCHOR_RUN_ID}`" in matrix
    assert "sorted-order code path" in matrix
    assert "trained fresh under the current sorted-order code path" in matrix
    assert "medium_v4" in matrix
    assert "keeps the original `medium_v4` control as the carried comparator" in matrix
    assert "no missingness promotion" in matrix
    assert EXPECTED_CORPUS_REF in matrix
    assert "final_log_loss_at_matched_regime_budget" in matrix
    assert RUN_ID in matrix
    assert "final log loss `0.6849`" in matrix


def test_tf_rd_010_classification_evolution_medium_v5_inspection_resolves_the_sorted_control_surface() -> None:
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
    assert resolved_data["recipe_id"] == EXPECTED_CORPUS_REF
    assert str(resolved_data["corpus_ref"]).startswith(EXPECTED_CORPUS_REF)
    assert resolved_training["schedule_stages"][0]["steps"] == 2500
    assert resolved_training["task_batch_size"] == 16
    assert payload["row"]["status"] == "completed"
    assert payload["row"]["run_id"] == RUN_ID
    assert payload["row"]["decision"] == "defer"
    assert payload["row"]["benchmark_metrics"]["final_log_loss"] == EXPECTED_FINAL_LOG_LOSS
