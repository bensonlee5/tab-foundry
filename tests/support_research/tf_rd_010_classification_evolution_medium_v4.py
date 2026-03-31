from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_medium_v4"
PREVIOUS_SWEEP_ID = "tf_rd_010_classification_evolution_medium_v3"
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]
EXPECTED_CORPUS_REFS = [
    "tf_rd_010_dagzoo_medium_control_v3",
    "tf_rd_010_missingness_mcar_v3",
    "tf_rd_010_missingness_mar_v3",
    "tf_rd_010_missingness_mnar_v3",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_classification_evolution_medium_v4_is_registered_and_medium_v3_is_preserved() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)

    previous_entry = sweeps[PREVIOUS_SWEEP_ID]
    assert previous_entry["parent_sweep_id"] == "tf_rd_010_classification_evolution_medium_v2"
    assert previous_entry["status"] == "superseded"
    assert previous_entry["anchor_run_id"] == (
        "sd_tf_rd_010_classification_evolution_medium_v3_01_"
        "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1"
    )

    entry = sweeps[SWEEP_ID]
    assert entry["parent_sweep_id"] == PREVIOUS_SWEEP_ID
    assert entry["status"] == "blocked_on_synthetic_adequacy"
    assert entry["anchor_run_id"] is None
    assert entry["complexity_level"] == "classification_md"
    assert entry["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
    )
    assert entry["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert entry["external_benchmarks"] == []


def test_tf_rd_010_classification_evolution_medium_v4_records_the_accumulation_lr_rerun_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == PREVIOUS_SWEEP_ID
    assert sweep["status"] == "blocked_on_synthetic_adequacy"
    assert sweep["anchor_run_id"] is None
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert "medium_v3" in notes[0]
    assert "blocked pending `tf_rd_010_synthetic_adequacy_v1`" in notes[0]
    assert any("task_batch_size=16" in note for note in notes)
    assert any("grad_accum_steps=4" in note for note in notes)
    assert any("optimizer.min_lr=1e-5" in note for note in notes)
    assert any("lr_max=1e-3" in note for note in notes)
    assert any("final_log_loss_at_matched_regime_budget" in note for note in notes)
    assert any("label-target log loss per test cell" in note for note in notes)
    assert any("training.loss_surface=classification" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["blocked_on_synthetic_adequacy"] * len(EXPECTED_ROWS)
    assert [row["run_id"] for row in rows] == [None] * len(EXPECTED_ROWS)
    assert [row["decision"] for row in rows] == [None] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["blocked"] * len(EXPECTED_ROWS)
    assert [row["data"]["corpus_ref"] for row in rows] == EXPECTED_CORPUS_REFS
    assert "Block this row pending `tf_rd_010_synthetic_adequacy_v1`" in rows[0]["next_action"]
    assert "do not retune or rerun `medium_v4`" in rows[0]["next_action"]
    assert all("Block this row pending `tf_rd_010_synthetic_adequacy_v1`" in row["next_action"] for row in rows[1:])
    assert all(
        any("task_batch_size=16" in note and "grad_accum_steps=4" in note for note in row["notes"])
        for row in rows
    )
    assert all(
        any("training.loss_surface=classification" in note for note in row["notes"])
        for row in rows
    )
    assert all(
        any("medium_v3" in note and "historical no-clipping evidence only" in note for note in row["notes"])
        for row in rows
    )
    assert any("step `1324`" in note and "step_001200.pt" in note for note in rows[0]["notes"])
    assert all(
        any("blocked until `tf_rd_010_synthetic_adequacy_v1` is interpreted" in note for note in row["notes"])
        for row in rows[1:]
    )

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] is None
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 2500 for row in materialized["rows"])
    assert all(
        row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
        for row in materialized["rows"]
    )
    assert all(row["training"]["task_batch_size"] == 16 for row in materialized["rows"])
    assert all(
        row["training"]["overrides"]["optimizer"]["min_lr"] == 1.0e-5
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 2500
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 1.0e-3
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["lr_schedule"] == "linear"
        for row in materialized["rows"]
    )
    assert all(
        row["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.10
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


def test_tf_rd_010_classification_evolution_medium_v4_resolved_queue_captures_runtime_surface() -> None:
    resolved = _load_yaml(
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / SWEEP_ID
        / "resolved_queue.yaml"
    )

    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["sweep_status"] == "blocked_on_synthetic_adequacy"
    assert resolved["anchor_run_id"] is None

    row = resolved["rows"][0]
    assert row["status"] == "blocked_on_synthetic_adequacy"
    assert row["run_id"] is None
    assert row["decision"] is None
    assert row["interpretation_status"] == "blocked"

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


def test_tf_rd_010_classification_evolution_medium_v4_matrix_and_medium_v3_matrix_record_the_pilot_transition() -> None:
    v4_matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")
    v3_matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / PREVIOUS_SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert SWEEP_ID in v4_matrix
    assert "Sweep status: `blocked_on_synthetic_adequacy`" in v4_matrix
    assert "Anchor run id: `null`" in v4_matrix
    assert "Block this row pending `tf_rd_010_synthetic_adequacy_v1`" in v4_matrix
    assert "grad_accum_steps': 4" in v4_matrix
    assert "task_batch_size=16" in v4_matrix
    assert "optimizer.min_lr=1e-5" in v4_matrix
    assert "warmup_ratio=0.10" in v4_matrix
    assert "lr_max=1e-3" in v4_matrix
    assert "final_log_loss_at_matched_regime_budget" in v4_matrix
    assert "label-target log loss per test cell" in v4_matrix
    assert "step `1324`" in v4_matrix
    assert "step_001200.pt" in v4_matrix
    assert "Block this row pending `tf_rd_010_synthetic_adequacy_v1`" in v4_matrix

    assert PREVIOUS_SWEEP_ID in v3_matrix
    assert "Sweep status: `superseded`" in v3_matrix
    assert "historical no-clipping overfit evidence only" in v3_matrix
    assert "`medium_v4` row 1" in v3_matrix
