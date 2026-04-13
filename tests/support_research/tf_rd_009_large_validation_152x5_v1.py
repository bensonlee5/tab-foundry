from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_009_large_validation_152x5_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_large_v2_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1"
)
TRAIN_RUN_ID = (
    "sd_tf_rd_009_width_depth_medium_v1_04_"
    "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1"
)
VALIDATION_RUN_ID = (
    "sd_tf_rd_009_large_validation_152x5_v1_01_"
    "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2"
)
EXPECTED_ROW = "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1"
TRAINING_SURFACE_FINGERPRINT = "8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253"
REUSE_TRAIN_ARTIFACT = {
    "run_dir": (
        "outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/"
        "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/"
        "sd_tf_rd_009_width_depth_medium_v1_04_"
        "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1/train"
    ),
    "training_surface_fingerprint": TRAINING_SURFACE_FINGERPRINT,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_large_validation_152x5_v1_is_registered() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_width_depth_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_lg",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_large_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_large_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_009_large_validation_152x5_v1_tracks_the_reused_large_gate() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_width_depth_medium_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/openml_classification_large_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("issue 257" in note for note in notes)
    assert any("152x5" in note for note in notes)
    assert any("[363685, 363699, 363707]" in note for note in notes)
    assert any("0.8974410961" in note for note in notes)
    assert any("do not retrain or add extra seeds" in note for note in notes)
    assert any("tf_rd_009_rtx8000_44gb_classification_medium_v1" in note for note in notes)
    assert any("issues 259 and 260" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert row["delta_ref"] == EXPECTED_ROW
    assert row["status"] == "completed"
    assert row["run_id"] == VALIDATION_RUN_ID
    assert row["decision"] == "keep"
    assert row["interpretation_status"] == "completed"
    assert row["reuse_train_artifact"] == REUSE_TRAIN_ARTIFACT
    assert row["benchmark_checkpoint_selection"] == "all"
    benchmark_metrics = row["benchmark_metrics"]
    assert benchmark_metrics["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert benchmark_metrics["final_log_loss"] == 0.7436636567836484
    assert benchmark_metrics["delta_final_log_loss"] == -0.15377743935891375
    assert benchmark_metrics["final_roc_auc"] == 0.765094033761902
    assert benchmark_metrics["delta_final_roc_auc"] == 0.1327435146887801
    assert row["model"]["d_icl"] == 152
    assert row["model"]["sandwich_layers"] == 5
    assert row["data"]["corpus_ref"] == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert "benchmark-only validate" in row["rationale"].lower()
    assert "large clean-control anchor" in row["rationale"]
    assert "no retraining" in row["anchor_delta"]
    assert "no extra seeds" in row["anchor_delta"]
    assert "0.8974410961" in " ".join(row["parameter_adequacy_plan"])
    assert "[363685, 363699, 363707]" in " ".join(row["parameter_adequacy_plan"])
    assert "frozen hardware baseline" in row["next_action"]
    assert any("do not retrain this row" in note for note in row["notes"])
    assert any("training-surface fingerprint" in note for note in row["notes"])
    assert any("do not add the large validation row to the medium constraint model" in note for note in row["notes"])
    assert any("current local workstation does not expose the required GPU surface" in note for note in row["notes"])
    assert any("Execution attempt" in note for note in row["notes"])
    assert any("latest.pt" in note for note in row["notes"])
    assert any("25/25" in note for note in row["notes"])
    assert any("Hardware baseline `tf_rd_009_rtx8000_44gb_classification_medium_v1` is now frozen" in note for note in row["notes"])
    assert any("Canonical rerun registered" in note for note in row["notes"])


def test_tf_rd_009_large_validation_152x5_v1_resolved_queue_captures_reuse_surface() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    assert_training_surface_semantics(
        queue,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    row = queue["rows"][0]
    assert row["delta_id"] == EXPECTED_ROW
    assert row["reuse_train_artifact"] == REUSE_TRAIN_ARTIFACT
    runtime_overrides = row["training"]["overrides"]["runtime"]
    assert runtime_overrides["mixed_precision"] == "bf16"
    assert runtime_overrides["grad_accum_steps"] == 4
    assert runtime_overrides["activation_checkpointing"] is True
    assert runtime_overrides["max_steps"] == 2500

    resolved = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "resolved_queue.yaml")
    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert resolved["sweep_status"] == "draft"

    resolved_row = resolved["rows"][0]
    assert resolved_row["status"] == "completed"
    assert resolved_row["run_id"] == VALIDATION_RUN_ID
    assert resolved_row["decision"] == "keep"
    assert resolved_row["reuse_train_artifact"] == REUSE_TRAIN_ARTIFACT
    assert resolved_row["resolved_surface_fingerprint"] == TRAINING_SURFACE_FINGERPRINT
    assert resolved_row["benchmark_metrics"]["final_log_loss"] == 0.7436636567836484
    assert resolved_row["benchmark_metrics"]["delta_final_log_loss"] == -0.15377743935891375
    assert resolved_row["data"]["corpus_ref"] == "tf_rd_010_dagzoo_medium_control_curated_v5"

    resolved_surface = resolved_row["resolved_surface"]
    training = resolved_surface["training"]
    runtime = resolved_surface["runtime"]
    assert training["task_batch_size"] == 16
    assert training["optimizer_min_lr"] == 1.0e-5
    assert training["schedule_stages"][0]["steps"] == 2500
    assert training["schedule_stages"][0]["lr_max"] == 1.0e-3
    assert training["schedule_stages"][0]["lr_schedule"] == "linear"
    assert training["schedule_stages"][0]["warmup_ratio"] == 0.10
    assert runtime["mixed_precision"] == "bf16"
    assert runtime["grad_accum_steps"] == 4
    assert runtime["activation_checkpointing"] is True
    assert runtime["compile_model"] is True
    assert runtime["compile_dynamic"] is True
    assert runtime["compile_backend"] == "eager"
    assert runtime["max_steps"] == 2500


def test_tf_rd_009_large_validation_152x5_v1_matrix_records_the_single_candidate_gate() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `draft`" in matrix
    assert f"Anchor run id: `{ANCHOR_RUN_ID}`" in matrix
    assert "openml_classification_large_v1" in matrix
    assert "`0.8974410961`" in matrix
    assert TRAIN_RUN_ID in matrix
    assert VALIDATION_RUN_ID in matrix
    assert REUSE_TRAIN_ARTIFACT["run_dir"] in matrix
    assert TRAINING_SURFACE_FINGERPRINT in matrix
    assert "do not retrain this row" in matrix
    assert "tf_rd_009_rtx8000_44gb_classification_medium_v1" in matrix
    assert "Decision: `keep`" in matrix
    assert "latest.pt" in matrix
    assert "do not add the large validation row to the medium constraint model" in matrix
    assert "current local workstation does not expose the required GPU surface" in matrix
