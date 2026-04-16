from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
NS_SWEEP_ID = "tf_rd_009_muon_ns_one_epoch_medium_v1"
BATCH_SWEEP_ID = "tf_rd_009_muon_batch_critical_one_epoch_medium_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
GEOMETRIES = ["72x1", "112x3", "144x4", "192x5", "264x6"]
STEP_LADDER = [625, 1250, 2500, 5000]
BATCH_LADDER = [1, 2, 4, 8, 16]
MUON_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
CORPUS_V6 = "tf_rd_010_dagzoo_medium_control_curated_v6"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_muon_phase2_sweeps_are_registered() -> None:
    index = _load_yaml(INDEX_PATH)

    assert index["sweeps"][NS_SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_muon_width_depth_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }
    assert index["sweeps"][BATCH_SWEEP_ID] == {
        "parent_sweep_id": NS_SWEEP_ID,
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_009_muon_ns_one_epoch_medium_v1_tracks_the_exact_geometry_step_matrix() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / NS_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == NS_SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_muon_width_depth_medium_v1"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law_phase2_ns",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == len(GEOMETRIES) * len(STEP_LADDER)
    assert {row["status"] for row in rows} == {"completed"}
    assert {row["interpretation_status"] for row in rows} == {"completed"}
    assert {
        row["run_id"] for row in rows
    } == {
        f"sd_tf_rd_009_muon_ns_one_epoch_medium_v1_{int(row['order']):02d}_{row['delta_ref']}_v1"
        for row in rows
    }
    geometry_counts = Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows)
    assert dict(geometry_counts) == {label: len(STEP_LADDER) for label in GEOMETRIES}
    step_counts = Counter(row["training"]["overrides"]["runtime"]["max_steps"] for row in rows)
    assert dict(step_counts) == {step: len(GEOMETRIES) for step in STEP_LADDER}
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows} == {4}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}
    assert {row["training"]["overrides"]["optimizer"]["name"] for row in rows} == {"muon"}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}
    best_row = min(rows, key=lambda row: row["benchmark_metrics"]["final_log_loss"])
    assert best_row["order"] == 12
    assert f"{best_row['model']['d_icl']}x{best_row['model']['sandwich_layers']}" == "144x4"
    assert best_row["training"]["overrides"]["runtime"]["max_steps"] == 5000
    assert best_row["benchmark_metrics"]["final_log_loss"] == 0.3971900010756594


def test_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_tracks_the_exact_batch_step_ladder() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / BATCH_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == BATCH_SWEEP_ID
    assert sweep["parent_sweep_id"] == NS_SWEEP_ID
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law_phase2_batch",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == len(BATCH_LADDER) * len(STEP_LADDER)
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["run_id"] for row in rows} == {None}
    assert {f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows} == {"264x6"}
    step_counts = Counter(row["training"]["overrides"]["runtime"]["max_steps"] for row in rows)
    assert dict(step_counts) == {step: len(BATCH_LADDER) for step in STEP_LADDER}
    batch_counts = Counter(row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows)
    assert dict(batch_counts) == {batch: len(STEP_LADDER) for batch in BATCH_LADDER}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}
    assert {row["training"]["overrides"]["optimizer"]["name"] for row in rows} == {"muon"}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}


def test_tf_rd_009_muon_phase2_study_config_references_the_new_sweeps_and_variables() -> None:
    study = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / "tf_rd_009_muon_phase2_one_epoch_v1.yaml")

    assert study["schema"] == "tab-foundry-scaling-study-v1"
    assert study["study_id"] == "tf_rd_009_muon_phase2_one_epoch_v1"
    assert study["phase1_reference_sweep_id"] == "tf_rd_009_muon_width_depth_medium_v1"
    assert study["geometry_row_labels"] == GEOMETRIES
    assert study["step_ladder"] == STEP_LADDER
    assert study["batch_grad_accum_ladder"] == BATCH_LADDER
    assert study["sweeps"] == [
        {"name": "ns_core", "sweep_id": NS_SWEEP_ID, "family": "ns_core"},
        {"name": "batch_critical", "sweep_id": BATCH_SWEEP_ID, "family": "batch_critical"},
    ]
    assert study["primary_fit"] == {"law": "L(N,S)", "target": "validation_loss"}
    assert study["historical_context_studies"] == ["tf_rd_009_phase2", "tf_rd_009_phase2_one_epoch_v1"]
