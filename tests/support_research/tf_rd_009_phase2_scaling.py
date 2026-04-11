from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
NS_SWEEP_ID = "tf_rd_009_ns_medium_v1"
BATCH_SWEEP_ID = "tf_rd_009_batch_critical_medium_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1"
GEOMETRIES = ["72x1", "96x2", "112x3", "128x4", "152x5", "176x6"]
STEP_LADDER = [625, 1250, 2500, 5000]
BATCH_LADDER = [1, 2, 4, 8, 16]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_phase2_sweeps_are_registered() -> None:
    index = _load_yaml(INDEX_PATH)

    assert index["sweeps"][NS_SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_width_depth_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }
    assert index["sweeps"][BATCH_SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_ns_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_009_ns_medium_v1_tracks_the_exact_geometry_step_matrix() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / NS_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == NS_SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_width_depth_medium_v1"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        training_config_profile="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law_phase2_ns",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == len(GEOMETRIES) * len(STEP_LADDER)
    geometry_counts = Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows)
    assert dict(geometry_counts) == {label: len(STEP_LADDER) for label in GEOMETRIES}
    step_counts = Counter(row["training"]["overrides"]["runtime"]["max_steps"] for row in rows)
    assert dict(step_counts) == {step: len(GEOMETRIES) for step in STEP_LADDER}
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows} == {4}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}


def test_tf_rd_009_batch_critical_medium_v1_tracks_the_exact_batch_step_ladder() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / BATCH_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == BATCH_SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_ns_medium_v1"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        training_config_profile="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law_phase2_batch",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == len(BATCH_LADDER) * len(STEP_LADDER)
    assert {f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows} == {"96x2"}
    step_counts = Counter(row["training"]["overrides"]["runtime"]["max_steps"] for row in rows)
    assert dict(step_counts) == {step: len(BATCH_LADDER) for step in STEP_LADDER}
    batch_counts = Counter(row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows)
    assert dict(batch_counts) == {batch: len(STEP_LADDER) for batch in BATCH_LADDER}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}


def test_tf_rd_009_phase2_study_config_references_the_new_sweeps_and_variables() -> None:
    study = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / "tf_rd_009_phase2.yaml")

    assert study["schema"] == "tab-foundry-scaling-study-v1"
    assert study["study_id"] == "tf_rd_009_phase2"
    assert study["phase1_reference_sweep_id"] == "tf_rd_009_width_depth_medium_v1"
    assert study["geometry_row_labels"] == GEOMETRIES
    assert study["step_ladder"] == STEP_LADDER
    assert study["batch_grad_accum_ladder"] == BATCH_LADDER
    assert study["sweeps"] == [
        {"name": "ns_core", "sweep_id": NS_SWEEP_ID, "family": "ns_core"},
        {"name": "batch_critical", "sweep_id": BATCH_SWEEP_ID, "family": "batch_critical"},
    ]
    assert study["canonical_variables"] == {
        "N": "parameter_accounting.canonical_non_embedding_params",
        "D": "regime_budget.tokens_seen",
        "S": "tab_foundry_metrics.final_step",
        "B_eff": "regime_budget.tokens_per_step",
        "C": "compute_accounting.total_train_flops",
    }
