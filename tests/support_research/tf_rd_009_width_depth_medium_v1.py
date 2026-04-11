from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.research.tf_rd_009_width_depth_derivation import (
    collect_tf_rd_009_completed_measured_fit_points,
)
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_009_width_depth_medium_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1"
EXPECTED_ROWS = [
    "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
    "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
    "delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1",
    "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1",
    "delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1",
]
EXPECTED_COMPLETED_RUN_IDS = [
    "sd_tf_rd_009_width_depth_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1",
    "sd_tf_rd_009_width_depth_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1",
    "sd_tf_rd_009_width_depth_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1",
    "sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1",
    "sd_tf_rd_009_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_009_width_depth_medium_v1_is_registered_as_the_pending_joint_family() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_width_transfer_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_009_width_depth_medium_v1_tracks_the_corrected_dense_diagonal() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_width_transfer_medium_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert [row["interpretation_status"] for row in rows] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert [row["run_id"] for row in rows] == EXPECTED_COMPLETED_RUN_IDS

    lower = _row_by_ref(queue, "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1")
    assert lower["model"]["d_icl"] == 72
    assert lower["model"]["sandwich_layers"] == 1
    assert "empirical depth-aware parameter fit" in lower["parameter_adequacy_plan"][0]
    assert "72x1" in lower["parameter_adequacy_plan"][2]

    upper_seed = _row_by_ref(queue, "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1")
    assert upper_seed["model"]["d_icl"] == 112
    assert upper_seed["model"]["sandwich_layers"] == 3
    assert "128x2" in upper_seed["parameter_adequacy_plan"][0]

    interpolation = _row_by_ref(queue, "delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1")
    assert interpolation["model"]["d_icl"] == 128
    assert interpolation["model"]["sandwich_layers"] == 4

    penultimate = _row_by_ref(queue, "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1")
    assert penultimate["model"]["d_icl"] == 152
    assert penultimate["model"]["sandwich_layers"] == 5

    ceiling = _row_by_ref(queue, "delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1")
    assert ceiling["model"]["d_icl"] == 176
    assert ceiling["model"]["sandwich_layers"] == 6
    assert "32-33 GB" in ceiling["parameter_adequacy_plan"][0]


def test_tf_rd_009_width_depth_medium_v1_materialized_queue_and_matrix_match_canonical_artifacts() -> None:
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
    rows = queue["rows"]
    assert [row["delta_id"] for row in rows] == EXPECTED_ROWS
    assert [row["model"]["d_icl"] for row in rows] == [72, 112, 128, 152, 176]
    assert [row["model"]["sandwich_layers"] for row in rows] == [1, 3, 4, 5, 6]

    resolved = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "resolved_queue.yaml")
    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in resolved["rows"]] == EXPECTED_ROWS

    matrix = (REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md").read_text(
        encoding="utf-8"
    )
    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1" in matrix
    assert "88x1" not in matrix
    assert "104x3" not in matrix


def test_tf_rd_009_width_depth_medium_v1_reported_fit_inputs_use_completed_in_family_rows_only() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    points = collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=default_benchmark_run_registry_path(),
    )

    assert [point.row_label for point in points] == ["72x1", "96x2", "112x3", "128x4", "152x5", "176x6"]
    assert [point.run_id for point in points] == [
        EXPECTED_COMPLETED_RUN_IDS[0],
        ANCHOR_RUN_ID,
        EXPECTED_COMPLETED_RUN_IDS[1],
        EXPECTED_COMPLETED_RUN_IDS[2],
        EXPECTED_COMPLETED_RUN_IDS[3],
        EXPECTED_COMPLETED_RUN_IDS[4],
    ]
    assert all(point.row_label not in {"88x1", "104x3", "112x4", "128x5", "144x6"} for point in points)
