from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_021b_sandwich_width_capacity_sensitivity_v1"
ANCHOR_RUN_ID = "tf_rd_021b_hybrid_full_cell_compact_prior_v1"
EXPECTED_ROWS = [
    "delta_tf_rd_021b_sandwich_dicl48_v1",
    "delta_tf_rd_021b_sandwich_dicl96_v1",
    "delta_tf_rd_021b_sandwich_headhidden64_v1",
    "delta_tf_rd_021b_sandwich_headhidden128_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_021b_sandwich_width_capacity_sensitivity_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_021b_sandwich_knob_sensitivity_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": [],
    }


def test_tf_rd_021b_sandwich_width_capacity_sensitivity_v1_matches_the_followup_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_021b_sandwich_knob_sensitivity_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["external_benchmarks"] == []
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_hybrid_prior"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_hybrid_prior"
    assert sweep["surface_role"] == "architecture_screen"
    assert sweep["comparison_policy"] == "anchor_only"
    assert sweep["upstream_reference"] == {
        "name": "PerceiverIO",
        "model_source": "https://openreview.net/forum?id=fILj7WpI-g",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("external comparator" in note for note in notes)
    assert any("did not identify a simpler winning topology setting" in note for note in notes)
    assert any("least harmful row was `d_icl=96`" in note for note in notes)
    assert any("The empirical power-curve phase comes after" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)

    width_low = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_dicl48_v1")
    assert "d_icl" in width_low["anchor_delta"]
    assert "now ready to execute" in " ".join(width_low["parameter_adequacy_plan"])
    assert width_low["run_id"] == "sd_tf_rd_021b_sandwich_width_capacity_sensitivity_v1_01_delta_tf_rd_021b_sandwich_dicl48_v1_v1"
    assert width_low["decision"] == "defer"
    assert width_low["benchmark_metrics"]["delta_final_log_loss"] > 0.0

    width_high = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_dicl96_v1")
    assert "d_icl" in width_high["anchor_delta"]
    assert "first live post-topology capacity axis" in width_high["hypothesis"]
    assert width_high["benchmark_metrics"]["delta_final_log_loss"] < width_low["benchmark_metrics"]["delta_final_log_loss"]

    head_high = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_headhidden128_v1")
    assert "head_hidden_dim" in head_high["anchor_delta"]
    assert "power-curve phase opens" in " ".join(head_high["parameter_adequacy_plan"])
    assert head_high["benchmark_metrics"]["delta_final_log_loss"] > 0.0

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["training_experiment"] == "cls_benchmark_sandwich_hybrid_prior"
    assert materialized["training_config_profile"] == "cls_benchmark_sandwich_hybrid_prior"
    assert materialized["surface_role"] == "architecture_screen"
    assert materialized["external_benchmarks"] == []
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS

    materialized_width_high = next(
        row for row in materialized["rows"] if row["delta_id"] == "delta_tf_rd_021b_sandwich_dicl96_v1"
    )
    assert materialized_width_high["model"]["d_icl"] == 96
    assert materialized_width_high["model"]["sandwich_heads"] == 4


def test_tf_rd_021b_sandwich_width_capacity_sensitivity_v1_matrix_records_the_completed_followup() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "External benchmarks: `none`" in matrix
    assert "Status | Recipe alias | Effective change | Next action" in matrix
    assert "| 1 | `delta_tf_rd_021b_sandwich_dicl48_v1` | width_capacity | yes | completed |" in matrix
    assert "delta_tf_rd_021b_sandwich_dicl96_v1" in matrix
    assert "delta_tf_rd_021b_sandwich_headhidden128_v1" in matrix
    assert "before the later power-curve phase." in matrix
    assert "Registered run: `sd_tf_rd_021b_sandwich_width_capacity_sensitivity_v1_02_delta_tf_rd_021b_sandwich_dicl96_v1_v1`" in matrix
