from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_021b_sandwich_knob_sensitivity_v1"
ANCHOR_RUN_ID = "tf_rd_021b_hybrid_full_cell_compact_prior_v1"
EXPECTED_ROWS = [
    "delta_tf_rd_021b_sandwich_latents12_v1",
    "delta_tf_rd_021b_sandwich_layers1_v1",
    "delta_tf_rd_021b_sandwich_heads2_v1",
    "delta_tf_rd_021b_sandwich_ffexp1_v1",
    "delta_tf_rd_021b_sandwich_summarytokens1_v1",
    "delta_tf_rd_021b_sandwich_selfattn1_v1",
    "delta_tf_rd_021b_sandwich_prerow0_v1",
    "delta_tf_rd_021b_sandwich_precol0_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_021b_sandwich_knob_sensitivity_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_021a_sandwich_openml_screen_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": [],
    }


def test_tf_rd_021b_sandwich_knob_sensitivity_v1_matches_the_screen_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_021a_sandwich_openml_screen_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_hybrid_prior",
        surface_role="architecture_screen",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert sweep["upstream_reference"] == {
        "name": "PerceiverIO",
        "model_source": "https://openreview.net/forum?id=fILj7WpI-g",
    }
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"] == {
        "arch": "tabfoundry_sandwich",
        "benchmark_profile": "sandwich_hybrid_compact_prior",
        "stage": None,
        "stage_label": "sandwich_hybrid_compact_prior",
        "module_selection": None,
    }
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "prior_dump",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#178" in note for note in notes)
    assert any("does not run any external comparator" in note for note in notes)
    assert any("All eight stage-1 ablations underperformed" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)

    latents = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_latents12_v1")
    assert latents["run_id"] == "sd_tf_rd_021b_sandwich_knob_sensitivity_v1_01_delta_tf_rd_021b_sandwich_latents12_v1_v1"
    assert "sandwich_latents" in latents["anchor_delta"]
    assert "2500" in " ".join(latents["parameter_adequacy_plan"])
    assert latents["decision"] == "defer"
    assert latents["benchmark_metrics"]["delta_final_log_loss"] > 0.0

    summary_tokens = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_summarytokens1_v1")
    assert "sandwich_summary_tokens_per_axis" in summary_tokens["anchor_delta"]
    assert "raw-cell bypass" in summary_tokens["hypothesis"]
    assert summary_tokens["benchmark_metrics"]["delta_final_log_loss"] > 0.0

    pre_row = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_prerow0_v1")
    assert "sandwich_pre_row_attention_layers" in pre_row["anchor_delta"]
    assert pre_row["status"] == "completed"

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment="cls_benchmark_sandwich_hybrid_prior",
        surface_role="architecture_screen",
        external_benchmarks=[],
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS

    materialized_latents = next(
        row for row in materialized["rows"] if row["delta_id"] == "delta_tf_rd_021b_sandwich_latents12_v1"
    )
    assert materialized_latents["model"]["sandwich_latents"] == 12
    assert materialized_latents["training"]["surface_label"] == "prior_cosine_warmup"


def test_tf_rd_021b_sandwich_knob_sensitivity_v1_matrix_records_local_only_benchmarking() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "PerceiverIO" in matrix
    assert "External benchmarks: `none`" in matrix
    assert "cls_benchmark_sandwich_hybrid_prior" in matrix
    assert "delta_tf_rd_021b_sandwich_latents12_v1" in matrix
    assert "delta_tf_rd_021b_sandwich_precol0_v1" in matrix
    assert "Locked medium binary bundle with no external comparator." in matrix
    assert "Registered run: `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_06_delta_tf_rd_021b_sandwich_selfattn1_v1_v1`" in matrix
