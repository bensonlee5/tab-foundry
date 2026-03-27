from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_021b_sandwich_feature_removal_v1"
ANCHOR_RUN_ID = "tf_rd_021b_hybrid_full_cell_compact_prior_v1"
EXPECTED_ROWS = [
    "delta_tf_rd_021b_sandwich_selfattn0_v1",
    "delta_tf_rd_021b_sandwich_ffexp1_v1",
    "delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1",
    "delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_021b_sandwich_feature_removal_v1_is_registered_without_global_active_sweep() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_021b_sandwich_width_capacity_sensitivity_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": [],
    }


def test_tf_rd_021b_sandwich_feature_removal_v1_matches_the_removal_first_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_021b_sandwich_width_capacity_sensitivity_v1"
    assert sweep["status"] == "draft"
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
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "prior_dump",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#184" in note for note in notes)
    assert any("`sandwich_self_attention_per_cross=0`" in note for note in notes)
    assert any("Do not rerun `sandwich_pre_row_attention_layers=0`" in note for note in notes)
    assert any("benchmark and workstation sandwich profiles" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["pending"] * len(EXPECTED_ROWS)
    assert all(row["run_id"] is None for row in rows)
    assert all(row["decision"] is None for row in rows)

    selfattn0 = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_selfattn0_v1")
    assert selfattn0["model"]["sandwich_self_attention_per_cross"] == 0
    assert selfattn0["model"]["sandwich_ff_expansion"] == 2
    assert "replacement for the earlier self-attention-depth ablation" in " ".join(
        selfattn0["parameter_adequacy_plan"]
    )
    assert "do not execute or promote in this pass" in selfattn0["next_action"]

    ffexp1 = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_ffexp1_v1")
    assert ffexp1["model"]["sandwich_ff_expansion"] == 1
    assert ffexp1["model"]["sandwich_self_attention_per_cross"] == 4
    assert "`sandwich_ff_expansion=0`" in " ".join(ffexp1["parameter_adequacy_plan"])

    compound = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1")
    assert compound["model"]["sandwich_self_attention_per_cross"] == 0
    assert compound["model"]["sandwich_ff_expansion"] == 1

    smallest = _row_by_ref(queue, "delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1")
    assert smallest["model"]["sandwich_self_attention_per_cross"] == 0
    assert smallest["model"]["sandwich_ff_expansion"] == 1
    assert smallest["model"]["sandwich_summary_tokens_per_axis"] == 1

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


def test_tf_rd_021b_sandwich_feature_removal_v1_matrix_records_the_draft_queue() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "External benchmarks: `none`" in matrix
    assert "delta_tf_rd_021b_sandwich_selfattn0_v1" in matrix
    assert "delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1" in matrix
    assert "Remove latent self-attention refinement entirely" in matrix
    assert "do not execute or promote in this pass" in matrix
