from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_024_classification_knob_sweep_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v4_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8"
)
EXPECTED_ROWS = [
    "delta_tf_rd_024_cls_sandwich_latents12_v1",
    "delta_tf_rd_024_cls_sandwich_heads2_v1",
    "delta_tf_rd_024_cls_sandwich_ffexp1_v1",
    "delta_tf_rd_024_cls_sandwich_summarytokens1_v1",
    "delta_tf_rd_024_cls_sandwich_selfattn1_v1",
    "delta_tf_rd_024_cls_sandwich_headhidden64_v1",
    "delta_tf_rd_024_cls_sandwich_headhidden128_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_024_classification_knob_sweep_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_010_classification_evolution_medium_v4",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_024_classification_knob_sweep_v1_matches_the_post_performance_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_010_classification_evolution_medium_v4"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1",
        surface_role="classification_architecture_followup",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert sweep["upstream_reference"] == {
        "name": "PerceiverIO",
        "model_source": "https://openreview.net/forum?id=fILj7WpI-g",
    }
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_010_dagzoo_medium_control_curated_v5",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#168" in note for note in notes)
    assert any("dagzoo RD-002/RD-005" in note for note in notes)
    assert any("Keep `d_icl`, `sandwich_layers`, batch size, LR, clipping" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["blocked_on_runtime_policy"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["blocked"] * len(EXPECTED_ROWS)

    latents = _row_by_ref(queue, "delta_tf_rd_024_cls_sandwich_latents12_v1")
    assert "sandwich_latents" in latents["anchor_delta"]
    assert "overprovisioned" in latents["hypothesis"]

    summary_tokens = _row_by_ref(queue, "delta_tf_rd_024_cls_sandwich_summarytokens1_v1")
    assert "raw-cell bypass" in summary_tokens["hypothesis"]
    assert "sandwich_summary_tokens_per_axis" in summary_tokens["anchor_delta"]

    headhidden128 = _row_by_ref(queue, "delta_tf_rd_024_cls_sandwich_headhidden128_v1")
    assert "binding readout bottleneck" in headhidden128["hypothesis"]
    assert "head_hidden_dim" in headhidden128["anchor_delta"]

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1",
        surface_role="classification_architecture_followup",
        external_benchmarks=[],
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS

    for row in materialized["rows"]:
        runtime = row["training"]["overrides"]["runtime"]
        assert row["status"] == "blocked_on_runtime_policy"
        assert row["interpretation_status"] == "blocked"
        assert row["training"]["task_batch_size"] == 16
        assert row["training"]["prior_dump_batch_size"] == 64
        assert runtime["mixed_precision"] == "bf16"
        assert runtime["trace_activations"] is False
        assert runtime["activation_checkpointing"] is False
        assert runtime["grad_accum_steps"] == 4
        assert row["model"]["d_icl"] == 60
        assert row["model"]["sandwich_layers"] == 2


def test_tf_rd_024_classification_knob_sweep_v1_matrix_records_the_runtime_policy_handoff() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "PerceiverIO" in matrix
    assert "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1" in matrix
    assert "delta_tf_rd_024_cls_sandwich_headhidden128_v1" in matrix
    assert "mixed_precision': 'bf16'" in matrix
    assert "TF-RD-010 medium contract screens rows first; any keep must then validate on the closed TF-RD-010 large contract." in matrix
    assert "`d_icl`" in matrix
    assert "`sandwich_layers`" in matrix
