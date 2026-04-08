from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_024_classification_heads_prerow_followup_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_024_classification_knob_sweep_v1_"
    "anchor_compile_eager_dynamic_v1"
)
EXPECTED_ROWS = [
    "delta_tf_rd_024_followup_cls_sandwich_heads1_v1",
    "delta_tf_rd_024_followup_cls_sandwich_prerow2_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_024_heads_prerow_followup_is_registered_and_completed() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_024_classification_knob_sweep_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_024_heads_prerow_followup_matches_the_two_row_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_024_classification_knob_sweep_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_architecture_followup",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert sweep["upstream_reference"] == {
        "name": "PerceiverIO",
        "model_source": "https://openreview.net/forum?id=fILj7WpI-g",
    }
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_010_dagzoo_medium_control_curated_v5",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("two-row TF-RD-024 follow-up" in note for note in notes)
    assert any("medium-only by decision" in note for note in notes)
    assert any("delta_tf_rd_024_cls_sandwich_heads2_v1" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed"]
    assert [row["interpretation_status"] for row in rows] == ["completed", "completed"]
    assert all(row["execution_policy"] == "benchmark_full" for row in rows)
    assert all(row["benchmark_checkpoint_selection"] == "all" for row in rows)

    heads1 = _row_by_ref(queue, "delta_tf_rd_024_followup_cls_sandwich_heads1_v1")
    assert heads1["model"]["sandwich_heads"] == 1
    assert heads1["model"]["sandwich_pre_row_attention_layers"] == 1
    assert heads1["decision"] == "keep"
    assert heads1["run_id"] == (
        "sd_tf_rd_024_classification_heads_prerow_followup_v1_01_"
        "delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1"
    )
    assert "TF-RD-009" in heads1["next_action"]
    assert heads1["benchmark_metrics"]["final_log_loss"] == 0.6603575332789623

    prerow2 = _row_by_ref(queue, "delta_tf_rd_024_followup_cls_sandwich_prerow2_v1")
    assert prerow2["model"]["sandwich_heads"] == 4
    assert prerow2["model"]["sandwich_pre_row_attention_layers"] == 2
    assert prerow2["decision"] == "defer"
    assert "independent topology probe" in prerow2["parameter_adequacy_plan"][0]
    assert prerow2["benchmark_metrics"]["final_log_loss"] == 0.6780725432447957

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_architecture_followup",
        external_benchmarks=[],
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS

    for row in materialized["rows"]:
        runtime = row["training"]["overrides"]["runtime"]
        assert row["status"] == "completed"
        assert row["interpretation_status"] == "completed"
        assert row["training"]["task_batch_size"] == 16
        assert row["training"]["prior_dump_batch_size"] == 64
        assert runtime["mixed_precision"] == "bf16"
        assert runtime["trace_activations"] is False
        assert runtime["activation_checkpointing"] is True
        assert runtime["grad_accum_steps"] == 4
        assert row["model"]["d_icl"] == 60
        assert row["model"]["sandwich_layers"] == 2
        resolved_runtime = row["resolved_surface"]["runtime"]
        assert resolved_runtime["compile_model"] is True
        assert resolved_runtime["compile_backend"] == "eager"
        assert resolved_runtime["compile_dynamic"] is True


def test_tf_rd_024_heads_prerow_followup_resolved_artifacts_capture_medium_only_bridge() -> None:
    resolved = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "resolved_queue.yaml"
    )

    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["sweep_status"] == "completed"
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["status"] for row in resolved["rows"]] == ["completed", "completed"]
    assert [row["run_id"] for row in resolved["rows"]] == [
        "sd_tf_rd_024_classification_heads_prerow_followup_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1",
        "sd_tf_rd_024_classification_heads_prerow_followup_v1_02_delta_tf_rd_024_followup_cls_sandwich_prerow2_v1_v1",
    ]

    heads1_runtime = resolved["rows"][0]["resolved_surface"]["runtime"]
    assert heads1_runtime["compile_model"] is True
    assert heads1_runtime["compile_backend"] == "eager"
    assert heads1_runtime["compile_dynamic"] is True
    assert resolved["rows"][0]["resolved_surface"]["model"]["architecture"]["heads"] == 1

    prerow2_runtime = resolved["rows"][1]["resolved_surface"]["runtime"]
    assert prerow2_runtime["compile_model"] is True
    assert prerow2_runtime["compile_backend"] == "eager"
    assert prerow2_runtime["compile_dynamic"] is True
    assert (
        resolved["rows"][1]["resolved_surface"]["model"]["architecture"]["pre_row_attention_layers"]
        == 2
    )

    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")
    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `completed`" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "sandwich_heads=2" in matrix
    assert "Medium-only is sufficient here" in matrix
    assert "delta_tf_rd_024_followup_cls_sandwich_heads1_v1" in matrix
    assert "delta_tf_rd_024_followup_cls_sandwich_prerow2_v1" in matrix
    assert "Final log loss: `0.6604`" in matrix
