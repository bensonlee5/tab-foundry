from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_022_runtime_policy_medium_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v4_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8"
)
EXPECTED_ROWS = [
    "delta_tf_rd_022_cls_runtime_control_noamp_v1",
    "delta_tf_rd_022_cls_runtime_bf16_v1",
    "delta_tf_rd_022_cls_runtime_trace_v1",
    "delta_tf_rd_022_cls_runtime_checkpoint_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_022_runtime_policy_medium_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_010_classification_evolution_medium_v4",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_022_runtime_policy_medium_v1_matches_the_runtime_ladder_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_010_classification_evolution_medium_v4"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_v1",
        surface_role="classification_runtime_policy",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert sweep["upstream_reference"] == {
        "name": "PyTorch AMP",
        "model_source": "https://pytorch.org/docs/stable/amp.html",
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
    assert any("#169" in note for note in notes)
    assert any("Rows 2 through 4 vary only" in note for note in notes)
    assert any("peak_vram_reserved" in note for note in notes)
    assert any("measured defer" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["interpretation_status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert all(row["execution_policy"] == "benchmark_full" for row in rows)
    assert all(row["benchmark_checkpoint_selection"] == "all" for row in rows)

    control = _row_by_ref(queue, "delta_tf_rd_022_cls_runtime_control_noamp_v1")
    assert control["training"]["overrides"]["runtime"]["mixed_precision"] == "no"
    assert control["training"]["overrides"]["runtime"]["trace_activations"] is False
    assert control["training"]["overrides"]["runtime"]["activation_checkpointing"] is False

    bf16 = _row_by_ref(queue, "delta_tf_rd_022_cls_runtime_bf16_v1")
    assert bf16["training"]["overrides"]["runtime"]["mixed_precision"] == "bf16"
    assert bf16["training"]["overrides"]["runtime"]["trace_activations"] is False
    assert bf16["training"]["overrides"]["runtime"]["activation_checkpointing"] is False

    trace = _row_by_ref(queue, "delta_tf_rd_022_cls_runtime_trace_v1")
    assert trace["training"]["overrides"]["runtime"]["mixed_precision"] == "bf16"
    assert trace["training"]["overrides"]["runtime"]["trace_activations"] is True
    assert trace["training"]["overrides"]["runtime"]["activation_checkpointing"] is False

    checkpoint = _row_by_ref(queue, "delta_tf_rd_022_cls_runtime_checkpoint_v1")
    assert checkpoint["training"]["overrides"]["runtime"]["mixed_precision"] == "bf16"
    assert checkpoint["training"]["overrides"]["runtime"]["trace_activations"] is False
    assert checkpoint["training"]["overrides"]["runtime"]["activation_checkpointing"] is True
    assert checkpoint["decision"] == "keep"
    assert checkpoint["run_id"] == (
        "sd_tf_rd_022_runtime_policy_medium_v1_04_"
        "delta_tf_rd_022_cls_runtime_checkpoint_v1_v2"
    )
    assert checkpoint["benchmark_metrics"]["final_log_loss"] == 0.6765953231883223

    assert bf16["decision"] == "defer"
    assert bf16["run_id"] == (
        "sd_tf_rd_022_runtime_policy_medium_v1_02_delta_tf_rd_022_cls_runtime_bf16_v1_v1"
    )
    assert bf16["benchmark_metrics"]["peak_vram_reserved"] == 5312086016

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment="cls_benchmark_sandwich_classification_evolution_v1",
        surface_role="classification_runtime_policy",
        external_benchmarks=[],
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS

    for row in materialized["rows"]:
        runtime = row["training"]["overrides"]["runtime"]
        assert row["status"] == "completed"
        assert row["interpretation_status"] == "completed"
        assert row["training"]["task_batch_size"] == 16
        assert row["training"]["prior_dump_batch_size"] == 64
        assert row["training"]["synthetic_epoch_budget"]["resolved_task_count"] == 159984
        assert row["training"]["synthetic_epoch_budget"]["resolved_max_steps"] == 2500
        assert runtime["grad_accum_steps"] == 4
        assert runtime["grad_clip"] == 0.0
        assert runtime["max_steps"] == 2500
        assert runtime["eval_every"] == 25
        assert runtime["checkpoint_every"] == 25
        assert row["model"]["d_icl"] == 60
        assert row["model"]["sandwich_layers"] == 2


def test_tf_rd_022_runtime_policy_medium_v1_resolved_queue_captures_runtime_surfaces() -> None:
    resolved = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "resolved_queue.yaml"
    )

    assert resolved["schema"] == "tab-foundry-system-delta-resolved-queue-v1"
    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["sweep_status"] == "completed"
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["status"] for row in resolved["rows"]] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["run_id"] for row in resolved["rows"]] == [
        "sd_tf_rd_022_runtime_policy_medium_v1_01_delta_tf_rd_022_cls_runtime_control_noamp_v1_v4",
        "sd_tf_rd_022_runtime_policy_medium_v1_02_delta_tf_rd_022_cls_runtime_bf16_v1_v1",
        "sd_tf_rd_022_runtime_policy_medium_v1_03_delta_tf_rd_022_cls_runtime_trace_v1_v2",
        "sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2",
    ]
    assert [row["decision"] for row in resolved["rows"]] == ["defer", "defer", "defer", "keep"]

    control_runtime = resolved["rows"][0]["resolved_surface"]["runtime"]
    assert control_runtime["mixed_precision"] == "no"
    assert control_runtime["trace_activations"] is False
    assert control_runtime["activation_checkpointing"] is False

    bf16_runtime = resolved["rows"][1]["resolved_surface"]["runtime"]
    assert bf16_runtime["mixed_precision"] == "bf16"
    assert bf16_runtime["trace_activations"] is False
    assert bf16_runtime["activation_checkpointing"] is False

    trace_runtime = resolved["rows"][2]["resolved_surface"]["runtime"]
    assert trace_runtime["mixed_precision"] == "bf16"
    assert trace_runtime["trace_activations"] is True
    assert trace_runtime["activation_checkpointing"] is False

    checkpoint_runtime = resolved["rows"][3]["resolved_surface"]["runtime"]
    assert checkpoint_runtime["mixed_precision"] == "bf16"
    assert checkpoint_runtime["trace_activations"] is False
    assert checkpoint_runtime["activation_checkpointing"] is True


def test_tf_rd_022_runtime_policy_medium_v1_matrix_records_the_benchmark_first_keep_bar() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `completed`" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "PyTorch AMP" in matrix
    assert "classification_runtime_policy" in matrix
    assert "delta_tf_rd_022_cls_runtime_bf16_v1" in matrix
    assert "delta_tf_rd_022_cls_runtime_trace_v1" in matrix
    assert "delta_tf_rd_022_cls_runtime_checkpoint_v1" in matrix
    assert "final_log_loss_at_matched_regime_budget" in matrix
    assert "peak_vram_reserved" in matrix
    assert "throughput_tokens_per_second" in matrix
    assert "diagnostic loser unless it is benchmark-safe" in matrix
    assert "Decision: `keep`" in matrix
    assert "sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2" in matrix
