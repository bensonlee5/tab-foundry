from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_018_optimizer_family_v1"
ANCHOR_RUN_ID = "sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2"
EXPECTED_ROWS = [
    "delta_training_adamw",
    "delta_training_muon",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def _assert_row_state(row: dict[str, Any], *, delta_ref: str) -> None:
    status = row["status"]
    assert status in {"ready", "completed"}
    if status == "ready":
        assert row.get("run_id") is None
        return
    assert row["interpretation_status"] == "completed"
    run_id = row.get("run_id")
    assert isinstance(run_id, str)
    assert run_id.startswith(f"sd_{SWEEP_ID}_")
    assert delta_ref in run_id
    benchmark_metrics = row.get("benchmark_metrics")
    assert isinstance(benchmark_metrics, dict)
    assert benchmark_metrics["final_log_loss"] is not None


def test_tf_rd_018_optimizer_family_v1_is_registered_and_completed() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["tf_rd_020_harder_dagzoo_ladder_v1"]["status"] == "completed"
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_020_harder_dagzoo_ladder_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": ["nanotabpfn"],
    }


def test_tf_rd_018_optimizer_family_v1_inherits_noise_drift_anchor_for_optimizer_comparison() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["parent_sweep_id"] == "tf_rd_020_harder_dagzoo_ladder_v1"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_staged_corpus"
    assert sweep["training_config_profile"] == "cls_benchmark_staged_corpus"
    assert sweep["surface_role"] == "architecture_screen"
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_020_shift_noise_drift",
        "model": "delta_qass_no_column_v3",
        "preprocessing": "runtime_default",
        "training": "linear_warmup_decay",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("issue `#137`" in note for note in notes)
    assert any("tf_rd_020_shift_noise_drift_v1" in note for note in notes)
    assert any("sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2" in note for note in notes)
    assert any("does not replay or promote a separate schedulefree row" in note for note in notes)
    assert any("tf_rd_020_noise_mixture_v1" in note for note in notes)
    assert any("inherited the same short-run mismatch" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    for row in rows:
        _assert_row_state(row, delta_ref=str(row["delta_ref"]))

    row1 = _row_by_ref(queue, "delta_training_adamw")
    assert "issue `#137`" in row1["rationale"]
    assert "locked TF-RD-020 noise-drift winner" in row1["rationale"]
    assert row1["model"] == {
        "stage": "qass_context",
        "stage_label": "delta_qass_no_column_v3",
        "module_overrides": {"column_encoder": "none"},
    }
    assert row1["data"] == {
        "surface_label": "tf_rd_020_shift_noise_drift",
        "source": "manifest",
        "corpus_ref": "tf_rd_020_shift_noise_drift_v1",
    }
    assert row1["training"]["surface_label"] == "linear_warmup_decay"
    assert row1["training"]["task_batch_size"] == 1
    assert row1["training"]["overrides"]["optimizer"] == {
        "name": "adamw",
        "require_requested": True,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "min_lr": 0.0004,
        "muon_per_parameter_lr": False,
    }
    assert row1["training"]["overrides"]["runtime"] == {
        "grad_accum_steps": 4,
        "max_steps": 400,
        "target_train_seconds": None,
        "eval_every": 25,
        "checkpoint_every": 25,
        "trace_activations": False,
        "val_batches": 0,
    }
    assert row1["training"]["overrides"]["schedule"] == {
        "stages": [
            {
                "name": "stage1",
                "steps": 2500,
                "lr_max": 0.004,
                "lr_schedule": "linear",
                "warmup_ratio": 0.05,
            }
        ]
    }
    assert "carry the locked `schedulefree_adamw` anchor into TF-RD-018 issue `#138`" in row1["next_action"]
    assert "parent_delta_ref" not in row1

    row2 = _row_by_ref(queue, "delta_training_muon")
    assert row2["data"] == row1["data"]
    assert row2["training"]["surface_label"] == "linear_warmup_decay"
    assert row2["training"]["task_batch_size"] == 1
    assert row2["training"]["overrides"]["runtime"] == row1["training"]["overrides"]["runtime"]
    assert row2["training"]["overrides"]["schedule"] == row1["training"]["overrides"]["schedule"]
    assert row2["training"]["overrides"]["optimizer"]["name"] == "muon"
    assert row2["training"]["overrides"]["optimizer"]["muon_per_parameter_lr"] is True
    assert "parent_delta_ref" not in row2


def test_tf_rd_018_optimizer_family_v1_materialized_queue_and_matrix_match_canonical_artifacts() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    rows = queue["rows"]
    assert [row["delta_id"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        _row_by_ref(_load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "queue.yaml"), delta_ref)["status"]
        for delta_ref in EXPECTED_ROWS
    ]

    row1 = next(row for row in rows if row["delta_id"] == "delta_training_adamw")
    assert row1["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert row1["training"]["task_batch_size"] == 1
    assert row1["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert row1["training"]["overrides"]["runtime"]["max_steps"] == 400
    assert row1["training"]["overrides"]["optimizer"]["name"] == "adamw"

    row2 = next(row for row in rows if row["delta_id"] == "delta_training_muon")
    assert row2["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert row2["training"]["task_batch_size"] == 1
    assert row2["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert row2["training"]["overrides"]["runtime"]["max_steps"] == 400
    assert row2["training"]["overrides"]["optimizer"]["name"] == "muon"
    if row1["status"] == "completed":
        assert row1["run_id"] is not None
        assert row1["benchmark_metrics"]["final_log_loss"] is not None
    if row2["status"] == "completed":
        assert row2["run_id"] is not None
        assert row2["benchmark_metrics"]["final_log_loss"] is not None

    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / SWEEP_ID
        / "matrix.md"
    ).read_text(encoding="utf-8")
    program = (REPO_ROOT / "program.md").read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_training_adamw" in matrix
    assert "delta_training_muon" in matrix
    assert "tf_rd_020_shift_noise_drift_v1" in matrix
    assert not (REPO_ROOT / "reference" / "system_delta_queue.yaml").exists()
    assert not (REPO_ROOT / "reference" / "system_delta_matrix.md").exists()
    assert "`reference/system_delta_sweeps/<sweep_id>/queue.yaml`" in program
    assert "There is no repo-global active sweep" in program
