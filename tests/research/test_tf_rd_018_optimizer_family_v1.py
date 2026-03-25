from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_018_optimizer_family_v1"
ANCHOR_RUN_ID = "sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1"
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_020_shift_noise_drift",
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


def test_tf_rd_018_optimizer_family_v1_is_registered_and_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == SWEEP_ID

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["tf_rd_020_harder_dagzoo_ladder_v1"]["status"] == "completed"
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_020_harder_dagzoo_ladder_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": ["nanotabpfn"],
    }


def test_tf_rd_018_optimizer_family_v1_replays_noise_drift_before_optimizer_comparison() -> None:
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
        "data": "tf_rd_013_dagzoo_shape_aware_size_medium",
        "model": "delta_qass_no_column_v3",
        "preprocessing": "runtime_default",
        "training": "linear_warmup_decay",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("issue `#137`" in note for note in notes)
    assert any("tf_rd_020_shift_noise_drift_v1" in note for note in notes)
    assert any("not the original TF-RD-020 `400`-step executed row" in note for note in notes)
    assert any("tf_rd_020_noise_mixture_v1" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "ready", "ready"]
    assert not any("noise_mixture" in row["delta_ref"] for row in rows)

    row1 = _row_by_ref(queue, "delta_data_manifest_root_tf_rd_020_shift_noise_drift")
    assert "issue `#137`" in row1["rationale"]
    assert "harmonized `400`-step recipe" in row1["hypothesis"]
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
    assert row1["training"]["task_batch_size"] == 4
    assert row1["training"]["overrides"]["optimizer"] == {
        "name": "schedulefree_adamw",
        "require_requested": True,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "min_lr": 0.0004,
        "muon_per_parameter_lr": False,
    }
    assert row1["training"]["overrides"]["runtime"] == {
        "grad_accum_steps": 1,
        "max_steps": 2500,
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
    assert "--promote-first-executed-row-to-anchor" in row1["next_action"]

    row2 = _row_by_ref(queue, "delta_training_adamw")
    assert row2["parent_delta_ref"] == "delta_data_manifest_root_tf_rd_020_shift_noise_drift"
    assert row2["data"] == row1["data"]
    assert row2["training"]["surface_label"] == "linear_warmup_decay"
    assert row2["training"]["task_batch_size"] == 4
    assert row2["training"]["overrides"]["runtime"] == row1["training"]["overrides"]["runtime"]
    assert row2["training"]["overrides"]["schedule"] == row1["training"]["overrides"]["schedule"]
    assert row2["training"]["overrides"]["optimizer"]["name"] == "adamw"
    assert row2["training"]["overrides"]["optimizer"]["weight_decay"] == 0.0

    row3 = _row_by_ref(queue, "delta_training_muon")
    assert row3["parent_delta_ref"] == "delta_data_manifest_root_tf_rd_020_shift_noise_drift"
    assert row3["data"] == row1["data"]
    assert row3["training"]["surface_label"] == "linear_warmup_decay"
    assert row3["training"]["task_batch_size"] == 4
    assert row3["training"]["overrides"]["runtime"] == row1["training"]["overrides"]["runtime"]
    assert row3["training"]["overrides"]["schedule"] == row1["training"]["overrides"]["schedule"]
    assert row3["training"]["overrides"]["optimizer"]["name"] == "muon"
    assert row3["training"]["overrides"]["optimizer"]["muon_per_parameter_lr"] is True


def test_tf_rd_018_optimizer_family_v1_materialized_queue_and_matrix_match_active_alias() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    rows = queue["rows"]
    assert [row["delta_id"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "ready", "ready"]

    row1 = next(row for row in rows if row["delta_id"] == "delta_data_manifest_root_tf_rd_020_shift_noise_drift")
    assert row1["data"]["surface_label"] == "tf_rd_020_shift_noise_drift"
    assert row1["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert row1["training"]["task_batch_size"] == 4
    assert row1["training"]["overrides"]["optimizer"]["name"] == "schedulefree_adamw"

    row2 = next(row for row in rows if row["delta_id"] == "delta_training_adamw")
    assert row2["parent_delta_ref"] == "delta_data_manifest_root_tf_rd_020_shift_noise_drift"
    assert row2["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert row2["training"]["overrides"]["optimizer"]["name"] == "adamw"

    row3 = next(row for row in rows if row["delta_id"] == "delta_training_muon")
    assert row3["parent_delta_ref"] == "delta_data_manifest_root_tf_rd_020_shift_noise_drift"
    assert row3["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert row3["training"]["overrides"]["optimizer"]["name"] == "muon"

    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / SWEEP_ID
        / "matrix.md"
    ).read_text(encoding="utf-8")
    active_queue = (REPO_ROOT / "reference" / "system_delta_queue.yaml").read_text(encoding="utf-8")
    program = (REPO_ROOT / "program.md").read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_data_manifest_root_tf_rd_020_shift_noise_drift" in matrix
    assert "delta_training_adamw" in matrix
    assert "delta_training_muon" in matrix
    assert "tf_rd_020_shift_noise_drift_v1" in matrix
    assert "generated_from_sweep_id: tf_rd_018_optimizer_family_v1" in active_queue
    assert "canonical_queue_path: reference/system_delta_sweeps/tf_rd_018_optimizer_family_v1/queue.yaml" in active_queue
    assert "- active sweep id: `tf_rd_018_optimizer_family_v1`" in program
    assert "- canonical sweep queue: `reference/system_delta_sweeps/tf_rd_018_optimizer_family_v1/queue.yaml`" in program
