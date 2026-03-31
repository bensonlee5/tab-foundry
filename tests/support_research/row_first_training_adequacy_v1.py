from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "row_first_training_adequacy_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_013_dagzoo_size_ladder_v1_03_"
    "delta_data_manifest_root_dagzoo_shape_aware_size_medium_v1"
)
EXPECTED_ROWS = [
    "delta_training_task_batch4",
    "delta_training_task_batch8",
    "delta_training_task_batch16",
    "delta_training_task_batch32",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_row_first_training_adequacy_v1_is_registered_on_the_tf_rd_013_medium_anchor() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    entry = sweeps[SWEEP_ID]
    assert entry["parent_sweep_id"] == "tf_rd_013_dagzoo_size_ladder_v1"
    assert entry["status"] == "completed"
    assert entry["anchor_run_id"] == ANCHOR_RUN_ID
    assert entry["complexity_level"] == "binary_md"
    assert entry["benchmark_manifest_path"] == (
        "src/tab_foundry/bench/openml_binary_medium_v1.json"
    )
    assert entry["control_baseline_id"] == "cls_benchmark_linear_v2"
    assert entry["external_benchmarks"] is None


def test_row_first_training_adequacy_v1_rebases_the_queue_on_task_batch_rungs() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    catalog = _load_yaml(REPO_ROOT / "reference" / "system_delta_catalog.yaml")

    assert sweep["parent_sweep_id"] == "tf_rd_013_dagzoo_size_ladder_v1"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged"
    assert sweep["benchmark_manifest_path"] == "src/tab_foundry/bench/openml_binary_medium_v1.json"
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_013_dagzoo_shape_aware_size_medium",
        "model": "delta_qass_no_column_v3",
        "preprocessing": "runtime_default",
        "training": "prior_linear_warmup_decay",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("dataset batching" in note for note in notes)
    assert any("task_batch_size=4" in note for note in notes)
    assert any("task_batch_size=8" in note for note in notes)
    assert any("<=900s" in note for note in notes)
    assert not any("schedule and warmup shape" in note for note in notes)
    assert not any("batch-size or accumulation policy" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed", "blocked", "blocked"]
    assert [row["interpretation_status"] for row in rows] == ["completed", "completed", "blocked", "blocked"]
    assert [row["decision"] for row in rows] == ["keep", "defer", None, None]

    row1 = _row_by_ref(queue, "delta_training_task_batch4")
    assert "#137" in row1["next_action"]
    assert "#138" in row1["next_action"]
    assert "#139" in row1["next_action"]
    assert any("699.4s" in note for note in row1["notes"])

    row2 = _row_by_ref(queue, "delta_training_task_batch8")
    assert any("Reused the saved nanoTabPFN curve" in note for note in row2["notes"])
    assert any("1109.3s" in note for note in row2["notes"])

    row3 = _row_by_ref(queue, "delta_training_task_batch16")
    assert "#137" in row3["next_action"]
    assert any("1109.3s" in note for note in row3["notes"])

    for delta_ref, task_batch_size in (
        ("delta_training_task_batch4", 4),
        ("delta_training_task_batch8", 8),
        ("delta_training_task_batch16", 16),
        ("delta_training_task_batch32", 32),
    ):
        row = _row_by_ref(queue, delta_ref)
        assert row["data"] == {
            "surface_label": "tf_rd_013_dagzoo_shape_aware_size_medium",
            "source": "manifest",
            "corpus_ref": "tf_rd_013_dagzoo_shape_aware_size_medium_v1",
        }
        assert row["training"]["surface_label"] == "linear_warmup_decay"
        assert row["training"]["task_batch_size"] == task_batch_size
        assert row["training"]["overrides"]["runtime"] == {
            "grad_accum_steps": 1,
            "max_steps": 2500,
            "target_train_seconds": None,
            "eval_every": 25,
            "checkpoint_every": 25,
            "trace_activations": False,
            "val_batches": 0,
        }
        assert row["training"]["overrides"]["optimizer"] == {
            "name": "schedulefree_adamw",
            "require_requested": True,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "min_lr": 0.0004,
            "muon_per_parameter_lr": False,
        }
        assert row["training"]["overrides"]["schedule"] == {
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

    assert _row_by_ref(queue, "delta_training_task_batch16")["parent_delta_ref"] == "delta_training_task_batch8"
    assert _row_by_ref(queue, "delta_training_task_batch32")["parent_delta_ref"] == "delta_training_task_batch16"

    old_delta_refs = {
        "delta_training_linear_decay",
        "delta_training_adamw",
        "delta_training_muon",
        "delta_training_batch64_sqrt",
        "delta_training_clip05",
        "delta_training_budget_5k",
    }
    assert old_delta_refs.isdisjoint({row["delta_ref"] for row in rows})

    for delta_ref, task_batch_size in (
        ("delta_training_task_batch4", 4),
        ("delta_training_task_batch8", 8),
        ("delta_training_task_batch16", 16),
        ("delta_training_task_batch32", 32),
    ):
        entry = catalog["deltas"][delta_ref]
        assert entry["family"] == "batch_size"
        assert entry["default_effective_surface"]["data"] == {
            "surface_label": "tf_rd_013_dagzoo_shape_aware_size_medium",
            "surface_overrides": {
                "source": "manifest",
                "corpus_ref": "tf_rd_013_dagzoo_shape_aware_size_medium_v1",
            },
        }
        assert entry["default_effective_surface"]["training"]["task_batch_size"] == task_batch_size


def test_row_first_training_adequacy_v1_materialized_queue_preserves_task_batch_ladder() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    rows = queue["rows"]
    assert [row["delta_id"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed", "blocked", "blocked"]
    assert [row["decision"] for row in rows] == ["keep", "defer", None, None]

    for delta_id, task_batch_size in (
        ("delta_training_task_batch4", 4),
        ("delta_training_task_batch8", 8),
        ("delta_training_task_batch16", 16),
        ("delta_training_task_batch32", 32),
    ):
        row = next(row for row in rows if row["delta_id"] == delta_id)
        assert row["data"]["surface_label"] == "tf_rd_013_dagzoo_shape_aware_size_medium"
        assert row["data"]["corpus_ref"] == "tf_rd_013_dagzoo_shape_aware_size_medium_v1"
        assert row["training"]["task_batch_size"] == task_batch_size
        assert row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 1


def test_row_first_training_adequacy_v1_matrix_records_the_task_batch_rebase() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / SWEEP_ID
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_training_task_batch4" in matrix
    assert "delta_training_task_batch8" in matrix
    assert "delta_training_task_batch16" in matrix
    assert "delta_training_task_batch32" in matrix
    assert "sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1" in matrix
    assert "sd_row_first_training_adequacy_v1_02_delta_training_task_batch8_v1" in matrix
    assert "#137" in matrix
    assert "1109.3s" in matrix
    assert "tf_rd_013_dagzoo_shape_aware_size_medium_v1" in matrix
    assert "openml_binary_medium_v1.json" in matrix
    assert "delta_training_linear_decay" not in matrix
    assert "delta_training_adamw" not in matrix
    assert "delta_training_muon" not in matrix
    assert "delta_training_clip05" not in matrix
    assert "delta_training_budget_5k" not in matrix
