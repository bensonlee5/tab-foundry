from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_018_lr_warmup_shape_v1"
PARENT_SWEEP_ID = "tf_rd_018_optimizer_family_v1"
ANCHOR_RUN_ID = "sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2"
EXPECTED_ROWS = [
    "delta_training_linear_warmup_decay",
    "delta_training_linear_warmup_decay_warm0",
    "delta_training_linear_warmup_decay_lr3e3",
    "delta_training_linear_warmup_decay_minlr1e4",
    "delta_training_linear_warmup_decay_warm20",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_018_lr_warmup_shape_v1_is_registered_and_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == SWEEP_ID

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[PARENT_SWEEP_ID]["status"] == "completed"
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": PARENT_SWEEP_ID,
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": ["nanotabpfn"],
    }


def test_tf_rd_018_lr_warmup_shape_v1_locks_runtime_and_reads_schedule_variants() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["parent_sweep_id"] == PARENT_SWEEP_ID
    assert sweep["status"] == "draft"
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
    assert any("issue `#138`" in note for note in notes)
    assert any("`#139`" in note for note in notes)
    assert any("schedulefree_adamw" in note for note in notes)
    assert any("tf_rd_020_noise_mixture_v1" in note for note in notes)
    assert any("corrects the inherited parent mismatch" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)

    for row in rows:
        assert row["decision"] is None
        assert row["run_id"] is None
        assert row["interpretation_status"] == "pending"
        assert row["model"] == {
            "stage": "qass_context",
            "stage_label": "delta_qass_no_column_v3",
            "module_overrides": {"column_encoder": "none"},
        }
        assert row["data"] == {
            "surface_label": "tf_rd_020_shift_noise_drift",
            "source": "manifest",
            "corpus_ref": "tf_rd_020_shift_noise_drift_v1",
        }
        assert row["preprocessing"] == {"surface_label": "runtime_default"}
        assert row["training"]["surface_label"] == "linear_warmup_decay"
        assert row["training"]["task_batch_size"] == 1
        assert row["training"]["overrides"]["optimizer"]["name"] == "schedulefree_adamw"
        assert row["training"]["overrides"]["runtime"] == {
            "grad_accum_steps": 4,
            "max_steps": 400,
            "target_train_seconds": None,
            "eval_every": 25,
            "checkpoint_every": 25,
            "trace_activations": False,
            "val_batches": 0,
        }
        assert (
            row["training"]["overrides"]["schedule"]["stages"][0]["steps"]
            == row["training"]["overrides"]["runtime"]["max_steps"]
        )

    baseline = _row_by_ref(queue, "delta_training_linear_warmup_decay")
    assert baseline["training"]["overrides"]["optimizer"]["min_lr"] == 0.0004
    assert baseline["training"]["overrides"]["schedule"] == {
        "stages": [
            {
                "name": "stage1",
                "steps": 400,
                "lr_max": 0.004,
                "lr_schedule": "linear",
                "warmup_ratio": 0.05,
            }
        ]
    }
    assert "corrected short-run LR/warmup baseline replay" in baseline["next_action"]

    no_warmup = _row_by_ref(queue, "delta_training_linear_warmup_decay_warm0")
    assert no_warmup["training"]["overrides"]["optimizer"]["min_lr"] == 0.0004
    assert no_warmup["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400
    assert no_warmup["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004
    assert no_warmup["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.0

    lower_ceiling = _row_by_ref(queue, "delta_training_linear_warmup_decay_lr3e3")
    assert lower_ceiling["training"]["overrides"]["optimizer"]["min_lr"] == 0.0004
    assert lower_ceiling["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400
    assert lower_ceiling["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.003
    assert lower_ceiling["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.05

    lower_floor = _row_by_ref(queue, "delta_training_linear_warmup_decay_minlr1e4")
    assert lower_floor["training"]["overrides"]["optimizer"]["min_lr"] == 0.0001
    assert lower_floor["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400
    assert lower_floor["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004
    assert lower_floor["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.05

    longer_warmup = _row_by_ref(queue, "delta_training_linear_warmup_decay_warm20")
    assert longer_warmup["training"]["overrides"]["optimizer"]["min_lr"] == 0.0004
    assert longer_warmup["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400
    assert longer_warmup["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004
    assert longer_warmup["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.20
    assert "materially longer-warmup probe" in longer_warmup["next_action"]


def test_tf_rd_018_lr_warmup_shape_v1_materialized_queue_matches_active_alias() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    rows = queue["rows"]
    assert [row["delta_id"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)

    baseline = next(row for row in rows if row["delta_id"] == "delta_training_linear_warmup_decay")
    assert baseline["data"]["corpus_ref"] == "tf_rd_020_shift_noise_drift_v1"
    assert baseline["training"]["task_batch_size"] == 1
    assert baseline["training"]["overrides"]["optimizer"]["name"] == "schedulefree_adamw"
    assert baseline["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert baseline["training"]["overrides"]["runtime"]["max_steps"] == 400
    assert baseline["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400
    assert baseline["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004
    assert baseline["training"]["overrides"]["schedule"]["stages"][0]["warmup_ratio"] == 0.05

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
    assert "delta_training_linear_warmup_decay_lr3e3" in matrix
    assert "delta_training_linear_warmup_decay_warm0" in matrix
    assert "delta_training_linear_warmup_decay_warm20" in matrix
    assert "tf_rd_020_shift_noise_drift_v1" in matrix
    assert "generated_from_sweep_id: tf_rd_018_lr_warmup_shape_v1" in active_queue
    assert "canonical_queue_path: reference/system_delta_sweeps/tf_rd_018_lr_warmup_shape_v1/queue.yaml" in active_queue
    assert "- active sweep id: `tf_rd_018_lr_warmup_shape_v1`" in program
    assert "- canonical sweep queue: `reference/system_delta_sweeps/tf_rd_018_lr_warmup_shape_v1/queue.yaml`" in program
