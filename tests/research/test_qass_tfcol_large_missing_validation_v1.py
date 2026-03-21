from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_qass_tfcol_large_no_missing_validation_v1_01_delta_qass_no_column_v3_v1"
EXPECTED_ROWS = [
    "delta_qass_no_column_v3",
    "delta_qass_context_tfcol_heads4_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def _assert_full_replay_training_payload(row: dict[str, Any]) -> None:
    training = row["training"]
    assert training["surface_label"] == "prior_linear_warmup_decay"
    assert training["prior_dump_non_finite_policy"] == "skip"

    overrides = training["overrides"]
    assert overrides["apply_schedule"] is True
    assert overrides["runtime"] == {
        "max_steps": 2500,
        "eval_every": 25,
        "checkpoint_every": 25,
        "trace_activations": False,
    }
    assert overrides["optimizer"] == {
        "name": "schedulefree_adamw",
        "require_requested": True,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "min_lr": 0.0004,
        "muon_per_parameter_lr": False,
    }
    assert overrides["schedule"] == {
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


def _assert_row_execution_state(
    row: dict[str, Any],
    *,
    allowed_statuses: tuple[str, ...],
) -> None:
    assert row["status"] in allowed_statuses
    if row["status"] == "ready":
        assert row["decision"] is None
        assert row["interpretation_status"] == "pending"
        assert row["run_id"] is None
        assert row.get("benchmark_metrics") is None
        return
    assert row["status"] == "completed"
    assert row["decision"] is not None
    assert row["interpretation_status"] == "completed"
    assert row["run_id"] is not None
    assert row["benchmark_metrics"] is not None


def test_qass_tfcol_large_missing_validation_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["qass_tfcol_large_missing_validation_v1"] == {
        "parent_sweep_id": "qass_tfcol_large_no_missing_validation_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_qass_tfcol_large_missing_validation_v1_metadata_and_rows_match_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "qass_tfcol_large_missing_validation_v1"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "qass_tfcol_large_missing_validation_v1"
    assert sweep["parent_sweep_id"] == "qass_tfcol_large_no_missing_validation_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "qass_context"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("TF-RD-008 missing-data generalization sweep" in note for note in notes)
    assert any("reuse signature" in note for note in notes)
    assert any("missingness_followup" in note for note in notes)
    assert any("mixed result" in note for note in notes)
    assert any("default row-first classification anchor" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS

    row1 = _row_by_ref(queue, "delta_qass_no_column_v3")
    assert row1.get("parent_delta_ref") is None
    _assert_row_execution_state(row1, allowed_statuses=("completed",))
    assert row1["decision"] == "keep"
    assert "default row-first classification anchor" in row1["next_action"]
    assert any("simpler" in note for note in row1["notes"])
    assert row1["model"] == {
        "stage": "qass_context",
        "stage_label": "delta_qass_no_column_v3",
        "module_overrides": {"column_encoder": "none"},
    }
    _assert_full_replay_training_payload(row1)

    row2 = _row_by_ref(queue, "delta_qass_context_tfcol_heads4_v1")
    assert row2["parent_delta_ref"] == "delta_qass_no_column_v3"
    _assert_row_execution_state(row2, allowed_statuses=("completed",))
    assert row2["decision"] == "defer"
    assert "calibration-oriented alternative" in row2["next_action"]
    assert any("slightly worse" in note for note in row2["notes"])
    assert row2["model"] == {
        "stage": "qass_context",
        "stage_label": "delta_qass_context_tfcol_heads4_v1",
        "tfcol_n_heads": 4,
    }
    _assert_full_replay_training_payload(row2)

    materialized = load_system_delta_queue(
        sweep_id="qass_tfcol_large_missing_validation_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS

    materialized_row1 = next(row for row in materialized_rows if row["delta_id"] == "delta_qass_no_column_v3")
    assert materialized_row1["model"]["stage"] == "qass_context"
    assert materialized_row1["model"]["module_overrides"] == {"column_encoder": "none"}
    assert materialized_row1["training"]["overrides"]["runtime"]["max_steps"] == 2500

    materialized_row2 = next(
        row for row in materialized_rows if row["delta_id"] == "delta_qass_context_tfcol_heads4_v1"
    )
    assert materialized_row2["parent_delta_ref"] == "delta_qass_no_column_v3"
    assert materialized_row2["model"]["stage"] == "qass_context"
    assert materialized_row2["model"]["tfcol_n_heads"] == 4
    assert materialized_row2["training"]["overrides"]["runtime"]["max_steps"] == 2500


def test_qass_tfcol_large_missing_validation_v1_matrix_records_the_tf_rd_008_closure() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "qass_tfcol_large_missing_validation_v1"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "qass_tfcol_large_missing_validation_v1" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_qass_no_column_v3" in matrix
    assert "delta_qass_context_tfcol_heads4_v1" in matrix
    assert "prior_linear_warmup_decay" in matrix
    assert "nanotabpfn_openml_binary_large_v1.json" in matrix
    assert "TF-RD-008" in matrix
    assert "completed" in matrix
    assert "no-TFCol row becomes the default" in matrix
