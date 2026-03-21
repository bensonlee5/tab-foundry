from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1"
EXPECTED_ROWS = [
    "delta_row_embeddings_no_context_v2",
    "delta_row_embeddings_plain_context_v2",
    "delta_column_set_plain_context_v2",
    "delta_qass_context_v2",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def _load_registry() -> dict[str, Any]:
    payload = json.loads(
        (REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(payload, dict)
    return payload


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


def test_row_embedding_attribution_v2_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["row_embedding_attribution_v2"] == {
        "parent_sweep_id": "tokenization_migration_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_row_embedding_attribution_v2_metadata_and_rows_match_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "row_embedding_attribution_v2"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "row_embedding_attribution_v2"
    assert sweep["parent_sweep_id"] == "tokenization_migration_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "grouped_tokens"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("2500-step" in note for note in notes)
    assert any("reuse" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed", "completed", "completed"]

    registry = _load_registry()
    registry_runs = registry["runs"]
    assert isinstance(registry_runs, dict)
    for order, row in enumerate(rows, start=1):
        assert row["interpretation_status"] == "completed"
        assert row["decision"] == "defer"
        run_id = row["run_id"]
        assert isinstance(run_id, str)
        assert re.fullmatch(
            rf"sd_row_embedding_attribution_v2_{order:02d}_{re.escape(str(row['delta_ref']))}_v\d+",
            run_id,
        )
        benchmark_metrics = row["benchmark_metrics"]
        assert isinstance(benchmark_metrics, dict)
        registry_run = registry_runs[run_id]
        assert registry_run["sweep"]["queue_order"] == order
        assert registry_run["sweep"]["delta_id"] == row["delta_ref"]
        assert registry_run["sweep"]["run_kind"] == "primary"

    row1 = _row_by_ref(queue, "delta_row_embeddings_no_context_v2")
    assert row1["model"] == {
        "stage": "row_cls_pool",
        "stage_label": "delta_row_embeddings_no_context_v2",
        "module_overrides": {"context_encoder": "none"},
    }
    _assert_full_replay_training_payload(row1)

    row2 = _row_by_ref(queue, "delta_row_embeddings_plain_context_v2")
    assert row2["parent_delta_ref"] == "delta_row_embeddings_no_context_v2"
    assert row2["model"] == {
        "stage": "row_cls_pool",
        "stage_label": "delta_row_embeddings_plain_context_v2",
    }
    _assert_full_replay_training_payload(row2)

    row3 = _row_by_ref(queue, "delta_column_set_plain_context_v2")
    assert row3["parent_delta_ref"] == "delta_row_embeddings_plain_context_v2"
    assert row3["model"] == {
        "stage": "column_set",
        "stage_label": "delta_column_set_plain_context_v2",
    }
    _assert_full_replay_training_payload(row3)

    row4 = _row_by_ref(queue, "delta_qass_context_v2")
    assert row4["parent_delta_ref"] == "delta_column_set_plain_context_v2"
    assert row4["model"] == {
        "stage": "qass_context",
        "stage_label": "delta_qass_context_v2",
    }
    _assert_full_replay_training_payload(row4)

    materialized = load_system_delta_queue(
        sweep_id="row_embedding_attribution_v2",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS

    materialized_row2 = next(
        row for row in materialized_rows if row["delta_id"] == "delta_row_embeddings_plain_context_v2"
    )
    assert materialized_row2["parent_delta_ref"] == "delta_row_embeddings_no_context_v2"
    assert materialized_row2["training"]["overrides"]["runtime"]["max_steps"] == 2500

    materialized_row4 = next(
        row for row in materialized_rows if row["delta_id"] == "delta_qass_context_v2"
    )
    assert materialized_row4["parent_delta_ref"] == "delta_column_set_plain_context_v2"
    assert materialized_row4["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert materialized_row4["model"]["stage"] == "qass_context"


def test_row_embedding_attribution_v2_matrix_records_the_full_stage_native_chain() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "row_embedding_attribution_v2"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "row_embedding_attribution_v2" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_row_embeddings_no_context_v2" in matrix
    assert "delta_row_embeddings_plain_context_v2" in matrix
    assert "delta_column_set_plain_context_v2" in matrix
    assert "delta_qass_context_v2" in matrix
    assert "prior_linear_warmup_decay" in matrix
    assert "2500" in matrix
