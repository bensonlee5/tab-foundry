from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1"
EXPECTED_ROWS = [
    "delta_row_cls_pool",
    "delta_plain_context_encoder",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_row_embedding_attribution_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["row_embedding_attribution_v1"] == {
        "parent_sweep_id": "tokenization_migration_v1",
        "status": "superseded",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_row_embedding_attribution_metadata_and_rows_match_tf_rd_005_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "row_embedding_attribution_v1"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "row_embedding_attribution_v1"
    assert sweep["parent_sweep_id"] == "tokenization_migration_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "grouped_tokens"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("TF-RD-005" in note for note in notes)
    assert any("control surface" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed"]

    row_pool_row = _row_by_ref(queue, "delta_row_cls_pool")
    assert row_pool_row["model"]["module_overrides"] == {"row_pool": "row_cls"}
    assert row_pool_row["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert row_pool_row["run_id"] == "sd_row_embedding_attribution_v1_01_delta_row_cls_pool_v1"
    assert row_pool_row["benchmark_metrics"] is not None

    context_row = _row_by_ref(queue, "delta_plain_context_encoder")
    assert context_row["parent_delta_ref"] == "delta_row_cls_pool"
    assert context_row["model"]["module_overrides"] == {"context_encoder": "plain"}
    assert context_row["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert context_row["run_id"] == "sd_row_embedding_attribution_v1_02_delta_plain_context_encoder_v1"
    assert context_row["benchmark_metrics"] is not None

    materialized = load_system_delta_queue(
        sweep_id="row_embedding_attribution_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS
    materialized_context_row = next(
        row for row in materialized_rows if row["delta_id"] == "delta_plain_context_encoder"
    )
    assert materialized_context_row["parent_delta_ref"] == "delta_row_cls_pool"
    assert materialized_context_row["model"]["module_overrides"] == {"context_encoder": "plain"}
    assert materialized_context_row["anchor_delta"].startswith("Starting from the resolved")
    assert materialized_context_row["entangled_legacy_stage"] == "row_cls_pool"
    assert materialized_context_row["training"]["surface_label"] == "prior_linear_warmup_decay"


def test_row_embedding_attribution_matrix_records_the_three_surface_comparison() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "row_embedding_attribution_v1"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "row_embedding_attribution_v1" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "delta_row_cls_pool" in matrix
    assert "delta_plain_context_encoder" in matrix
    assert "prior_linear_warmup_decay" in matrix
    assert "row-first line open" in matrix
