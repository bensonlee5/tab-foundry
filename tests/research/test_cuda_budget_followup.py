from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
EXPECTED_ROWS = [
    "dpnb_cuda_budget_5k",
    "dpnb_cuda_budget_10k",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_cuda_budget_followup_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["cuda_budget_followup"] == {
        "parent_sweep_id": "cuda_capacity_pilot",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_cuda_budget_followup_metadata_and_rows_match_the_blocked_budget_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_budget_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "cuda_budget_followup"
    assert sweep["parent_sweep_id"] == "cuda_capacity_pilot"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        "blocked_on_anchor_selection",
        "blocked_on_anchor_selection",
    ]
    assert [row["interpretation_status"] for row in rows] == ["blocked", "blocked"]

    budget_5k = _row_by_ref(queue, "dpnb_cuda_budget_5k")
    assert budget_5k["training"]["prior_dump_batch_size"] == 64
    assert budget_5k["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert budget_5k["training"]["overrides"]["runtime"]["max_steps"] == 5000
    assert budget_5k["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 5000
    assert budget_5k["run_id"] is None

    budget_10k = _row_by_ref(queue, "dpnb_cuda_budget_10k")
    assert budget_10k["training"]["prior_dump_batch_size"] == 64
    assert budget_10k["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert budget_10k["training"]["overrides"]["runtime"]["max_steps"] == 10000
    assert budget_10k["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 10000
    assert budget_10k["run_id"] is None

    materialized = load_system_delta_queue(
        sweep_id="cuda_budget_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_cuda_budget_followup_matrix_records_the_blocked_budget_rows() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_budget_followup" / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "cuda_budget_followup" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "blocked_on_anchor_selection" in matrix
    assert "dpnb_cuda_budget_5k" in matrix
    assert "dpnb_cuda_budget_10k" in matrix
    assert "5000" in matrix
    assert "10000" in matrix
    assert "batch64" in matrix
