from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1"
EXPECTED_ROWS = [
    "dpnb_cuda_large_anchor",
    "dpnb_cuda_large_width_x2",
    "dpnb_cuda_large_depth_plus4",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_cuda_capacity_pilot_is_registered_and_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_capacity_pilot"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["cuda_capacity_pilot"] == {
        "parent_sweep_id": "input_norm_followup",
        "status": "active",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_cuda_capacity_pilot_metadata_and_rows_match_the_capacity_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_capacity_pilot"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "cuda_capacity_pilot"
    assert sweep["parent_sweep_id"] == "input_norm_followup"
    assert sweep["status"] == "active"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "ready", "ready"]

    baseline = _row_by_ref(queue, "dpnb_cuda_large_anchor")
    assert baseline["model"]["stage_label"] == "dpnb_cuda_large_anchor"
    assert baseline["model"]["d_col"] == 128
    assert baseline["model"]["d_icl"] == 512
    assert baseline["model"]["input_normalization"] == "train_zscore_clip"
    assert baseline["model"]["tficl_n_heads"] == 8
    assert baseline["model"]["tficl_n_layers"] == 12
    assert baseline["model"]["head_hidden_dim"] == 1024
    assert baseline["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert baseline["training"]["prior_dump_batch_size"] == 64
    assert baseline["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert baseline["training"]["prior_dump_batch_reference_size"] == 32
    assert baseline["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert baseline["run_id"] is None
    assert baseline["interpretation_status"] == "pending"

    width = _row_by_ref(queue, "dpnb_cuda_large_width_x2")
    assert width["model"]["d_col"] == 256
    assert width["model"]["d_icl"] == 1024
    assert width["model"]["tficl_n_layers"] == 12
    assert width["model"]["head_hidden_dim"] == 2048
    assert width["training"]["overrides"]["runtime"]["max_steps"] == 2500

    depth = _row_by_ref(queue, "dpnb_cuda_large_depth_plus4")
    assert depth["model"]["d_col"] == 128
    assert depth["model"]["d_icl"] == 512
    assert depth["model"]["tficl_n_layers"] == 16
    assert depth["model"]["head_hidden_dim"] == 1024
    assert depth["training"]["overrides"]["runtime"]["max_steps"] == 2500

    materialized = load_system_delta_queue(
        sweep_id="cuda_capacity_pilot",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_cuda_capacity_pilot_matrix_records_the_three_row_capacity_probe() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_capacity_pilot" / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "cuda_capacity_pilot" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "dpnb_cuda_large_anchor" in matrix
    assert "dpnb_cuda_large_width_x2" in matrix
    assert "dpnb_cuda_large_depth_plus4" in matrix
    assert "2500" in matrix
    assert "batch64" in matrix
    assert "budget follow-up" in matrix
