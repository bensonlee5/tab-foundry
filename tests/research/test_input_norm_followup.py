from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = [
    "dpnb_input_norm_anchor_replay",
    "dpnb_input_norm_zscore",
    "dpnb_input_norm_winsorize_zscore",
    "dpnb_input_norm_zscore_tanh",
    "dpnb_input_norm_robust_tanh",
    "dpnb_input_norm_anchor_replay_batch16_sqrt",
    "dpnb_input_norm_anchor_replay_batch64_sqrt",
    "dpnb_input_norm_zscore_tanh_batch16_sqrt",
    "dpnb_input_norm_zscore_tanh_batch64_sqrt",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_input_norm_followup_is_registered_and_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "input_norm_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["input_norm_followup"] == {
        "parent_sweep_id": "stability_followup",
        "status": "draft",
        "anchor_run_id": None,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_input_norm_followup_metadata_and_rows_match_the_bridge_baseline_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "input_norm_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "input_norm_followup"
    assert sweep["parent_sweep_id"] == "stability_followup"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] is None
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert sweep["anchor_context"]["model"]["module_selection"] == {
        "allow_test_self_attention": False,
        "column_encoder": "none",
        "context_encoder": "none",
        "feature_encoder": "nano",
        "head": "binary_direct",
        "post_encoder_norm": "none",
        "row_pool": "row_cls",
        "table_block_style": "prenorm",
        "target_conditioner": "mean_padded_linear",
        "tokenizer": "scalar_per_feature",
    }
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    assert any(
        "metadata-only bridge baseline" in note
        for note in sweep["anchor_surface"]["notes"]
    )

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)
    assert all(row["training"]["prior_dump_non_finite_policy"] == "skip" for row in rows)
    assert all(row["training"]["overrides"]["runtime"]["trace_activations"] is True for row in rows)

    baseline = _row_by_ref(queue, "dpnb_input_norm_anchor_replay")
    assert baseline["model"]["stage_label"] == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert baseline["model"]["input_normalization"] == "train_zscore_clip"
    assert baseline["training"]["prior_dump_batch_size"] == 32
    assert baseline["training"]["prior_dump_lr_scale_rule"] == "none"
    assert baseline["training"]["prior_dump_batch_reference_size"] == 32
    assert baseline["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert baseline["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "prior_dump",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.05,
        }
    ]

    zscore = _row_by_ref(queue, "dpnb_input_norm_zscore")
    assert zscore["model"]["input_normalization"] == "train_zscore"

    winsor = _row_by_ref(queue, "dpnb_input_norm_winsorize_zscore")
    assert winsor["model"]["input_normalization"] == "train_winsorize_zscore"

    zscore_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh")
    assert zscore_tanh["model"]["input_normalization"] == "train_zscore_tanh"

    robust_tanh = _row_by_ref(queue, "dpnb_input_norm_robust_tanh")
    assert robust_tanh["model"]["input_normalization"] == "train_robust_tanh"

    batch16 = _row_by_ref(queue, "dpnb_input_norm_anchor_replay_batch16_sqrt")
    assert batch16["training"]["prior_dump_batch_size"] == 16
    assert batch16["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert batch16["training"]["prior_dump_batch_reference_size"] == 32
    assert "0.7071" in batch16["notes"][0]

    batch64 = _row_by_ref(queue, "dpnb_input_norm_anchor_replay_batch64_sqrt")
    assert batch64["training"]["prior_dump_batch_size"] == 64
    assert batch64["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert batch64["training"]["prior_dump_batch_reference_size"] == 32
    assert "1.4142" in batch64["notes"][0]

    batch16_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh_batch16_sqrt")
    assert batch16_tanh["model"]["input_normalization"] == "train_zscore_tanh"
    assert batch16_tanh["training"]["prior_dump_batch_size"] == 16

    batch64_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh_batch64_sqrt")
    assert batch64_tanh["model"]["input_normalization"] == "train_zscore_tanh"
    assert batch64_tanh["training"]["prior_dump_batch_size"] == 64

    materialized = load_system_delta_queue(
        sweep_id="input_norm_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_input_norm_followup_matrix_records_the_bridge_norm_and_batch_queue() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "input_norm_followup"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# Input Norm Follow-Up Comparison Matrix" in matrix
    assert "dpnb_row_cls_cls2_linear_warmup_decay" in matrix
    assert "metadata-only bridge baseline" in matrix
    assert "train_zscore_tanh" in matrix
    assert "prior_dump_batch_size=16" in matrix
    assert "prior_dump_batch_size=64" in matrix
    assert "RMSNorm is intentionally excluded" in matrix
