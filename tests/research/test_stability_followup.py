from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_stability_followup_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "binary_md_v3"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["stability_followup"] == {
        "parent_sweep_id": "stability_ladder",
        "status": "draft",
        "anchor_run_id": None,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_stability_followup_metadata_and_rows_match_the_retargeted_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "stability_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "stability_followup"
    assert sweep["parent_sweep_id"] == "stability_ladder"
    assert sweep["status"] == "draft"
    assert sweep["anchor_context"]["model"]["arch"] == "tabfoundry_staged"
    assert sweep["anchor_context"]["model"]["stage"] == "prenorm_block"
    assert sweep["anchor_context"]["model"]["module_selection"] == {
        "allow_test_self_attention": False,
        "column_encoder": "none",
        "context_encoder": "none",
        "feature_encoder": "shared",
        "head": "binary_direct",
        "post_encoder_norm": "layernorm",
        "row_pool": "target_column",
        "table_block_style": "prenorm",
        "target_conditioner": "mean_padded_linear",
        "tokenizer": "scalar_per_feature",
    }
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_cosine_warmup"

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == [
        "horizon_10000_baseline",
        "schedule_linear_decay_prenorm",
        "schedule_linear_warmup_decay_prenorm",
        "rowpool_row_cls_cls2",
    ]
    assert [row["status"] for row in rows] == ["ready", "ready", "ready", "ready"]

    horizon_row = _row_by_ref(queue, "horizon_10000_baseline")
    assert horizon_row["training"]["surface_label"] == "prior_cosine_warmup"
    assert horizon_row["training"]["overrides"]["apply_schedule"] is True
    assert horizon_row["training"]["overrides"]["runtime"] == {"grad_clip": 1.0, "max_steps": 10000}
    assert horizon_row["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "prior_dump",
            "steps": 10000,
            "lr_max": 0.004,
            "lr_schedule": "cosine",
            "warmup_ratio": 0.05,
        }
    ]

    linear_row = _row_by_ref(queue, "schedule_linear_decay_prenorm")
    assert linear_row["training"]["surface_label"] == "prior_linear_decay"
    assert linear_row["training"]["overrides"] == {
        "apply_schedule": True,
        "optimizer": {"min_lr": 0.0004},
        "runtime": {"grad_clip": 1.0, "max_steps": 2500},
        "schedule": {
            "stages": [
                {
                    "name": "prior_dump",
                    "steps": 2500,
                    "lr_max": 0.004,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.0,
                }
            ]
        },
    }

    linear_warmup_row = _row_by_ref(queue, "schedule_linear_warmup_decay_prenorm")
    assert linear_warmup_row["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert linear_warmup_row["training"]["overrides"] == {
        "apply_schedule": True,
        "optimizer": {"min_lr": 0.0004},
        "runtime": {"grad_clip": 1.0, "max_steps": 2500},
        "schedule": {
            "stages": [
                {
                    "name": "prior_dump",
                    "steps": 2500,
                    "lr_max": 0.004,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.05,
                }
            ]
        },
    }

    rowpool_row = _row_by_ref(queue, "rowpool_row_cls_cls2")
    assert rowpool_row["training"]["surface_label"] == "prior_cosine_warmup"
    assert rowpool_row["model"]["tfrow_n_heads"] == 8
    assert rowpool_row["model"]["tfrow_n_layers"] == 3
    assert rowpool_row["model"]["tfrow_cls_tokens"] == 2
    assert rowpool_row["model"]["tfrow_norm"] == "layernorm"
    assert rowpool_row["model"]["module_overrides"] == {
        "feature_encoder": "shared",
        "post_encoder_norm": "layernorm",
        "row_pool": "row_cls",
        "table_block_style": "prenorm",
    }


def test_stability_followup_matrix_records_gradient_ratio_schedule_rows_and_rowpool() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "stability_followup"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# Stability Follow-Up Comparison Matrix" in matrix
    assert "encoder/head gradient ratio" in matrix
    assert "steps `1-25`" in matrix
    assert "prior_linear_decay" in matrix
    assert "prior_linear_warmup_decay" in matrix
    assert "rowpool_row_cls_cls2" in matrix
    assert "--handoff-root" not in matrix
    assert "pre_encoder_clip=10.0" not in matrix
