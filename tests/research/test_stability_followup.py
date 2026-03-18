from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = [
    "dpnb_baseline_cosine_warmup_2500",
    "dpnb_linear_decay_lr4e3",
    "dpnb_linear_warmup_decay_lr4e3_warm5",
    "dpnb_linear_warmup_decay_lr3e3_warm5",
    "dpnb_linear_warmup_decay_lr5e3_warm5",
    "dpnb_linear_warmup_decay_lr4e3_warm2",
    "dpnb_linear_warmup_decay_lr4e3_warm10",
    "dpnb_linear_warmup_decay_lr4e3_warm5_clip05",
    "dpnb_linear_warmup_decay_lr4e3_warm5_wd5e4",
    "dpnb_linear_warmup_decay_lr4e3_warm5_adamw",
    "dpnb_row_cls_cls2_linear_warmup_decay",
    "dpnb_row_cls_cls2_linear_warmup_decay_warm10",
]


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

    assert index["active_sweep_id"] == "input_norm_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["stability_followup"] == {
        "parent_sweep_id": "stability_ladder",
        "status": "completed",
        "anchor_run_id": None,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_stability_followup_metadata_and_rows_match_the_delta_prenorm_bridge_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "stability_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "stability_followup"
    assert sweep["parent_sweep_id"] == "stability_ladder"
    assert sweep["status"] == "completed"
    assert sweep["anchor_context"]["model"]["arch"] == "tabfoundry_staged"
    assert sweep["anchor_context"]["model"]["stage"] == "nano_exact"
    assert sweep["anchor_context"]["model"]["stage_label"] == "delta_prenorm_block"
    assert sweep["anchor_context"]["model"]["module_selection"] == {
        "allow_test_self_attention": False,
        "column_encoder": "none",
        "context_encoder": "none",
        "feature_encoder": "nano",
        "head": "binary_direct",
        "post_encoder_norm": "none",
        "row_pool": "target_column",
        "table_block_style": "prenorm",
        "target_conditioner": "mean_padded_linear",
        "tokenizer": "scalar_per_feature",
    }
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_cosine_warmup"
    assert sweep["anchor_run_id"] is None
    assert any(
        "row 11 is the promoted bridge-surface default and row 12 is confirmatory evidence only."
        in note
        for note in sweep["anchor_surface"]["notes"]
    )
    assert any(
        "Promotion is metadata-only on this machine because the winning artifacts were produced elsewhere;"
        in note
        for note in sweep["anchor_surface"]["notes"]
    )

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)
    assert all(row["training"]["prior_dump_non_finite_policy"] == "skip" for row in rows)
    assert all(row["training"]["overrides"]["runtime"]["trace_activations"] is True for row in rows)

    baseline = _row_by_ref(queue, "dpnb_baseline_cosine_warmup_2500")
    assert baseline["model"] == {
        "stage": "nano_exact",
        "stage_label": "dpnb_baseline_cosine_warmup_2500",
        "module_overrides": {
            "table_block_style": "prenorm",
            "allow_test_self_attention": False,
        },
    }
    assert baseline["training"]["surface_label"] == "prior_cosine_warmup"
    assert baseline["training"]["overrides"]["runtime"] == {
        "grad_clip": 1.0,
        "max_steps": 2500,
        "trace_activations": True,
    }

    canonical = _row_by_ref(queue, "dpnb_linear_warmup_decay_lr4e3_warm5")
    assert canonical["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert canonical["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert canonical["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "prior_dump",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.05,
        }
    ]

    low_lr = _row_by_ref(queue, "dpnb_linear_warmup_decay_lr3e3_warm5")
    assert low_lr["training"]["overrides"]["optimizer"] == {"min_lr": 0.0003}
    assert low_lr["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.003

    clip_row = _row_by_ref(queue, "dpnb_linear_warmup_decay_lr4e3_warm5_clip05")
    assert clip_row["training"]["overrides"]["runtime"]["grad_clip"] == 0.5

    wd_row = _row_by_ref(queue, "dpnb_linear_warmup_decay_lr4e3_warm5_wd5e4")
    assert wd_row["training"]["overrides"]["optimizer"] == {
        "min_lr": 0.0004,
        "weight_decay": 0.0005,
    }

    adamw_row = _row_by_ref(queue, "dpnb_linear_warmup_decay_lr4e3_warm5_adamw")
    assert adamw_row["training"]["overrides"]["optimizer"] == {
        "name": "adamw",
        "min_lr": 0.0004,
        "weight_decay": 0.0,
    }

    rowpool_row = _row_by_ref(queue, "dpnb_row_cls_cls2_linear_warmup_decay")
    assert rowpool_row["model"]["tfrow_n_heads"] == 8
    assert rowpool_row["model"]["tfrow_n_layers"] == 3
    assert rowpool_row["model"]["tfrow_cls_tokens"] == 2
    assert rowpool_row["model"]["tfrow_norm"] == "layernorm"
    assert rowpool_row["decision"] == "promote"
    assert rowpool_row["interpretation_status"] == "completed"
    assert (
        rowpool_row["next_action"]
        == "Defer canonical anchor registration until the remote artifacts are synced locally; until then, treat this as the metadata-only bridge-surface default."
    )
    assert rowpool_row["notes"] == [
        "Matrix outcome: best/final ROC AUC 0.7650 with zero drift; row-cls cls2 is the promoted bridge-surface default.",
        "Remote artifacts were produced on a different machine, so local benchmark registry promotion is intentionally deferred.",
    ]
    assert rowpool_row["model"]["module_overrides"] == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }

    rowpool_warm10 = _row_by_ref(queue, "dpnb_row_cls_cls2_linear_warmup_decay_warm10")
    assert rowpool_warm10["model"]["tfrow_n_heads"] == 8
    assert rowpool_warm10["model"]["tfrow_n_layers"] == 3
    assert rowpool_warm10["model"]["tfrow_cls_tokens"] == 2
    assert rowpool_warm10["model"]["tfrow_norm"] == "layernorm"
    assert rowpool_warm10["decision"] == "accept_as_evidence"
    assert rowpool_warm10["interpretation_status"] == "completed"
    assert (
        rowpool_warm10["next_action"]
        == "Keep this row as confirmatory evidence only; do not spend more bridge-surface budget on warmup-only follow-up until row 11 is replicated or synced locally."
    )
    assert rowpool_warm10["notes"] == [
        "Matrix outcome: confirms that row-cls is reopened on the bridge surface, but does not materially improve over row 11 and reintroduces small drift plus a higher clipped-step fraction.",
        "Remote artifacts were produced on a different machine, so this row remains metadata-only on this machine.",
    ]
    assert rowpool_warm10["model"]["module_overrides"] == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert rowpool_warm10["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "prior_dump",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.10,
        }
    ]

    materialized = load_system_delta_queue(
        sweep_id="stability_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_stability_followup_matrix_records_the_expanded_bridge_queue() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "stability_followup"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# Stability Follow-Up Comparison Matrix" in matrix
    assert "delta_prenorm_block" in matrix
    assert "early_1_25" in matrix
    assert "post_warmup_100" in matrix
    assert "final_10pct" in matrix
    assert "dpnb_linear_warmup_decay_lr4e3_warm5_adamw" in matrix
    assert "dpnb_linear_warmup_decay_lr4e3_warm5_wd5e4" in matrix
    assert "dpnb_row_cls_cls2_linear_warmup_decay" in matrix
    assert "dpnb_row_cls_cls2_linear_warmup_decay_warm10" in matrix
    assert "Promoted `dpnb_row_cls_cls2_linear_warmup_decay`" in matrix
    assert "This promotion is formalized as sweep metadata only on this machine" in matrix
    assert "Dagzoo" not in matrix
    assert "10000-step horizon" not in matrix
