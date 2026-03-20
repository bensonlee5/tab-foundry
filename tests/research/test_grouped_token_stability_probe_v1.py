from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue
from tab_foundry.research.system_delta_execute import _compose_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "grouped_token_stability_probe_v1"
ANCHOR_RUN_ID = "sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2"
EXPECTED_ROWS = [
    "delta_anchor_activation_trace_baseline",
    "delta_training_linear_decay",
    "delta_training_linear_warmup_decay",
]
EXPECTED_MODULE_OVERRIDES = {
    "allow_test_self_attention": True,
    "column_encoder": "none",
    "context_encoder": "none",
    "feature_encoder": "shared",
    "head": "small_class",
    "post_encoder_norm": "none",
    "post_stack_norm": "none",
    "row_pool": "target_column",
    "table_block_residual_scale": "none",
    "table_block_style": "prenorm",
    "target_conditioner": "label_token",
    "tokenizer": "shifted_grouped",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_grouped_token_stability_probe_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tokenization_migration_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_grouped_token_stability_probe_v1_metadata_and_rows_match_the_grouped_token_probe_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tokenization_migration_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_staged_prior"
    assert sweep["training_config_profile"] == "cls_benchmark_staged_prior"
    assert sweep["surface_role"] == "hybrid_diagnostic"
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged"
    assert sweep["anchor_context"]["model"]["stage"] == "grouped_tokens"
    assert sweep["anchor_context"]["model"]["stage_label"] == "delta_architecture_screen_grouped_tokens"
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "anchor_manifest_default",
        "model": "delta_architecture_screen_grouped_tokens",
        "preprocessing": "runtime_default",
        "training": "training_default",
    }
    assert any("bounded hybrid-diagnostic probe" in note for note in sweep["anchor_surface"]["notes"])
    assert any("TF-RD-005, TF-RD-006, and TF-RD-007 remain blocked" in note for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed", "completed"]
    assert [row["interpretation_status"] for row in rows] == ["completed", "completed", "completed"]

    trace_row = _row_by_ref(queue, "delta_anchor_activation_trace_baseline")
    assert trace_row["model"] == {
        "stage": "grouped_tokens",
        "stage_label": "delta_architecture_screen_grouped_tokens",
        "module_overrides": EXPECTED_MODULE_OVERRIDES,
    }
    assert trace_row["training"] == {
        "surface_label": "prior_constant_lr_trace_activations",
        "overrides": {
            "runtime": {
                "trace_activations": True,
            }
        },
    }
    assert "grouped-token anchor" in trace_row["anchor_delta"]
    assert "telemetry anchor" in trace_row["next_action"]
    assert any("recovered the mixed architecture-screen result" in note for note in trace_row["notes"])

    decay_row = _row_by_ref(queue, "delta_training_linear_decay")
    assert decay_row["model"] == {
        "stage": "grouped_tokens",
        "stage_label": "delta_architecture_screen_grouped_tokens",
        "module_overrides": EXPECTED_MODULE_OVERRIDES,
    }
    assert decay_row["training"]["surface_label"] == "prior_linear_decay"
    assert decay_row["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert decay_row["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "stage1",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.0,
        }
    ]
    assert "grouped-token model" in decay_row["anchor_delta"]
    assert "ROC-oriented tradeoff reference" in decay_row["next_action"]
    assert any("not the preferred grouped-token training surface" in note for note in decay_row["notes"])

    warmup_row = _row_by_ref(queue, "delta_training_linear_warmup_decay")
    assert warmup_row["model"] == {
        "stage": "grouped_tokens",
        "stage_label": "delta_architecture_screen_grouped_tokens",
        "module_overrides": EXPECTED_MODULE_OVERRIDES,
    }
    assert warmup_row["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert warmup_row["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert warmup_row["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "stage1",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.05,
        }
    ]
    assert "grouped-token model" in warmup_row["anchor_delta"]
    assert "benchmark-facing grouped-token replay" in warmup_row["next_action"]
    assert warmup_row["decision"] == "keep"
    assert any("preferred grouped-token adequacy surface" in note for note in warmup_row["notes"])

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_grouped_token_stability_probe_v1_compose_cfg_preserves_grouped_token_surface(
    tmp_path: Path,
) -> None:
    queue = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "queue.yaml")
    trace_row = _row_by_ref(queue, "delta_anchor_activation_trace_baseline")

    cfg = _compose_cfg(
        row=trace_row,
        run_dir=tmp_path / "train",
        device="cpu",
        training_experiment="cls_benchmark_staged_prior",
    )

    assert str(cfg.model.stage) == "grouped_tokens"
    assert str(cfg.model.stage_label) == "delta_architecture_screen_grouped_tokens"
    assert OmegaConf.to_container(cfg.model.module_overrides, resolve=True) == EXPECTED_MODULE_OVERRIDES
    assert str(cfg.model.module_overrides.row_pool) == "target_column"
    assert bool(cfg.model.module_overrides.allow_test_self_attention) is True
    assert bool(cfg.runtime.trace_activations) is True


def test_grouped_token_stability_probe_v1_matrix_records_the_grouped_token_probe() -> None:
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
    assert "Training experiment: `cls_benchmark_staged_prior`" in matrix
    assert "Surface role: `hybrid_diagnostic`" in matrix
    assert "delta_anchor_activation_trace_baseline" in matrix
    assert "delta_training_linear_decay" in matrix
    assert "delta_training_linear_warmup_decay" in matrix
    assert "Shifted grouped tokenizer" in matrix
    assert "warmup-decay grouped-token surface" in matrix
    assert "preferred grouped-token adequacy surface" in matrix
