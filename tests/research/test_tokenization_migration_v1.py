from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue
from tab_foundry.research.system_delta_execute import _compose_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1"
EXPECTED_ROWS = [
    "delta_architecture_screen_grouped_tokens",
    "delta_training_linear_warmup_decay",
]
HISTORICAL_RUN_ID = "sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v1"
GROUPED_RUN_ID = "sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2"
REPLAY_RUN_ID = "sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1"
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


def test_tokenization_migration_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["tokenization_migration_v1"] == {
        "parent_sweep_id": "shared_surface_bridge_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_tokenization_migration_v1_metadata_and_rows_match_the_grouped_token_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "tokenization_migration_v1"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "tokenization_migration_v1"
    assert sweep["parent_sweep_id"] == "shared_surface_bridge_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_staged"
    assert sweep["training_config_profile"] == "cls_benchmark_staged"
    assert sweep["surface_role"] == "architecture_screen"
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "prenorm_block"
    assert sweep["anchor_context"]["model"]["stage_label"] == "delta_architecture_screen_prenorm_block"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "training_default"
    assert any("first stage-native TF-RD-004 sweep" in note for note in sweep["anchor_surface"]["notes"])
    assert any("grouped_token_stability_probe_v1" in note for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed"]
    assert [row["interpretation_status"] for row in rows] == ["completed", "completed"]

    grouped_row = _row_by_ref(queue, "delta_architecture_screen_grouped_tokens")
    assert grouped_row["model"] == {
        "stage": "grouped_tokens",
        "stage_label": "delta_architecture_screen_grouped_tokens",
    }
    assert grouped_row["data"] == {"surface_label": "anchor_manifest_default"}
    assert grouped_row["preprocessing"] == {"surface_label": "runtime_default"}
    assert grouped_row["training"] == {
        "surface_label": "training_default",
        "overrides": {},
    }
    assert grouped_row["run_id"] == GROUPED_RUN_ID
    assert "locked `prenorm_block` bridge row" in grouped_row["anchor_delta"]
    assert "grouped_token_stability_probe_v1" in grouped_row["next_action"]
    assert grouped_row["decision"] == "defer"
    assert grouped_row["benchmark_metrics"]["best_step"] == 400
    assert grouped_row["notes"][0] == f"Canonical rerun registered as `{HISTORICAL_RUN_ID}`."
    assert grouped_row["notes"][-1] == f"Canonical rerun registered as `{GROUPED_RUN_ID}`."

    replay_row = _row_by_ref(queue, "delta_training_linear_warmup_decay")
    assert replay_row["model"] == {
        "stage": "grouped_tokens",
        "stage_label": "delta_architecture_screen_grouped_tokens",
        "module_overrides": EXPECTED_MODULE_OVERRIDES,
    }
    assert replay_row["data"] == {"surface_label": "anchor_manifest_default"}
    assert replay_row["preprocessing"] == {"surface_label": "runtime_default"}
    assert replay_row["training"] == {
        "surface_label": "prior_linear_warmup_decay",
        "prior_dump_non_finite_policy": "skip",
        "overrides": {
            "apply_schedule": True,
            "optimizer": {
                "name": "schedulefree_adamw",
                "require_requested": True,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "min_lr": 0.0004,
                "muon_per_parameter_lr": False,
            },
            "runtime": {
                "max_steps": 2500,
                "eval_every": 25,
                "checkpoint_every": 25,
                "trace_activations": False,
            },
            "schedule": {
                "stages": [
                    {
                        "name": "stage1",
                        "steps": 2500,
                        "lr_max": 0.004,
                        "lr_schedule": "linear",
                        "warmup_ratio": 0.05,
                    }
                ]
            },
        },
    }
    assert replay_row["run_id"] == REPLAY_RUN_ID
    assert replay_row["decision"] == "keep"
    assert REPLAY_RUN_ID in replay_row["next_action"]
    assert any(note == f"Canonical rerun registered as `{REPLAY_RUN_ID}`." for note in replay_row["notes"])
    assert any("benchmark-facing grouped-token warmup-decay replay" in note for note in replay_row["notes"])

    materialized = load_system_delta_queue(
        sweep_id="tokenization_migration_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_tokenization_migration_v1_matrix_records_the_grouped_token_handoff_row() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "tokenization_migration_v1"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "tokenization_migration_v1" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "Training experiment: `cls_benchmark_staged`" in matrix
    assert "Surface role: `architecture_screen`" in matrix
    assert "delta_architecture_screen_grouped_tokens" in matrix
    assert "delta_training_linear_warmup_decay" in matrix
    assert "grouped_tokens" in matrix
    assert REPLAY_RUN_ID in matrix
    assert "canonical grouped-token predecessor for TF-RD-005, TF-RD-006, and TF-RD-007" in matrix
    assert "first stage-native tokenizer change on the shared-surface handoff" in matrix


def test_tokenization_migration_v1_compose_cfg_resolves_architecture_screen_warmup_decay_replay(
    tmp_path: Path,
) -> None:
    queue = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "tokenization_migration_v1" / "queue.yaml")
    replay_row = _row_by_ref(queue, "delta_training_linear_warmup_decay")

    cfg = _compose_cfg(
        row=replay_row,
        run_dir=tmp_path / "train",
        device="cpu",
        training_experiment="cls_benchmark_staged",
    )

    assert str(cfg.model.stage) == "grouped_tokens"
    assert str(cfg.model.stage_label) == "delta_architecture_screen_grouped_tokens"
    assert OmegaConf.to_container(cfg.model.module_overrides, resolve=True) == EXPECTED_MODULE_OVERRIDES
    assert str(cfg.training.surface_label) == "prior_linear_warmup_decay"
    assert str(cfg.training.prior_dump_non_finite_policy) == "skip"
    assert bool(cfg.training.apply_schedule) is True
    assert str(cfg.optimizer.name) == "schedulefree_adamw"
    assert float(cfg.optimizer.min_lr) == 0.0004
    assert int(cfg.runtime.max_steps) == 2500
    assert int(cfg.runtime.eval_every) == 25
    assert int(cfg.runtime.checkpoint_every) == 25
    assert bool(cfg.runtime.trace_activations) is False
    assert OmegaConf.to_container(cfg.schedule.stages, resolve=True) == [
        {
            "name": "stage1",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.05,
        }
    ]
