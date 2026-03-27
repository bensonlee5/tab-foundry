from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_021a_sandwich_nanotabpfn_screen_v1"
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
EXPECTED_ROWS = [
    "delta_tf_rd_021a_sandwich_replay_v1",
    "delta_tf_rd_021a_sandwich_latents24_v1",
    "delta_tf_rd_021a_sandwich_latents96_v1",
    "delta_tf_rd_021a_sandwich_width128_latents48_v1",
    "delta_tf_rd_021a_sandwich_width128_latents96_v1",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_021a_sandwich_nanotabpfn_screen_v1_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    assert "active_sweep_id" not in index

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": None,
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
        "external_benchmarks": ["nanotabpfn"],
    }


def test_tf_rd_021a_sandwich_nanotabpfn_screen_v1_matches_the_screen_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] is None
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_prior"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_prior"
    assert sweep["surface_role"] == "architecture_screen"
    assert sweep["upstream_reference"] == {
        "name": "Perceiver",
        "model_source": "https://proceedings.mlr.press/v139/jaegle21a.html",
    }
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["arch"] == "tabfoundry_staged"
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "anchor_manifest_default",
        "model": "dpnb_input_norm_anchor_replay_batch64_sqrt",
        "preprocessing": "runtime_default",
        "training": "prior_linear_warmup_decay",
    }

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#179" in note for note in notes)
    assert any("materially underpowered" in note for note in notes)
    assert any("batch64-sqrt" in note for note in notes)
    assert any("rows `02` through `05`" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        "completed",
        "deferred_separate_workstream",
        "deferred_separate_workstream",
        "deferred_separate_workstream",
        "deferred_separate_workstream",
    ]
    assert [row["interpretation_status"] for row in rows] == [
        "completed",
        "blocked",
        "blocked",
        "blocked",
        "blocked",
    ]

    replay = _row_by_ref(queue, "delta_tf_rd_021a_sandwich_replay_v1")
    assert replay.get("parent_delta_ref") is None
    assert replay["model"] == {
        "arch": "tabfoundry_sandwich",
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "head_hidden_dim": 128,
        "norm_type": "layernorm",
        "sandwich_latents": 48,
        "sandwich_layers": 8,
        "sandwich_heads": 8,
        "sandwich_ff_expansion": 2,
    }
    assert replay["training"]["surface_label"] == "prior_linear_warmup_decay"
    assert replay["training"]["prior_dump_batch_size"] == 64
    assert replay["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert replay["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert replay["run_id"] == "sd_tf_rd_021a_sandwich_nanotabpfn_screen_v1_01_delta_tf_rd_021a_sandwich_replay_v1_v1"
    assert replay["benchmark_metrics"]["final_roc_auc"] == 0.6224489632492666
    assert "underperformed the locked staged anchor" in " ".join(replay["notes"])

    low_latent = _row_by_ref(queue, "delta_tf_rd_021a_sandwich_latents24_v1")
    assert low_latent["parent_delta_ref"] == "delta_tf_rd_021a_sandwich_replay_v1"
    assert low_latent["model"]["sandwich_latents"] == 24
    assert low_latent["run_id"] is None
    assert low_latent["status"] == "deferred_separate_workstream"
    assert "Do not run inside TF-RD-021A." in low_latent["next_action"]

    high_latent = _row_by_ref(queue, "delta_tf_rd_021a_sandwich_latents96_v1")
    assert high_latent["parent_delta_ref"] == "delta_tf_rd_021a_sandwich_replay_v1"
    assert high_latent["model"]["sandwich_latents"] == 96
    assert high_latent["run_id"] is None
    assert high_latent["status"] == "deferred_separate_workstream"
    assert "successor replay" in high_latent["next_action"]

    width_replay = _row_by_ref(queue, "delta_tf_rd_021a_sandwich_width128_latents48_v1")
    assert width_replay["parent_delta_ref"] == "delta_tf_rd_021a_sandwich_replay_v1"
    assert width_replay["model"]["d_icl"] == 128
    assert width_replay["model"]["sandwich_latents"] == 48
    assert width_replay["status"] == "deferred_separate_workstream"
    assert "Do not execute in TF-RD-021A." in width_replay["next_action"]

    width_high = _row_by_ref(queue, "delta_tf_rd_021a_sandwich_width128_latents96_v1")
    assert width_high["parent_delta_ref"] == "delta_tf_rd_021a_sandwich_latents96_v1"
    assert width_high["model"]["d_icl"] == 128
    assert width_high["model"]["sandwich_latents"] == 96
    assert width_high["status"] == "deferred_separate_workstream"
    assert "TF-RD-021B" in " ".join([width_high["next_action"], *width_high["notes"]]) or "successor replay" in width_high["next_action"]

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["training_experiment"] == "cls_benchmark_sandwich_prior"
    assert materialized["training_config_profile"] == "cls_benchmark_sandwich_prior"
    assert materialized["surface_role"] == "architecture_screen"
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_tf_rd_021a_sandwich_nanotabpfn_screen_v1_matrix_records_the_blocked_followup() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "Perceiver" in matrix
    assert "cls_benchmark_sandwich_prior" in matrix
    assert "deferred_separate_workstream" in matrix
    assert "delta_tf_rd_021a_sandwich_latents24_v1" in matrix
    assert "delta_tf_rd_021a_sandwich_latents96_v1" in matrix
    assert "delta_tf_rd_021a_sandwich_width128_latents48_v1" in matrix
    assert "delta_tf_rd_021a_sandwich_width128_latents96_v1" in matrix
    assert "underperformed the locked staged anchor badly enough" in matrix
