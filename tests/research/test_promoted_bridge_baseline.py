from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.promoted_bridge_baseline import promoted_bridge_baseline_payload


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_promoted_bridge_baseline_stays_aligned_across_config_and_sweeps() -> None:
    payload = promoted_bridge_baseline_payload()
    stability_queue = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / "stability_followup" / "queue.yaml"
    )
    input_norm_sweep = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / "input_norm_followup" / "sweep.yaml"
    )
    input_norm_queue = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / "input_norm_followup" / "queue.yaml"
    )

    promoted_row = _row_by_ref(stability_queue, "dpnb_row_cls_cls2_linear_warmup_decay")
    anchor_replay_row = _row_by_ref(input_norm_queue, "dpnb_input_norm_anchor_replay")

    expected_model = {
        "stage": payload["model"]["stage"],
        "stage_label": payload["model"]["stage_label"],
        "input_normalization": payload["model"]["input_normalization"],
        "module_overrides": payload["model"]["module_overrides"],
        "tfrow_n_heads": payload["model"]["tfrow_n_heads"],
        "tfrow_n_layers": payload["model"]["tfrow_n_layers"],
        "tfrow_cls_tokens": payload["model"]["tfrow_cls_tokens"],
        "tfrow_norm": payload["model"]["tfrow_norm"],
    }
    assert promoted_row["model"] == {
        key: value
        for key, value in expected_model.items()
        if key != "input_normalization"
    }
    assert anchor_replay_row["model"] == expected_model

    expected_training = {
        "surface_label": payload["training"]["surface_label"],
        "prior_dump_non_finite_policy": payload["training"]["prior_dump_non_finite_policy"],
        "prior_dump_batch_size": payload["training"]["prior_dump_batch_size"],
        "prior_dump_lr_scale_rule": payload["training"]["prior_dump_lr_scale_rule"],
        "prior_dump_batch_reference_size": payload["training"]["prior_dump_batch_reference_size"],
        "overrides": {
            "apply_schedule": True,
            "optimizer": {
                "min_lr": payload["training"]["optimizer_min_lr"],
            },
            "runtime": payload["training"]["runtime"],
            "schedule": {
                "stages": [payload["training"]["schedule_stage"]],
            },
        },
    }
    assert anchor_replay_row["training"] == expected_training
    assert input_norm_sweep["anchor_context"]["experiment"] == "cls_benchmark_staged_prior"
    assert input_norm_sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged_prior"
    assert input_norm_sweep["anchor_run_id"] == "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1"
    assert input_norm_sweep["anchor_context"]["model"]["stage_label"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert input_norm_sweep["anchor_context"]["surface_labels"]["training"] == payload["training"]["surface_label"]


def test_promoted_bridge_baseline_logging_name_matches_stage_label() -> None:
    payload = promoted_bridge_baseline_payload()
    assert payload["logging"]["run_name"] == payload["model"]["stage_label"]
