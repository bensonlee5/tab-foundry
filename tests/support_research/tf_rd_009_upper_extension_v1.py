from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

import tab_foundry.research.tf_rd_009_upper_extension_design as upper_extension_module
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tab_foundry.research.tf_rd_009_upper_extension_design import (
    TF_RD_009_UPPER_EXTENSION_GATE_STEPS,
    TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID,
    TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID,
    TF_RD_009_UPPER_EXTENSION_SELECTION_JSON,
    TF_RD_009_UPPER_EXTENSION_STEP_LADDER,
    TF_RD_009_UPPER_EXTENSION_STUDY_ID,
    build_tf_rd_009_upper_extension_gate_queue_rows,
    build_tf_rd_009_upper_extension_ns_queue_rows,
    promoted_tf_rd_009_upper_extension_row_labels,
    select_tf_rd_009_upper_extension,
)
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
GATE_SELECTED_ROWS = [
    "delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1",
    "delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1",
    "delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1",
    "delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1",
]
ALL_CANDIDATE_ROWS = [
    ["216x7", "272x8"],
    ["200x7", "224x8", "256x9"],
    ["192x7", "208x8", "224x9", "248x10"],
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_upper_extension_selection_is_deterministic_and_feasible() -> None:
    selection = select_tf_rd_009_upper_extension()

    assert [list(candidate.row_labels) for candidate in selection.candidates] == [
        ["192x7", "208x8", "224x9", "248x10"],
        ["200x7", "224x8", "256x9"],
        ["216x7", "272x8"],
    ]
    assert sorted(
        [sorted(candidate.row_labels) for candidate in selection.candidates],
    ) == sorted([sorted(candidate_rows) for candidate_rows in ALL_CANDIDATE_ROWS])
    assert list(selection.selected_row_labels) == ["192x7", "208x8", "224x9", "248x10"]
    assert selection.selected_candidate_id == "192x7->208x8->224x9->248x10"
    assert all(candidate.feasible_under_corrected_ceiling for candidate in selection.candidates)
    assert selection.candidates[0].d_optimal_gain > selection.candidates[1].d_optimal_gain
    assert selection.candidates[1].d_optimal_gain > selection.candidates[2].d_optimal_gain
    assert selection.candidates[0].alpha_uncertainty_width < selection.candidates[1].alpha_uncertainty_width
    assert selection.candidates[1].alpha_uncertainty_width < selection.candidates[2].alpha_uncertainty_width


def test_tf_rd_009_upper_extension_gate_helper_matches_selected_continuation() -> None:
    rows = build_tf_rd_009_upper_extension_gate_queue_rows()

    assert [row["delta_ref"] for row in rows] == GATE_SELECTED_ROWS
    assert [row["order"] for row in rows] == [1, 2, 3, 4]
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in rows} == {
        TF_RD_009_UPPER_EXTENSION_GATE_STEPS
    }
    assert {
        row["training"]["overrides"]["schedule"]["stages"][0]["steps"]
        for row in rows
    } == {TF_RD_009_UPPER_EXTENSION_GATE_STEPS}
    assert all("health=`ok`" in " ".join(row["parameter_adequacy_plan"]) for row in rows)
    assert all("152x5" in " ".join(row["parameter_adequacy_plan"]) for row in rows)


def test_tf_rd_009_upper_extension_warn_rows_do_not_auto_promote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_rows = [
        {
            "status": "completed",
            "run_id": "run_ok",
            "model": {"d_icl": 192, "sandwich_layers": 7},
        },
        {
            "status": "completed",
            "run_id": "run_warn",
            "model": {"d_icl": 208, "sandwich_layers": 8},
        },
        {
            "status": "ready",
            "run_id": None,
            "model": {"d_icl": 224, "sandwich_layers": 9},
        },
    ]

    monkeypatch.setattr(
        upper_extension_module,
        "load_benchmark_run_entry",
        lambda run_id, path=None: {"run_id": run_id, "artifacts": {"run_dir": "outputs/synthetic"}},
    )
    monkeypatch.setattr(
        upper_extension_module,
        "_run_health_verdict",
        lambda entry: "ok" if entry["run_id"] == "run_ok" else "warn",
    )

    promoted = promoted_tf_rd_009_upper_extension_row_labels(gate_rows=gate_rows)

    assert promoted == ("192x7",)


def test_tf_rd_009_upper_extension_ns_helper_materializes_only_promoted_rows() -> None:
    rows = build_tf_rd_009_upper_extension_ns_queue_rows(
        promoted_row_labels=("208x8", "248x10"),
    )

    assert len(rows) == 2 * len(TF_RD_009_UPPER_EXTENSION_STEP_LADDER)
    assert {row["delta_ref"] for row in rows} == {
        "delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1",
        "delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1",
    }
    assert {
        row["training"]["overrides"]["runtime"]["max_steps"]
        for row in rows
    } == set(TF_RD_009_UPPER_EXTENSION_STEP_LADDER)
    assert all("health=`ok`" in " ".join(row["parameter_adequacy_plan"]) for row in rows)
    assert build_tf_rd_009_upper_extension_ns_queue_rows(promoted_row_labels=()) == ()


def test_tf_rd_009_upper_extension_selection_artifact_is_checked_in() -> None:
    selection_payload = json.loads(
        (REPO_ROOT / TF_RD_009_UPPER_EXTENSION_SELECTION_JSON).read_text(encoding="utf-8")
    )

    assert selection_payload["selected_candidate_id"] == "192x7->208x8->224x9->248x10"
    assert selection_payload["selected_row_labels"] == ["192x7", "208x8", "224x9", "248x10"]
    assert [candidate["row_labels"] for candidate in selection_payload["candidates"]] == [
        ["192x7", "208x8", "224x9", "248x10"],
        ["200x7", "224x8", "256x9"],
        ["216x7", "272x8"],
    ]


def test_tf_rd_009_upper_extension_sweeps_are_registered() -> None:
    index = _load_yaml(INDEX_PATH)

    assert index["sweeps"][TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_009_width_depth_medium_v1",
        "status": "draft",
        "anchor_run_id": "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1",
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }
    assert index["sweeps"][TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID] == {
        "parent_sweep_id": TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID,
        "status": "draft",
        "anchor_run_id": "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1",
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_009_upper_extension_gate_sweep_tracks_the_selected_gate_rows() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_009_width_depth_medium_v1"
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        training_config_profile="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert [row["delta_ref"] for row in rows] == GATE_SELECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "ready", "ready", "ready"]
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in rows} == {
        TF_RD_009_UPPER_EXTENSION_GATE_STEPS
    }

    materialized = load_system_delta_queue(
        sweep_id=TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == GATE_SELECTED_ROWS
    assert all(
        row["model"]["d_icl"] == expected
        for row, expected in zip(materialized["rows"], [192, 208, 224, 248], strict=True)
    )
    assert all(
        row["model"]["sandwich_layers"] == expected
        for row, expected in zip(materialized["rows"], [7, 8, 9, 10], strict=True)
    )


def test_tf_rd_009_upper_extension_ns_sweep_stays_empty_until_promotion() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID
    assert sweep["parent_sweep_id"] == TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        training_config_profile="cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        surface_role="classification_scaling_law_phase2_ns",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert queue["rows"] == []

    materialized = load_system_delta_queue(
        sweep_id=TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["rows"] == []


def test_tf_rd_009_upper_extension_study_config_tracks_ns_extension_only() -> None:
    study = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / f"{TF_RD_009_UPPER_EXTENSION_STUDY_ID}.yaml")

    assert study["study_id"] == TF_RD_009_UPPER_EXTENSION_STUDY_ID
    assert study["phase1_reference_sweep_id"] == TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID
    assert study["geometry_row_labels"] == ["192x7", "208x8", "224x9", "248x10"]
    assert study["step_ladder"] == list(TF_RD_009_UPPER_EXTENSION_STEP_LADDER)
    assert study["batch_grad_accum_ladder"] == []
    assert study["sweeps"] == [
        {"name": "ns_core", "sweep_id": TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID, "family": "ns_core"}
    ]
    assert study["canonical_loss_axes"]["validation"] == "validation_loss"
    assert study["primary_fit"] == {"law": "L(N,S)", "target": "validation_loss"}

