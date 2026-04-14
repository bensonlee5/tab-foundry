from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

from tab_foundry.data.corpus_loading import load_corpus_recipe
from tab_foundry.research.scaling.study import load_scaling_study_config
from tab_foundry.research.sweep.configuration import validate_one_epoch_contract
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_V5 = "tf_rd_010_dagzoo_medium_control_curated_v5"
CORPUS_V6 = "tf_rd_010_dagzoo_medium_control_curated_v6"
NS_OLD = "tf_rd_009_ns_medium_v1"
BCRIT_OLD = "tf_rd_009_batch_critical_medium_v1"
NS_NEW = "tf_rd_009_ns_one_epoch_medium_v1"
BCRIT_NEW = "tf_rd_009_batch_critical_one_epoch_medium_v1"
PHASE2_NEW = "tf_rd_009_phase2_one_epoch_v1"
UPPER_GATE_NEW = "tf_rd_009_width_depth_upper_extension_one_epoch_medium_v1"
UPPER_NS_NEW = "tf_rd_009_ns_upper_extension_one_epoch_medium_v1"
UPPER_STUDY_NEW = "tf_rd_009_phase2_upper_extension_one_epoch_v1"
STEP_LADDER = [625, 1250, 2500, 5000]
BATCH_LADDER = [1, 2, 4, 8, 16]
GEOMETRIES = ["72x1", "96x2", "112x3", "128x4", "152x5", "176x6"]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _sweep_root(sweep_id: str) -> Path:
    return REPO_ROOT / "reference" / "system_delta_sweeps" / sweep_id


def _queue(sweep_id: str) -> dict[str, Any]:
    return _load_yaml(_sweep_root(sweep_id) / "queue.yaml")


def test_tf_rd_009_v6_corpus_preserves_v5_distribution_and_expands_count() -> None:
    v5 = _load_yaml(REPO_ROOT / "reference" / "corpus_recipes" / f"{CORPUS_V5}.yaml")
    v6 = _load_yaml(REPO_ROOT / "reference" / "corpus_recipes" / f"{CORPUS_V6}.yaml")

    assert v6["recipe_id"] == CORPUS_V6
    assert v6["manifest"] == v5["manifest"]
    assert v6["surface_label"] == v5["surface_label"]
    assert v6["generator"]["module"] == v5["generator"]["module"]
    assert v6["generator"]["callable"] == v5["generator"]["callable"]
    assert {
        key: value
        for key, value in v6["generator"]["inputs"].items()
        if key != "num_datasets"
    } == {
        key: value
        for key, value in v5["generator"]["inputs"].items()
        if key != "num_datasets"
    }
    assert v6["generator"]["inputs"]["num_datasets"] == 10371
    assert v6["review_summary"] == {
        **v5["review_summary"],
        "manifest_record_count": 1493424,
        "num_datasets_per_invocation": 10371,
    }

    recipe = load_corpus_recipe(CORPUS_V6)
    assert len(recipe.invocations) == 144
    assert sum(invocation.num_datasets for invocation in recipe.invocations) == 1493424
    assert int(1493424 * recipe.manifest_policy.train_ratio) == 1344081


def test_tf_rd_009_one_epoch_guard_fails_curated_v5_high_budget_rows() -> None:
    old_ns_row = _queue(NS_OLD)["rows"][23]
    old_bcrit_row = _queue(BCRIT_OLD)["rows"][19]

    with pytest.raises(RuntimeError, match="required_train_tasks=320000 exceeds"):
        validate_one_epoch_contract(
            old_ns_row,
            repo_root=REPO_ROOT,
            sweep_id=NS_OLD,
            sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
            require_declared_contract=False,
        )
    with pytest.raises(RuntimeError, match="required_train_tasks=1280000 exceeds"):
        validate_one_epoch_contract(
            old_bcrit_row,
            repo_root=REPO_ROOT,
            sweep_id=BCRIT_OLD,
            sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
            require_declared_contract=False,
        )


def test_tf_rd_009_corrected_one_epoch_rows_pass_contract() -> None:
    for sweep_id in (NS_NEW, BCRIT_NEW):
        materialized = load_system_delta_queue(
            sweep_id=sweep_id,
            index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
            catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        )
        assert materialized["rows"]
        for row in materialized["rows"]:
            result = validate_one_epoch_contract(
                row,
                repo_root=REPO_ROOT,
                sweep_id=sweep_id,
                sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
            )
            assert result is not None
            assert result["corpus_ref"] == CORPUS_V6
            assert result["required_train_tasks"] <= result["train_records_available"]


def test_tf_rd_009_corrected_ns_queue_preserves_original_ladder_on_v6() -> None:
    queue = _queue(NS_NEW)
    rows = queue["rows"]

    assert queue["sweep_id"] == NS_NEW
    assert len(rows) == len(GEOMETRIES) * len(STEP_LADDER)
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["run_id"] for row in rows} == {None}
    assert {
        row["data"]["corpus_ref"]
        for row in rows
    } == {CORPUS_V6}
    assert {
        row["training"]["one_epoch_contract"]["scope"]
        for row in rows
    } == {"train_manifest_unique_tasks"}
    assert Counter(
        f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows
    ) == {label: len(STEP_LADDER) for label in GEOMETRIES}
    assert Counter(
        row["training"]["overrides"]["runtime"]["max_steps"] for row in rows
    ) == {step: len(GEOMETRIES) for step in STEP_LADDER}
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows} == {4}


def test_tf_rd_009_corrected_bcrit_queue_preserves_original_ladder_on_v6() -> None:
    queue = _queue(BCRIT_NEW)
    rows = queue["rows"]

    assert queue["sweep_id"] == BCRIT_NEW
    assert len(rows) == len(BATCH_LADDER) * len(STEP_LADDER)
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}
    assert {f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows} == {"96x2"}
    assert Counter(
        row["training"]["overrides"]["runtime"]["max_steps"] for row in rows
    ) == {step: len(BATCH_LADDER) for step in STEP_LADDER}
    assert Counter(
        row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows
    ) == {batch: len(STEP_LADDER) for batch in BATCH_LADDER}


def test_tf_rd_009_historical_phase2_artifacts_remain_unchanged() -> None:
    old_ns = _queue(NS_OLD)
    old_bcrit = _queue(BCRIT_OLD)

    assert old_ns["sweep_id"] == NS_OLD
    assert old_bcrit["sweep_id"] == BCRIT_OLD
    assert {row["data"]["corpus_ref"] for row in old_ns["rows"]} == {CORPUS_V5}
    assert {row["data"]["corpus_ref"] for row in old_bcrit["rows"]} == {CORPUS_V5}
    assert "one_epoch_contract" not in old_ns["rows"][23]["training"]
    assert "one_epoch_contract" not in old_bcrit["rows"][19]["training"]


def test_tf_rd_009_corrected_studies_exclude_historical_sweeps() -> None:
    phase2 = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / f"{PHASE2_NEW}.yaml")

    assert load_scaling_study_config(study_id=PHASE2_NEW).study_id == PHASE2_NEW
    assert phase2["sweeps"] == [
        {"name": "ns_core", "sweep_id": NS_NEW, "family": "ns_core"},
        {"name": "batch_critical", "sweep_id": BCRIT_NEW, "family": "batch_critical"},
    ]
    assert NS_OLD not in {entry["sweep_id"] for entry in phase2["sweeps"]}
    assert BCRIT_OLD not in {entry["sweep_id"] for entry in phase2["sweeps"]}
    assert phase2["primary_fit"] == {"law": "L(N,S)", "target": "validation_loss"}
    assert phase2["validation_overlay_path"].endswith(
        "tf_rd_009_phase2_one_epoch_v1_validation_backfill_v1.json"
    )


def test_tf_rd_009_upper_one_epoch_extension_scaffolds_wait_for_corrected_fit() -> None:
    gate_queue = _queue(UPPER_GATE_NEW)
    ns_queue = _queue(UPPER_NS_NEW)
    upper_study = _load_yaml(
        REPO_ROOT / "reference" / "scaling_studies" / f"{UPPER_STUDY_NEW}.yaml"
    )

    assert gate_queue["rows"] == []
    assert ns_queue["rows"] == []
    assert upper_study["selection_dependency"] == PHASE2_NEW
    assert upper_study["phase1_reference_sweep_id"] == UPPER_GATE_NEW
    assert upper_study["sweeps"] == [
        {"name": "ns_core", "sweep_id": UPPER_NS_NEW, "family": "ns_core"}
    ]
    assert upper_study["geometry_row_labels"] == []
