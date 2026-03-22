from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.inspect as inspect_module
from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_013_data_source_contract_v1"
ANCHOR_RUN_ID = "sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1"
EXPECTED_ROWS = [
    "delta_data_filter_policy_accepted_only",
    "delta_data_manifest_root_curated_dagzoo",
    "delta_data_manifest_curated_realdata_comparator",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_013_data_source_contract_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "qass_tfcol_large_missing_validation_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_tf_rd_013_data_source_contract_metadata_and_rows_match_issue_100_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "qass_tfcol_large_missing_validation_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "qass_context"
    assert sweep["anchor_context"]["surface_labels"]["data"] == "anchor_manifest_default"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("TF-RD-013 contract layer" in note for note in notes)
    assert any("historical precursor evidence" in note for note in notes)
    assert any("dagzoo_provenance" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["blocked_on_artifacts"] * 3

    accepted_only_row = _row_by_ref(queue, "delta_data_filter_policy_accepted_only")
    assert accepted_only_row["data"]["surface_label"] == "tf_rd_013_dagzoo_accepted_only"
    accepted_overrides = accepted_only_row["data"]["surface_overrides"]
    assert accepted_overrides["manifest_path"] == "outputs/staged_ladder_support/tf_rd_013/dagzoo_accepted_only/manifest.parquet"
    assert accepted_overrides["filter_policy"] == "accepted_only"
    assert accepted_overrides["dagzoo_provenance"] == {
        "corpus_variant": "dagzoo_accepted_only",
        "comparator_role": "promoted_anchor_candidate",
        "commands": [],
        "config_refs": ["configs/default.yaml"],
        "curated_root_lineage": [],
        "materialization_issue": 120,
    }

    curated_dagzoo_row = _row_by_ref(queue, "delta_data_manifest_root_curated_dagzoo")
    assert curated_dagzoo_row["parent_delta_ref"] == "delta_data_filter_policy_accepted_only"
    assert curated_dagzoo_row["data"]["surface_label"] == "tf_rd_013_curated_dagzoo_root"
    assert curated_dagzoo_row["data"]["surface_overrides"]["dagzoo_provenance"]["curated_root_lineage"] == [
        "dagzoo_accepted_only",
        "curated_root_selection",
    ]
    assert any("supersedes the old `binary_md_v1`" in note for note in curated_dagzoo_row["notes"])

    realdata_row = _row_by_ref(queue, "delta_data_manifest_curated_realdata_comparator")
    assert realdata_row["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"
    assert realdata_row["data"]["surface_overrides"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet"
    )
    assert all("license" not in key for key in realdata_row["data"]["surface_overrides"].keys())

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS
    assert all(row["status"] == "blocked_on_artifacts" for row in materialized_rows)


def test_tf_rd_013_data_source_contract_inspect_and_diff_resolve_explicit_data_surfaces() -> None:
    inspect_payload = inspect_module.inspect_sweep_row(
        order=2,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    resolved_data = inspect_payload["target"]["resolved"]["data"]
    assert resolved_data["surface_label"] == "tf_rd_013_curated_dagzoo_root"
    assert resolved_data["source"] == "manifest"
    assert resolved_data["dagzoo_provenance"]["corpus_variant"] == "dagzoo_curated_root"
    assert resolved_data["dagzoo_provenance"]["materialization_issue"] == 120
    assert resolved_data["overrides"]["manifest_path"].endswith(
        "outputs/staged_ladder_support/tf_rd_013/curated_dagzoo/manifest.parquet"
    )

    diff_payload = diff_module.diff_sweep_row(
        order=2,
        sweep_id=SWEEP_ID,
        against="anchor",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert diff_payload["difference_count"] > 0
    differences = diff_payload["differences"]
    assert differences["resolved.data.surface_label"] == {
        "target": "tf_rd_013_curated_dagzoo_root",
        "against": "anchor_manifest_default",
    }
    assert differences["resolved.data.dagzoo_provenance"]["target"]["corpus_variant"] == "dagzoo_curated_root"
    assert differences["resolved.data.dagzoo_provenance"]["against"] is None


def test_binary_md_v1_curated_dagzoo_row_is_now_marked_as_historical_precursor() -> None:
    queue = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "binary_md_v1" / "queue.yaml")
    row = _row_by_ref(queue, "delta_data_manifest_root_curated_dagzoo")

    assert row["status"] == "superseded"
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "blocked"
    assert "tf_rd_013_data_source_contract_v1" in row["next_action"]
    assert any("historical precursor evidence only" in note for note in row["notes"])
