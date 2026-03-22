from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.inspect as inspect_module
import tab_foundry.research.sweep.matrix as matrix_module
from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_013_data_source_contract_v1"
ANCHOR_RUN_ID = "sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1"
MATERIALIZATION_ISSUE_NUMBER = 120
FILTERING_POLICY_ISSUE_NUMBER = 124
EXPECTED_ROWS = [
    "delta_data_manifest_root_dagzoo_generated_source",
    "delta_data_manifest_curated_realdata_comparator",
]
SUPPORT_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "support"
MATRIX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
MATERIALIZATION_SUMMARY_PATH = SUPPORT_ROOT / "materialization_summary.json"
MANIFEST_CHARACTERISTICS_SUMMARY_PATH = SUPPORT_ROOT / "manifest_characteristics_summary.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def test_tf_rd_013_data_source_contract_metadata_and_rows_match_unfiltered_support_bundle() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    materialization_summary = _load_json(MATERIALIZATION_SUMMARY_PATH)
    manifest_characteristics_summary = _load_json(MANIFEST_CHARACTERISTICS_SUMMARY_PATH)

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "qass_tfcol_large_missing_validation_v1"
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["upstream_reference"] == {
        "name": "TabICLv2",
        "model_source": "https://arxiv.org/abs/2602.11139",
    }
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage"] == "qass_context"
    assert sweep["anchor_context"]["surface_labels"]["data"] == "anchor_manifest_default"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("TabICLv2" in note for note in notes)
    assert any("nanoTabPFN" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "blocked_on_artifacts"]

    generated_row = _row_by_ref(queue, "delta_data_manifest_root_dagzoo_generated_source")
    generated_overrides = generated_row["data"]["surface_overrides"]
    generated_provenance = generated_overrides["dagzoo_provenance"]
    assert generated_row["data"]["surface_label"] == "tf_rd_013_dagzoo_generated_source"
    assert generated_overrides["manifest_path"] == "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    assert generated_provenance["commands"]
    assert generated_provenance["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert generated_provenance == materialization_summary["surfaces"]["dagzoo_generated_source"]["dagzoo_provenance"]
    assert "issue 124" in generated_row["next_action"]
    assert any("support/materialization_summary.json" in note for note in generated_row["notes"])

    realdata_row = _row_by_ref(queue, "delta_data_manifest_curated_realdata_comparator")
    assert realdata_row["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"
    assert realdata_row["data"]["surface_overrides"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet"
    )

    assert materialization_summary["issues"]["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert materialization_summary["issues"]["subsequent_filtering_policy_issue"] == FILTERING_POLICY_ISSUE_NUMBER
    assert materialization_summary["env_hints"]["dagzoo_root"] == "../dagzoo"
    assert materialization_summary["artifacts"]["generated_source_manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )
    assert set(materialization_summary["surfaces"]) == {"dagzoo_generated_source"}
    assert materialization_summary["surfaces"]["dagzoo_generated_source"]["state"] == "materialized_local_only"

    steps = {step["step_id"]: step for step in materialization_summary["steps"]}
    assert set(steps) == {"dagzoo_generate", "build_manifest_generated_source"}
    assert steps["dagzoo_generate"]["status"] == "completed"
    assert steps["build_manifest_generated_source"]["status"] == "completed"

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS
    assert [row["status"] for row in materialized_rows] == ["ready", "blocked_on_artifacts"]

    comparisons = manifest_characteristics_summary["comparisons"]
    assert set(comparisons) == {"anchor_vs_generated_source"}
    assert comparisons["anchor_vs_generated_source"]["status"] == "available"


def test_tf_rd_013_data_source_contract_inspect_and_diff_resolve_explicit_data_surfaces() -> None:
    inspect_generated = inspect_module.inspect_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    resolved_generated = inspect_generated["target"]["resolved"]["data"]
    assert resolved_generated["surface_label"] == "tf_rd_013_dagzoo_generated_source"
    assert resolved_generated["source"] == "manifest"
    assert resolved_generated["dagzoo_provenance"]["corpus_variant"] == "dagzoo_generated_source"
    assert resolved_generated["dagzoo_provenance"]["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert resolved_generated["overrides"]["manifest_path"].endswith(
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )

    inspect_realdata = inspect_module.inspect_sweep_row(
        order=2,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )
    assert inspect_realdata["target"]["identity"]["status"] == "blocked_on_artifacts"
    assert inspect_realdata["target"]["resolved"]["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"

    diff_payload = diff_module.diff_sweep_row(
        order=1,
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
        "target": "tf_rd_013_dagzoo_generated_source",
        "against": "anchor_manifest_default",
    }
    assert differences["resolved.data.dagzoo_provenance"]["target"]["corpus_variant"] == "dagzoo_generated_source"
    assert differences["resolved.data.dagzoo_provenance"]["against"] is None


def test_binary_md_v1_curated_dagzoo_row_is_now_marked_as_historical_precursor() -> None:
    queue = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "binary_md_v1" / "queue.yaml")
    row = _row_by_ref(queue, "delta_data_manifest_root_curated_dagzoo")

    assert row["status"] == "superseded"
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "blocked"
    assert "tf_rd_013_data_source_contract_v1" in row["next_action"]
    assert any("historical precursor evidence only" in note for note in row["notes"])


def test_tf_rd_013_support_bundle_and_catalog_defaults_are_tracked_separately() -> None:
    materialization_summary = _load_json(MATERIALIZATION_SUMMARY_PATH)
    manifest_characteristics_summary = _load_json(MANIFEST_CHARACTERISTICS_SUMMARY_PATH)
    catalog = _load_yaml(REPO_ROOT / "reference" / "system_delta_catalog.yaml")

    assert MATERIALIZATION_SUMMARY_PATH.exists()
    assert MANIFEST_CHARACTERISTICS_SUMMARY_PATH.exists()
    assert SUPPORT_ROOT.joinpath("README.md").exists()
    assert materialization_summary["config_refs"]["dagzoo"] == ["configs/default.yaml"]
    assert materialization_summary["config_refs"]["tab_foundry"] == ["configs/data/default.yaml"]
    assert manifest_characteristics_summary["manifests"]["anchor_manifest_default"]["manifest_path"] == "data/manifests/default.parquet"
    assert manifest_characteristics_summary["manifests"]["dagzoo_generated_source"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )
    for path in (
        MATERIALIZATION_SUMMARY_PATH,
        MANIFEST_CHARACTERISTICS_SUMMARY_PATH,
        SUPPORT_ROOT / "README.md",
    ):
        assert "/Users/" not in path.read_text(encoding="utf-8")

    comparator_default = catalog["deltas"]["delta_data_manifest_curated_realdata_comparator"]["default_effective_surface"]["data"]
    assert comparator_default == {
        "surface_label": "curated_realdata_comparator",
        "surface_overrides": {
            "source": "manifest",
            "manifest_path": "outputs/staged_ladder_support/curated_realdata/openml_baseline/manifest.parquet",
        },
    }


def test_tf_rd_013_matrix_uses_tabiclv2_as_dynamic_upstream_reference(tmp_path: Path) -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    out_path = tmp_path / "matrix.md"
    matrix_module.render_and_write_system_delta_matrix(
        queue=queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        out_path=out_path,
    )

    rendered = out_path.read_text(encoding="utf-8")
    checked_in = MATRIX_PATH.read_text(encoding="utf-8")
    assert "Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`." in rendered
    assert "| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |" in rendered
    assert "nanoTabPFN" in rendered
    assert "| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |" in checked_in
