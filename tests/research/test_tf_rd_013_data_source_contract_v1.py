from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.inspect as inspect_module
import tab_foundry.research.sweep.matrix as matrix_module
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_013_data_source_contract_v1"
ANCHOR_RUN_ID = "sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1"
MATERIALIZATION_ISSUE_NUMBER = 120
EXECUTION_ISSUE_NUMBER = 122
FILTERING_POLICY_ISSUE_NUMBER = 124
MULTI_INVOCATION_ISSUE_NUMBER = 127
EXPECTED_ROWS = [
    "delta_data_manifest_root_dagzoo_generated_source",
    "delta_data_manifest_curated_realdata_comparator",
]
GENERATED_RUN_ID = "sd_tf_rd_013_data_source_contract_v1_01_delta_data_manifest_root_dagzoo_generated_source_v2"
CURATED_RUN_ID = "sd_tf_rd_013_data_source_contract_v1_02_delta_data_manifest_curated_realdata_comparator_v2"
REGISTRY_PATH = default_benchmark_run_registry_path()
SUPPORT_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "support"
MATRIX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
MATERIALIZATION_SUMMARY_PATH = SUPPORT_ROOT / "materialization_summary.json"
MANIFEST_CHARACTERISTICS_SUMMARY_PATH = SUPPORT_ROOT / "manifest_characteristics_summary.json"
EXPECTED_NUM_DATASETS = 8192
EXPECTED_CURATED_TASK_COUNT = 12


def _assert_full_replay_training_payload(
    row: dict[str, Any],
    *,
    expected_val_batches: int | None = None,
) -> None:
    training = row["training"]
    assert training["surface_label"] == "linear_warmup_decay"

    overrides = training["overrides"]
    assert overrides["apply_schedule"] is True
    runtime = {
        "max_steps": 2500,
        "eval_every": 25,
        "checkpoint_every": 25,
        "trace_activations": False,
    }
    if expected_val_batches is not None:
        runtime["val_batches"] = expected_val_batches
    assert overrides["runtime"] == runtime
    assert overrides["optimizer"] == {
        "name": "schedulefree_adamw",
        "require_requested": True,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "min_lr": 0.0004,
        "muon_per_parameter_lr": False,
    }
    assert overrides["schedule"] == {
        "stages": [
            {
                "name": "stage1",
                "steps": 2500,
                "lr_max": 0.004,
                "lr_schedule": "linear",
                "warmup_ratio": 0.05,
            }
        ]
    }


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


def _assert_row_execution_state(
    row: dict[str, Any],
    *,
    expected_run_id: str,
) -> None:
    assert row["status"] == "completed"
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "completed"
    assert row["run_id"] == expected_run_id
    benchmark_metrics = row["benchmark_metrics"]
    assert isinstance(benchmark_metrics, dict)
    assert int(benchmark_metrics["best_step"]) > 0
    for metric_key in (
        "final_log_loss",
        "final_brier_score",
        "final_roc_auc",
        "best_log_loss",
        "best_brier_score",
        "best_roc_auc",
    ):
        assert math.isfinite(float(benchmark_metrics[metric_key]))


def _ci_like_registry_path(tmp_path: Path) -> Path:
    registry_payload = _load_json(REGISTRY_PATH)
    for run_id in (ANCHOR_RUN_ID, GENERATED_RUN_ID, CURATED_RUN_ID):
        run_payload = registry_payload["runs"][run_id]
        artifacts = run_payload["artifacts"]
        assert isinstance(artifacts, dict)
        for artifact_key, artifact_value in list(artifacts.items()):
            if not isinstance(artifact_value, str):
                continue
            artifacts[artifact_key] = f"/tmp/nonexistent-ci-artifacts/{run_id}/{artifact_key}"
    registry_copy = tmp_path / "benchmark_run_registry_v1.ci.json"
    registry_copy.write_text(json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8")
    return registry_copy


def test_tf_rd_013_data_source_contract_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "tf_rd_020_harder_dagzoo_ladder_v1"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "qass_tfcol_large_missing_validation_v1",
        "status": "completed",
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
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["upstream_reference"] == {
        "name": "TabICLv2",
        "model_source": "https://arxiv.org/abs/2602.11139",
    }
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged"
    assert sweep["anchor_context"]["model"]["stage"] == "qass_context"
    assert sweep["anchor_context"]["surface_labels"]["data"] == "anchor_manifest_default"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("TabICLv2" in note for note in notes)
    assert any("nanoTabPFN" in note for note in notes)
    assert any("issue 122" in note for note in notes)
    assert any("issue 127" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed"]

    generated_row = _row_by_ref(queue, "delta_data_manifest_root_dagzoo_generated_source")
    generated_provenance = generated_row["data"]["dagzoo_provenance"]
    assert generated_row["data"]["surface_label"] == "tf_rd_013_dagzoo_generated_source"
    assert generated_row["data"]["source"] == "manifest"
    assert generated_row["data"]["manifest_path"] == "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    _assert_full_replay_training_payload(generated_row)
    _assert_row_execution_state(generated_row, expected_run_id=GENERATED_RUN_ID)
    assert generated_provenance["commands"]
    assert generated_provenance["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert generated_provenance == materialization_summary["surfaces"]["dagzoo_generated_source"]["dagzoo_provenance"]
    assert f"issue {MULTI_INVOCATION_ISSUE_NUMBER}" in generated_row["next_action"]
    assert f"issue {FILTERING_POLICY_ISSUE_NUMBER}" in generated_row["next_action"]
    assert any("support/materialization_summary.json" in note for note in generated_row["notes"])
    assert any("issue 127" in note for note in generated_row["notes"])
    assert any("neutral" in note for note in generated_row["notes"])
    assert any("Fitness_Club" in confounder for confounder in generated_row["confounders"])

    realdata_row = _row_by_ref(queue, "delta_data_manifest_curated_realdata_comparator")
    assert realdata_row["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"
    assert realdata_row["data"]["source"] == "manifest"
    assert realdata_row["data"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet"
    )
    _assert_full_replay_training_payload(realdata_row, expected_val_batches=0)
    _assert_row_execution_state(realdata_row, expected_run_id=CURATED_RUN_ID)
    assert f"issue {MULTI_INVOCATION_ISSUE_NUMBER}" in realdata_row["next_action"]
    assert "issue 107" in realdata_row["next_action"]
    assert any("evidence-only" in note for note in realdata_row["notes"])
    assert any("materially worse than the anchor" in note for note in realdata_row["notes"])
    assert any("Fitness_Club" in confounder for confounder in realdata_row["confounders"])

    assert materialization_summary["issues"]["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert materialization_summary["issues"]["execution_issue"] == EXECUTION_ISSUE_NUMBER
    assert materialization_summary["issues"]["subsequent_filtering_policy_issue"] == FILTERING_POLICY_ISSUE_NUMBER
    assert materialization_summary["env_hints"]["dagzoo_root"] == "../dagzoo"
    assert materialization_summary["artifacts"]["generated_source_manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )
    assert materialization_summary["artifacts"]["curated_openml_baseline_manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet"
    )
    assert set(materialization_summary["surfaces"]) == {
        "dagzoo_generated_source",
        "curated_realdata_openml_baseline",
    }
    assert materialization_summary["surfaces"]["dagzoo_generated_source"]["state"] == "materialized_local_only"
    assert materialization_summary["surfaces"]["curated_realdata_openml_baseline"]["state"] == "materialized_local_only"
    curated_provenance = materialization_summary["surfaces"]["curated_realdata_openml_baseline"][
        "curated_realdata_provenance"
    ]
    assert curated_provenance["execution_issue"] == EXECUTION_ISSUE_NUMBER
    assert curated_provenance["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json"
    )
    assert curated_provenance["benchmark_bundle"]["task_count"] == EXPECTED_CURATED_TASK_COUNT
    assert curated_provenance["approved_external_augmentations"] == []
    assert curated_provenance["split_policy"] == {
        "name": "deterministic_holdout",
        "test_size": 0.2,
        "seed": 0,
        "preferred_mode": "stratified",
        "fallback_mode": "unstratified_fallback",
    }
    assert len(curated_provenance["task_ids"]) == EXPECTED_CURATED_TASK_COUNT
    assert len(curated_provenance["task_summaries"]) == EXPECTED_CURATED_TASK_COUNT

    steps = {step["step_id"]: step for step in materialization_summary["steps"]}
    assert set(steps) == {
        "dagzoo_generate",
        "build_manifest_generated_source",
        "prepare_openml_baseline_shards",
        "build_manifest_curated_realdata_openml_baseline",
    }
    assert steps["dagzoo_generate"]["status"] == "completed"
    assert steps["build_manifest_generated_source"]["status"] == "completed"
    assert steps["prepare_openml_baseline_shards"]["status"] == "completed"
    assert steps["build_manifest_curated_realdata_openml_baseline"]["status"] == "completed"
    assert f"--num-datasets {EXPECTED_NUM_DATASETS}" in steps["dagzoo_generate"]["command"]
    assert materialization_summary["handoff"]["generate_invocation"]["config_path"] == "../dagzoo/configs/default.yaml"
    assert materialization_summary["handoff"]["generate_invocation"]["overrides"]["num_datasets"] == EXPECTED_NUM_DATASETS
    assert materialization_summary["handoff"]["summary"]["generated_datasets"] == EXPECTED_NUM_DATASETS
    assert (
        materialization_summary["handoff"]["throughput"]["generation_stage"]["generated_datasets"]
        == EXPECTED_NUM_DATASETS
    )

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS
    assert [row["status"] for row in materialized_rows] == ["completed", "completed"]
    _assert_full_replay_training_payload(materialized_rows[0])
    _assert_full_replay_training_payload(materialized_rows[1], expected_val_batches=0)

    comparisons = manifest_characteristics_summary["comparisons"]
    assert set(comparisons) == {
        "anchor_vs_generated_source",
        "anchor_vs_curated_realdata_openml_baseline",
        "dagzoo_generated_source_vs_curated_realdata_openml_baseline",
    }
    assert comparisons["anchor_vs_generated_source"]["status"] == "available"
    assert comparisons["anchor_vs_curated_realdata_openml_baseline"]["status"] == "available"
    assert comparisons["dagzoo_generated_source_vs_curated_realdata_openml_baseline"]["status"] == "available"


def test_tf_rd_013_data_source_contract_inspect_and_diff_resolve_explicit_data_surfaces() -> None:
    inspect_generated = inspect_module.inspect_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REGISTRY_PATH,
    )

    resolved_generated = inspect_generated["target"]["resolved"]["data"]
    assert resolved_generated["surface_label"] == "tf_rd_013_dagzoo_generated_source"
    assert resolved_generated["source"] == "manifest"
    assert resolved_generated["dagzoo_provenance"]["corpus_variant"] == "dagzoo_generated_source"
    assert resolved_generated["dagzoo_provenance"]["materialization_issue"] == MATERIALIZATION_ISSUE_NUMBER
    assert resolved_generated["manifest"]["manifest_path"].endswith(
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )

    inspect_realdata = inspect_module.inspect_sweep_row(
        order=2,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REGISTRY_PATH,
    )
    assert inspect_realdata["target"]["identity"]["status"] == "completed"
    assert inspect_realdata["target"]["resolved"]["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"
    assert inspect_generated["target"]["resolved"]["training"]["optimizer_name"] == "schedulefree_adamw"
    assert inspect_generated["target"]["resolved"]["training"]["apply_schedule"] is True
    assert inspect_generated["target"]["resolved"]["training"]["backend"] == "manifest"
    assert inspect_generated["target"]["resolved"]["training"]["surface_label"] == "linear_warmup_decay"

    diff_payload = diff_module.diff_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        against="anchor",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REGISTRY_PATH,
    )

    assert diff_payload["difference_count"] > 0
    differences = diff_payload["differences"]
    assert differences["resolved.data.surface_label"] == {
        "target": "tf_rd_013_dagzoo_generated_source",
        "against": "anchor_manifest_default",
    }
    assert differences["resolved.data.dagzoo_provenance"]["target"]["corpus_variant"] == "dagzoo_generated_source"
    assert differences["resolved.data.dagzoo_provenance"]["against"] is None
    assert "resolved.training.optimizer_name" not in differences
    assert "resolved.training.apply_schedule" not in differences
    assert "resolved.training.legacy_prior" not in differences
    assert "resolved.training.schedule_stages" not in differences

    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    assert "Sweep status: `completed`" in matrix
    assert GENERATED_RUN_ID in matrix
    assert CURATED_RUN_ID in matrix
    assert "issue 127" in matrix
    assert "Fitness_Club" in matrix


def test_tf_rd_013_diff_uses_originating_anchor_row_when_anchor_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    diff_payload = diff_module.diff_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        against="anchor",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=_ci_like_registry_path(tmp_path),
    )

    assert diff_payload["against"]["source"] == "originating_sweep_row"
    differences = diff_payload["differences"]
    assert "resolved.training.optimizer_name" not in differences
    assert "resolved.training.apply_schedule" not in differences
    assert "resolved.training.legacy_prior" not in differences
    assert "resolved.training.schedule_stages" not in differences


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
    assert materialization_summary["config_refs"]["benchmark_bundle"] == [
        "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json"
    ]
    assert f"--num-datasets {EXPECTED_NUM_DATASETS}" in materialization_summary["steps"][0]["command"]
    assert manifest_characteristics_summary["manifests"]["anchor_manifest_default"]["manifest_path"] == "data/manifests/default.parquet"
    assert manifest_characteristics_summary["manifests"]["dagzoo_generated_source"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet"
    )
    assert manifest_characteristics_summary["manifests"]["curated_realdata_openml_baseline"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet"
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
        "allow_missing_values": True,
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
        registry_path=REGISTRY_PATH,
        out_path=out_path,
    )

    rendered = out_path.read_text(encoding="utf-8")
    checked_in = MATRIX_PATH.read_text(encoding="utf-8")
    assert "Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`." in rendered
    assert "| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |" in rendered
    assert "nanoTabPFN" in rendered
    assert "| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |" in checked_in
