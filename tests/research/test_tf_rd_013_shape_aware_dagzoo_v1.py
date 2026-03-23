from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.inspect as inspect_module
from tab_foundry.research.sweep.core import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = default_benchmark_run_registry_path()
SWEEP_ID = "tf_rd_013_shape_aware_dagzoo_v1"
ANCHOR_RUN_ID = "sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1"
SHAPE_AWARE_ISSUE_NUMBER = 127
FILTERING_POLICY_ISSUE_NUMBER = 124
EXPECTED_ROWS = [
    "delta_data_manifest_root_dagzoo_shape_aware_multi_invocation",
    "delta_data_manifest_curated_realdata_comparator",
]
GENERATED_RUN_ID = (
    "sd_tf_rd_013_shape_aware_dagzoo_v1_01_"
    "delta_data_manifest_root_dagzoo_shape_aware_multi_invocation_v2"
)
CURATED_RUN_ID = "sd_tf_rd_013_shape_aware_dagzoo_v1_02_delta_data_manifest_curated_realdata_comparator_v1"
SUPPORT_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "support"
MATRIX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
MATERIALIZATION_SUMMARY_PATH = SUPPORT_ROOT / "materialization_summary.json"
MANIFEST_CHARACTERISTICS_SUMMARY_PATH = SUPPORT_ROOT / "manifest_characteristics_summary.json"


def _assert_training_payload(row: dict[str, Any], *, expected_val_batches: int) -> None:
    training = row["training"]
    assert training["surface_label"] == "prior_linear_warmup_decay"
    assert training["prior_dump_non_finite_policy"] == "skip"
    overrides = training["overrides"]
    assert overrides["apply_schedule"] is True
    assert overrides["runtime"] == {
        "max_steps": 2500,
        "eval_every": 25,
        "checkpoint_every": 25,
        "trace_activations": False,
        "val_batches": expected_val_batches,
    }
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


def _assert_completed_row(row: dict[str, Any], *, expected_run_id: str) -> None:
    assert row["status"] == "completed"
    assert row["run_id"] == expected_run_id
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "completed"
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


def test_tf_rd_013_shape_aware_sweep_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_013_data_source_contract_v1",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_tf_rd_013_shape_aware_metadata_rows_and_support_bundle_match() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    materialization_summary = _load_json(MATERIALIZATION_SUMMARY_PATH)
    manifest_characteristics_summary = _load_json(MANIFEST_CHARACTERISTICS_SUMMARY_PATH)

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "tf_rd_013_data_source_contract_v1"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["upstream_reference"] == {
        "name": "TabICLv2",
        "model_source": "https://arxiv.org/abs/2602.11139",
    }
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("issue 127" in note for note in notes)
    assert any("invocations" in note for note in notes)

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed", "completed"]

    generated_row = _row_by_ref(queue, EXPECTED_ROWS[0])
    _assert_completed_row(generated_row, expected_run_id=GENERATED_RUN_ID)
    _assert_training_payload(generated_row, expected_val_batches=0)
    assert generated_row["data"]["surface_label"] == "tf_rd_013_dagzoo_shape_aware_multi_invocation"
    assert generated_row["data"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
        "dagzoo_shape_aware_multi_invocation/manifest.parquet"
    )
    generated_provenance = generated_row["data"]["dagzoo_provenance"]
    assert generated_provenance["corpus_variant"] == "dagzoo_shape_aware_multi_invocation"
    assert generated_provenance["materialization_issue"] == SHAPE_AWARE_ISSUE_NUMBER
    assert generated_provenance["config_refs"] == [
        "configs/benchmark_cpu.yaml",
        "configs/default.yaml",
        "configs/benchmark_cuda_h100_large_shape.yaml",
    ]
    assert [entry["invocation_id"] for entry in generated_provenance["invocations"]] == [
        "benchmark_cpu",
        "default_medium",
        "large_shape",
    ]
    assert any("issue 127" in note for note in generated_row["notes"])
    assert f"Issue {FILTERING_POLICY_ISSUE_NUMBER}" in "\n".join(generated_row["notes"])
    assert "issue 127" in generated_row["next_action"]
    assert "issue 96" in generated_row["next_action"]
    assert "issue 107" in generated_row["next_action"]
    assert any("Fitness_Club" in confounder for confounder in generated_row["confounders"])

    curated_row = _row_by_ref(queue, EXPECTED_ROWS[1])
    _assert_completed_row(curated_row, expected_run_id=CURATED_RUN_ID)
    _assert_training_payload(curated_row, expected_val_batches=0)
    assert curated_row["data"]["surface_label"] == "tf_rd_013_curated_realdata_comparator"
    assert curated_row["data"]["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
        "curated_realdata/openml_baseline/manifest.parquet"
    )
    assert any("evidence-only" in note for note in curated_row["notes"])
    assert any("materially worse than the anchor" in note for note in curated_row["notes"])
    assert "issue 127" in curated_row["next_action"]
    assert "96" in curated_row["next_action"]
    assert "107" in curated_row["next_action"]
    assert any("Fitness_Club" in confounder for confounder in curated_row["confounders"])

    assert materialization_summary["issues"]["materialization_issue"] == SHAPE_AWARE_ISSUE_NUMBER
    assert materialization_summary["issues"]["execution_issue"] == SHAPE_AWARE_ISSUE_NUMBER
    assert materialization_summary["issues"]["epic_issue"] == 96
    assert materialization_summary["issues"]["downstream_training_surface_issue"] == 107
    assert materialization_summary["shape_program"]["kind"] == "config_ladder"
    assert materialization_summary["assembly"]["invocation_count"] == 3
    assert materialization_summary["assembly"]["persisted_summary"]["total_records"] > 0
    assert len(
        materialization_summary["surfaces"]["dagzoo_shape_aware_multi_invocation"]["invocation_handoffs"]
    ) == 3

    generated_manifest = manifest_characteristics_summary["manifests"]["dagzoo_shape_aware_multi_invocation"]
    assert generated_manifest["manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
        "dagzoo_shape_aware_multi_invocation/manifest.parquet"
    )
    assert generated_manifest["inspection"]["unique_source_root_count"] == 3
    assert set(manifest_characteristics_summary["comparisons"]) == {
        "anchor_vs_curated_realdata_openml_baseline",
        "anchor_vs_dagzoo_shape_aware_multi_invocation",
        "dagzoo_shape_aware_multi_invocation_vs_curated_realdata_openml_baseline",
    }

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    materialized_rows = materialized["rows"]
    assert [row["delta_id"] for row in materialized_rows] == EXPECTED_ROWS
    assert [row["status"] for row in materialized_rows] == ["completed", "completed"]


def test_tf_rd_013_shape_aware_inspect_and_diff_resolve_broader_data_surface() -> None:
    inspect_generated = inspect_module.inspect_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REGISTRY_PATH,
    )

    resolved_generated = inspect_generated["target"]["resolved"]["data"]
    assert resolved_generated["surface_label"] == "tf_rd_013_dagzoo_shape_aware_multi_invocation"
    assert resolved_generated["source"] == "manifest"
    assert resolved_generated["dagzoo_provenance"]["corpus_variant"] == "dagzoo_shape_aware_multi_invocation"
    assert len(resolved_generated["dagzoo_provenance"]["invocations"]) == 3
    assert inspect_generated["target"]["resolved"]["training"]["backend"] == "manifest"
    assert resolved_generated["manifest"]["manifest_path"].endswith(
        "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
        "dagzoo_shape_aware_multi_invocation/manifest.parquet"
    )

    diff_payload = diff_module.diff_sweep_row(
        order=1,
        sweep_id=SWEEP_ID,
        against="anchor",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=REPO_ROOT / "reference" / "system_delta_sweeps",
        registry_path=REGISTRY_PATH,
    )

    differences = diff_payload["differences"]
    assert differences["resolved.data.surface_label"] == {
        "target": "tf_rd_013_dagzoo_shape_aware_multi_invocation",
        "against": "anchor_manifest_default",
    }
    assert differences["resolved.data.dagzoo_provenance"]["target"]["corpus_variant"] == (
        "dagzoo_shape_aware_multi_invocation"
    )
    assert len(differences["resolved.data.dagzoo_provenance"]["target"]["invocations"]) == 3

    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    assert "Sweep status: `completed`" in matrix
    assert GENERATED_RUN_ID in matrix
    assert CURATED_RUN_ID in matrix
    assert "helper_failed_on_missing_bundle" in matrix
    assert "Fitness_Club" in matrix
    assert "issue 127" in matrix


def test_tf_rd_013_shape_aware_support_bundle_and_catalog_defaults_are_tracked_separately() -> None:
    materialization_summary = _load_json(MATERIALIZATION_SUMMARY_PATH)
    manifest_characteristics_summary = _load_json(MANIFEST_CHARACTERISTICS_SUMMARY_PATH)
    catalog = _load_yaml(REPO_ROOT / "reference" / "system_delta_catalog.yaml")

    assert MATERIALIZATION_SUMMARY_PATH.exists()
    assert MANIFEST_CHARACTERISTICS_SUMMARY_PATH.exists()
    assert SUPPORT_ROOT.joinpath("README.md").exists()
    assert materialization_summary["config_refs"]["dagzoo"] == [
        "configs/benchmark_cpu.yaml",
        "configs/default.yaml",
        "configs/benchmark_cuda_h100_large_shape.yaml",
    ]
    assert materialization_summary["artifacts"]["dagzoo_shape_aware_manifest_path"] == (
        "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
        "dagzoo_shape_aware_multi_invocation/manifest.parquet"
    )
    assert set(manifest_characteristics_summary["comparisons"]) == {
        "anchor_vs_curated_realdata_openml_baseline",
        "anchor_vs_dagzoo_shape_aware_multi_invocation",
        "dagzoo_shape_aware_multi_invocation_vs_curated_realdata_openml_baseline",
    }
    for path in (
        MATERIALIZATION_SUMMARY_PATH,
        MANIFEST_CHARACTERISTICS_SUMMARY_PATH,
        SUPPORT_ROOT / "README.md",
    ):
        assert "/Users/" not in path.read_text(encoding="utf-8")

    dagzoo_default = catalog["deltas"]["delta_data_manifest_root_dagzoo_shape_aware_multi_invocation"][
        "default_effective_surface"
    ]["data"]
    assert dagzoo_default == {
        "surface_label": "tf_rd_013_dagzoo_shape_aware_multi_invocation",
        "surface_overrides": {
            "source": "manifest",
            "manifest_path": (
                "outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/"
                "dagzoo_shape_aware_multi_invocation/manifest.parquet"
            ),
            "dagzoo_provenance": {
                "commands": [],
                "config_refs": [],
                "curated_root_lineage": [],
                "invocations": [],
            },
        },
    }
