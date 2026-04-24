from __future__ import annotations

# ruff: noqa: F401

from pathlib import Path

from tab_foundry.bench.openml_benchmark.bundle import (
    benchmark_bundle_summary,
    canonical_benchmark_bundle_source_path,
    default_anchor_benchmark_summary,
    default_anchor_control_baseline_id,
    validate_default_anchor_benchmark_summary,
)
from tests.support.openml_benchmark_compare_cases import (
    test_default_benchmark_manifest_path_resolves_to_missing_wide_multiclass_bundle,
    test_explicit_benchmark_manifest_paths_accept_checked_in_legacy_and_medium_multiclass_bundles,
    test_load_benchmark_bundle_requires_full_selection,
)


def test_benchmark_bundle_summary_persists_repo_relative_source_path() -> None:
    source_path = Path(__file__).resolve().parents[2] / "src" / "tab_foundry" / "bench" / "openml_classification_medium_v1.json"
    summary = benchmark_bundle_summary(
        {
            "name": "bundle",
            "version": 1,
            "selection": {
                "new_instances": 200,
                "task_type": "supervised_classification",
                "max_features": 10,
                "max_classes": 10,
                "max_missing_pct": 20.0,
                "min_minority_class_pct": 1.0,
            },
            "task_ids": [1],
        },
        source_path=source_path,
    )

    assert summary["source_path"] == "src/tab_foundry/bench/openml_classification_medium_v1.json"


def test_canonical_benchmark_bundle_source_path_matches_foreign_checkout_repo_tracked_bundle(
    tmp_path: Path,
) -> None:
    foreign_bundle_path = tmp_path / "foreign_checkout" / "src" / "tab_foundry" / "bench" / "openml_classification_medium_v1.json"

    assert canonical_benchmark_bundle_source_path(foreign_bundle_path) == (
        "src/tab_foundry/bench/openml_classification_medium_v1.json"
    )


def test_canonical_benchmark_bundle_source_path_uses_known_sibling_relative_form() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling_bundle_path = (
        repo_root.parent
        / "tab-realdata-hub"
        / "src"
        / "tab_realdata_hub"
        / "bench"
        / "openml_classification_medium_v1.json"
    )

    assert canonical_benchmark_bundle_source_path(
        sibling_bundle_path,
        repo_root=repo_root,
    ) == "../tab-realdata-hub/src/tab_realdata_hub/bench/openml_classification_medium_v1.json"


def test_default_anchor_benchmark_summary_is_missing_wide_with_observed_missingness_contract() -> None:
    summary = default_anchor_benchmark_summary()

    assert summary["name"] == "openml_classification_missing_wide"
    assert summary["source_path"] == "src/tab_foundry/bench/openml_classification_missing_wide_v1.json"
    assert summary["task_count"] == 65
    assert summary["allow_missing_values"] is True
    assert summary["selection"]["max_features"] == 100
    assert summary["selection"]["min_missing_pct"] == 0.5
    assert default_anchor_control_baseline_id() == "cls_benchmark_linear_multiclass_missing_wide_v1"


def test_validate_default_anchor_benchmark_summary_rejects_binary_regression() -> None:
    issues = validate_default_anchor_benchmark_summary(
        {
            "name": "openml_binary_medium",
            "version": 1,
            "source_path": "src/tab_foundry/bench/openml_binary_medium_v1.json",
            "task_count": 10,
            "task_ids": [42, 3638],
            "selection": {
                "new_instances": 200,
                "task_type": "supervised_classification",
                "max_features": 10,
                "max_classes": 2,
                "max_missing_pct": 0.0,
                "min_minority_class_pct": 2.5,
            },
            "allow_missing_values": False,
            "all_tasks_no_missing": True,
        }
    )

    assert issues
    assert any("openml_classification_missing_wide" in issue for issue in issues)
