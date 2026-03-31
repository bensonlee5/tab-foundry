from __future__ import annotations

# ruff: noqa: F401

from pathlib import Path

from tab_foundry.bench.openml_benchmark.bundle import (
    benchmark_bundle_summary,
    canonical_benchmark_bundle_source_path,
)
from tests.support.openml_benchmark_compare_cases import (
    test_default_benchmark_manifest_path_resolves_to_medium_binary_bundle,
    test_explicit_benchmark_manifest_paths_accept_checked_in_legacy_and_medium_binary_bundles,
    test_load_benchmark_bundle_requires_full_selection,
)


def test_benchmark_bundle_summary_persists_repo_relative_source_path() -> None:
    source_path = Path(__file__).resolve().parents[2] / "src" / "tab_foundry" / "bench" / "openml_binary_medium_v1.json"
    summary = benchmark_bundle_summary(
        {
            "name": "bundle",
            "version": 1,
            "selection": {
                "new_instances": 200,
                "task_type": "supervised_classification",
                "max_features": 10,
                "max_classes": 2,
                "max_missing_pct": 0.0,
                "min_minority_class_pct": 2.5,
            },
            "task_ids": [1],
        },
        source_path=source_path,
    )

    assert summary["source_path"] == "src/tab_foundry/bench/openml_binary_medium_v1.json"


def test_canonical_benchmark_bundle_source_path_matches_foreign_checkout_repo_tracked_bundle(
    tmp_path: Path,
) -> None:
    foreign_bundle_path = tmp_path / "foreign_checkout" / "src" / "tab_foundry" / "bench" / "openml_binary_medium_v1.json"

    assert canonical_benchmark_bundle_source_path(foreign_bundle_path) == (
        "src/tab_foundry/bench/openml_binary_medium_v1.json"
    )
