from __future__ import annotations

from pathlib import Path

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.control_baseline_registry import default_control_baseline_registry_path
from tab_foundry.research.sweep.artifacts import ExecutionPaths, PromotionPaths
from tab_foundry.research.sweep import paths_io


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sweep_path_defaults_follow_shared_repo_root() -> None:
    assert paths_io.repo_root() == REPO_ROOT
    assert paths_io.default_catalog_path() == REPO_ROOT / "reference" / "system_delta_catalog.yaml"
    assert paths_io.default_sweeps_root() == REPO_ROOT / "reference" / "system_delta_sweeps"
    assert paths_io.default_sweep_index_path() == REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
    assert paths_io.default_registry_path() == default_benchmark_run_registry_path()


def test_render_path_keeps_repo_relative_paths_relative(tmp_path: Path) -> None:
    assert paths_io._render_path(REPO_ROOT / "reference" / "system_delta_catalog.yaml") == "reference/system_delta_catalog.yaml"
    assert paths_io._render_path(tmp_path / "outside.md") == str((tmp_path / "outside.md").resolve())


def test_execution_paths_default_uses_shared_control_baseline_registry_path() -> None:
    assert ExecutionPaths.default().control_baseline_registry_path == default_control_baseline_registry_path()


def test_execution_paths_promote_conversion_stays_within_sweep_artifacts() -> None:
    promotion_paths = ExecutionPaths.default().promotion_paths()

    assert isinstance(promotion_paths, PromotionPaths)
    assert promotion_paths.registry_path == default_benchmark_run_registry_path()
