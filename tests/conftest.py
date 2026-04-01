"""Repo-wide pytest configuration and shared defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from hypothesis import settings
import tab_foundry.research.sweep.anchor as anchor_module


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_MANIFEST_TASK_COUNTS = {
    "openml_classification_small_v1": 5,
    "openml_classification_medium_v1": 10,
    "openml_classification_large_v1": 10,
}


def _configure_hypothesis_profiles() -> None:
    settings.register_profile("ci", deadline=None, max_examples=25)
    settings.register_profile("extended", deadline=None, max_examples=35)
    settings.register_profile("stress", deadline=None, max_examples=40)
    settings.register_profile("tiny", deadline=None, max_examples=10)
    settings.load_profile(os.environ.get("TAB_FOUNDRY_HYPOTHESIS_PROFILE", "ci"))


_configure_hypothesis_profiles()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply repo-wide test markers from stable path-based policy."""

    for item in items:
        test_path = Path(str(item.fspath)).resolve()
        relpath = test_path.relative_to(ROOT).as_posix()

        if relpath.startswith("tests/research/"):
            item.add_marker(pytest.mark.research)
        if relpath.startswith("tests/smoke/") or relpath.startswith("tests/audit/"):
            item.add_marker(pytest.mark.integration)
        if relpath == "tests/benchmark/test_stability_ladder_h5.py":
            item.add_marker(pytest.mark.slow)
        if (
            relpath == "tests/model/test_attention_sdpa.py"
            and item.name == "test_multihead_attention_sdpa_cuda_long_row_backward_smoke"
        ):
            item.add_marker(pytest.mark.gpu)


def _fake_benchmark_manifest_surface(
    benchmark_manifest_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    manifest_key = benchmark_manifest_path.parent.name
    task_count = _MANIFEST_TASK_COUNTS.get(manifest_key, 1)
    bundle_name = manifest_key.removesuffix("_v1")
    allow_missing_values = manifest_key == "openml_classification_large_v1"
    task_records = [
        {
            "task": "classification",
            "metadata": {
                "benchmark_bundle": {
                    "name": bundle_name,
                    "version": 1,
                    "source_path": str(benchmark_manifest_path),
                    "task_id": task_id,
                    "allow_missing_values": allow_missing_values,
                    "selection": None,
                }
            },
        }
        for task_id in range(1, task_count + 1)
    ]
    return {}, task_records, {
        "allow_missing_values": allow_missing_values,
        "benchmark_bundle": {"name": bundle_name},
    }


@pytest.fixture(autouse=True)
def stub_research_benchmark_manifest_surface(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_path = Path(str(request.node.fspath)).resolve()
    relpath = test_path.relative_to(ROOT).as_posix()
    if not (
        relpath.startswith("tests/research/")
        or relpath.startswith("tests/support_research/")
    ):
        return
    monkeypatch.setattr(
        anchor_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path, allow_missing_values=None: _fake_benchmark_manifest_surface(
            Path(benchmark_manifest_path)
        ),
    )
