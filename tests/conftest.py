"""Repo-wide pytest configuration and shared defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from hypothesis import settings


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
