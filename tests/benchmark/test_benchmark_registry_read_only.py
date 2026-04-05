from __future__ import annotations

import json
from pathlib import Path

import pytest

import tab_foundry.benchmark_registry as benchmark_registry


def _write_registry(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": benchmark_registry.REGISTRY_SCHEMA,
        "version": benchmark_registry.REGISTRY_VERSION,
        "runs": {
            "run_001": {
                "run_id": "run_001",
                "artifacts": {"run_dir": "outputs/run_001/train"},
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_load_benchmark_run_entry_reads_one_registry_entry(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path / "benchmark_run_registry_v1.json")

    entry = benchmark_registry.load_benchmark_run_entry("run_001", path=registry_path)

    assert entry["run_id"] == "run_001"
    assert entry["artifacts"]["run_dir"] == "outputs/run_001/train"


def test_benchmark_registry_helpers_roundtrip_with_explicit_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_path = (repo_root / "outputs" / "run_001" / "train").resolve()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark_registry, "repo_root", lambda: repo_root)

        assert benchmark_registry.default_benchmark_run_registry_path() == (
            repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
        )
        normalized = benchmark_registry.normalize_registry_path_value(artifact_path, root=repo_root)
        assert normalized == "outputs/run_001/train"
        assert benchmark_registry.resolve_registry_path_value(normalized, root=repo_root) == artifact_path


def test_benchmark_registry_helpers_roundtrip_known_sibling_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "tab-foundry"
    sibling_path = (tmp_path / "nanoTabPFN" / "300k_150x5_2.h5").resolve()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark_registry, "repo_root", lambda: repo_root)

        normalized = benchmark_registry.normalize_registry_path_value(sibling_path, root=repo_root)

    assert normalized == "../nanoTabPFN/300k_150x5_2.h5"
    assert benchmark_registry.resolve_registry_path_value(normalized, root=repo_root) == sibling_path
