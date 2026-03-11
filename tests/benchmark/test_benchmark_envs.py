from __future__ import annotations

from pathlib import Path

import pytest

import tab_foundry.benchmark_envs as env_module


def test_bootstrap_benchmark_envs_creates_nanotabpfn_pyproject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nano_root = tmp_path / "nano"
    tabpfn_root = tmp_path / "tabpfn"
    tabicl_root = tmp_path / "tabicl"
    for root in (nano_root, tabpfn_root, tabicl_root):
        root.mkdir(parents=True)
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    synced: list[Path] = []
    validated: list[tuple[Path, str]] = []

    monkeypatch.setattr(env_module, "_sync_repo", lambda root: synced.append(root))
    monkeypatch.setattr(
        env_module,
        "_validate_import",
        lambda python_path, module_name: validated.append((python_path, module_name)),
    )

    summary = env_module.bootstrap_benchmark_envs(
        env_module.BenchmarkEnvConfig(
            nanotabpfn_root=nano_root,
            tabpfn_root=tabpfn_root,
            tabicl_root=tabicl_root,
        )
    )

    pyproject_path = nano_root / "pyproject.toml"
    assert pyproject_path.exists()
    assert "schedulefree" in pyproject_path.read_text(encoding="utf-8")
    assert synced == [nano_root.resolve(), tabpfn_root.resolve(), tabicl_root.resolve()]
    assert len(validated) == 6
    assert summary["nanotabpfn_python"].endswith("/nano/.venv/bin/python")
