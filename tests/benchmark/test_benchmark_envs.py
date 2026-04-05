from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

import tab_foundry.bench.envs as env_module
import tab_foundry.cli.bench_env_bootstrap as env_bootstrap_cli_module


def test_bootstrap_benchmark_envs_creates_nanotabpfn_pyproject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nano_root = tmp_path / "nano"
    tabpfn_root = tmp_path / "tabpfn"
    tabicl_root = tmp_path / "tabicl"
    hub_root = tmp_path / "tab-realdata-hub"
    for root in (nano_root, tabpfn_root, tabicl_root, hub_root):
        root.mkdir(parents=True)
    (hub_root / "pyproject.toml").write_text("[project]\nname='tab-realdata-hub'\n", encoding="utf-8")
    (hub_root / "src" / "tab_realdata_hub").mkdir(parents=True)
    for root in (nano_root, tabpfn_root, tabicl_root):
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    synced: list[Path] = []
    installed: list[tuple[Path, str]] = []
    validated: list[tuple[Path, str]] = []
    linked_src_roots: list[tuple[Path, Path, str]] = []
    linked_src_roots: list[tuple[Path, Path, str]] = []
    linked_src_roots: list[tuple[Path, Path, str]] = []

    monkeypatch.setattr(env_module, "_sync_repo", lambda root: synced.append(root))
    monkeypatch.setattr(env_module, "_python_version_info", lambda _python_path: (3, 14))
    monkeypatch.setattr(
        env_module,
        "_install_python_package",
        lambda python_path, package_spec: installed.append((python_path, package_spec)),
    )
    monkeypatch.setattr(
        env_module,
        "_validate_import",
        lambda python_path, module_name: validated.append((python_path, module_name)),
    )
    monkeypatch.setattr(
        env_module,
        "_install_explicit_src_root_path",
        lambda python_path, *, src_root, module_name: linked_src_roots.append(
            (python_path, src_root, module_name)
        ),
    )
    monkeypatch.setattr(
        env_module,
        "_install_explicit_src_root_path",
        lambda python_path, *, src_root, module_name: linked_src_roots.append(
            (python_path, src_root, module_name)
        ),
    )
    monkeypatch.setattr(
        env_module,
        "_install_explicit_src_root_path",
        lambda python_path, *, src_root, module_name: linked_src_roots.append(
            (python_path, src_root, module_name)
        ),
    )

    summary = env_module.bootstrap_benchmark_envs(
        env_module.BenchmarkEnvConfig(
            nanotabpfn_root=nano_root,
            tabpfn_root=tabpfn_root,
            tabicl_root=tabicl_root,
            tab_realdata_hub_root=hub_root,
        )
    )

    pyproject_path = nano_root / "pyproject.toml"
    assert pyproject_path.exists()
    assert "schedulefree" in pyproject_path.read_text(encoding="utf-8")
    assert synced == [nano_root.resolve(), tabpfn_root.resolve(), tabicl_root.resolve()]
    assert installed == [
        (
            nano_root.resolve() / ".venv" / "bin" / "python",
            str(hub_root.resolve()),
        ),
        (
            tabicl_root.resolve() / ".venv" / "bin" / "python",
            str(hub_root.resolve()),
        ),
    ]
    assert len(validated) == 10
    assert summary["nanotabpfn_python"].endswith("/nano/.venv/bin/python")


def test_tab_realdata_hub_install_spec_uses_published_package_by_default() -> None:
    assert env_module._tab_realdata_hub_install_spec() == env_module.TAB_REALDATA_HUB_INSTALL_SPEC


def test_bootstrap_benchmark_envs_uses_runtime_dependencies_for_py313_tabicl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nano_root = tmp_path / "nano"
    tabpfn_root = tmp_path / "tabpfn"
    tabicl_root = tmp_path / "tabicl"
    hub_root = tmp_path / "tab-realdata-hub"
    for root in (nano_root, tabpfn_root, tabicl_root, hub_root):
        root.mkdir(parents=True)
    (hub_root / "pyproject.toml").write_text("[project]\nname='tab-realdata-hub'\n", encoding="utf-8")
    (hub_root / "src" / "tab_realdata_hub").mkdir(parents=True)
    for root in (nano_root, tabpfn_root, tabicl_root):
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    installed: list[tuple[Path, str]] = []
    validated: list[tuple[Path, str]] = []
    linked_src_roots: list[tuple[Path, Path, str]] = []

    monkeypatch.setattr(env_module, "_sync_repo", lambda _root: None)
    monkeypatch.setattr(
        env_module,
        "_python_version_info",
        lambda python_path: (3, 13) if python_path.parent.parent.parent == tabicl_root.resolve() else (3, 14),
    )
    monkeypatch.setattr(
        env_module,
        "_install_python_package",
        lambda python_path, package_spec: installed.append((python_path, package_spec)),
    )
    monkeypatch.setattr(
        env_module,
        "_validate_import",
        lambda python_path, module_name: validated.append((python_path, module_name)),
    )
    monkeypatch.setattr(
        env_module,
        "_install_explicit_src_root_path",
        lambda python_path, *, src_root, module_name: linked_src_roots.append(
            (python_path, src_root, module_name)
        ),
    )

    env_module.bootstrap_benchmark_envs(
        env_module.BenchmarkEnvConfig(
            nanotabpfn_root=nano_root,
            tabpfn_root=tabpfn_root,
            tabicl_root=tabicl_root,
            tab_realdata_hub_root=hub_root,
        )
    )

    tabicl_python = tabicl_root.resolve() / ".venv" / "bin" / "python"
    assert installed == [
        (nano_root.resolve() / ".venv" / "bin" / "python", str(hub_root.resolve())),
        *((tabicl_python, dependency) for dependency in env_module.TAB_REALDATA_HUB_RUNTIME_DEPENDENCIES),
    ]
    assert linked_src_roots == [
        (
            tabicl_python,
            hub_root.resolve() / "src",
            "tab_realdata_hub",
        )
    ]
    assert (tabicl_python, "tab_realdata_hub") in validated


def test_bootstrap_benchmark_envs_requires_explicit_hub_root_for_py313_tabicl(
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

    monkeypatch.setattr(env_module, "_sync_repo", lambda _root: None)
    monkeypatch.setattr(env_module, "_python_version_info", lambda _python_path: (3, 13))

    with pytest.raises(RuntimeError, match="pass --tab-realdata-hub-root"):
        env_module.bootstrap_benchmark_envs(
            env_module.BenchmarkEnvConfig(
                nanotabpfn_root=nano_root,
                tabpfn_root=tabpfn_root,
                tabicl_root=tabicl_root,
            )
        )


def test_bench_env_bootstrap_cli_requires_explicit_roots() -> None:
    result = CliRunner().invoke(env_bootstrap_cli_module.COMMAND, [])

    assert result.exit_code == 2
    assert "Missing option '--nanotabpfn-root'" in result.output
