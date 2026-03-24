from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pre_commit_exec = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "pre_commit_exec.py",
    "pre_commit_exec_script",
)


def test_build_command_uses_primary_python_binary(tmp_path: Path) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("", encoding="utf-8")

    command = pre_commit_exec.build_command(["python", "-m", "mypy", "src"], venv_bin)

    assert command == [str(venv_bin / "python"), "-m", "mypy", "src"]


def test_build_command_uses_tool_binary_from_primary_venv(tmp_path: Path) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "ruff").write_text("", encoding="utf-8")

    command = pre_commit_exec.build_command(["ruff", "check", "src"], venv_bin)

    assert command == [str(venv_bin / "ruff"), "check", "src"]


def test_build_command_rejects_missing_binary(tmp_path: Path) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Missing"):
        pre_commit_exec.build_command(["mdformat", "README.md"], venv_bin)


def test_resolve_primary_venv_bin_requires_primary_python(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Bootstrap the original checkout"):
        pre_commit_exec.resolve_primary_venv_bin(tmp_path)

