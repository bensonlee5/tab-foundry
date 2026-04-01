from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
import yaml


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
    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (worktree_root / ".venv" / "bin").mkdir(parents=True)
    (worktree_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    roots = pre_commit_exec.resolve_tool_roots(
        env={
            "TAB_FOUNDRY_WORKTREE_ROOT": str(worktree_root),
            "TAB_FOUNDRY_PRIMARY_ROOT": str(primary_root),
        }
    )

    command = pre_commit_exec.build_command(["python", "-m", "mypy", "src"], tool_roots=roots)

    assert command == [str(worktree_root / ".venv" / "bin" / "python"), "-m", "mypy", "src"]


def test_build_command_uses_tool_binary_from_primary_venv(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (worktree_root / ".venv" / "bin").mkdir(parents=True)
    (worktree_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (worktree_root / ".venv" / "bin" / "ruff").write_text("", encoding="utf-8")
    roots = pre_commit_exec.resolve_tool_roots(
        env={
            "TAB_FOUNDRY_WORKTREE_ROOT": str(worktree_root),
            "TAB_FOUNDRY_PRIMARY_ROOT": str(primary_root),
        }
    )

    command = pre_commit_exec.build_command(["ruff", "check", "src"], tool_roots=roots)

    assert command == [str(worktree_root / ".venv" / "bin" / "ruff"), "check", "src"]


def test_build_command_rejects_missing_binary(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (worktree_root / ".venv" / "bin").mkdir(parents=True)
    (worktree_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    roots = pre_commit_exec.resolve_tool_roots(
        env={
            "TAB_FOUNDRY_WORKTREE_ROOT": str(worktree_root),
            "TAB_FOUNDRY_PRIMARY_ROOT": str(primary_root),
        }
    )

    with pytest.raises(RuntimeError, match="Missing"):
        pre_commit_exec.build_command(["missing-tool", "README.md"], tool_roots=roots)


def test_resolve_tool_roots_prefers_worktree_python(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (worktree_root / ".venv" / "bin").mkdir(parents=True)
    (primary_root / ".venv" / "bin").mkdir(parents=True)
    (worktree_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (primary_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")

    roots = pre_commit_exec.resolve_tool_roots(
        env={
            "TAB_FOUNDRY_WORKTREE_ROOT": str(worktree_root),
            "TAB_FOUNDRY_PRIMARY_ROOT": str(primary_root),
        }
    )

    assert roots.source == "worktree"
    assert roots.tool_root == worktree_root


def test_resolve_tool_roots_falls_back_to_primary_python(tmp_path: Path) -> None:
    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (primary_root / ".venv" / "bin").mkdir(parents=True)
    (primary_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")

    roots = pre_commit_exec.resolve_tool_roots(
        env={
            "TAB_FOUNDRY_WORKTREE_ROOT": str(worktree_root),
            "TAB_FOUNDRY_PRIMARY_ROOT": str(primary_root),
        }
    )

    assert roots.source == "primary"
    assert roots.tool_root == primary_root


def test_resolve_tool_roots_requires_bootstrapped_candidate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing bootstrap environment"):
        pre_commit_exec.resolve_tool_roots(
            env={
                "TAB_FOUNDRY_WORKTREE_ROOT": str(tmp_path / "worktree"),
                "TAB_FOUNDRY_PRIMARY_ROOT": str(tmp_path / "primary"),
            }
        )


def test_verify_paths_hook_entry_executes_dev_verify_through_pre_commit_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = payload["repos"][0]["hooks"]
    verify_hook = next(hook for hook in hooks if hook["id"] == "verify-paths")
    entry_argv = shlex.split(str(verify_hook["entry"]))
    script_index = entry_argv.index("scripts/audit/pre_commit_exec.py")
    forwarded_argv = entry_argv[script_index + 1 :]

    worktree_root = tmp_path / "worktree"
    primary_root = tmp_path / "primary"
    (worktree_root / ".venv" / "bin").mkdir(parents=True)
    (worktree_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    monkeypatch.setenv("TAB_FOUNDRY_WORKTREE_ROOT", str(worktree_root))
    monkeypatch.setenv("TAB_FOUNDRY_PRIMARY_ROOT", str(primary_root))

    calls: list[tuple[list[str], Path, dict[str, str], bool]] = []

    def _fake_run(command, *, cwd, env, check):
        calls.append((list(command), cwd, dict(env), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pre_commit_exec.subprocess, "run", _fake_run)

    result = pre_commit_exec.main(
        [
            *forwarded_argv,
            "tests/audit/test_scripts_dev.py",
            "tests/cli/test_app.py",
            "tests/training/test_prior_train_cli.py",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    command, cwd, env, check = calls[0]
    assert command == [
        str(worktree_root / ".venv" / "bin" / "python"),
        "scripts/audit/dev_verify.py",
        "verify",
        "paths",
        "--pre-commit",
        "tests/audit/test_scripts_dev.py",
        "tests/cli/test_app.py",
        "tests/training/test_prior_train_cli.py",
    ]
    assert cwd == worktree_root
    assert env["TAB_FOUNDRY_WORKTREE_ROOT"] == str(worktree_root)
    assert env["TAB_FOUNDRY_PRIMARY_ROOT"] == str(primary_root)
    assert check is False
