#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _git_rev_parse(*args: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "git rev-parse failed"
        raise RuntimeError(message)
    return completed.stdout.strip()


def resolve_repo_roots() -> tuple[Path, Path]:
    worktree_root = Path(_git_rev_parse("--show-toplevel"))
    common_git_dir = Path(_git_rev_parse("--git-common-dir"))
    primary_root = common_git_dir.parent.resolve()
    return worktree_root, primary_root


def resolve_primary_venv_bin(primary_root: Path) -> Path:
    venv_bin = primary_root / ".venv" / "bin"
    if not (venv_bin / "python").is_file():
        raise RuntimeError(
            f"Missing {venv_bin / 'python'}. "
            "Bootstrap the original checkout before running pre-commit in a worktree."
        )
    return venv_bin


def build_command(command: Sequence[str], venv_bin: Path) -> list[str]:
    if not command:
        raise ValueError("expected a command")
    tool = command[0]
    executable = venv_bin / ("python" if tool == "python" else tool)
    if not executable.is_file():
        raise RuntimeError(f"Missing {executable}.")
    return [str(executable), *command[1:]]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    worktree_root, primary_root = resolve_repo_roots()
    venv_bin = resolve_primary_venv_bin(primary_root)
    command = build_command(args, venv_bin)
    env = os.environ.copy()
    env.setdefault("TAB_FOUNDRY_WORKTREE_ROOT", str(worktree_root))
    env.setdefault("TAB_FOUNDRY_PRIMARY_ROOT", str(primary_root))
    completed = subprocess.run(command, cwd=worktree_root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
