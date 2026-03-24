#!/usr/bin/env python3
"""Shared worktree-aware tool-root resolution for audit scripts and hooks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ToolRoots:
    worktree_root: Path
    primary_root: Path
    tool_root: Path
    source: str

    @property
    def venv_bin(self) -> Path:
        return self.tool_root / ".venv" / "bin"

    @property
    def python_path(self) -> Path:
        return self.venv_bin / "python"


def _git_rev_parse(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "git rev-parse failed"
        raise RuntimeError(message)
    return completed.stdout.strip()


def resolve_repo_roots(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    env_map = os.environ if env is None else env

    raw_worktree_root = env_map.get("TAB_FOUNDRY_WORKTREE_ROOT")
    if raw_worktree_root:
        worktree_root = Path(raw_worktree_root).expanduser().resolve()
    else:
        worktree_root = Path(_git_rev_parse("--show-toplevel", cwd=cwd)).resolve()

    raw_primary_root = env_map.get("TAB_FOUNDRY_PRIMARY_ROOT")
    if raw_primary_root:
        primary_root = Path(raw_primary_root).expanduser().resolve()
    else:
        common_git_dir = Path(_git_rev_parse("--git-common-dir", cwd=worktree_root)).resolve()
        primary_root = common_git_dir.parent.resolve()

    return worktree_root, primary_root


def _python_path(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def _has_venv_python(root: Path) -> bool:
    return _python_path(root).is_file()


def resolve_tool_roots(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ToolRoots:
    worktree_root, primary_root = resolve_repo_roots(cwd=cwd, env=env)
    candidates: list[tuple[str, Path]] = [("worktree", worktree_root)]
    if primary_root != worktree_root:
        candidates.append(("primary", primary_root))

    for source, root in candidates:
        if _has_venv_python(root):
            return ToolRoots(
                worktree_root=worktree_root,
                primary_root=primary_root,
                tool_root=root,
                source=source,
            )

    checked = ", ".join(str(_python_path(root)) for _, root in candidates)
    raise RuntimeError(
        "Missing bootstrap environment. Checked "
        f"{checked}. Run ./scripts/dev bootstrap in this worktree, or bootstrap the primary checkout."
    )


def resolve_tool_executable(tool: str, *, tool_roots: ToolRoots) -> Path:
    normalized_tool = "python" if tool == "python" else str(tool)
    executable = tool_roots.venv_bin / normalized_tool
    if not executable.is_file():
        raise RuntimeError(f"Missing {executable}.")
    return executable


def build_command(command: Sequence[str], *, tool_roots: ToolRoots) -> list[str]:
    if not command:
        raise ValueError("expected a command")
    executable = resolve_tool_executable(command[0], tool_roots=tool_roots)
    return [str(executable), *command[1:]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_parser = subparsers.add_parser("print", help="Print one resolved tool-root field")
    print_parser.add_argument(
        "--field",
        required=True,
        choices=(
            "worktree-root",
            "primary-root",
            "tool-root",
            "source",
            "python",
            "venv-bin",
        ),
    )
    return parser.parse_args(list(argv))


def _print_field(args: argparse.Namespace) -> int:
    tool_roots = resolve_tool_roots()
    value_by_field = {
        "worktree-root": tool_roots.worktree_root,
        "primary-root": tool_roots.primary_root,
        "tool-root": tool_roots.tool_root,
        "source": tool_roots.source,
        "python": tool_roots.python_path,
        "venv-bin": tool_roots.venv_bin,
    }
    print(value_by_field[args.field])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "print":
        return _print_field(args)
    raise RuntimeError(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
