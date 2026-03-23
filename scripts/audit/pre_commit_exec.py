#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_tool_roots_module():
    spec = importlib.util.spec_from_file_location("tool_roots_script", SCRIPT_DIR / "tool_roots.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load scripts/audit/tool_roots.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool_roots = _load_tool_roots_module()
build_command = tool_roots.build_command
resolve_tool_roots = tool_roots.resolve_tool_roots

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    resolved_tool_roots = tool_roots.resolve_tool_roots()
    command = tool_roots.build_command(args, tool_roots=resolved_tool_roots)
    env = os.environ.copy()
    env.setdefault("TAB_FOUNDRY_WORKTREE_ROOT", str(resolved_tool_roots.worktree_root))
    env.setdefault("TAB_FOUNDRY_PRIMARY_ROOT", str(resolved_tool_roots.primary_root))
    completed = subprocess.run(command, cwd=resolved_tool_roots.worktree_root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
