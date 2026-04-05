from __future__ import annotations

import subprocess
from pathlib import Path

from tests.support.paths import REPO_ROOT
from tests.support.portability import find_banned_local_path_markers


def _tracked_text_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "grep", "-Il", ".", "--", "."],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        REPO_ROOT / line
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def test_tracked_text_files_do_not_embed_local_development_paths() -> None:
    offenders: list[str] = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in find_banned_local_path_markers(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)} :: {marker}")
    assert offenders == []
