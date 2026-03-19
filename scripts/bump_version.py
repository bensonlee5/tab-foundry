#!/usr/bin/env python3
"""Bump the version in pyproject.toml and finalize the CHANGELOG.md [Unreleased] section.

Usage:
    python scripts/bump_version.py patch
    python scripts/bump_version.py minor
    python scripts/bump_version.py major
"""

from __future__ import annotations

import argparse
import re
import sys
import datetime
from pathlib import Path
import subprocess
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PYPROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\s*$", re.MULTILINE)


def parse_version(version: str) -> tuple[int, int, int]:
    match = SEMVER_RE.fullmatch(version.strip())
    if match is None:
        raise ValueError(f"Expected MAJOR.MINOR.PATCH, got {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def bump(version: str, level: str) -> str:
    major, minor, patch = parse_version(version)
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "major":
        return f"{major + 1}.0.0"
    raise ValueError(f"Unknown bump level: {level!r}")


def update_pyproject(path: Path, new_version: str) -> str:
    text = path.read_text(encoding="utf-8")
    updated, count = PYPROJECT_VERSION_RE.subn(rf"\g<1>{new_version}\3", text)
    if count != 1:
        raise RuntimeError("Could not find exactly one version = \"...\" in pyproject.toml")
    path.write_text(updated, encoding="utf-8")
    return new_version


def update_changelog(path: Path, new_version: str) -> None:
    text = path.read_text(encoding="utf-8")
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    replacement = f"## [Unreleased]\n\n## [{new_version}] - {today}"
    updated, count = UNRELEASED_RE.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not find ## [Unreleased] heading in CHANGELOG.md")
    path.write_text(updated, encoding="utf-8")


def refresh_uv_lock(repo_root: Path) -> None:
    result = subprocess.run(
        ["uv", "lock"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "`uv lock` failed"
        raise RuntimeError(f"`uv lock` failed: {detail}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=["patch", "minor", "major"], help="Bump level")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    old_version = None
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = PYPROJECT_VERSION_RE.search(text)
    if match:
        old_version = match.group(2)

    if old_version is None:
        print("Could not read current version from pyproject.toml")
        return 1

    new_version = bump(old_version, args.level)
    update_pyproject(PYPROJECT_PATH, new_version)
    update_changelog(CHANGELOG_PATH, new_version)
    refresh_uv_lock(REPO_ROOT)

    print(f"{old_version} -> {new_version}")
    print("  Updated: pyproject.toml")
    print("  Updated: CHANGELOG.md")
    print("  Refreshed: uv.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
