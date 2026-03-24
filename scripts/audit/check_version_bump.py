#!/usr/bin/env python3
"""Validate that pyproject.toml only bumps one semantic-version step beyond main."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

VersionTuple = tuple[int, int, int]


def parse_version(version: str) -> VersionTuple:
    match = SEMVER_RE.fullmatch(version.strip())
    if match is None:
        raise ValueError(f"Expected MAJOR.MINOR.PATCH version, got {version!r}")
    return tuple(int(part) for part in match.groups())


def format_version(version: VersionTuple) -> str:
    return ".".join(str(part) for part in version)


def load_version_from_pyproject_text(pyproject_text: str) -> str:
    payload = tomllib.loads(pyproject_text)
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing a [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml is missing project.version")
    return version.strip()


def read_version_from_pyproject_path(pyproject_path: Path) -> str:
    return load_version_from_pyproject_text(pyproject_path.read_text(encoding="utf-8"))


def read_version_from_git_ref(
    repo_root: Path,
    *,
    ref: str,
    pyproject_path: str = "pyproject.toml",
) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{pyproject_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git show failed"
        raise RuntimeError(f"Unable to read {pyproject_path} from git ref {ref!r}: {detail}")
    return load_version_from_pyproject_text(result.stdout)


def allowed_version_bumps(version: str) -> tuple[VersionTuple, VersionTuple, VersionTuple]:
    major, minor, patch = parse_version(version)
    return (
        (major, minor, patch + 1),
        (major, minor + 1, 0),
        (major + 1, 0, 0),
    )


def validate_version_bump(base_version: str, candidate_version: str) -> str | None:
    parsed_base = parse_version(base_version)
    parsed_candidate = parse_version(candidate_version)
    if parsed_candidate == parsed_base:
        return None

    allowed = allowed_version_bumps(base_version)
    if parsed_candidate in allowed:
        return None

    allowed_versions = ", ".join([base_version, *(format_version(version) for version in allowed)])
    if parsed_candidate < parsed_base:
        return (
            f"project.version {candidate_version} must not be lower than {base_version}; "
            f"allowed versions: {allowed_versions}"
        )
    return (
        f"project.version {candidate_version} must be unchanged or exactly one patch, minor, "
        f"or major step ahead of {base_version}; allowed versions: {allowed_versions}"
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Git ref used as the version baseline.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Repo-relative pyproject file to validate.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    pyproject_path = (REPO_ROOT / args.pyproject).resolve()

    try:
        base_version = read_version_from_git_ref(REPO_ROOT, ref=args.base_ref, pyproject_path=args.pyproject)
        current_version = read_version_from_pyproject_path(pyproject_path)
        error = validate_version_bump(base_version, current_version)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Version bump check failed: {exc}")
        return 1

    if error is not None:
        print(f"Version bump check failed against {args.base_ref}: {error}")
        return 1

    print(
        "Version bump check passed: "
        f"current={current_version}, base_ref={args.base_ref}, base_version={base_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
