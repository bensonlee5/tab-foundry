"""Shared repository-root and repo-relative path helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root for repo-local workflows."""

    return Path(__file__).resolve().parents[2]


def resolve_repo_relative_path(
    value: str,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve a repo-relative or absolute path value."""

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved_root = (root or repo_root()).expanduser().resolve()
    return (resolved_root / path).resolve()


def normalize_repo_relative_path(
    path: Path,
    *,
    root: Path | None = None,
) -> str:
    """Normalize one path to a repo-relative form when possible."""

    resolved_path = path.expanduser().resolve()
    resolved_root = (root or repo_root()).expanduser().resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)
