"""Shared path helpers for registry modules."""

from __future__ import annotations

from pathlib import Path

from tab_foundry.repo_paths import normalize_repo_relative_path, resolve_repo_relative_path


def resolve_registry_path_value(
    value: str,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve one registry-stored path value."""

    return resolve_repo_relative_path(value, root=root)


def normalize_registry_path_value(
    path: Path,
    *,
    root: Path | None = None,
) -> str:
    """Normalize one absolute path into the repo-relative registry form when possible."""

    return normalize_repo_relative_path(path, root=root)
