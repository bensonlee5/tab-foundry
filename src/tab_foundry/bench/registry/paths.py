"""Shared path-resolution helpers for benchmark-facing registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tab_foundry.bench.registry_common import (
    normalize_path_value as _normalize_path_value_common,
    project_root as _project_root_common,
    resolve_config_path as _resolve_config_path_common,
    resolve_registry_path_value as _resolve_registry_path_value_common,
)


def project_root() -> Path:
    """Return the repository root for registry-style repo-relative paths."""

    return _project_root_common()


def normalize_path_value(
    path: Path,
    *,
    root_fn: Callable[[], Path] = project_root,
) -> str:
    """Normalize one absolute path into the repo-relative registry form when possible."""

    return _normalize_path_value_common(path, root=root_fn())


def resolve_registry_path_value(
    value: str,
    *,
    root_fn: Callable[[], Path] = project_root,
) -> Path:
    """Resolve one registry path value into an absolute path."""

    return _resolve_registry_path_value_common(value, root=root_fn())


def resolve_config_path(
    raw_value: Any,
    *,
    root_fn: Callable[[], Path] = project_root,
) -> Path:
    """Resolve one manifest/config path stored in checkpoint-style metadata."""

    return _resolve_config_path_common(raw_value, root=root_fn())
