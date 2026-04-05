"""Shared repository-root and repo-relative path helpers."""

from __future__ import annotations

from pathlib import Path

KNOWN_SIBLING_WORKSPACES = (
    "dagzoo",
    "nanoTabPFN",
    "TabPFN",
    "tabicl",
    "tab-realdata-hub",
)


def repo_root() -> Path:
    """Return the repository root for repo-local workflows."""

    return Path(__file__).resolve().parents[2]


def repo_root_from_sweeps_root(sweeps_root: Path | None) -> Path | None:
    """Derive the repo root from `<repo>/reference/system_delta_sweeps`."""

    if sweeps_root is None:
        return None
    resolved_sweeps_root = sweeps_root.expanduser().resolve()
    if resolved_sweeps_root.name != "system_delta_sweeps":
        return None
    reference_root = resolved_sweeps_root.parent
    if reference_root.name != "reference":
        return None
    return reference_root.parent


def repo_root_from_catalog_path(catalog_path: Path | None) -> Path | None:
    """Derive the repo root from `<repo>/reference/system_delta_catalog.yaml`."""

    if catalog_path is None:
        return None
    resolved_catalog_path = catalog_path.expanduser().resolve()
    if resolved_catalog_path.name != "system_delta_catalog.yaml":
        return None
    reference_root = resolved_catalog_path.parent
    if reference_root.name != "reference":
        return None
    return reference_root.parent


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
    """Normalize one path to a repo- or sibling-relative form when possible."""

    resolved_path = path.expanduser().resolve()
    resolved_root = (root or repo_root()).expanduser().resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        pass
    sibling_parent = resolved_root.parent
    for sibling_name in KNOWN_SIBLING_WORKSPACES:
        sibling_root = (sibling_parent / sibling_name).resolve()
        try:
            relative_suffix = resolved_path.relative_to(sibling_root)
        except ValueError:
            continue
        if str(relative_suffix) == ".":
            return str(Path("..") / sibling_name)
        return str(Path("..") / sibling_name / relative_suffix)
    return str(resolved_path)
