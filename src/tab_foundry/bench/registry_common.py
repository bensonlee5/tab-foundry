"""Shared helpers for bench registry-style modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast


def project_root() -> Path:
    """Return the repository root for repo-relative artifact paths."""

    return Path(__file__).resolve().parents[3]


def copy_jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return one JSON-safe deep copy of a mapping payload."""

    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))


def normalize_path_value(path: Path, *, root: Path) -> str:
    """Normalize one path to a repo-relative registry value when possible."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def resolve_registry_path_value(value: str, *, root: Path) -> Path:
    """Resolve a registry path value to an absolute path."""

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def resolve_config_path(raw_value: Any, *, root: Path) -> Path:
    """Resolve the manifest path stored in a checkpoint config."""

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise RuntimeError("checkpoint config must include a non-empty data.manifest_path")
    return resolve_registry_path_value(str(raw_value), root=root)


def load_comparison_summary(path: Path) -> dict[str, Any]:
    """Load the standard comparison-summary payload used by bench registries."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"comparison summary must be a JSON object: {path}")
    benchmark_bundle = payload.get("benchmark_bundle")
    tab_foundry = payload.get("tab_foundry")
    if not isinstance(benchmark_bundle, dict):
        raise RuntimeError(f"comparison summary missing benchmark_bundle: {path}")
    if not isinstance(tab_foundry, dict):
        raise RuntimeError(f"comparison summary missing tab_foundry section: {path}")
    return cast(dict[str, Any], payload)
