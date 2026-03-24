"""Dependency-light read-only helpers for the benchmark run registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from .repo_paths import normalize_repo_relative_path, repo_root, resolve_repo_relative_path


REGISTRY_SCHEMA = "tab-foundry-benchmark-runs-v1"
REGISTRY_VERSION = 1
_TOP_LEVEL_KEYS = {"schema", "version", "runs"}


def default_benchmark_run_registry_path() -> Path:
    """Return the repo-tracked benchmark-run registry path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"


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


def _validate_run_entry(entry: Any, *, run_id: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"benchmark run entry {run_id!r} must be a mapping")
    entry_payload = {str(key): value for key, value in entry.items()}
    actual_run_id = entry_payload.get("run_id")
    if not isinstance(actual_run_id, str) or not actual_run_id.strip():
        raise RuntimeError(f"benchmark run entry {run_id!r} must include a non-empty run_id")
    if str(actual_run_id) != run_id:
        raise RuntimeError(
            "benchmark run entry run_id mismatch: "
            f"expected={run_id!r}, actual={actual_run_id!r}"
        )
    return entry_payload


def load_benchmark_run_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the benchmark run registry."""

    registry_path = (path or default_benchmark_run_registry_path()).expanduser().resolve()
    if not registry_path.exists():
        raise RuntimeError(f"benchmark run registry does not exist: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"benchmark run registry must be a JSON object: {registry_path}")

    actual_keys = set(payload.keys())
    if actual_keys != _TOP_LEVEL_KEYS:
        raise RuntimeError(
            "benchmark run registry keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}"
        )
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError(
            "benchmark run registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload.get('schema')!r}"
        )
    if int(payload.get("version", -1)) != REGISTRY_VERSION:
        raise RuntimeError(
            "benchmark run registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload.get('version')!r}"
        )

    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        raise RuntimeError("benchmark run registry runs must be an object")
    normalized_runs: dict[str, Any] = {}
    for run_id, entry in runs.items():
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError("benchmark run registry run ids must be non-empty strings")
        normalized_runs[str(run_id)] = _validate_run_entry(entry, run_id=str(run_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": cast(dict[str, Any], normalized_runs),
    }


def load_benchmark_run_entry(
    run_id: str,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load one minimally validated benchmark-run registry entry."""

    registry = load_benchmark_run_registry(path)
    runs = cast(dict[str, Any], registry["runs"])
    try:
        entry = runs[str(run_id)]
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark registry run_id: {run_id!r}") from exc
    if not isinstance(entry, dict):
        raise RuntimeError(f"benchmark run entry {run_id!r} must be a mapping")
    return cast(dict[str, Any], entry)
