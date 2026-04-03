"""Dependency-light read-only helpers for the benchmark run registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from tab_foundry.registry.paths import (  # noqa: F401 - re-exported
    normalize_registry_path_value,
    resolve_registry_path_value,
)
from tab_foundry.registry.storage import load_json_object_payload
from tab_foundry.repo_paths import repo_root


REGISTRY_SCHEMA = "tab-foundry-benchmark-runs-v1"
REGISTRY_VERSION = 1
_TOP_LEVEL_KEYS = {"schema", "version", "runs"}


class _ReadOnlyBenchmarkRunEntryPayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    run_id: StrictStr


def default_benchmark_run_registry_path() -> Path:
    """Return the repo-tracked benchmark-run registry path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"


def _validate_run_entry(entry: Any, *, run_id: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"benchmark run entry {run_id!r} must be a mapping")
    entry_payload = {str(key): value for key, value in entry.items()}
    try:
        validated = _ReadOnlyBenchmarkRunEntryPayload.model_validate(entry_payload)
    except ValidationError as exc:
        raise RuntimeError(f"benchmark run entry {run_id!r} is invalid: {exc}") from exc
    actual_run_id = str(validated.run_id)
    if actual_run_id != run_id:
        raise RuntimeError(
            "benchmark run entry run_id mismatch: "
            f"expected={run_id!r}, actual={actual_run_id!r}"
        )
    return entry_payload


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": {},
    }


def load_benchmark_run_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the benchmark run registry."""

    registry_path = (path or default_benchmark_run_registry_path()).expanduser().resolve()
    payload = load_json_object_payload(
        registry_path,
        allow_missing=False,
        empty_payload=_empty_registry(),
        payload_label="benchmark run registry",
    )
    actual_keys = set(payload.keys())
    if actual_keys != _TOP_LEVEL_KEYS:
        raise RuntimeError(
            "benchmark run registry keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}"
        )
    if payload["schema"] != REGISTRY_SCHEMA:
        raise RuntimeError(
            "benchmark run registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload['schema']!r}"
        )
    if int(payload["version"]) != REGISTRY_VERSION:
        raise RuntimeError(
            "benchmark run registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload['version']}"
        )
    runs = payload["runs"]
    if not isinstance(runs, dict):
        raise RuntimeError("benchmark run registry runs must be an object")
    for run_id, entry in runs.items():
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError("benchmark run registry run_id ids must be non-empty strings")
        _validate_run_entry(entry, run_id=str(run_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": {str(key): value for key, value in cast(dict[str, Any], runs).items()},
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
