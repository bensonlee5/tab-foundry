"""Shared storage helpers for JSON-backed registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_versioned_registry_payload(
    path: Path,
    *,
    allow_missing: bool,
    empty_payload: dict[str, Any],
    top_level_keys: set[str],
    schema: str,
    version: int,
    entries_key: str,
    registry_label: str,
    validate_entry_fn: Any,
    entry_label: str,
) -> dict[str, Any]:
    registry_path = path.expanduser().resolve()
    if not registry_path.exists():
        if allow_missing:
            return empty_payload
        raise RuntimeError(f"{registry_label} does not exist: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{registry_label} must be a JSON object: {registry_path}")
    actual_keys = set(payload.keys())
    if actual_keys != top_level_keys:
        raise RuntimeError(
            f"{registry_label} keys mismatch: "
            f"missing={sorted(top_level_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - top_level_keys)}"
        )
    if payload["schema"] != schema:
        raise RuntimeError(
            f"{registry_label} schema mismatch: "
            f"expected={schema!r}, actual={payload['schema']!r}"
        )
    if int(payload["version"]) != version:
        raise RuntimeError(
            f"{registry_label} version mismatch: expected={version}, actual={payload['version']}"
        )
    entries = payload[entries_key]
    if not isinstance(entries, dict):
        raise RuntimeError(f"{registry_label} {entries_key} must be an object")
    for entry_id, entry in entries.items():
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise RuntimeError(f"{registry_label} {entry_label} ids must be non-empty strings")
        validate_entry_fn(entry, **{entry_label: str(entry_id)})
    normalized_entries = {str(key): value for key, value in cast(dict[str, Any], entries).items()}
    return {
        "schema": schema,
        "version": version,
        entries_key: normalized_entries,
    }
