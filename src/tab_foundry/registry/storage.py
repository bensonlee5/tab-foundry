"""Shared storage helpers for JSON-backed registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_json_object_payload(
    path: Path,
    *,
    allow_missing: bool,
    empty_payload: dict[str, Any],
    payload_label: str,
) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        if allow_missing:
            return empty_payload
        raise RuntimeError(f"{payload_label} does not exist: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{payload_label} must be a JSON object: {resolved_path}")
    return cast(dict[str, Any], payload)
