"""Shared UTC timestamp formatting helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    """Return one canonical repo-wide UTC timestamp string."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
