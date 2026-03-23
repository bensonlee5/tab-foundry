"""Manual benchmark comparison helpers against external baselines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_benchmark_{stamp}"


def _optional_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()
