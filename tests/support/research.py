from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.support.io import load_json, load_yaml, write_yaml


def load_research_yaml(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def load_research_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def write_research_yaml(path: Path, payload: dict[str, Any]) -> None:
    write_yaml(path, payload)


def row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)
