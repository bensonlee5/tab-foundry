"""Sweep row synchronization helpers."""

from __future__ import annotations

from typing import Any, cast

from .artifacts import ExecutionPaths
from .queue_loading import load_system_delta_queue
from .matrix import render_and_write_system_delta_matrix


def materialized_row_map(*, sweep_id: str, paths: ExecutionPaths) -> dict[str, dict[str, Any]]:
    materialized = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["order"]): row for row in rows}


def sync_sweep_matrix(*, sweep_id: str, paths: ExecutionPaths) -> None:
    _ = render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
