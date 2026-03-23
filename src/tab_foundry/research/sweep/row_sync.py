"""Sweep row synchronization helpers."""

from __future__ import annotations

from typing import Any, cast

from . import core as sweep_core
from .artifacts import ExecutionPaths


def materialized_row_map(*, sweep_id: str, paths: ExecutionPaths) -> dict[str, dict[str, Any]]:
    materialized = sweep_core.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["delta_id"]): row for row in rows}


def sync_sweep_matrix(*, sweep_id: str, paths: ExecutionPaths) -> None:
    _ = sweep_core.render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )


def sync_active_aliases_if_active(*, sweep_id: str, paths: ExecutionPaths) -> None:
    index = sweep_core.load_system_delta_index(paths.index_path)
    if str(index["active_sweep_id"]) != sweep_id:
        return
    _ = sweep_core.sync_active_sweep_aliases(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        registry_path=paths.registry_path,
        sweeps_root=paths.sweeps_root,
    )
