"""Shared queue helpers for sweep inspection and diffing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from .materialize import load_system_delta_queue_for_inspection, ordered_rows


def load_inspection_queue(
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        load_system_delta_queue_for_inspection(
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        ),
    )


def queue_metadata_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sweep_id": str(queue["sweep_id"]),
        "anchor_run_id": None if queue.get("anchor_run_id") is None else str(queue["anchor_run_id"]),
        "training_experiment": str(queue["training_experiment"]),
        "training_config_profile": str(queue["training_config_profile"]),
        "surface_role": str(queue["surface_role"]),
        "comparison_policy": str(queue["comparison_policy"]),
        "benchmark_bundle_path": str(queue["benchmark_bundle_path"]),
        "control_baseline_id": str(queue["control_baseline_id"]),
        "external_benchmarks": list(cast(list[Any], queue.get("external_benchmarks", []))),
        "canonical_sweep_path": str(queue["canonical_sweep_path"]),
        "canonical_queue_path": str(queue["canonical_queue_path"]),
        "canonical_matrix_path": str(queue["canonical_matrix_path"]),
    }


def find_row(queue: Mapping[str, Any], *, order: int) -> dict[str, Any]:
    for row in ordered_rows(queue):
        if int(row["order"]) == int(order):
            return row
    raise RuntimeError(f"unknown sweep order: {order}")


def queue_anchor_row(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    anchor_run_id = queue.get("anchor_run_id")
    if not isinstance(anchor_run_id, str) or not anchor_run_id.strip():
        return None
    for row in ordered_rows(queue):
        row_run_id = row.get("run_id")
        if isinstance(row_run_id, str) and row_run_id == anchor_run_id:
            return row
    return None
