"""Promote a completed system-delta run to the canonical sweep anchor."""

from __future__ import annotations

from typing import Any, cast

from .anchor import anchor_context_from_registry_run
from .artifacts import PromotionPaths
from .catalog import load_system_delta_index, load_system_delta_queue_instance
from .matrix import render_and_write_system_delta_matrix
from .paths_io import load_yaml_mapping, sweep_matrix_path, sweep_metadata_path, write_yaml


def _render_sweep_matrix(*, sweep_id: str, paths: PromotionPaths) -> None:
    _ = render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )


def _read_yaml(path, *, context: str) -> dict[str, Any]:
    return load_yaml_mapping(path, context=context)


def resolve_run_id_for_order(*, sweep_id: str, order: int, paths: PromotionPaths | None = None) -> str:
    resolved_paths = PromotionPaths.default() if paths is None else paths
    queue = load_system_delta_queue_instance(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    for row in cast(list[dict[str, Any]], queue["rows"]):
        if int(row["order"]) != int(order):
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError(
                f"sweep {sweep_id!r} row {order} does not have a completed run_id to promote"
            )
        return run_id
    raise RuntimeError(f"sweep {sweep_id!r} does not contain queue order {order}")


def promote_anchor(
    *,
    sweep_id: str,
    anchor_run_id: str,
    paths: PromotionPaths | None = None,
) -> dict[str, str]:
    resolved_paths = PromotionPaths.default() if paths is None else paths
    normalized_sweep_id = str(sweep_id).strip()
    normalized_anchor_run_id = str(anchor_run_id).strip()
    if not normalized_sweep_id:
        raise RuntimeError("sweep_id must be non-empty")
    if not normalized_anchor_run_id:
        raise RuntimeError("anchor_run_id must be non-empty")

    _ = anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=resolved_paths.registry_path,
    )

    sweep_path = sweep_metadata_path(
        normalized_sweep_id,
        sweeps_root=resolved_paths.sweeps_root,
    )
    sweep = _read_yaml(sweep_path, context=f"sweep {normalized_sweep_id!r}")
    sweep["anchor_run_id"] = normalized_anchor_run_id
    sweep["anchor_context"] = anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=resolved_paths.registry_path,
    )
    write_yaml(sweep_path, sweep)

    index = load_system_delta_index(resolved_paths.index_path)
    sweeps = cast(dict[str, Any], index["sweeps"])
    if normalized_sweep_id not in sweeps:
        raise RuntimeError(f"unknown sweep_id: {normalized_sweep_id}")
    cast(dict[str, Any], sweeps[normalized_sweep_id])["anchor_run_id"] = normalized_anchor_run_id
    write_yaml(resolved_paths.index_path, index)

    _render_sweep_matrix(sweep_id=normalized_sweep_id, paths=resolved_paths)

    return {
        "sweep_id": normalized_sweep_id,
        "anchor_run_id": normalized_anchor_run_id,
        "sweep_path": str(sweep_path.resolve()),
        "index_path": str(resolved_paths.index_path.resolve()),
        "matrix_path": str(
            sweep_matrix_path(
                normalized_sweep_id,
                sweeps_root=resolved_paths.sweeps_root,
            ).resolve()
        ),
    }
