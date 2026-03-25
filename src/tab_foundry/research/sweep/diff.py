"""Diff resolved surfaces for one sweep row against another target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from .inspect import (
    _find_row,
    _queue_metadata_payload,
    resolve_anchor_target,
    resolve_row_target,
)
from .materialize import load_system_delta_queue_for_inspection
from .paths_io import (
    default_registry_path,
)


def _diff_values(
    left: Any,
    right: Any,
    *,
    prefix: str,
    differences: dict[str, dict[str, Any]],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted({*(str(key) for key in left.keys()), *(str(key) for key in right.keys())})
        for key in keys:
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _diff_values(left.get(key), right.get(key), prefix=next_prefix, differences=differences)
        return
    if left == right:
        return
    differences[prefix] = {
        "target": left,
        "against": right,
    }


def diff_sweep_row(
    *,
    order: int,
    sweep_id: str | None = None,
    against: str = "anchor",
    against_order: int | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    resolved_registry_path = registry_path or default_registry_path()
    queue = load_system_delta_queue_for_inspection(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    row = _find_row(queue, order=int(order))
    target = resolve_row_target(
        queue=queue,
        row=row,
        registry_path=resolved_registry_path,
        sweeps_root=sweeps_root,
    )

    if against_order is not None:
        baseline_row = _find_row(queue, order=int(against_order))
        baseline = resolve_row_target(
            queue=queue,
            row=baseline_row,
            registry_path=resolved_registry_path,
            sweeps_root=sweeps_root,
        )
    else:
        if str(against).strip().lower() != "anchor":
            raise RuntimeError("only --against anchor or --against-order <N> are supported")
        baseline = resolve_anchor_target(
            queue=queue,
            registry_path=resolved_registry_path,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )

    differences: dict[str, dict[str, Any]] = {}
    _diff_values(
        cast(Mapping[str, Any], target["resolved"]),
        cast(Mapping[str, Any], baseline["resolved"]),
        prefix="resolved",
        differences=differences,
    )
    _diff_values(
        target.get("metrics"),
        baseline.get("metrics"),
        prefix="metrics",
        differences=differences,
    )
    return {
        "queue": _queue_metadata_payload(queue),
        "target": target["identity"],
        "against": baseline["identity"],
        "difference_count": len(differences),
        "differences": differences,
    }


def render_sweep_diff_text(payload: Mapping[str, Any]) -> str:
    target = cast(Mapping[str, Any], payload["target"])
    against = cast(Mapping[str, Any], payload["against"])
    differences = cast(Mapping[str, Any], payload["differences"])
    lines = [
        "Sweep row diff.",
        f"target={json.dumps(dict(target), sort_keys=True)}",
        f"against={json.dumps(dict(against), sort_keys=True)}",
        f"difference_count={payload['difference_count']}",
    ]
    for path in sorted(differences):
        diff_entry = cast(Mapping[str, Any], differences[path])
        lines.append(
            f"{path}: target={json.dumps(diff_entry['target'], sort_keys=True)} "
            f"against={json.dumps(diff_entry['against'], sort_keys=True)}"
        )
    return "\n".join(lines)
