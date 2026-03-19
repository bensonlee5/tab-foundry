"""Diff resolved surfaces for one sweep row against another target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .inspect import (
    _find_row,
    _queue_metadata_payload,
    resolve_anchor_target,
    resolve_row_target,
)
from .materialize import load_system_delta_queue
from .paths_io import (
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
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
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    row = _find_row(queue, order=int(order))
    target = resolve_row_target(queue=queue, row=row, registry_path=resolved_registry_path)

    if against_order is not None:
        baseline_row = _find_row(queue, order=int(against_order))
        baseline = resolve_row_target(queue=queue, row=baseline_row, registry_path=resolved_registry_path)
    else:
        if str(against).strip().lower() != "anchor":
            raise RuntimeError("only --against anchor or --against-order <N> are supported")
        baseline = resolve_anchor_target(queue=queue, registry_path=resolved_registry_path)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff one system-delta sweep row against anchor or another row")
    parser.add_argument("--order", type=int, required=True, help="Row order to diff")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--against",
        default="anchor",
        help="Baseline target; only 'anchor' is supported when --against-order is omitted",
    )
    parser.add_argument(
        "--against-order",
        type=int,
        default=None,
        help="Compare against another sweep row order instead of the anchor",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to the benchmark run registry",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = diff_sweep_row(
        order=int(args.order),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        against=str(args.against),
        against_order=None if args.against_order is None else int(args.against_order),
        index_path=Path(str(args.index_path)).expanduser().resolve(),
        catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
        sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
        registry_path=Path(str(args.registry_path)).expanduser().resolve(),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_sweep_diff_text(payload))
    return 0
