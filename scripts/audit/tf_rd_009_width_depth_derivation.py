#!/usr/bin/env python3
"""Recompute and verify the TF-RD-009 width-depth derivation from code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.research.tf_rd_009_width_depth_derivation import (
    derive_tf_rd_009_width_depth_family,
)
from tab_foundry.repo_paths import repo_root


REPO_ROOT = repo_root()
DEFAULT_QUEUE_PATH = (
    REPO_ROOT
    / "reference"
    / "system_delta_sweeps"
    / "tf_rd_009_width_depth_medium_v1"
    / "queue.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping in {path}")
    return payload


def verify_canonical_queue(
    *,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    registry_path: Path | None = None,
) -> tuple[bool, list[str]]:
    derivation = derive_tf_rd_009_width_depth_family(registry_path=registry_path)
    queue = _load_yaml(queue_path)
    rows = queue.get("rows")
    if not isinstance(rows, list):
        return False, [f"queue rows must be a list in {queue_path}"]
    expected_rows = [(row.d_icl, row.layers, row.delta_id) for row in derivation.queue_rows]
    observed_rows: list[tuple[int, int, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            return False, [f"queue row must be a mapping in {queue_path}"]
        model = row.get("model")
        if not isinstance(model, dict):
            return False, [f"queue row is missing model payload in {queue_path}"]
        observed_rows.append(
            (
                int(model["d_icl"]),
                int(model["sandwich_layers"]),
                str(row["delta_ref"]),
            )
        )
    if observed_rows == expected_rows:
        return True, []
    return (
        False,
        [
            "TF-RD-009 queue rows do not match the derived family:",
            f"expected={expected_rows}",
            f"observed={observed_rows}",
        ],
    )


def _text_report(registry_path: Path | None = None) -> str:
    derivation = derive_tf_rd_009_width_depth_family(registry_path=registry_path)
    lines = [
        "TF-RD-009 width-depth derivation",
        f"queue_parameter_bridge: {derivation.parameter_bridge.expression()}",
        (
            "reported_fit_policy: fit the law only on measured "
            "`model_size.total_params` from completed benchmark-backed in-family runs"
        ),
        (
            "vram_fit: "
            f"reserved_gb ≈ {derivation.vram_fit.intercept:.2f} + "
            f"{derivation.vram_fit.slope:.3e} * params"
        ),
        (
            "train_fit: "
            f"train_wall_seconds ≈ {derivation.train_fit.intercept:.2f} + "
            f"{derivation.train_fit.slope:.3e} * params"
        ),
        "derived rows:",
    ]
    for row in derivation.queue_rows:
        lines.append(
            "  - "
            f"{row.row_label}: raw_d={row.raw_d_icl:.1f}, "
            f"target_params={row.target_params:.0f}, "
            f"predicted_params={row.predicted_total_params:.0f}, "
            f"predicted_reserved_gb={row.predicted_reserved_vram_gb:.2f}, "
            f"predicted_train_wall_seconds={row.predicted_train_wall_seconds:.0f}"
        )
    return "\n".join(lines)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=default_benchmark_run_registry_path(),
        help="Path to the benchmark run registry used for the width-only evidence rows.",
    )
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="Path to the canonical TF-RD-009 queue to verify.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the derived report as JSON.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail if the checked-in TF-RD-009 queue drifts from the derived family.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    registry_path = Path(args.registry_path).expanduser().resolve()
    queue_path = Path(args.queue_path).expanduser().resolve()
    derivation = derive_tf_rd_009_width_depth_family(registry_path=registry_path)

    if args.json:
        print(json.dumps(derivation.as_dict(), indent=2, sort_keys=True))
    else:
        print(_text_report(registry_path))

    if not args.verify:
        return 0
    ok, messages = verify_canonical_queue(queue_path=queue_path, registry_path=registry_path)
    if ok:
        print(f"verification passed: {queue_path}")
        return 0
    for message in messages:
        print(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
