"""CLI wiring for `tab-foundry research sweep execute`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.research.sweep.execute import execute_sweep
from tab_foundry.research.sweep.row_execution import (
    ALLOWED_DECISIONS,
    DEFAULT_CONCLUSION,
    DEFAULT_DECISION,
    DEFAULT_DEVICE,
    DEFAULT_NANOTABPFN_ROOT,
)
from tab_foundry.research.sweep.runtime_env import absolute_path_without_resolving_symlinks
from tab_foundry.research.sweep.selection import parse_order_overrides

from tab_foundry.research.sweep.paths_io import repo_root as _repo_root


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sweep-id", default=None, help="Sweep id to execute; defaults to the active sweep")
    parser.add_argument(
        "--order",
        type=int,
        action="append",
        default=[],
        help="Explicit queue order to execute; repeatable",
    )
    parser.add_argument(
        "--start-order",
        type=int,
        default=None,
        help="Optional starting queue order for a contiguous range",
    )
    parser.add_argument(
        "--stop-after-order",
        type=int,
        default=None,
        help="Optional inclusive last queue order for a contiguous range",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Allow explicitly selected completed rows to run again",
    )
    parser.add_argument(
        "--promote-first-executed-row-to-anchor",
        action="store_true",
        help="Promote the first executed row to the sweep anchor after it completes",
    )
    parser.add_argument(
        "--nanotabpfn-prior-dump",
        default=None,
        help="Optional path to the nanoTabPFN prior dump",
    )
    parser.add_argument(
        "--nanotabpfn-root",
        default=str(DEFAULT_NANOTABPFN_ROOT),
        help="Path to the nanoTabPFN checkout",
    )
    parser.add_argument(
        "--reuse-nanotabpfn-only",
        action="store_true",
        help=(
            "Do not launch a fresh nanoTabPFN helper; reuse a cached curve/error when available "
            "and otherwise record a synthetic nanoTabPFN reuse-missing outcome."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Sweep execution device: cpu, cuda, or auto. Sweeps do not support mps.",
    )
    parser.add_argument(
        "--tab-foundry-python",
        default=str(_repo_root() / ".venv" / "bin" / "python"),
        help="Interpreter to expose under nanoTabPFN/.venv/bin/python",
    )
    parser.add_argument("--decision-default", default=DEFAULT_DECISION, choices=sorted(ALLOWED_DECISIONS))
    parser.add_argument(
        "--conclusion-default",
        default=DEFAULT_CONCLUSION,
        help="Default conclusion recorded for executed rows",
    )
    parser.add_argument(
        "--decision-override",
        action="append",
        default=[],
        help="Per-order override like 7=keep",
    )
    parser.add_argument(
        "--conclusion-override",
        action="append",
        default=[],
        help="Per-order override like 7=Promote this surface.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser(description="Execute system-delta sweep rows"))


def run_from_args(args: argparse.Namespace) -> int:
    prior_dump = None
    if args.nanotabpfn_prior_dump is not None:
        prior_dump = Path(str(args.nanotabpfn_prior_dump)).expanduser().resolve()
    nanotabpfn_root = Path(str(args.nanotabpfn_root)).expanduser().resolve()
    fallback_python = absolute_path_without_resolving_symlinks(Path(str(args.tab_foundry_python)))
    if prior_dump is not None and not prior_dump.exists():
        raise RuntimeError(f"prior dump does not exist: {prior_dump}")
    if not fallback_python.exists():
        raise RuntimeError(f"tab-foundry interpreter does not exist: {fallback_python}")

    decision_overrides = parse_order_overrides(
        list(args.decision_override),
        arg_name="--decision-override",
    )
    conclusion_overrides = parse_order_overrides(
        list(args.conclusion_override),
        arg_name="--conclusion-override",
    )
    for decision in decision_overrides.values():
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")

    executed = execute_sweep(
        sweep_id=(None if args.sweep_id is None else str(args.sweep_id)),
        prior_dump=prior_dump,
        nanotabpfn_root=nanotabpfn_root,
        reuse_nanotabpfn_only=bool(args.reuse_nanotabpfn_only),
        device=str(args.device),
        fallback_python=fallback_python,
        orders=list(args.order),
        start_order=(None if args.start_order is None else int(args.start_order)),
        stop_after_order=(None if args.stop_after_order is None else int(args.stop_after_order)),
        include_completed=bool(args.include_completed),
        decision_default=str(args.decision_default),
        conclusion_default=str(args.conclusion_default),
        decision_overrides=decision_overrides,
        conclusion_overrides=conclusion_overrides,
        promote_first_executed_row_to_anchor=bool(args.promote_first_executed_row_to_anchor),
    )
    target_sweep = "active" if args.sweep_id is None else str(args.sweep_id)
    print(
        "Queue execution complete.",
        f"sweep_id={target_sweep}",
        f"executed_rows={len(executed)}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
