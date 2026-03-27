"""CLI wiring for `tab-foundry research sweep promote`."""

from __future__ import annotations

import argparse
import sys

from tab_foundry.research.sweep.artifacts import PromotionPaths
from tab_foundry.research.sweep.promote import promote_anchor, resolve_run_id_for_order


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sweep-id", required=True, help="Sweep id whose anchor should be updated")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id", help="Benchmark registry run id to promote")
    target.add_argument("--order", type=int, help="Queue order whose run_id should be promoted")
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(
        argparse.ArgumentParser(description="Promote a completed system-delta run to the sweep anchor")
    )


def run_from_args(args: argparse.Namespace) -> int:
    paths = PromotionPaths.default()
    run_id = (
        str(args.run_id)
        if args.run_id is not None
        else resolve_run_id_for_order(sweep_id=str(args.sweep_id), order=int(args.order), paths=paths)
    )
    result = promote_anchor(
        sweep_id=str(args.sweep_id),
        anchor_run_id=run_id,
        paths=paths,
    )
    print(
        "Promotion complete.",
        f"sweep_id={result['sweep_id']}",
        f"anchor_run_id={result['anchor_run_id']}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
