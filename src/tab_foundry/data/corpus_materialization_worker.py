"""Internal subprocess entrypoint for one corpus invocation materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from .corpus_materialization import materialize_recipe_invocation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal worker for one tab-foundry corpus invocation materialization",
    )
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--dagzoo-root", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--materialize-worker-threads", type=int, default=None)
    parser.add_argument("--initial-expected-acceptance-rate", type=float, default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--sweeps-root", default=None)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    materialize_recipe_invocation(
        recipe_id=str(args.recipe_id),
        invocation_id=str(args.invocation_id),
        dagzoo_root=Path(str(args.dagzoo_root)).expanduser().resolve(),
        corpus_root=Path(str(args.corpus_root)).expanduser().resolve(),
        materialize_worker_threads=(
            None
            if args.materialize_worker_threads is None
            else int(args.materialize_worker_threads)
        ),
        initial_expected_acceptance_rate=(
            None
            if args.initial_expected_acceptance_rate is None
            else float(args.initial_expected_acceptance_rate)
        ),
        repo_root=Path(str(args.repo_root)).expanduser().resolve(),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        sweeps_root=(
            None
            if args.sweeps_root is None
            else Path(str(args.sweeps_root)).expanduser().resolve()
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
