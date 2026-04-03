"""Internal subprocess entrypoint for one corpus recipe materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus_materialization_recipe import materialize_corpus_recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal worker for one tab-foundry corpus recipe materialization",
    )
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--dagzoo-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--materialize-processes", type=int, required=True)
    parser.add_argument("--materialize-worker-threads", type=int, default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--sweeps-root", default=None)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    record = materialize_corpus_recipe(
        recipe_id=str(args.recipe_id),
        dagzoo_root=Path(str(args.dagzoo_root)).expanduser().resolve(),
        force=bool(args.force),
        materialize_processes=int(args.materialize_processes),
        materialize_worker_threads=(
            None
            if args.materialize_worker_threads is None
            else int(args.materialize_worker_threads)
        ),
        repo_root=Path(str(args.repo_root)).expanduser().resolve(),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        sweeps_root=(
            None
            if args.sweeps_root is None
            else Path(str(args.sweeps_root)).expanduser().resolve()
        ),
    )
    Path(str(args.result_path)).expanduser().resolve().write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
