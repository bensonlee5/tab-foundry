"""Eval CLI command."""

from __future__ import annotations

import argparse

from tab_foundry.config import compose_config
from tab_foundry.training.evaluate import evaluate_checkpoint


def _run(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.checkpoint is not None:
        overrides.append(f"eval.checkpoint={args.checkpoint}")
    if args.split is not None:
        overrides.append(f"eval.split={args.split}")

    cfg = compose_config(overrides)
    result = evaluate_checkpoint(cfg)
    print("Evaluation complete:", f"checkpoint={result.checkpoint}", f"metrics={result.metrics}")
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Evaluate checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override")
    parser.add_argument("--split", default=None, help="Eval split override")
    parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    parser.set_defaults(func=_run)

