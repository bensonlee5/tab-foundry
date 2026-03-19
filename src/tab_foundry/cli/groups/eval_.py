"""Evaluation CLI group."""

from __future__ import annotations

import argparse

from tab_foundry.config import compose_config
from tab_foundry.training.evaluate import evaluate_checkpoint


def _run_checkpoint(args: argparse.Namespace) -> int:
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
    parser = subparsers.add_parser("eval", help="Evaluation workflows")
    nested = parser.add_subparsers(dest="eval_command", required=True)

    checkpoint_parser = nested.add_parser("checkpoint", help="Evaluate checkpoint")
    checkpoint_parser.add_argument("--checkpoint", default=None, help="Checkpoint override")
    checkpoint_parser.add_argument("--split", default=None, help="Eval split override")
    checkpoint_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    checkpoint_parser.set_defaults(func=_run_checkpoint)
