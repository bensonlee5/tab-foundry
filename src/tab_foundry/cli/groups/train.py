"""Training CLI group."""

from __future__ import annotations

import argparse

from tab_foundry.config import compose_config
import tab_foundry.cli.train_prior as train_prior_cli
from tab_foundry.training.trainer import train as run_training


def _run_training_command(args: argparse.Namespace) -> int:
    cfg = compose_config(args.overrides)
    result = run_training(cfg)
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"best={result.best_checkpoint}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0

def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Training workflows")
    nested = parser.add_subparsers(dest="train_command", required=True)

    run_parser = nested.add_parser("run", help="Train from Hydra config")
    run_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    run_parser.set_defaults(func=_run_training_command)

    prior_parser = nested.add_parser("legacy-prior", help="Legacy exact-prior training workflows")
    prior_nested = prior_parser.add_subparsers(dest="prior_command", required=True)
    prior_simple_parser = prior_nested.add_parser(
        "simple",
        help="Train the exact-prior simple benchmark family",
    )
    train_prior_cli.configure_parser(prior_simple_parser)
    prior_simple_parser.set_defaults(func=train_prior_cli.run_from_args)

    prior_staged_parser = prior_nested.add_parser(
        "staged",
        help="Train the exact-prior staged benchmark family",
    )
    train_prior_cli.configure_parser(prior_staged_parser)
    prior_staged_parser.set_defaults(func=train_prior_cli.run_staged_from_args)
