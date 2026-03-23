"""Training CLI group."""

from __future__ import annotations

import argparse

import tab_foundry.bench.prior_train as prior_train_module
from tab_foundry.config import compose_config
from tab_foundry.training.trainer import train as run_training


_STAGED_PRIOR_EXPERIMENT = "experiment=cls_benchmark_staged_prior"


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


def _run_prior_staged(args: argparse.Namespace) -> int:
    overrides = [str(value) for value in args.overrides]
    if not any(value.startswith("experiment=") for value in overrides):
        overrides.append(_STAGED_PRIOR_EXPERIMENT)
    staged_args = argparse.Namespace(**vars(args))
    staged_args.overrides = overrides
    return prior_train_module.run_from_args(staged_args)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Training workflows")
    nested = parser.add_subparsers(dest="train_command", required=True)

    run_parser = nested.add_parser("run", help="Train from Hydra config")
    run_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    run_parser.set_defaults(func=_run_training_command)

    prior_parser = nested.add_parser("prior", help="Exact-prior training workflows")
    prior_nested = prior_parser.add_subparsers(dest="prior_command", required=True)
    prior_simple_parser = prior_nested.add_parser(
        "simple",
        help="Train the exact-prior simple benchmark family",
    )
    prior_train_module.configure_parser(prior_simple_parser)
    prior_simple_parser.set_defaults(func=prior_train_module.run_from_args)

    prior_staged_parser = prior_nested.add_parser(
        "staged",
        help="Train the exact-prior staged benchmark family",
    )
    prior_train_module.configure_parser(prior_staged_parser)
    prior_staged_parser.set_defaults(func=_run_prior_staged)
