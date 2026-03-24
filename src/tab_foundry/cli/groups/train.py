"""Training CLI group."""

from __future__ import annotations

import argparse
from typing import Sequence

from tab_foundry.config import compose_config
from tab_foundry.training.trainer import train as run_training

from ..helpers import register_delegate_leaf


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


def _run_prior_simple(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.prior_train import main as prior_main

    return prior_main(argv)


def _run_prior_staged(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.prior_train import main as prior_main

    resolved_argv = [] if argv is None else list(argv)
    if not any(argument in {"-h", "--help"} for argument in resolved_argv):
        if not any(str(argument).startswith("experiment=") for argument in resolved_argv):
            resolved_argv.append(_STAGED_PRIOR_EXPERIMENT)
    return prior_main(None if not resolved_argv else resolved_argv)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Training workflows")
    nested = parser.add_subparsers(dest="train_command", required=True)

    run_parser = nested.add_parser("run", help="Train from Hydra config")
    run_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    run_parser.set_defaults(func=_run_training_command)

    prior_parser = nested.add_parser("prior", help="Exact-prior training workflows")
    prior_nested = prior_parser.add_subparsers(dest="prior_command", required=True)
    register_delegate_leaf(
        prior_nested,
        "simple",
        help="Train the exact-prior simple benchmark family",
        delegate=_run_prior_simple,
    )
    register_delegate_leaf(
        prior_nested,
        "staged",
        help="Train the exact-prior staged benchmark family",
        delegate=_run_prior_staged,
    )
