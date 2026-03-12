"""Train CLI command."""

from __future__ import annotations

import argparse

from tab_foundry.config import compose_config
from tab_foundry.training.trainer import train


def _run(args: argparse.Namespace) -> int:
    cfg = compose_config(args.overrides)
    result = train(cfg)
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
    parser = subparsers.add_parser("train", help="Train from Hydra config")
    parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    parser.set_defaults(func=_run)

