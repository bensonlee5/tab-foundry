"""CLI wiring for `tab-foundry train legacy-prior` commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from omegaconf import DictConfig

from tab_foundry.config import compose_config
from tab_foundry.training.prior_train import (
    DEFAULT_EXPERIMENT,
    DEFAULT_PRIOR_DUMP_PATH,
    train_tabfoundry_simple_prior,
)


_STAGED_PRIOR_EXPERIMENT = "experiment=cls_benchmark_staged_prior"


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prior-dump",
        default=str(DEFAULT_PRIOR_DUMP_PATH),
        help="Path to the nanoTabPFN prior dump (.h5)",
    )
    parser.add_argument("overrides", nargs="*", help="Optional Hydra override strings")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a legacy exact-prior tabfoundry classifier on the nanoTabPFN prior dump"
    )
    configure_parser(parser)
    return parser


def _resolved_overrides(
    overrides: Sequence[str],
    *,
    default_experiment: str | None,
    append_default_experiment: bool = False,
) -> list[str]:
    resolved_overrides = [str(override) for override in overrides]
    if (
        default_experiment is not None
        and not any(override.startswith("experiment=") for override in resolved_overrides)
    ):
        if append_default_experiment:
            resolved_overrides.append(str(default_experiment))
        else:
            resolved_overrides.insert(0, str(default_experiment))
    return resolved_overrides


def _compose_prior_cfg(overrides: Sequence[str]) -> DictConfig:
    return compose_config([str(override) for override in overrides])


def _run_prior_command(args: argparse.Namespace, *, overrides: Sequence[str]) -> int:
    cfg = _compose_prior_cfg(overrides)
    result = train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=Path(str(args.prior_dump)),
    )
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


def run_from_args(args: argparse.Namespace) -> int:
    return _run_prior_command(
        args,
        overrides=_resolved_overrides(
            args.overrides,
            default_experiment=DEFAULT_EXPERIMENT,
            append_default_experiment=False,
        ),
    )


def run_staged_from_args(args: argparse.Namespace) -> int:
    staged_args = argparse.Namespace(**vars(args))
    staged_args.overrides = _resolved_overrides(
        args.overrides,
        default_experiment=_STAGED_PRIOR_EXPERIMENT,
        append_default_experiment=True,
    )
    return run_from_args(staged_args)


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
