"""CLI wiring for `tab-foundry train legacy-prior` commands."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import click
from omegaconf import DictConfig

from tab_foundry.config import compose_config
from tab_foundry.training.prior_train import (
    DEFAULT_EXPERIMENT,
    train_tabfoundry_simple_prior,
)
from tab_foundry.cli.click_utils import run_click_command


_STAGED_PRIOR_EXPERIMENT = "experiment=cls_benchmark_staged_prior"


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


def _run_prior_command(*, prior_dump: Path, overrides: Sequence[str]) -> int:
    cfg = _compose_prior_cfg(overrides)
    result = train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=prior_dump,
    )
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


def _run_simple_command(*, prior_dump: Path, overrides: Sequence[str]) -> int:
    return _run_prior_command(
        prior_dump=prior_dump,
        overrides=_resolved_overrides(
            overrides,
            default_experiment=DEFAULT_EXPERIMENT,
            append_default_experiment=False,
        ),
    )


def _run_staged_command(*, prior_dump: Path, overrides: Sequence[str]) -> int:
    return _run_prior_command(
        prior_dump=prior_dump,
        overrides=_resolved_overrides(
            overrides,
            default_experiment=_STAGED_PRIOR_EXPERIMENT,
            append_default_experiment=True,
        ),
    )


@click.command(name="simple", help="Train the exact-prior simple benchmark family")
@click.option(
    "--prior-dump",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the nanoTabPFN prior dump (.h5)",
)
@click.argument("overrides", nargs=-1, type=str)
def COMMAND(prior_dump: Path, overrides: tuple[str, ...]) -> int:
    return _run_simple_command(prior_dump=prior_dump, overrides=overrides)


@click.command(name="staged", help="Train the exact-prior staged benchmark family")
@click.option(
    "--prior-dump",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the nanoTabPFN prior dump (.h5)",
)
@click.argument("overrides", nargs=-1, type=str)
def STAGED_COMMAND(prior_dump: Path, overrides: tuple[str, ...]) -> int:
    return _run_staged_command(prior_dump=prior_dump, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry train legacy-prior simple")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
