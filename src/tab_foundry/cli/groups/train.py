"""Training CLI group."""

from __future__ import annotations

import click

from tab_foundry.config import compose_config
import tab_foundry.cli.train_prior as train_prior_cli
from tab_foundry.training.trainer import train as run_training
from tab_foundry.cli.click_utils import GROUP_KWARGS


def _run_training_command(*, overrides: tuple[str, ...]) -> int:
    cfg = compose_config(list(overrides))
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


@click.group(name="train", help="Training workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Training workflows."""


@click.command(name="run", help="Train from Hydra config")
@click.argument("overrides", nargs=-1, type=str)
def RUN_COMMAND(overrides: tuple[str, ...]) -> int:
    return _run_training_command(overrides=overrides)


@click.group(name="legacy-prior", help="Legacy exact-prior training workflows", **GROUP_KWARGS)
def _legacy_prior_group() -> None:
    """Legacy exact-prior training workflows."""


_legacy_prior_group.add_command(train_prior_cli.COMMAND)
_legacy_prior_group.add_command(train_prior_cli.STAGED_COMMAND)


GROUP.add_command(RUN_COMMAND)
GROUP.add_command(_legacy_prior_group)
