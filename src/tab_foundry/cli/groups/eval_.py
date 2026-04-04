"""Evaluation CLI group."""

from __future__ import annotations

import click

from tab_foundry.config import compose_config
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.cli.click_utils import GROUP_KWARGS


def _run_checkpoint(
    *,
    checkpoint: str | None,
    split: str | None,
    overrides: tuple[str, ...],
) -> int:
    resolved_overrides = list(overrides)
    if checkpoint is not None:
        resolved_overrides.append(f"eval.checkpoint={checkpoint}")
    if split is not None:
        resolved_overrides.append(f"eval.split={split}")

    cfg = compose_config(resolved_overrides)
    result = evaluate_checkpoint(cfg)
    print("Evaluation complete:", f"checkpoint={result.checkpoint}", f"metrics={result.metrics}")
    return 0


@click.group(name="eval", help="Evaluation workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Evaluation workflows."""


@click.command(name="checkpoint", help="Evaluate checkpoint")
@click.option("--checkpoint", default=None, help="Checkpoint override")
@click.option("--split", default=None, help="Eval split override")
@click.argument("overrides", nargs=-1, type=str)
def CHECKPOINT_COMMAND(
    checkpoint: str | None,
    split: str | None,
    overrides: tuple[str, ...],
) -> int:
    return _run_checkpoint(checkpoint=checkpoint, split=split, overrides=overrides)


GROUP.add_command(CHECKPOINT_COMMAND)
