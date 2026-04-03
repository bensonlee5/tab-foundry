"""CLI wiring for `tab-foundry research sweep promote`."""

from __future__ import annotations

import sys

import click

from tab_foundry.research.sweep.artifacts import PromotionPaths
from tab_foundry.research.sweep.promote import promote_anchor, resolve_run_id_for_order
from tab_foundry.cli.click_utils import run_click_command


def _promote_command(*, sweep_id: str, run_id: str | None, order: int | None) -> int:
    paths = PromotionPaths.default()
    if run_id is None and order is None:
        raise click.UsageError("exactly one of --run-id or --order must be provided")
    if run_id is None:
        assert order is not None
        run_id = resolve_run_id_for_order(sweep_id=sweep_id, order=order, paths=paths)
    result = promote_anchor(
        sweep_id=sweep_id,
        anchor_run_id=run_id,
        paths=paths,
    )
    print(
        "Promotion complete.",
        f"sweep_id={result['sweep_id']}",
        f"anchor_run_id={result['anchor_run_id']}",
        flush=True,
    )
    return 0


@click.command(name="promote", help="Promote a completed run to the sweep anchor")
@click.option("--sweep-id", required=True, help="Sweep id whose anchor should be updated")
@click.option("--run-id", default=None, help="Benchmark registry run id to promote")
@click.option("--order", default=None, type=int, help="Queue order whose run_id should be promoted")
def COMMAND(sweep_id: str, run_id: str | None, order: int | None) -> int:
    if (run_id is None) == (order is None):
        raise click.UsageError("exactly one of --run-id or --order must be provided")
    return _promote_command(sweep_id=sweep_id, run_id=run_id, order=order)


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep promote")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
