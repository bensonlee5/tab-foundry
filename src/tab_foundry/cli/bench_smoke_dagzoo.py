"""CLI wiring for `tab-foundry bench smoke dagzoo`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

import tab_foundry.bench.dagzoo_smoke as smoke_module
from tab_foundry.cli.click_utils import DEVICE_CHOICES, dagzoo_root_option, run_click_command


def _dagzoo_smoke_command(
    *,
    dagzoo_root: Path,
    out_root: Path | None,
    num_datasets: int,
    rows: int,
    device: str,
    seed: int,
    train_steps: int,
    checkpoint_every: int,
) -> int:
    resolved_out_root = smoke_module._default_out_root() if out_root is None else out_root
    telemetry = smoke_module.run_dagzoo_smoke(
        smoke_module.SmokeConfig(
            dagzoo_root=dagzoo_root,
            out_root=resolved_out_root,
            num_datasets=num_datasets,
            rows=rows,
            device=device,
            seed=seed,
            train_steps=train_steps,
            checkpoint_every=checkpoint_every,
        )
    )
    print("dagzoo smoke complete:")
    print(f"  out_root={resolved_out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0


@click.command(name="dagzoo", help="Run the dagzoo smoke harness")
@dagzoo_root_option()
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Output directory root")
@click.option(
    "--num-datasets",
    default=smoke_module.DEFAULT_NUM_DATASETS,
    show_default=True,
    type=int,
    help="Number of dagzoo datasets to generate",
)
@click.option("--rows", default=smoke_module.DEFAULT_ROWS, show_default=True, type=int, help="Rows per generated dataset")
@click.option(
    "--device",
    default=smoke_module.DEFAULT_DEVICE,
    show_default=True,
    type=click.Choice(DEVICE_CHOICES),
    help="Generation and training device",
)
@click.option("--seed", default=smoke_module.DEFAULT_SEED, show_default=True, type=int, help="Shared run seed")
@click.option(
    "--train-steps",
    default=smoke_module.DEFAULT_TRAIN_STEPS,
    show_default=True,
    type=int,
    help="Training steps for the smoke harness",
)
@click.option(
    "--checkpoint-every",
    default=smoke_module.DEFAULT_CHECKPOINT_EVERY,
    show_default=True,
    type=int,
    help="Checkpoint snapshot cadence in steps",
)
def COMMAND(
    dagzoo_root: Path,
    out_root: Path | None,
    num_datasets: int,
    rows: int,
    device: str,
    seed: int,
    train_steps: int,
    checkpoint_every: int,
) -> int:
    return _dagzoo_smoke_command(
        dagzoo_root=dagzoo_root,
        out_root=out_root,
        num_datasets=num_datasets,
        rows=rows,
        device=device,
        seed=seed,
        train_steps=train_steps,
        checkpoint_every=checkpoint_every,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench smoke dagzoo")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
