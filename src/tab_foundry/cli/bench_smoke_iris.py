"""CLI wiring for `tab-foundry bench smoke iris`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

import tab_foundry.bench.iris_smoke as iris_smoke_module
from tab_foundry.bench.iris_smoke.config import default_out_root
from tab_foundry.cli.click_utils import DEVICE_CHOICES, run_click_command


def _iris_smoke_command(
    *,
    out_root: Path | None,
    device: str,
    seed: int,
    initial_num_tasks: int,
    max_num_tasks: int,
    iris_benchmark_seeds: int,
    checkpoint_every: int,
) -> int:
    resolved_out_root = default_out_root() if out_root is None else out_root
    telemetry = iris_smoke_module.run_iris_smoke(
        iris_smoke_module.IrisSmokeConfig(
            out_root=resolved_out_root,
            device=device,
            seed=seed,
            initial_num_tasks=initial_num_tasks,
            max_num_tasks=max_num_tasks,
            iris_benchmark_seeds=iris_benchmark_seeds,
            checkpoint_every=checkpoint_every,
        )
    )
    print("iris smoke complete:")
    print(f"  out_root={resolved_out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  iris_benchmark_means={telemetry['iris_benchmark']['means']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0


@click.command(name="iris", help="Run the Iris smoke harness")
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Output directory root")
@click.option(
    "--device",
    default=iris_smoke_module.DEFAULT_DEVICE,
    show_default=True,
    type=click.Choice(DEVICE_CHOICES),
    help="Training and evaluation device",
)
@click.option("--seed", default=iris_smoke_module.DEFAULT_SEED, show_default=True, type=int, help="Shared run seed")
@click.option(
    "--initial-num-tasks",
    default=iris_smoke_module.DEFAULT_INITIAL_NUM_TASKS,
    show_default=True,
    type=int,
    help="Initial number of derived Iris tasks to materialize",
)
@click.option(
    "--max-num-tasks",
    default=iris_smoke_module.DEFAULT_MAX_NUM_TASKS,
    show_default=True,
    type=int,
    help="Maximum number of derived Iris tasks to materialize",
)
@click.option(
    "--iris-benchmark-seeds",
    default=iris_smoke_module.DEFAULT_IRIS_BENCHMARK_SEEDS,
    show_default=True,
    type=int,
    help="Number of binary Iris benchmark splits for the final checkpoint",
)
@click.option(
    "--checkpoint-every",
    default=iris_smoke_module.DEFAULT_CHECKPOINT_EVERY,
    show_default=True,
    type=int,
    help="Checkpoint snapshot cadence in steps",
)
def COMMAND(
    out_root: Path | None,
    device: str,
    seed: int,
    initial_num_tasks: int,
    max_num_tasks: int,
    iris_benchmark_seeds: int,
    checkpoint_every: int,
) -> int:
    return _iris_smoke_command(
        out_root=out_root,
        device=device,
        seed=seed,
        initial_num_tasks=initial_num_tasks,
        max_num_tasks=max_num_tasks,
        iris_benchmark_seeds=iris_benchmark_seeds,
        checkpoint_every=checkpoint_every,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench smoke iris")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
