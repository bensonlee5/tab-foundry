"""CLI wiring for `tab-foundry bench smoke iris`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tab_foundry.bench.iris_smoke as iris_smoke_module


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default=iris_smoke_module.DEFAULT_DEVICE,
        choices=("cpu", "cuda", "mps", "auto"),
        help="Training and evaluation device",
    )
    parser.add_argument("--seed", type=int, default=iris_smoke_module.DEFAULT_SEED, help="Shared run seed")
    parser.add_argument(
        "--initial-num-tasks",
        type=int,
        default=iris_smoke_module.DEFAULT_INITIAL_NUM_TASKS,
        help="Initial number of derived Iris tasks to materialize",
    )
    parser.add_argument(
        "--max-num-tasks",
        type=int,
        default=iris_smoke_module.DEFAULT_MAX_NUM_TASKS,
        help="Maximum number of derived Iris tasks to materialize",
    )
    parser.add_argument(
        "--iris-benchmark-seeds",
        type=int,
        default=iris_smoke_module.DEFAULT_IRIS_BENCHMARK_SEEDS,
        help="Number of binary Iris benchmark splits for the final checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=iris_smoke_module.DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint snapshot cadence in steps",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Iris-backed tab-foundry smoke harness")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    out_root = iris_smoke_module._default_out_root() if args.out_root is None else Path(str(args.out_root))
    telemetry = iris_smoke_module.run_iris_smoke(
        iris_smoke_module.IrisSmokeConfig(
            out_root=out_root,
            device=str(args.device),
            seed=int(args.seed),
            initial_num_tasks=int(args.initial_num_tasks),
            max_num_tasks=int(args.max_num_tasks),
            iris_benchmark_seeds=int(args.iris_benchmark_seeds),
            checkpoint_every=int(args.checkpoint_every),
        )
    )
    print("iris smoke complete:")
    print(f"  out_root={out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  iris_benchmark_means={telemetry['iris_benchmark']['means']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
