"""CLI wiring for `tab-foundry bench smoke dagzoo`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tab_foundry.bench.dagzoo_smoke as smoke_module


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dagzoo-root", default="~/dev/dagzoo", help="Local dagzoo checkout root")
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=smoke_module.DEFAULT_NUM_DATASETS,
        help="Number of dagzoo datasets to generate",
    )
    parser.add_argument("--rows", type=int, default=smoke_module.DEFAULT_ROWS, help="Rows per generated dataset")
    parser.add_argument(
        "--device",
        default=smoke_module.DEFAULT_DEVICE,
        choices=("cpu", "cuda", "mps", "auto"),
        help="Generation and training device",
    )
    parser.add_argument("--seed", type=int, default=smoke_module.DEFAULT_SEED, help="Shared run seed")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=smoke_module.DEFAULT_TRAIN_STEPS,
        help="Training steps for the smoke harness",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=smoke_module.DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint snapshot cadence in steps",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dagzoo-backed tab-foundry smoke harness")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    out_root = smoke_module._default_out_root() if args.out_root is None else Path(str(args.out_root))
    telemetry = smoke_module.run_dagzoo_smoke(
        smoke_module.SmokeConfig(
            dagzoo_root=Path(str(args.dagzoo_root)),
            out_root=out_root,
            num_datasets=int(args.num_datasets),
            rows=int(args.rows),
            device=str(args.device),
            seed=int(args.seed),
            train_steps=int(args.train_steps),
            checkpoint_every=int(args.checkpoint_every),
        )
    )
    print("dagzoo smoke complete:")
    print(f"  out_root={out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
