#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tab_foundry.bench.iris import evaluate_iris_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tab-foundry checkpoint on binary Iris")
    parser.add_argument("--checkpoint", required=True, help="Classification checkpoint path")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Inference device for the checkpointed model",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of train/test splits")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate_iris_checkpoint(
        Path(str(args.checkpoint)),
        device=str(args.device),
        seeds=int(args.seeds),
    )
    print(f"Iris evaluation for checkpoint={summary.checkpoint}")
    print(f"{'Method':<15} {'ROC AUC':>8} {'Std':>8}")
    print("-" * 33)
    for name, values in sorted(summary.results.items(), key=lambda item: -np.mean(item[1])):
        print(f"{name:<15} {np.mean(values):>8.3f} {np.std(values):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
