"""Standalone Iris benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


if str(_repo_src_root()) not in sys.path:
    sys.path.insert(0, str(_repo_src_root()))

import tab_foundry.bench.iris as iris_module  # noqa: E402


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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = iris_module.evaluate_iris_checkpoint(
        Path(str(args.checkpoint)),
        device=str(args.device),
        seeds=int(args.seeds),
    )
    print(iris_module.render_iris_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
