"""Standalone nanoTabPFN benchmark helper entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


if str(_repo_src_root()) not in sys.path:
    sys.path.insert(0, str(_repo_src_root()))

import tab_foundry.bench.nanotabpfn_helper as helper_module  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate nanoTabPFN on cached benchmark datasets")
    parser.add_argument("--tab-foundry-src", required=True, help="tab-foundry src directory for shared helpers")
    parser.add_argument("--dataset-cache", required=True, help="Path to cached benchmark datasets (.npz)")
    parser.add_argument("--prior-dump", required=True, help="Path to nanoTabPFN prior dump (.h5)")
    parser.add_argument("--out-path", required=True, help="Output JSONL path")
    parser.add_argument("--device", default="auto", help="Device override")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluation cadence in steps")
    parser.add_argument("--seeds", type=int, default=2, help="Number of random seeds")
    parser.add_argument("--batch-size", type=int, default=32, help="Prior batch size")
    parser.add_argument("--lr", type=float, default=4.0e-3, help="Learning rate")
    parser.add_argument(
        "--allow-missing-values",
        action="store_true",
        help="Permit missing-valued benchmark inputs when the bundle explicitly allows them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return helper_module.run_nanotabpfn_helper(
        tab_foundry_src=Path(str(args.tab_foundry_src)),
        dataset_cache=Path(str(args.dataset_cache)),
        prior_dump=Path(str(args.prior_dump)),
        out_path=Path(str(args.out_path)),
        device=str(args.device),
        steps=int(args.steps),
        eval_every=int(args.eval_every),
        seeds=int(args.seeds),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        allow_missing_values=bool(args.allow_missing_values),
    )


if __name__ == "__main__":
    raise SystemExit(main())
