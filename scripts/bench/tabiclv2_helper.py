"""Standalone TabICLv2 benchmark helper entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


if str(_repo_src_root()) not in sys.path:
    sys.path.insert(0, str(_repo_src_root()))

import tab_foundry.bench.tabiclv2_helper as helper_module  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TabICLv2 on a manifest-backed benchmark surface")
    parser.add_argument("--tab-foundry-src", required=True, help="tab-foundry src directory for shared helpers")
    parser.add_argument("--benchmark-manifest", required=True, help="Path to a manifest-backed benchmark surface")
    parser.add_argument("--out-path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--task-type",
        required=True,
        choices=("supervised_classification", "supervised_regression"),
        help="Benchmark task type",
    )
    parser.add_argument("--checkpoint-version", required=True, help="TabICLv2 checkpoint version")
    parser.add_argument("--device", default="auto", help="Device override")
    parser.add_argument(
        "--allow-missing-values",
        action="store_true",
        help="Permit missing-valued benchmark inputs when the bundle explicitly allows them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return helper_module.run_tabiclv2_helper(
        tab_foundry_src=Path(str(args.tab_foundry_src)),
        benchmark_manifest=Path(str(args.benchmark_manifest)),
        out_path=Path(str(args.out_path)),
        task_type=str(args.task_type),
        checkpoint_version=str(args.checkpoint_version),
        device=str(args.device),
        allow_missing_values=bool(args.allow_missing_values),
    )


if __name__ == "__main__":
    raise SystemExit(main())
