"""CLI wiring for `tab-foundry bench tune`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tab_foundry.bench.tune import TuneConfig, _default_out_root, _parse_float_list, run_tuning


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-path", required=True, help="Fixed manifest path used for every trial")
    parser.add_argument("--out-root", default=None, help="Output root for sweep artifacts")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Training device override",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed used for every trial")
    parser.add_argument(
        "--lr-max-values",
        default="4e-4,8e-4,1.2e-3",
        help="Comma-separated lr_max grid",
    )
    parser.add_argument(
        "--warmup-ratios",
        default="0.0,0.05,0.1",
        help="Comma-separated warmup_ratio grid",
    )
    parser.add_argument(
        "--grad-clip-values",
        default="0.5,1.0",
        help="Comma-separated grad_clip grid",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune tab-foundry against internal validation only")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = run_tuning(
        TuneConfig(
            manifest_path=Path(str(args.manifest_path)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            device=str(args.device),
            seed=int(args.seed),
            lr_max_values=_parse_float_list(str(args.lr_max_values)),
            warmup_ratios=_parse_float_list(str(args.warmup_ratios)),
            grad_clip_values=_parse_float_list(str(args.grad_clip_values)),
        )
    )
    print("tab-foundry tuning complete:")
    print(f"  trial_count={summary['trial_count']}")
    if summary["best_trial"] is not None:
        print(f"  best_trial={summary['best_trial']}")
    print(f"  artifacts={{'summary': '{Path(summary['out_root']) / 'sweep_summary.json'}', 'csv': '{Path(summary['out_root']) / 'sweep_results.csv'}'}}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
