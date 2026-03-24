"""CLI wiring for `tab-foundry bench env bootstrap`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tab_foundry.bench.envs import BenchmarkEnvConfig, bootstrap_benchmark_envs


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nanotabpfn-root", default="~/dev/nanoTabPFN", help="Local nanoTabPFN checkout")
    parser.add_argument("--tabpfn-root", default="~/dev/TabPFN", help="Local TabPFN checkout")
    parser.add_argument("--tabicl-root", default="~/dev/tabicl", help="Local tabicl checkout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap sibling benchmark environments")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = bootstrap_benchmark_envs(
        BenchmarkEnvConfig(
            nanotabpfn_root=Path(str(args.nanotabpfn_root)),
            tabpfn_root=Path(str(args.tabpfn_root)),
            tabicl_root=Path(str(args.tabicl_root)),
        )
    )
    print("Benchmark env bootstrap complete:")
    for key, value in summary.items():
        print(f"  {key}={value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
