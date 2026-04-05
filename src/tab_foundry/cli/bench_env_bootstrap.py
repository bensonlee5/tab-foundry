"""CLI wiring for `tab-foundry bench env bootstrap`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from tab_foundry.bench.envs import BenchmarkEnvConfig, bootstrap_benchmark_envs
from tab_foundry.cli.click_utils import run_click_command


def _bootstrap_command(
    *,
    nanotabpfn_root: Path,
    tabpfn_root: Path,
    tabicl_root: Path,
    tab_realdata_hub_root: Path | None,
) -> int:
    summary = bootstrap_benchmark_envs(
        BenchmarkEnvConfig(
            nanotabpfn_root=nanotabpfn_root,
            tabpfn_root=tabpfn_root,
            tabicl_root=tabicl_root,
            tab_realdata_hub_root=tab_realdata_hub_root,
        )
    )
    print("Benchmark env bootstrap complete:")
    for key, value in summary.items():
        print(f"  {key}={value}")
    return 0


@click.command(name="bootstrap", help="Bootstrap sibling benchmark environments")
@click.option("--nanotabpfn-root", required=True, type=click.Path(path_type=Path), help="Local nanoTabPFN checkout")
@click.option("--tabpfn-root", required=True, type=click.Path(path_type=Path), help="Local TabPFN checkout")
@click.option("--tabicl-root", required=True, type=click.Path(path_type=Path), help="Local tabicl checkout")
@click.option(
    "--tab-realdata-hub-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Explicit local tab-realdata-hub checkout used by benchmark helpers",
)
def COMMAND(
    nanotabpfn_root: Path,
    tabpfn_root: Path,
    tabicl_root: Path,
    tab_realdata_hub_root: Path | None,
) -> int:
    return _bootstrap_command(
        nanotabpfn_root=nanotabpfn_root,
        tabpfn_root=tabpfn_root,
        tabicl_root=tabicl_root,
        tab_realdata_hub_root=tab_realdata_hub_root,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench env bootstrap")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
