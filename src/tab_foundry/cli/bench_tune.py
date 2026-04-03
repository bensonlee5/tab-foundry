"""CLI wiring for `tab-foundry bench tune`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from tab_foundry.bench.tune import TuneConfig, _default_out_root, _parse_float_list, run_tuning
from tab_foundry.cli.click_utils import DEVICE_CHOICES, run_click_command


def _tune_command(
    *,
    manifest_path: Path,
    out_root: Path | None,
    device: str,
    seed: int,
    lr_max_values: str,
    warmup_ratios: str,
    grad_clip_values: str,
) -> int:
    summary = run_tuning(
        TuneConfig(
            manifest_path=manifest_path,
            out_root=_default_out_root() if out_root is None else out_root,
            device=device,
            seed=seed,
            lr_max_values=_parse_float_list(lr_max_values),
            warmup_ratios=_parse_float_list(warmup_ratios),
            grad_clip_values=_parse_float_list(grad_clip_values),
        )
    )
    print("tab-foundry tuning complete:")
    print(f"  trial_count={summary['trial_count']}")
    if summary["best_trial"] is not None:
        print(f"  best_trial={summary['best_trial']}")
    print(f"  artifacts={{'summary': '{Path(summary['out_root']) / 'sweep_summary.json'}', 'csv': '{Path(summary['out_root']) / 'sweep_results.csv'}'}}")
    return 0


@click.command(name="tune", help="Run the internal benchmark tuning sweep")
@click.option("--manifest-path", required=True, type=click.Path(path_type=Path), help="Fixed manifest path used for every trial")
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Output root for sweep artifacts")
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(DEVICE_CHOICES),
    help="Training device override",
)
@click.option("--seed", default=1, show_default=True, type=int, help="Base random seed used for every trial")
@click.option(
    "--lr-max-values",
    default="4e-4,8e-4,1.2e-3",
    show_default=True,
    help="Comma-separated lr_max grid",
)
@click.option(
    "--warmup-ratios",
    default="0.0,0.05,0.1",
    show_default=True,
    help="Comma-separated warmup_ratio grid",
)
@click.option(
    "--grad-clip-values",
    default="0.5,1.0",
    show_default=True,
    help="Comma-separated grad_clip grid",
)
def COMMAND(
    manifest_path: Path,
    out_root: Path | None,
    device: str,
    seed: int,
    lr_max_values: str,
    warmup_ratios: str,
    grad_clip_values: str,
) -> int:
    return _tune_command(
        manifest_path=manifest_path,
        out_root=out_root,
        device=device,
        seed=seed,
        lr_max_values=lr_max_values,
        warmup_ratios=warmup_ratios,
        grad_clip_values=grad_clip_values,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench tune")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
