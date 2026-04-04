"""CLI wiring for `tab-foundry research adequacy` commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import (
    dagzoo_root_option,
    device_option,
    materialize_worker_options,
    run_click_command,
)
from tab_foundry.research.adequacy.pilot import (
    finalize_adequacy_pilot,
    run_adequacy_pilot,
)

_CONTRACT_CHECK_CHOICES = ("fast", "full")


def _pilot_command(
    *,
    adequacy_id: str,
    dagzoo_root: Path,
    device: str,
    force: bool,
    out_root: Path | None,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    contract_check: str,
) -> int:
    summary = run_adequacy_pilot(
        adequacy_id=adequacy_id,
        dagzoo_root=dagzoo_root.expanduser().resolve(),
        device=device,
        force=force,
        out_root=None if out_root is None else out_root.expanduser().resolve(),
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        contract_check=contract_check,
    )
    summary_paths = summary.get("summary_paths", {})
    print(
        "Adequacy pilot complete.",
        f"adequacy_id={summary['adequacy_id']}",
        f"interpretation={summary['provisional_interpretation']['bucket']}",
        f"summary={summary_paths.get('summary_md')}",
        flush=True,
    )
    return 0


def _finalize_command(
    *,
    adequacy_id: str,
    dagzoo_root: Path,
    out_root: Path | None,
    contract_check: str,
) -> int:
    summary = finalize_adequacy_pilot(
        adequacy_id=adequacy_id,
        dagzoo_root=dagzoo_root.expanduser().resolve(),
        out_root=None if out_root is None else out_root.expanduser().resolve(),
        contract_check=contract_check,
    )
    summary_paths = summary.get("summary_paths", {})
    print(
        "Adequacy pilot complete.",
        f"adequacy_id={summary['adequacy_id']}",
        f"interpretation={summary['provisional_interpretation']['bucket']}",
        f"summary={summary_paths.get('summary_md')}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research adequacy pilot")


@click.command(name="pilot", help="Run the lean synthetic adequacy pilot")
@click.option("--adequacy-id", required=True, help="Synthetic adequacy spec id to execute")
@dagzoo_root_option(help="Path to the sibling dagzoo checkout used for corpus materialization")
@device_option(
    default="cpu",
    choices=("cpu",),
    help="Pilot execution device. The lean adequacy pilot supports CPU only.",
)
@click.option("--force", is_flag=True, help="Force corpus rematerialization and overwrite pilot-local outputs")
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Optional output root override for pilot artifacts")
@materialize_worker_options(
    processes_help="Maximum concurrent invocation subprocesses to use while materializing pilot corpora",
)
@click.option(
    "--contract-check",
    default="fast",
    show_default=True,
    type=click.Choice(_CONTRACT_CHECK_CHOICES),
    help="Latent-target contract verification level",
)
def COMMAND(
    adequacy_id: str,
    dagzoo_root: Path,
    device: str,
    force: bool,
    out_root: Path | None,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    contract_check: str,
) -> int:
    return _pilot_command(
        adequacy_id=adequacy_id,
        dagzoo_root=dagzoo_root,
        device=device,
        force=force,
        out_root=out_root,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        contract_check=contract_check,
    )


@click.command(
    name="finalize",
    help="Finalize the lean synthetic adequacy pilot from existing artifacts",
)
@click.option("--adequacy-id", required=True, help="Synthetic adequacy spec id to finalize")
@dagzoo_root_option(help="Path to the sibling dagzoo checkout used to resolve staged corpus previews")
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Optional output root override for pilot artifacts")
@click.option(
    "--contract-check",
    default="fast",
    show_default=True,
    type=click.Choice(_CONTRACT_CHECK_CHOICES),
    help="Latent-target contract verification level",
)
def FINALIZE_COMMAND(
    adequacy_id: str,
    dagzoo_root: Path,
    out_root: Path | None,
    contract_check: str,
) -> int:
    return _finalize_command(
        adequacy_id=adequacy_id,
        dagzoo_root=dagzoo_root,
        out_root=out_root,
        contract_check=contract_check,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
