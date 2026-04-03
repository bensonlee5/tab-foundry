"""CLI wiring for `tab-foundry research sweep ...` core commands."""

from __future__ import annotations

from pathlib import Path

import click
from omegaconf import OmegaConf

from tab_foundry.cli.click_utils import (
    GROUP_KWARGS,
    dagzoo_root_option,
    emit_payload,
    json_output_option,
    materialize_worker_options,
    run_click_command,
    sweep_id_option,
    sweep_path_options,
)
from tab_foundry.research.sweep import manage as sweep_manage
from tab_foundry.research.sweep import materialize as sweep_materialize
from tab_foundry.research.sweep import matrix as sweep_matrix


def _run_list_sweeps(*, index_path: Path) -> int:
    for sweep_info in sweep_manage.list_sweeps(index_path=index_path):
        print(
            f"{sweep_info['sweep_id']}  {sweep_info['status']:<8}  "
            f"{sweep_info['complexity_level']:<16}  anchor={sweep_info['anchor_run_id']}"
        )
    return 0


def _run_sweep_create(
    *,
    sweep_id: str,
    anchor_run_id: str,
    parent_sweep_id: str | None,
    complexity_level: str,
    benchmark_manifest_path: Path,
    control_baseline_id: str,
    external_benchmark: tuple[str, ...],
    training_experiment: str | None,
    training_config_profile: str | None,
    surface_role: str | None,
    delta_ref: tuple[str, ...],
    index_path: Path,
    catalog_path: Path,
    registry_path: Path,
) -> int:
    result = sweep_manage.create_sweep(
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        parent_sweep_id=parent_sweep_id,
        complexity_level=complexity_level,
        benchmark_manifest_path=str(benchmark_manifest_path),
        control_baseline_id=control_baseline_id,
        external_benchmarks=list(external_benchmark) or None,
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
        delta_refs=list(delta_ref) or None,
        index_path=index_path,
        catalog_path=catalog_path,
        registry_path=registry_path,
    )
    print("Sweep created:")
    print(f"  sweep_path={result['sweep_path']}")
    print(f"  queue_path={result['queue_path']}")
    print(f"  matrix_path={result['matrix_path']}")
    print(f"  index_path={result['index_path']}")
    return 0


def _load_queue(*, sweep_id: str, index_path: Path, catalog_path: Path) -> dict[str, object]:
    return sweep_materialize.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
    )


def _run_sweep_list(*, sweep_id: str, index_path: Path, catalog_path: Path) -> int:
    queue = _load_queue(sweep_id=sweep_id, index_path=index_path, catalog_path=catalog_path)
    for row in sweep_materialize.ordered_rows(queue):
        print(
            f"{int(row['order']):02d}  {row['status']:<28}  "
            f"{row['dimension_family']:<13}  {row['delta_id']}"
        )
    return 0


def _run_sweep_next(*, sweep_id: str, index_path: Path, catalog_path: Path) -> int:
    queue = _load_queue(sweep_id=sweep_id, index_path=index_path, catalog_path=catalog_path)
    next_row = sweep_materialize.next_ready_row(queue)
    if next_row is None:
        print("No ready rows.")
        return 0
    print(OmegaConf.to_yaml(next_row, resolve=True).strip())
    return 0


def _run_sweep_render(
    *,
    sweep_id: str,
    out_path: Path | None,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
) -> int:
    resolved_out_path = sweep_matrix.render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        out_path=out_path,
    )
    print(f"Rendered system delta matrix to {resolved_out_path.expanduser().resolve()}")
    return 0


def _run_sweep_validate(
    *,
    sweep_id: str,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
) -> int:
    queue = _load_queue(sweep_id=sweep_id, index_path=index_path, catalog_path=catalog_path)
    issues = sweep_matrix.validate_system_delta_queue(queue, registry_path=registry_path)
    if not issues:
        print("System delta queue validation passed.")
        return 0
    for issue in issues:
        print(issue)
    return 1


def _run_sweep_materialize_corpora(
    *,
    sweep_id: str,
    dagzoo_root: Path,
    force: bool,
    json_mode: bool,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    index_path: Path,
    catalog_path: Path,
) -> int:
    payload = sweep_materialize.materialize_sweep_corpora(
        dagzoo_root=dagzoo_root,
        sweep_id=sweep_id,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        index_path=index_path,
        catalog_path=catalog_path,
    )
    if json_mode:
        emit_payload(payload, json_mode=True)
        return 0
    print(
        f"Sweep corpora materialized: sweep_id={payload['sweep_id']} "
        f"count={payload['recipe_count']}"
    )
    for record in payload["records"]:
        print(
            f"{record['recipe_id']}: corpus_ref={record['corpus_ref']} "
            f"manifest={record['manifest']['manifest_path']}"
        )
    return 0


@click.command(name="list-sweeps", help="List known sweeps")
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def LIST_SWEEPS_COMMAND(catalog_path: Path, index_path: Path, registry_path: Path) -> int:
    del catalog_path, registry_path
    return _run_list_sweeps(index_path=index_path)


@click.command(name="list", help="List queue rows in order")
@sweep_id_option()
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def LIST_COMMAND(sweep_id: str, catalog_path: Path, index_path: Path, registry_path: Path) -> int:
    del registry_path
    return _run_sweep_list(sweep_id=sweep_id, index_path=index_path, catalog_path=catalog_path)


@click.command(name="next", help="Print the next ready row")
@sweep_id_option()
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def NEXT_COMMAND(sweep_id: str, catalog_path: Path, index_path: Path, registry_path: Path) -> int:
    del registry_path
    return _run_sweep_next(sweep_id=sweep_id, index_path=index_path, catalog_path=catalog_path)


@click.command(name="render", help="Render the selected sweep matrix")
@click.option(
    "--out-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional alternate markdown output path",
)
@sweep_id_option()
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def RENDER_COMMAND(
    out_path: Path | None,
    sweep_id: str,
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
) -> int:
    return _run_sweep_render(
        out_path=out_path,
        sweep_id=sweep_id,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
    )


@click.command(name="validate", help="Validate completed rows for the selected sweep")
@sweep_id_option()
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def VALIDATE_COMMAND(
    sweep_id: str,
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
) -> int:
    return _run_sweep_validate(
        sweep_id=sweep_id,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
    )


@click.command(
    name="materialize-corpora",
    help="Materialize all unique data.corpus_ref surfaces for the selected sweep",
)
@sweep_id_option(help="Sweep id to materialize")
@dagzoo_root_option()
@click.option("--force", is_flag=True, help="Replace existing local materializations instead of reusing them")
@json_output_option
@materialize_worker_options(
    processes_help="Maximum concurrent invocation subprocesses to use while materializing each corpus",
)
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def MATERIALIZE_CORPORA_COMMAND(
    sweep_id: str,
    dagzoo_root: Path,
    force: bool,
    json_mode: bool,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
) -> int:
    del registry_path
    return _run_sweep_materialize_corpora(
        sweep_id=sweep_id,
        dagzoo_root=dagzoo_root,
        force=force,
        json_mode=json_mode,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        index_path=index_path,
        catalog_path=catalog_path,
    )


@click.command(name="create-sweep", help="Bootstrap a new sweep from the delta catalog")
@click.option("--sweep-id", required=True, help="New sweep id")
@click.option("--anchor-run-id", required=True, help="Anchor benchmark registry run id")
@click.option("--parent-sweep-id", default=None, help="Optional parent sweep id")
@click.option("--complexity-level", required=True, help="Complexity level label")
@click.option("--benchmark-manifest-path", required=True, type=click.Path(path_type=Path), help="Benchmark manifest path for the new sweep")
@click.option("--control-baseline-id", required=True, help="Control baseline id for the new sweep")
@click.option(
    "--external-benchmark",
    "external_benchmark",
    multiple=True,
    help="Ordered external benchmark id to record on the new sweep; repeat to add a secondary comparator. Defaults to tabiclv2.",
)
@click.option(
    "--training-experiment",
    default=None,
    help="Training experiment for new rows; required unless --parent-sweep-id is provided",
)
@click.option(
    "--training-config-profile",
    default=None,
    help="Training config profile for new rows; required unless --parent-sweep-id is provided",
)
@click.option(
    "--surface-role",
    default=None,
    help="Lane role label such as hybrid_diagnostic or architecture_screen; required unless --parent-sweep-id is provided",
)
@click.option(
    "--delta-ref",
    "delta_ref",
    multiple=True,
    help="Optional ordered delta id to include; repeat to build a curated subset",
)
@sweep_path_options(include_registry=True, include_sweeps_root=False)
def CREATE_SWEEP_COMMAND(
    sweep_id: str,
    anchor_run_id: str,
    parent_sweep_id: str | None,
    complexity_level: str,
    benchmark_manifest_path: Path,
    control_baseline_id: str,
    external_benchmark: tuple[str, ...],
    training_experiment: str | None,
    training_config_profile: str | None,
    surface_role: str | None,
    delta_ref: tuple[str, ...],
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
) -> int:
    return _run_sweep_create(
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        parent_sweep_id=parent_sweep_id,
        complexity_level=complexity_level,
        benchmark_manifest_path=benchmark_manifest_path,
        control_baseline_id=control_baseline_id,
        external_benchmark=external_benchmark,
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
        delta_ref=delta_ref,
        catalog_path=catalog_path,
        index_path=index_path,
        registry_path=registry_path,
    )


@click.group(name="research-sweep-core", help="Manage sweep-aware system-delta queues", **GROUP_KWARGS)
def GROUP() -> None:
    """Standalone sweep-core group."""


GROUP.add_command(LIST_SWEEPS_COMMAND)
GROUP.add_command(LIST_COMMAND)
GROUP.add_command(NEXT_COMMAND)
GROUP.add_command(RENDER_COMMAND)
GROUP.add_command(VALIDATE_COMMAND)
GROUP.add_command(MATERIALIZE_CORPORA_COMMAND)
GROUP.add_command(CREATE_SWEEP_COMMAND)


def main(argv: list[str] | None = None) -> int:
    return run_click_command(GROUP, argv, prog_name="tab-foundry research sweep")
