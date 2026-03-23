"""Research CLI group."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from tab_foundry.research import system_delta
from tab_foundry.research.sweep import diff as sweep_diff
from tab_foundry.research.sweep import graph as sweep_graph
from tab_foundry.research.sweep import inspect as sweep_inspect
from tab_foundry.research.sweep import summarize as sweep_summarize
from tab_foundry.research.system_delta_execute import run_from_args as run_execute_from_args
from tab_foundry.research.system_delta_execute import configure_parser as configure_execute_parser
from tab_foundry.research.system_delta_promote import run_from_args as run_promote_from_args
from tab_foundry.research.system_delta_promote import configure_parser as configure_promote_parser


def _catalog_path(args: argparse.Namespace) -> Path:
    return Path(str(args.catalog_path))


def _index_path(args: argparse.Namespace) -> Path:
    return Path(str(args.index_path))


def _registry_path(args: argparse.Namespace) -> Path:
    return Path(str(args.registry_path))


def _add_core_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog-path",
        default=str(system_delta.default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(system_delta.default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(system_delta.default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )


def _run_sweep_create(args: argparse.Namespace) -> int:
    result = system_delta.create_sweep(
        sweep_id=str(args.sweep_id),
        anchor_run_id=str(args.anchor_run_id),
        parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
        complexity_level=str(args.complexity_level),
        benchmark_bundle_path=str(args.benchmark_bundle_path),
        control_baseline_id=str(args.control_baseline_id),
        external_benchmarks=(
            None if args.external_benchmarks is None else [str(value) for value in args.external_benchmarks]
        ),
        training_experiment=(
            None if args.training_experiment is None else str(args.training_experiment)
        ),
        training_config_profile=(
            None if args.training_config_profile is None else str(args.training_config_profile)
        ),
        surface_role=None if args.surface_role is None else str(args.surface_role),
        delta_refs=None if args.delta_refs is None else [str(value) for value in args.delta_refs],
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
        registry_path=_registry_path(args),
    )
    print("Sweep created:")
    print(f"  sweep_path={result['sweep_path']}")
    print(f"  queue_path={result['queue_path']}")
    print(f"  matrix_path={result['matrix_path']}")
    print(f"  index_path={result['index_path']}")
    return 0


def _run_sweep_list(args: argparse.Namespace) -> int:
    queue = system_delta.load_system_delta_queue(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
    )
    for row in system_delta.ordered_rows(queue):
        print(
            f"{int(row['order']):02d}  {row['status']:<28}  "
            f"{row['dimension_family']:<13}  {row['delta_id']}"
        )
    return 0


def _run_sweep_next(args: argparse.Namespace) -> int:
    queue = system_delta.load_system_delta_queue(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
    )
    next_row = system_delta.next_ready_row(queue)
    if next_row is None:
        print("No ready rows.")
        return 0
    print(OmegaConf.to_yaml(next_row, resolve=True).strip())
    return 0


def _run_sweep_render(args: argparse.Namespace) -> int:
    queue = system_delta.load_system_delta_queue(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
    )
    resolved_out_path = system_delta.render_and_write_system_delta_matrix(
        sweep_id=str(queue["sweep_id"]),
        queue=queue,
        registry_path=_registry_path(args),
        out_path=None if args.out_path is None else Path(str(args.out_path)),
    )
    active_sweep_id = system_delta.ensure_non_empty_string(
        system_delta.load_system_delta_index(_index_path(args)).get("active_sweep_id"),
        context="active_sweep_id",
    )
    if str(queue["sweep_id"]) == active_sweep_id:
        system_delta.sync_active_sweep_aliases(
            sweep_id=str(queue["sweep_id"]),
            index_path=_index_path(args),
            catalog_path=_catalog_path(args),
            registry_path=_registry_path(args),
        )
    print(f"Rendered system delta matrix to {resolved_out_path.expanduser().resolve()}")
    return 0


def _run_sweep_validate(args: argparse.Namespace) -> int:
    queue = system_delta.load_system_delta_queue(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
    )
    issues = system_delta.validate_system_delta_queue(queue, registry_path=_registry_path(args))
    if not issues:
        print("System delta queue validation passed.")
        return 0
    for issue in issues:
        print(issue)
    return 1


def _run_sweep_set_active(args: argparse.Namespace) -> int:
    result = system_delta.set_active_sweep(
        str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
        registry_path=_registry_path(args),
    )
    print(f"Active sweep set to {args.sweep_id}")
    print(f"  queue_alias_path={result['queue_alias_path']}")
    print(f"  matrix_alias_path={result['matrix_alias_path']}")
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("research", help="Research workflows")
    nested = parser.add_subparsers(dest="research_command", required=True)

    sweep_parser = nested.add_parser("sweep", help="System-delta sweep workflows")
    sweep_nested = sweep_parser.add_subparsers(dest="sweep_command", required=True)

    create_parser = sweep_nested.add_parser("create", help="Create a new system-delta sweep")
    _add_core_paths(create_parser)
    create_parser.add_argument("--sweep-id", required=True, help="New sweep id")
    create_parser.add_argument("--anchor-run-id", required=True, help="Anchor benchmark registry run id")
    create_parser.add_argument("--parent-sweep-id", default=None, help="Optional parent sweep id")
    create_parser.add_argument("--complexity-level", required=True, help="Complexity level label")
    create_parser.add_argument(
        "--benchmark-bundle-path",
        required=True,
        help="Benchmark bundle path for the new sweep",
    )
    create_parser.add_argument(
        "--control-baseline-id",
        required=True,
        help="Control baseline id for the new sweep",
    )
    create_parser.add_argument(
        "--external-benchmark",
        action="append",
        dest="external_benchmarks",
        default=None,
        help="Ordered external benchmark id to record on the new sweep; repeat to add a secondary comparator. Defaults to tabiclv2.",
    )
    create_parser.add_argument(
        "--training-experiment",
        default=None,
        help="Optional training experiment for new rows; defaults to the parent sweep contract",
    )
    create_parser.add_argument(
        "--training-config-profile",
        default=None,
        help="Optional training config profile for new rows; defaults to the parent sweep contract",
    )
    create_parser.add_argument(
        "--surface-role",
        default=None,
        help="Optional lane role label such as hybrid_diagnostic or architecture_screen",
    )
    create_parser.add_argument(
        "--delta-ref",
        action="append",
        dest="delta_refs",
        default=None,
        help="Optional ordered delta id to include; repeat to build a curated subset",
    )
    create_parser.set_defaults(func=_run_sweep_create)

    list_parser = sweep_nested.add_parser("list", help="List rows in a system-delta sweep")
    _add_core_paths(list_parser)
    list_parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional sweep id; defaults to the active sweep",
    )
    list_parser.set_defaults(func=_run_sweep_list)

    next_parser = sweep_nested.add_parser("next", help="Print the next ready row in a system-delta sweep")
    _add_core_paths(next_parser)
    next_parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional sweep id; defaults to the active sweep",
    )
    next_parser.set_defaults(func=_run_sweep_next)

    render_parser = sweep_nested.add_parser("render", help="Render the system-delta matrix markdown")
    _add_core_paths(render_parser)
    render_parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional sweep id; defaults to the active sweep",
    )
    render_parser.add_argument(
        "--out-path",
        default=None,
        help="Optional alternate markdown output path",
    )
    render_parser.set_defaults(func=_run_sweep_render)

    validate_parser = sweep_nested.add_parser(
        "validate",
        help="Validate completed rows in a system-delta sweep",
    )
    _add_core_paths(validate_parser)
    validate_parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional sweep id; defaults to the active sweep",
    )
    validate_parser.set_defaults(func=_run_sweep_validate)

    set_active_parser = sweep_nested.add_parser(
        "set-active",
        help="Set the active system-delta sweep",
    )
    _add_core_paths(set_active_parser)
    set_active_parser.add_argument("--sweep-id", required=True, help="Sweep id to activate")
    set_active_parser.set_defaults(func=_run_sweep_set_active)

    execute_parser = sweep_nested.add_parser("execute", help="Execute selected system-delta sweep rows")
    configure_execute_parser(execute_parser)
    execute_parser.set_defaults(func=run_execute_from_args)

    graph_parser = sweep_nested.add_parser("graph", help="Render torchview architecture graphs for sweep targets")
    sweep_graph.configure_parser(graph_parser)
    graph_parser.set_defaults(func=sweep_graph.run_from_args)

    promote_parser = sweep_nested.add_parser("promote", help="Promote a completed run to the sweep anchor")
    configure_promote_parser(promote_parser)
    promote_parser.set_defaults(func=run_promote_from_args)

    summarize_parser = sweep_nested.add_parser(
        "summarize",
        help="Summarize local sweep results into one compact table",
    )
    sweep_summarize.configure_parser(summarize_parser)
    summarize_parser.set_defaults(func=sweep_summarize.run_from_args)

    inspect_parser = sweep_nested.add_parser(
        "inspect",
        help="Inspect one materialized sweep row and its resolved surfaces",
    )
    sweep_inspect.configure_parser(inspect_parser)
    inspect_parser.set_defaults(func=sweep_inspect.run_from_args)

    diff_parser = sweep_nested.add_parser(
        "diff",
        help="Diff one materialized sweep row against the anchor or another row",
    )
    sweep_diff.configure_parser(diff_parser)
    diff_parser.set_defaults(func=sweep_diff.run_from_args)
