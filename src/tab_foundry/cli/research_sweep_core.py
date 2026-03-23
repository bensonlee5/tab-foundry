"""CLI wiring for `tab-foundry research sweep ...` core commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from tab_foundry.research.sweep import core as sweep_core


def _catalog_path(args: argparse.Namespace) -> Path:
    return Path(str(args.catalog_path))


def _index_path(args: argparse.Namespace) -> Path:
    return Path(str(args.index_path))


def _registry_path(args: argparse.Namespace) -> Path:
    return Path(str(args.registry_path))


def configure_core_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog-path",
        default=str(sweep_core.default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(sweep_core.default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(sweep_core.default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )


def _run_list_sweeps(args: argparse.Namespace) -> int:
    for sweep_info in sweep_core.list_sweeps(index_path=_index_path(args)):
        marker = "*" if sweep_info["is_active"] else " "
        print(
            f"{marker} {sweep_info['sweep_id']}  {sweep_info['status']:<8}  "
            f"{sweep_info['complexity_level']:<16}  anchor={sweep_info['anchor_run_id']}"
        )
    return 0


def _run_show_active(args: argparse.Namespace) -> int:
    index = sweep_core.load_system_delta_index(_index_path(args))
    print(sweep_core.ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id"))
    return 0


def _run_set_active(args: argparse.Namespace) -> int:
    result = sweep_core.set_active_sweep(
        str(args.sweep_id),
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
        registry_path=_registry_path(args),
    )
    print(f"Active sweep set to {args.sweep_id}")
    print(f"  queue_alias_path={result['queue_alias_path']}")
    print(f"  matrix_alias_path={result['matrix_alias_path']}")
    return 0


def _run_sweep_create(args: argparse.Namespace) -> int:
    result = sweep_core.create_sweep(
        sweep_id=str(args.sweep_id),
        anchor_run_id=str(args.anchor_run_id),
        parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
        complexity_level=str(args.complexity_level),
        benchmark_bundle_path=str(args.benchmark_bundle_path),
        control_baseline_id=str(args.control_baseline_id),
        external_benchmarks=(
            None
            if args.external_benchmarks is None
            else [str(value) for value in args.external_benchmarks]
        ),
        training_experiment=(
            None if args.training_experiment is None else str(args.training_experiment)
        ),
        training_config_profile=(
            None if args.training_config_profile is None else str(args.training_config_profile)
        ),
        surface_role=(None if args.surface_role is None else str(args.surface_role)),
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


def _load_queue_from_args(args: argparse.Namespace) -> dict[str, object]:
    selected_sweep_id = None if getattr(args, "sweep_id", None) is None else str(args.sweep_id)
    return sweep_core.load_system_delta_queue(
        sweep_id=selected_sweep_id,
        index_path=_index_path(args),
        catalog_path=_catalog_path(args),
    )


def _run_sweep_list(args: argparse.Namespace) -> int:
    queue = _load_queue_from_args(args)
    for row in sweep_core.ordered_rows(queue):
        print(
            f"{int(row['order']):02d}  {row['status']:<28}  "
            f"{row['dimension_family']:<13}  {row['delta_id']}"
        )
    return 0


def _run_sweep_next(args: argparse.Namespace) -> int:
    queue = _load_queue_from_args(args)
    next_row = sweep_core.next_ready_row(queue)
    if next_row is None:
        print("No ready rows.")
        return 0
    print(OmegaConf.to_yaml(next_row, resolve=True).strip())
    return 0


def _run_sweep_render(args: argparse.Namespace) -> int:
    queue = _load_queue_from_args(args)
    resolved_out_path = sweep_core.render_and_write_system_delta_matrix(
        sweep_id=str(queue["sweep_id"]),
        queue=queue,
        registry_path=_registry_path(args),
        out_path=None if args.out_path is None else Path(str(args.out_path)),
    )
    active_sweep_id = sweep_core.ensure_non_empty_string(
        sweep_core.load_system_delta_index(_index_path(args)).get("active_sweep_id"),
        context="active_sweep_id",
    )
    if str(queue["sweep_id"]) == active_sweep_id:
        sweep_core.sync_active_sweep_aliases(
            sweep_id=str(queue["sweep_id"]),
            index_path=_index_path(args),
            catalog_path=_catalog_path(args),
            registry_path=_registry_path(args),
        )
    print(f"Rendered system delta matrix to {resolved_out_path.expanduser().resolve()}")
    return 0


def _run_sweep_validate(args: argparse.Namespace) -> int:
    queue = _load_queue_from_args(args)
    issues = sweep_core.validate_system_delta_queue(queue, registry_path=_registry_path(args))
    if not issues:
        print("System delta queue validation passed.")
        return 0
    for issue in issues:
        print(issue)
    return 1


def register_core_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    add_core_paths_to_subcommands: bool = False,
) -> None:
    def _configure_paths(parser: argparse.ArgumentParser) -> None:
        if add_core_paths_to_subcommands:
            configure_core_path_arguments(parser)

    list_sweeps_parser = subparsers.add_parser("list-sweeps", help="List known sweeps")
    _configure_paths(list_sweeps_parser)
    list_sweeps_parser.set_defaults(func=_run_list_sweeps)

    show_active_parser = subparsers.add_parser("show-active", help="Print the active sweep id")
    _configure_paths(show_active_parser)
    show_active_parser.set_defaults(func=_run_show_active)

    set_active_parser = subparsers.add_parser(
        "set-active",
        help="Set the active sweep and regenerate aliases",
    )
    _configure_paths(set_active_parser)
    set_active_parser.add_argument("--sweep-id", required=True, help="Sweep id to activate")
    set_active_parser.set_defaults(func=_run_set_active)

    for command_name, help_text in (
        ("list", "List queue rows in order"),
        ("next", "Print the next ready row"),
        ("render", "Render the selected sweep matrix"),
        ("validate", "Validate completed rows for the selected sweep"),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
        _configure_paths(command_parser)
        command_parser.add_argument(
            "--sweep-id",
            default=None,
            help="Optional sweep id; defaults to the active sweep",
        )
        if command_name == "render":
            command_parser.add_argument(
                "--out-path",
                default=None,
                help="Optional alternate markdown output path",
            )
        command_parser.set_defaults(
            func={
                "list": _run_sweep_list,
                "next": _run_sweep_next,
                "render": _run_sweep_render,
                "validate": _run_sweep_validate,
            }[command_name]
        )

    create_parser = subparsers.add_parser(
        "create-sweep",
        help="Bootstrap a new sweep from the delta catalog",
    )
    _configure_paths(create_parser)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage sweep-aware system-delta queues")
    configure_core_path_arguments(parser)
    register_core_subparsers(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
