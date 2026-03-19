"""Sweep-aware helpers for the anchor-only system-delta workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from . import anchor as _anchor
from . import catalog as _catalog
from . import manage as _manage
from . import materialize as _materialize
from . import matrix as _matrix
from . import paths_io as _paths_io
from . import validation as _validation


LEGACY_PRIOR_CONSTANT_LR_LABEL = _anchor.LEGACY_PRIOR_CONSTANT_LR_LABEL
UNAVAILABLE_TRAINING_LABEL = _anchor.UNAVAILABLE_TRAINING_LABEL
CATALOG_SCHEMA = _catalog.CATALOG_SCHEMA
SWEEP_INDEX_SCHEMA = _catalog.SWEEP_INDEX_SCHEMA
SWEEP_QUEUE_SCHEMA = _catalog.SWEEP_QUEUE_SCHEMA
SWEEP_SCHEMA = _catalog.SWEEP_SCHEMA
DEFAULT_SWEEP_STATUS = _manage.DEFAULT_SWEEP_STATUS
MATERIALIZED_QUEUE_SCHEMA = _materialize.MATERIALIZED_QUEUE_SCHEMA

load_system_delta_catalog = _catalog.load_system_delta_catalog
load_system_delta_index = _catalog.load_system_delta_index
load_system_delta_queue_instance = _catalog.load_system_delta_queue_instance
load_system_delta_sweep = _catalog.load_system_delta_sweep
resolve_selected_sweep_id = _catalog.resolve_selected_sweep_id
create_sweep = _manage.create_sweep
instantiate_queue_row = _manage.instantiate_queue_row
list_sweeps = _manage.list_sweeps
set_active_sweep = _manage.set_active_sweep
sync_active_sweep_aliases = _manage.sync_active_sweep_aliases
evaluate_applicability_guard = _materialize.evaluate_applicability_guard
guarded_initial_state = _materialize.guarded_initial_state
load_system_delta_queue = _materialize.load_system_delta_queue
materialize_row = _materialize.materialize_row
materialize_system_delta_queue = _materialize.materialize_system_delta_queue
next_ready_row = _materialize.next_ready_row
ordered_rows = _materialize.ordered_rows
metric_summary = _matrix.metric_summary
render_and_write_system_delta_matrix = _matrix.render_and_write_system_delta_matrix
render_model_change_payload = _matrix.render_model_change_payload
render_system_delta_matrix = _matrix.render_system_delta_matrix
result_card_path = _matrix.result_card_path
validate_system_delta_queue = _matrix.validate_system_delta_queue
default_catalog_path = _paths_io.default_catalog_path
default_matrix_path = _paths_io.default_matrix_path
default_queue_path = _paths_io.default_queue_path
default_registry_path = _paths_io.default_registry_path
default_sweep_index_path = _paths_io.default_sweep_index_path
default_sweeps_root = _paths_io.default_sweeps_root
repo_root = _paths_io.repo_root
sweep_dir = _paths_io.sweep_dir
sweep_matrix_path = _paths_io.sweep_matrix_path
sweep_metadata_path = _paths_io.sweep_metadata_path
sweep_queue_path = _paths_io.sweep_queue_path
ensure_mapping = _validation.ensure_mapping
ensure_non_empty_string = _validation.ensure_non_empty_string
ensure_rows = _validation.ensure_rows
ensure_string_list = _validation.ensure_string_list
validate_prose_fields = _validation.validate_prose_fields

_copy_jsonable = _paths_io._copy_jsonable
_load_yaml_mapping = _paths_io._load_yaml_mapping
_render_path = _paths_io._render_path
_write_text = _paths_io._write_text
_write_yaml = _paths_io._write_yaml
_ensure_non_empty_string = _validation.ensure_non_empty_string
_ensure_mapping = _validation.ensure_mapping
_ensure_rows = _validation.ensure_rows
_ensure_string_list = _validation.ensure_string_list
_validate_prose_fields = _validation.validate_prose_fields
_resolve_selected_sweep_id = _catalog.resolve_selected_sweep_id
_staged_module_selection_from_run_model = _anchor.staged_module_selection_from_run_model
_anchor_context_from_registry_run = _anchor.anchor_context_from_registry_run
_surface_label_from_anchor_context = _anchor.surface_label_from_anchor_context
_anchor_training_surface_label = _anchor.anchor_training_surface_label
_anchor_module_selection = _anchor.anchor_module_selection
_module_choice = _anchor.module_choice
_describe_feature_encoder = _anchor.describe_feature_encoder
_describe_target_conditioner = _anchor.describe_target_conditioner
_describe_table_block = _anchor.describe_table_block
_describe_tokenizer = _anchor.describe_tokenizer
_describe_column_encoder = _anchor.describe_column_encoder
_describe_row_pool = _anchor.describe_row_pool
_describe_context_encoder = _anchor.describe_context_encoder
_describe_head = _anchor.describe_head
_build_anchor_surface = _anchor.build_anchor_surface
_evaluate_applicability_guard = _materialize.evaluate_applicability_guard
_guarded_initial_state = _materialize.guarded_initial_state
_materialize_row = _materialize.materialize_row
_render_model_change_payload = _matrix.render_model_change_payload
_metric_summary = _matrix.metric_summary
_result_card_path = _matrix.result_card_path
_instantiate_queue_row = _manage.instantiate_queue_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage sweep-aware system-delta queues")
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-sweeps", help="List known sweeps")
    subparsers.add_parser("show-active", help="Print the active sweep id")

    set_active_parser = subparsers.add_parser(
        "set-active",
        help="Set the active sweep and regenerate aliases",
    )
    set_active_parser.add_argument("--sweep-id", required=True, help="Sweep id to activate")

    for command_name, help_text in (
        ("list", "List queue rows in order"),
        ("next", "Print the next ready row"),
        ("render", "Render the selected sweep matrix"),
        ("validate", "Validate completed rows for the selected sweep"),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
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

    create_parser = subparsers.add_parser(
        "create-sweep",
        help="Bootstrap a new sweep from the delta catalog",
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = Path(str(args.catalog_path))
    index_path = Path(str(args.index_path))
    registry_path = Path(str(args.registry_path))

    if args.command == "list-sweeps":
        for sweep_info in list_sweeps(index_path=index_path):
            marker = "*" if sweep_info["is_active"] else " "
            print(
                f"{marker} {sweep_info['sweep_id']}  {sweep_info['status']:<8}  "
                f"{sweep_info['complexity_level']:<16}  anchor={sweep_info['anchor_run_id']}"
            )
        return 0

    if args.command == "show-active":
        index = load_system_delta_index(index_path)
        print(ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id"))
        return 0

    if args.command == "set-active":
        result = set_active_sweep(
            str(args.sweep_id),
            index_path=index_path,
            catalog_path=catalog_path,
            registry_path=registry_path,
        )
        print(f"Active sweep set to {args.sweep_id}")
        print(f"  queue_alias_path={result['queue_alias_path']}")
        print(f"  matrix_alias_path={result['matrix_alias_path']}")
        return 0

    if args.command == "create-sweep":
        result = create_sweep(
            sweep_id=str(args.sweep_id),
            anchor_run_id=str(args.anchor_run_id),
            parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
            complexity_level=str(args.complexity_level),
            benchmark_bundle_path=str(args.benchmark_bundle_path),
            control_baseline_id=str(args.control_baseline_id),
            training_experiment=(
                None if args.training_experiment is None else str(args.training_experiment)
            ),
            training_config_profile=(
                None if args.training_config_profile is None else str(args.training_config_profile)
            ),
            surface_role=(None if args.surface_role is None else str(args.surface_role)),
            delta_refs=None if args.delta_refs is None else [str(value) for value in args.delta_refs],
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

    selected_sweep_id = None if getattr(args, "sweep_id", None) is None else str(args.sweep_id)
    queue = load_system_delta_queue(
        sweep_id=selected_sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
    )
    if args.command == "list":
        for row in ordered_rows(queue):
            print(
                f"{int(row['order']):02d}  {row['status']:<28}  "
                f"{row['dimension_family']:<13}  {row['delta_id']}"
            )
        return 0
    if args.command == "next":
        next_row = next_ready_row(queue)
        if next_row is None:
            print("No ready rows.")
            return 0
        print(OmegaConf.to_yaml(next_row, resolve=True).strip())
        return 0
    if args.command == "render":
        resolved_out_path = render_and_write_system_delta_matrix(
            sweep_id=str(queue["sweep_id"]),
            queue=queue,
            registry_path=registry_path,
            out_path=None if args.out_path is None else Path(str(args.out_path)),
        )
        active_sweep_id = ensure_non_empty_string(
            load_system_delta_index(index_path).get("active_sweep_id"),
            context="active_sweep_id",
        )
        if str(queue["sweep_id"]) == active_sweep_id:
            sync_active_sweep_aliases(
                sweep_id=str(queue["sweep_id"]),
                index_path=index_path,
                catalog_path=catalog_path,
                registry_path=registry_path,
            )
        print(f"Rendered system delta matrix to {resolved_out_path.expanduser().resolve()}")
        return 0
    if args.command == "validate":
        issues = validate_system_delta_queue(queue, registry_path=registry_path)
        if not issues:
            print("System delta queue validation passed.")
            return 0
        for issue in issues:
            print(issue)
        return 1
    raise RuntimeError(f"Unsupported command: {args.command!r}")
