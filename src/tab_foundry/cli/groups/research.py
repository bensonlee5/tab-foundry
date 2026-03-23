"""Research CLI group."""

from __future__ import annotations

import argparse

from tab_foundry.research.sweep import core as sweep_core
from tab_foundry.research.sweep import diff as sweep_diff
from tab_foundry.research.sweep import graph as sweep_graph
from tab_foundry.research.sweep import inspect as sweep_inspect
from tab_foundry.research.sweep import summarize as sweep_summarize
from tab_foundry.research.system_delta_execute import configure_parser as configure_execute_parser
from tab_foundry.research.system_delta_execute import run_from_args as run_execute_from_args
from tab_foundry.research.system_delta_promote import configure_parser as configure_promote_parser
from tab_foundry.research.system_delta_promote import run_from_args as run_promote_from_args


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("research", help="Research workflows")
    nested = parser.add_subparsers(dest="research_command", required=True)

    sweep_parser = nested.add_parser("sweep", help="System-delta sweep workflows")
    sweep_nested = sweep_parser.add_subparsers(dest="sweep_command", required=True)
    sweep_core.register_core_subparsers(
        sweep_nested,
        add_core_paths_to_subcommands=True,
    )

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
