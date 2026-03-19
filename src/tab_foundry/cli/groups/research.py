"""Research CLI group."""

from __future__ import annotations

import argparse
from typing import Sequence

from ..helpers import register_delegate_leaf


def _prepend_command(command: str, argv: Sequence[str] | None = None) -> list[str]:
    return [command, *(list(argv) if argv is not None else [])]


def _run_sweep_create(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("create-sweep", argv))


def _run_sweep_list(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("list", argv))


def _run_sweep_next(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("next", argv))


def _run_sweep_render(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("render", argv))


def _run_sweep_validate(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("validate", argv))


def _run_sweep_set_active(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta import main as system_delta_main

    return system_delta_main(_prepend_command("set-active", argv))


def _run_sweep_execute(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta_execute import main as execute_main

    return execute_main(None if argv is None else list(argv))


def _run_sweep_graph(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.sweep.graph import main as graph_main

    return graph_main(None if argv is None else list(argv))


def _run_sweep_promote(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.system_delta_promote import main as promote_main

    return promote_main(None if argv is None else list(argv))


def _run_sweep_summarize(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.sweep.summarize import main as summarize_main

    return summarize_main(None if argv is None else list(argv))


def _run_sweep_inspect(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.sweep.inspect import main as inspect_main

    return inspect_main(None if argv is None else list(argv))


def _run_sweep_diff(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.research.sweep.diff import main as diff_main

    return diff_main(None if argv is None else list(argv))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("research", help="Research workflows")
    nested = parser.add_subparsers(dest="research_command", required=True)

    sweep_parser = nested.add_parser("sweep", help="System-delta sweep workflows")
    sweep_nested = sweep_parser.add_subparsers(dest="sweep_command", required=True)
    register_delegate_leaf(
        sweep_nested,
        "create",
        help="Create a new system-delta sweep",
        delegate=_run_sweep_create,
    )
    register_delegate_leaf(
        sweep_nested,
        "list",
        help="List rows in a system-delta sweep",
        delegate=_run_sweep_list,
    )
    register_delegate_leaf(
        sweep_nested,
        "next",
        help="Print the next ready row in a system-delta sweep",
        delegate=_run_sweep_next,
    )
    register_delegate_leaf(
        sweep_nested,
        "render",
        help="Render the system-delta matrix markdown",
        delegate=_run_sweep_render,
    )
    register_delegate_leaf(
        sweep_nested,
        "validate",
        help="Validate completed rows in a system-delta sweep",
        delegate=_run_sweep_validate,
    )
    register_delegate_leaf(
        sweep_nested,
        "set-active",
        help="Set the active system-delta sweep",
        delegate=_run_sweep_set_active,
    )
    register_delegate_leaf(
        sweep_nested,
        "execute",
        help="Execute selected system-delta sweep rows",
        delegate=_run_sweep_execute,
    )
    register_delegate_leaf(
        sweep_nested,
        "graph",
        help="Render torchview architecture graphs for sweep targets",
        delegate=_run_sweep_graph,
    )
    register_delegate_leaf(
        sweep_nested,
        "promote",
        help="Promote a completed run to the sweep anchor",
        delegate=_run_sweep_promote,
    )
    register_delegate_leaf(
        sweep_nested,
        "summarize",
        help="Summarize local sweep results into one compact table",
        delegate=_run_sweep_summarize,
    )
    register_delegate_leaf(
        sweep_nested,
        "inspect",
        help="Inspect one materialized sweep row and its resolved surfaces",
        delegate=_run_sweep_inspect,
    )
    register_delegate_leaf(
        sweep_nested,
        "diff",
        help="Diff one materialized sweep row against the anchor or another row",
        delegate=_run_sweep_diff,
    )
