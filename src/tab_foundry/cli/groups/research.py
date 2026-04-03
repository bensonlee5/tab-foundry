"""Research CLI group."""

from __future__ import annotations

import click

import tab_foundry.cli.research_adequacy as research_adequacy_cli
import tab_foundry.cli.research_diff as research_diff_cli
import tab_foundry.cli.research_execute as research_execute_cli
import tab_foundry.cli.research_graph as research_graph_cli
import tab_foundry.cli.research_inspect as research_inspect_cli
import tab_foundry.cli.research_promote as research_promote_cli
import tab_foundry.cli.research_summarize as research_summarize_cli
import tab_foundry.cli.research_sweep_core as research_sweep_core_cli
from tab_foundry.cli.click_utils import GROUP_KWARGS


@click.group(name="research", help="Research workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Research workflows."""


@click.group(name="adequacy", help="Synthetic adequacy workflows", **GROUP_KWARGS)
def _adequacy_group() -> None:
    """Synthetic adequacy workflows."""


_adequacy_group.add_command(research_adequacy_cli.COMMAND)
_adequacy_group.add_command(research_adequacy_cli.FINALIZE_COMMAND)


@click.group(name="sweep", help="System-delta sweep workflows", **GROUP_KWARGS)
def _sweep_group() -> None:
    """System-delta sweep workflows."""


_sweep_group.add_command(research_sweep_core_cli.LIST_SWEEPS_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.LIST_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.NEXT_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.RENDER_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.VALIDATE_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.MATERIALIZE_CORPORA_COMMAND)
_sweep_group.add_command(research_sweep_core_cli.CREATE_SWEEP_COMMAND)
_sweep_group.add_command(research_execute_cli.COMMAND)
_sweep_group.add_command(research_graph_cli.COMMAND)
_sweep_group.add_command(research_promote_cli.COMMAND)
_sweep_group.add_command(research_summarize_cli.COMMAND)
_sweep_group.add_command(research_inspect_cli.COMMAND)
_sweep_group.add_command(research_diff_cli.COMMAND)


GROUP.add_command(_adequacy_group)
GROUP.add_command(_sweep_group)
