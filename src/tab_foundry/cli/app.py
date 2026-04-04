"""Packaged click CLI assembly."""

from __future__ import annotations

from typing import Sequence

from .click_utils import GROUP_KWARGS, LazyCommandSpec, LazyGroup, run_click_command


cli = LazyGroup(
    name="tab-foundry",
    help="tab-foundry tooling",
    lazy_commands={
        "bench": LazyCommandSpec(
            module="tab_foundry.cli.groups.bench",
            attr="GROUP",
            help="Benchmark workflows",
        ),
        "data": LazyCommandSpec(
            module="tab_foundry.cli.groups.data",
            attr="GROUP",
            help="Data workflows",
        ),
        "dev": LazyCommandSpec(
            module="tab_foundry.cli.dev",
            attr="GROUP",
            help="Developer-focused inspection tools",
        ),
        "eval": LazyCommandSpec(
            module="tab_foundry.cli.groups.eval_",
            attr="GROUP",
            help="Evaluation workflows",
        ),
        "export": LazyCommandSpec(
            module="tab_foundry.cli.groups.export",
            attr="GROUP",
            help="Export workflows",
        ),
        "research": LazyCommandSpec(
            module="tab_foundry.cli.groups.research",
            attr="GROUP",
            help="Research workflows",
        ),
        "train": LazyCommandSpec(
            module="tab_foundry.cli.groups.train",
            attr="GROUP",
            help="Training workflows",
        ),
    },
    **GROUP_KWARGS,
)


def main(argv: Sequence[str] | None = None) -> int:
    return run_click_command(cli, argv, prog_name="tab-foundry")
