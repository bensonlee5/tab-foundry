"""Research CLI group."""

from __future__ import annotations

from tab_foundry.cli.click_utils import GROUP_KWARGS, LazyCommandSpec, LazyGroup


_ADEQUACY_GROUP = LazyGroup(
    name="adequacy",
    help="Synthetic adequacy workflows",
    lazy_commands={
        "finalize": LazyCommandSpec(
            module="tab_foundry.cli.research_adequacy",
            attr="FINALIZE_COMMAND",
            help="Finalize the lean synthetic adequacy pilot from existing artifacts",
        ),
        "pilot": LazyCommandSpec(
            module="tab_foundry.cli.research_adequacy",
            attr="COMMAND",
            help="Run the lean synthetic adequacy pilot",
        ),
    },
    **GROUP_KWARGS,
)

_SWEEP_GROUP = LazyGroup(
    name="sweep",
    help="System-delta sweep workflows",
    lazy_commands={
        "create-sweep": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="CREATE_SWEEP_COMMAND",
            help="Bootstrap a new sweep from the delta catalog",
        ),
        "diff": LazyCommandSpec(
            module="tab_foundry.cli.research_diff",
            attr="COMMAND",
            help="Diff one materialized sweep row against the anchor or another row",
        ),
        "execute": LazyCommandSpec(
            module="tab_foundry.cli.research_execute",
            attr="COMMAND",
            help="Execute selected system-delta sweep rows",
        ),
        "graph": LazyCommandSpec(
            module="tab_foundry.cli.research_graph",
            attr="COMMAND",
            help="Render torchview architecture graphs for sweep targets",
        ),
        "inspect": LazyCommandSpec(
            module="tab_foundry.cli.research_inspect",
            attr="COMMAND",
            help="Inspect one materialized sweep row and its resolved surfaces",
        ),
        "list": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="LIST_COMMAND",
            help="List queue rows in order",
        ),
        "list-sweeps": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="LIST_SWEEPS_COMMAND",
            help="List known sweeps",
        ),
        "materialize-corpora": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="MATERIALIZE_CORPORA_COMMAND",
            help="Materialize all unique data.corpus_ref surfaces for the selected sweep",
        ),
        "next": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="NEXT_COMMAND",
            help="Print the next ready row",
        ),
        "promote": LazyCommandSpec(
            module="tab_foundry.cli.research_promote",
            attr="COMMAND",
            help="Promote a completed run to the sweep anchor",
        ),
        "render": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="RENDER_COMMAND",
            help="Render the selected sweep matrix",
        ),
        "summarize": LazyCommandSpec(
            module="tab_foundry.cli.research_summarize",
            attr="COMMAND",
            help="Summarize local sweep results into one compact table",
        ),
        "validate": LazyCommandSpec(
            module="tab_foundry.cli.research_sweep_core",
            attr="VALIDATE_COMMAND",
            help="Validate completed rows for the selected sweep",
        ),
    },
    **GROUP_KWARGS,
)

_ROBUST_PRIOR_GROUP = LazyGroup(
    name="robust-prior",
    help="Adversarial dagzoo-prior workflows",
    lazy_commands={
        "inspect": LazyCommandSpec(
            module="tab_foundry.cli.research_robust_prior",
            attr="INSPECT_COMMAND",
            help="Inspect one robust-prior pilot and its completed rounds",
        ),
        "run": LazyCommandSpec(
            module="tab_foundry.cli.research_robust_prior",
            attr="RUN_COMMAND",
            help="Run one robust-prior pilot from an anchor checkpoint",
        ),
    },
    **GROUP_KWARGS,
)
GROUP = LazyGroup(
    name="research",
    help="Research workflows",
    lazy_commands={
        "adequacy": _ADEQUACY_GROUP,
        "robust-prior": _ROBUST_PRIOR_GROUP,
        "sweep": _SWEEP_GROUP,
    },
    **GROUP_KWARGS,
)
