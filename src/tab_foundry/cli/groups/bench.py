"""Benchmark CLI group."""

from __future__ import annotations

from tab_foundry.cli.click_utils import GROUP_KWARGS, LazyCommandSpec, LazyGroup


_SMOKE_GROUP = LazyGroup(
    name="smoke",
    help="Smoke harnesses",
    lazy_commands={
        "dagzoo": LazyCommandSpec(
            module="tab_foundry.cli.bench_smoke_dagzoo",
            attr="COMMAND",
            help="Run the dagzoo smoke harness",
        ),
        "iris": LazyCommandSpec(
            module="tab_foundry.cli.bench_smoke_iris",
            attr="COMMAND",
            help="Run the Iris smoke harness",
        ),
    },
    **GROUP_KWARGS,
)

_ENV_GROUP = LazyGroup(
    name="env",
    help="Benchmark environment helpers",
    lazy_commands={
        "bootstrap": LazyCommandSpec(
            module="tab_foundry.cli.bench_env_bootstrap",
            attr="COMMAND",
            help="Bootstrap sibling benchmark environments",
        ),
    },
    **GROUP_KWARGS,
)

_BUNDLE_GROUP = LazyGroup(
    name="bundle",
    help="Benchmark bundle workflows",
    lazy_commands={
        "build-openml": LazyCommandSpec(
            module="tab_foundry.cli.bench_bundle_openml",
            attr="COMMAND",
            help="Build an OpenML benchmark bundle",
        ),
    },
    **GROUP_KWARGS,
)

_REGISTRY_GROUP = LazyGroup(
    name="registry",
    help="Benchmark registry workflows",
    lazy_commands={
        "freeze-baseline": LazyCommandSpec(
            module="tab_foundry.cli.bench_control_baseline_freeze",
            attr="COMMAND",
            help="Freeze a control baseline",
        ),
        "freeze-hardware-baseline": LazyCommandSpec(
            module="tab_foundry.cli.bench_hardware_architecture_freeze",
            attr="COMMAND",
            help="Freeze a hardware architecture baseline",
        ),
        "register-run": LazyCommandSpec(
            module="tab_foundry.cli.bench_run_registration",
            attr="COMMAND",
            help="Register a benchmark run",
        ),
    },
    **GROUP_KWARGS,
)

_DIAGNOSE_GROUP = LazyGroup(
    name="diagnose",
    help="Benchmark diagnosis flows",
    lazy_commands={
        "bounce": LazyCommandSpec(
            module="tab_foundry.cli.bench_bounce_diagnosis",
            attr="COMMAND",
            help="Run the benchmark bounce diagnosis flow",
        ),
    },
    **GROUP_KWARGS,
)

GROUP = LazyGroup(
    name="bench",
    help="Benchmark workflows",
    lazy_commands={
        "bundle": _BUNDLE_GROUP,
        "compare": LazyCommandSpec(
            module="tab_foundry.cli.bench_compare",
            attr="COMMAND",
            help="Run the benchmark comparison against external baselines",
        ),
        "diagnose": _DIAGNOSE_GROUP,
        "env": _ENV_GROUP,
        "materialize-openml-bundle": LazyCommandSpec(
            module="tab_foundry.cli.bench_materialize_openml_bundle",
            attr="COMMAND",
            help="Materialize an OpenML bundle into a manifest-backed benchmark surface",
        ),
        "registry": _REGISTRY_GROUP,
        "smoke": _SMOKE_GROUP,
        "tune": LazyCommandSpec(
            module="tab_foundry.cli.bench_tune",
            attr="COMMAND",
            help="Run the internal benchmark tuning sweep",
        ),
    },
    **GROUP_KWARGS,
)
