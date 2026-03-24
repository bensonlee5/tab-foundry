"""Benchmark CLI group."""

from __future__ import annotations

import argparse
from typing import Sequence

from ..helpers import register_delegate_leaf


def _run_smoke_iris(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.iris_smoke import main as iris_smoke_main

    return iris_smoke_main(argv)


def _run_smoke_dagzoo(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.dagzoo_smoke import main as dagzoo_smoke_main

    return dagzoo_smoke_main(argv)


def _run_compare(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.compare import main as compare_main

    return compare_main(argv)


def _run_tune(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.tune import main as tune_main

    return tune_main(argv)


def _run_env_bootstrap(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.envs import main as env_main

    return env_main(argv)


def _run_bundle_build_openml(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.openml_benchmark_bundle import main as bundle_main

    return bundle_main(argv)


def _run_registry_register(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.benchmark_run_registry import main as registry_main

    return registry_main(argv)


def _run_registry_freeze(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.control_baseline import main as control_main

    return control_main(argv)


def _run_diagnose_bounce(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.bench.bounce_diagnosis import main as diagnosis_main

    return diagnosis_main(argv)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bench", help="Benchmark workflows")
    nested = parser.add_subparsers(dest="bench_command", required=True)

    smoke_parser = nested.add_parser("smoke", help="Smoke harnesses")
    smoke_nested = smoke_parser.add_subparsers(dest="smoke_command", required=True)
    register_delegate_leaf(
        smoke_nested,
        "iris",
        help="Run the Iris smoke harness",
        delegate=_run_smoke_iris,
    )
    register_delegate_leaf(
        smoke_nested,
        "dagzoo",
        help="Run the dagzoo smoke harness",
        delegate=_run_smoke_dagzoo,
    )

    register_delegate_leaf(
        nested,
        "tune",
        help="Run the internal benchmark tuning sweep",
        delegate=_run_tune,
    )
    register_delegate_leaf(
        nested,
        "compare",
        help="Run the benchmark comparison against external baselines",
        delegate=_run_compare,
    )

    env_parser = nested.add_parser("env", help="Benchmark environment helpers")
    env_nested = env_parser.add_subparsers(dest="env_command", required=True)
    register_delegate_leaf(
        env_nested,
        "bootstrap",
        help="Bootstrap sibling benchmark environments",
        delegate=_run_env_bootstrap,
    )

    bundle_parser = nested.add_parser("bundle", help="Benchmark bundle workflows")
    bundle_nested = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    register_delegate_leaf(
        bundle_nested,
        "build-openml",
        help="Build an OpenML benchmark bundle",
        delegate=_run_bundle_build_openml,
    )

    registry_parser = nested.add_parser("registry", help="Benchmark registry workflows")
    registry_nested = registry_parser.add_subparsers(dest="registry_command", required=True)
    register_delegate_leaf(
        registry_nested,
        "register-run",
        help="Register a benchmark run",
        delegate=_run_registry_register,
    )
    register_delegate_leaf(
        registry_nested,
        "freeze-baseline",
        help="Freeze a control baseline",
        delegate=_run_registry_freeze,
    )

    diagnose_parser = nested.add_parser("diagnose", help="Benchmark diagnosis flows")
    diagnose_nested = diagnose_parser.add_subparsers(dest="diagnose_command", required=True)
    register_delegate_leaf(
        diagnose_nested,
        "bounce",
        help="Run the benchmark bounce diagnosis flow",
        delegate=_run_diagnose_bounce,
    )
