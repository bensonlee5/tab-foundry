"""Benchmark CLI group."""

from __future__ import annotations

import argparse

import tab_foundry.cli.bench_control_baseline_freeze as control_baseline_freeze_cli
import tab_foundry.cli.bench_run_registration as run_registration_cli
import tab_foundry.bench.bounce_diagnosis as bounce_diagnosis_module
import tab_foundry.bench.compare as compare_module
import tab_foundry.bench.dagzoo_smoke as dagzoo_smoke_module
import tab_foundry.bench.envs as envs_module
import tab_foundry.bench.iris_smoke as iris_smoke_module
import tab_foundry.bench.openml_benchmark_bundle as openml_benchmark_bundle_module
import tab_foundry.bench.tune as tune_module


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bench", help="Benchmark workflows")
    nested = parser.add_subparsers(dest="bench_command", required=True)

    smoke_parser = nested.add_parser("smoke", help="Smoke harnesses")
    smoke_nested = smoke_parser.add_subparsers(dest="smoke_command", required=True)
    iris_parser = smoke_nested.add_parser("iris", help="Run the Iris smoke harness")
    iris_smoke_module.configure_parser(iris_parser)
    iris_parser.set_defaults(func=iris_smoke_module.run_from_args)

    dagzoo_parser = smoke_nested.add_parser("dagzoo", help="Run the dagzoo smoke harness")
    dagzoo_smoke_module.configure_parser(dagzoo_parser)
    dagzoo_parser.set_defaults(func=dagzoo_smoke_module.run_from_args)

    tune_parser = nested.add_parser("tune", help="Run the internal benchmark tuning sweep")
    tune_module.configure_parser(tune_parser)
    tune_parser.set_defaults(func=tune_module.run_from_args)

    compare_parser = nested.add_parser(
        "compare",
        help="Run the benchmark comparison against external baselines",
    )
    compare_module.configure_parser(compare_parser)
    compare_parser.set_defaults(func=compare_module.run_from_args)

    env_parser = nested.add_parser("env", help="Benchmark environment helpers")
    env_nested = env_parser.add_subparsers(dest="env_command", required=True)
    env_bootstrap_parser = env_nested.add_parser(
        "bootstrap",
        help="Bootstrap sibling benchmark environments",
    )
    envs_module.configure_parser(env_bootstrap_parser)
    env_bootstrap_parser.set_defaults(func=envs_module.run_from_args)

    bundle_parser = nested.add_parser("bundle", help="Benchmark bundle workflows")
    bundle_nested = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    bundle_build_parser = bundle_nested.add_parser(
        "build-openml",
        help="Build an OpenML benchmark bundle",
    )
    openml_benchmark_bundle_module.configure_parser(bundle_build_parser)
    bundle_build_parser.set_defaults(func=openml_benchmark_bundle_module.run_from_args)

    registry_parser = nested.add_parser("registry", help="Benchmark registry workflows")
    registry_nested = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_register_parser = registry_nested.add_parser(
        "register-run",
        help="Register a benchmark run",
    )
    run_registration_cli.configure_parser(registry_register_parser)
    registry_register_parser.set_defaults(func=run_registration_cli.run_from_args)

    registry_freeze_parser = registry_nested.add_parser(
        "freeze-baseline",
        help="Freeze a control baseline",
    )
    control_baseline_freeze_cli.configure_parser(registry_freeze_parser)
    registry_freeze_parser.set_defaults(func=control_baseline_freeze_cli.run_from_args)

    diagnose_parser = nested.add_parser("diagnose", help="Benchmark diagnosis flows")
    diagnose_nested = diagnose_parser.add_subparsers(dest="diagnose_command", required=True)
    diagnose_bounce_parser = diagnose_nested.add_parser(
        "bounce",
        help="Run the benchmark bounce diagnosis flow",
    )
    bounce_diagnosis_module.configure_parser(diagnose_bounce_parser)
    diagnose_bounce_parser.set_defaults(func=bounce_diagnosis_module.run_from_args)
