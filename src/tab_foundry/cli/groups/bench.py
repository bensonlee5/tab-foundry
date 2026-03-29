"""Benchmark CLI group."""

from __future__ import annotations

import argparse

import tab_foundry.cli.bench_bounce_diagnosis as bounce_diagnosis_cli
import tab_foundry.cli.bench_bundle_openml as bundle_openml_cli
import tab_foundry.cli.bench_compare as compare_cli
import tab_foundry.cli.bench_control_baseline_freeze as control_baseline_freeze_cli
import tab_foundry.cli.bench_env_bootstrap as env_bootstrap_cli
import tab_foundry.cli.bench_materialize_openml_bundle as materialize_openml_cli
import tab_foundry.cli.bench_run_registration as run_registration_cli
import tab_foundry.cli.bench_smoke_dagzoo as dagzoo_smoke_cli
import tab_foundry.cli.bench_smoke_iris as iris_smoke_cli
import tab_foundry.cli.bench_tune as tune_cli


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bench", help="Benchmark workflows")
    nested = parser.add_subparsers(dest="bench_command", required=True)

    smoke_parser = nested.add_parser("smoke", help="Smoke harnesses")
    smoke_nested = smoke_parser.add_subparsers(dest="smoke_command", required=True)
    iris_parser = smoke_nested.add_parser("iris", help="Run the Iris smoke harness")
    iris_smoke_cli.configure_parser(iris_parser)
    iris_parser.set_defaults(func=iris_smoke_cli.run_from_args)

    dagzoo_parser = smoke_nested.add_parser("dagzoo", help="Run the dagzoo smoke harness")
    dagzoo_smoke_cli.configure_parser(dagzoo_parser)
    dagzoo_parser.set_defaults(func=dagzoo_smoke_cli.run_from_args)

    tune_parser = nested.add_parser("tune", help="Run the internal benchmark tuning sweep")
    tune_cli.configure_parser(tune_parser)
    tune_parser.set_defaults(func=tune_cli.run_from_args)

    compare_parser = nested.add_parser(
        "compare",
        help="Run the benchmark comparison against external baselines",
    )
    compare_cli.configure_parser(compare_parser)
    compare_parser.set_defaults(func=compare_cli.run_from_args)

    materialize_parser = nested.add_parser(
        "materialize-openml-bundle",
        help="Materialize an OpenML bundle into a manifest-backed benchmark surface",
    )
    materialize_openml_cli.configure_parser(materialize_parser)
    materialize_parser.set_defaults(func=materialize_openml_cli.run_from_args)

    env_parser = nested.add_parser("env", help="Benchmark environment helpers")
    env_nested = env_parser.add_subparsers(dest="env_command", required=True)
    env_bootstrap_parser = env_nested.add_parser(
        "bootstrap",
        help="Bootstrap sibling benchmark environments",
    )
    env_bootstrap_cli.configure_parser(env_bootstrap_parser)
    env_bootstrap_parser.set_defaults(func=env_bootstrap_cli.run_from_args)

    bundle_parser = nested.add_parser("bundle", help="Benchmark bundle workflows")
    bundle_nested = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    bundle_build_parser = bundle_nested.add_parser(
        "build-openml",
        help="Build an OpenML benchmark bundle",
    )
    bundle_openml_cli.configure_parser(bundle_build_parser)
    bundle_build_parser.set_defaults(func=bundle_openml_cli.run_from_args)

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
    bounce_diagnosis_cli.configure_parser(diagnose_bounce_parser)
    diagnose_bounce_parser.set_defaults(func=bounce_diagnosis_cli.run_from_args)
