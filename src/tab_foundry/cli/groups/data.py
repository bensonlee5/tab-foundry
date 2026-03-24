"""Data CLI group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import tab_foundry.cli.data_inspect as data_inspect_module
from tab_foundry.data.corpus import (
    corpus_compare_payload,
    corpus_results_payload,
    list_corpus_recipes,
    load_corpus_record,
    materialize_corpus_recipe,
)
from tab_foundry.data.dagzoo_workflow import (
    DagzooGenerateManifestConfig,
    run_dagzoo_generate_manifest,
)
from tab_foundry.data.manifest import build_manifest


_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_HARDWARE_POLICY_CHOICES = ("none", "cuda_tiered_v1")
_MISSINGNESS_MECHANISM_CHOICES = ("none", "mcar", "mar", "mnar")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw}.")
    return value


def _seed_32bit_int(raw: str) -> int:
    value = int(raw)
    if value < 0 or value > 4_294_967_295:
        raise argparse.ArgumentTypeError(f"Expected a 32-bit unsigned seed, got {raw}.")
    return value


def _finite_float(raw: str, *, flag: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {flag} value {raw!r}.") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"Invalid {flag} value {raw!r}.")
    return value


def _missing_rate(raw: str) -> float:
    value = _finite_float(raw, flag="--missing-rate")
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Expected --missing-rate in [0, 1].")
    return value


def _missing_fraction(raw: str) -> float:
    value = _finite_float(raw, flag="--missing-mar-observed-fraction")
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Expected --missing-mar-observed-fraction in (0, 1].")
    return value


def _positive_float(raw: str, *, flag: str) -> float:
    value = _finite_float(raw, flag=flag)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected {flag} > 0.")
    return value


def _validate_split_ratios(*, train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise SystemExit(
            "invalid split ratios: expected --train-ratio > 0, "
            "--val-ratio >= 0, and --train-ratio + --val-ratio < 1"
        )


def _print_manifest_summary(summary) -> None:
    print(
        "Manifest built:",
        f"path={summary.out_path}",
        f"filter_policy={summary.filter_policy}",
        f"missing_value_policy={summary.missing_value_policy}",
        f"discovered={summary.discovered_records}",
        f"excluded={summary.excluded_records}",
        f"excluded_for_missing_values={summary.excluded_for_missing_values}",
        f"total={summary.total_records}",
        f"train={summary.train_records}",
        f"val={summary.val_records}",
        f"test={summary.test_records}",
    )
    if summary.filter_status_counts:
        counts = ", ".join(
            f"{status}={count}" for status, count in summary.filter_status_counts.items()
        )
        print("Filter status counts:", counts)
    if summary.missing_value_status_counts:
        counts = ", ".join(
            f"{status}={count}" for status, count in summary.missing_value_status_counts.items()
        )
        print("Missing-value status counts:", counts)
    for warning in summary.warnings:
        print("Warning:", warning)


def _run_build_manifest(args: argparse.Namespace) -> int:
    _validate_split_ratios(train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio))
    roots = [Path(path).expanduser() for path in args.data_root]
    summary = build_manifest(
        data_roots=roots,
        out_path=Path(args.out_manifest),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        filter_policy=str(args.filter_policy),
        missing_value_policy=str(args.missing_value_policy),
    )
    _print_manifest_summary(summary)
    return 0


def _run_dagzoo_generate_manifest(args: argparse.Namespace) -> int:
    _validate_split_ratios(train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio))
    try:
        result = run_dagzoo_generate_manifest(
            DagzooGenerateManifestConfig(
                dagzoo_root=Path(str(args.dagzoo_root)),
                dagzoo_config=Path(str(args.dagzoo_config)),
                handoff_root=Path(str(args.handoff_root)),
                out_manifest=Path(str(args.out_manifest)),
                num_datasets=int(args.num_datasets),
                seed=None if args.seed is None else int(args.seed),
                rows=None if args.rows is None else str(args.rows),
                device=None if args.device is None else str(args.device),
                hardware_policy=str(args.hardware_policy),
                diagnostics=bool(args.diagnostics),
                diagnostics_out_dir=(
                    None if args.diagnostics_out_dir is None else Path(str(args.diagnostics_out_dir))
                ),
                missing_rate=None if args.missing_rate is None else float(args.missing_rate),
                missing_mechanism=(
                    None if args.missing_mechanism is None else str(args.missing_mechanism)
                ),
                missing_mar_observed_fraction=(
                    None
                    if args.missing_mar_observed_fraction is None
                    else float(args.missing_mar_observed_fraction)
                ),
                missing_mar_logit_scale=(
                    None
                    if args.missing_mar_logit_scale is None
                    else float(args.missing_mar_logit_scale)
                ),
                missing_mnar_logit_scale=(
                    None
                    if args.missing_mnar_logit_scale is None
                    else float(args.missing_mnar_logit_scale)
                ),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                filter_policy=str(args.filter_policy),
                missing_value_policy=str(args.missing_value_policy),
            )
        )
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    print(f"Dagzoo handoff manifest: {result.handoff.handoff_manifest_path}")
    print(f"Dagzoo generated dir: {result.handoff.generated_dir}")
    print(f"Output manifest: {result.summary.out_path}")
    _print_manifest_summary(result.summary)
    return 0


def _print_json_payload(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_corpus_list_recipes(args: argparse.Namespace) -> int:
    recipes = [recipe.to_dict() for recipe in list_corpus_recipes()]
    if bool(args.json):
        _print_json_payload({"recipes": recipes})
        return 0
    for recipe in recipes:
        print(
            f"{recipe['recipe_id']}: "
            f"kind={recipe['kind']} "
            f"surface_label={recipe['surface_label']} "
            f"invocations={len(recipe['invocations'])}"
        )
    return 0


def _run_corpus_materialize(args: argparse.Namespace) -> int:
    record = materialize_corpus_recipe(
        recipe_id=str(args.recipe),
        dagzoo_root=Path(str(args.dagzoo_root)),
        force=bool(args.force),
    )
    if bool(args.json):
        _print_json_payload(record)
        return 0
    print(f"Corpus materialized: {record['corpus_ref']}")
    print(f"Recipe: {record['recipe_id']}")
    print(f"Manifest: {record['manifest']['manifest_path']}")
    print(f"Surface label: {record['surface_label']}")
    return 0


def _run_corpus_inspect(args: argparse.Namespace) -> int:
    record = load_corpus_record(str(args.corpus_ref))
    if bool(args.json):
        _print_json_payload(record)
        return 0
    manifest = record["manifest"]
    print(f"Corpus: {record['corpus_ref']}")
    print(f"Recipe: {record['recipe_id']}")
    print(f"Surface label: {record['surface_label']}")
    print(f"Manifest: {manifest['manifest_path']}")
    print(f"Records: {manifest['inspection']['total_records']}")
    print(f"Splits: {json.dumps(manifest['inspection']['split_counts'], sort_keys=True)}")
    return 0


def _run_corpus_compare(args: argparse.Namespace) -> int:
    payload = corpus_compare_payload(left=str(args.left), right=str(args.right))
    if bool(args.json):
        _print_json_payload(payload)
        return 0
    print(f"Left: {payload['left']['corpus_ref']}")
    print(f"Right: {payload['right']['corpus_ref']}")
    print(f"Differences: {payload['difference_count']}")
    for key in sorted(payload["differences"]):
        difference = payload["differences"][key]
        print(f"{key}: left={json.dumps(difference['left'], sort_keys=True)} right={json.dumps(difference['right'], sort_keys=True)}")
    return 0


def _run_corpus_results(args: argparse.Namespace) -> int:
    payload = corpus_results_payload(
        corpus_ref=str(args.corpus_ref),
        registry_path=None if args.registry_path is None else Path(str(args.registry_path)),
    )
    if bool(args.json):
        _print_json_payload(payload)
        return 0
    print(f"Corpus: {payload['corpus_ref']}")
    print(f"Runs: {payload['run_count']}")
    for run in payload["runs"]:
        sweep = run["sweep"]
        metrics = run["headline_metrics"] or {}
        print(
            f"{run['run_id']}: "
            f"sweep={sweep['sweep_id']} "
            f"delta={sweep['delta_id']} "
            f"best_roc_auc={metrics.get('best_roc_auc')} "
            f"final_roc_auc={metrics.get('final_roc_auc')}"
        )
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("data", help="Data workflows")
    nested = parser.add_subparsers(dest="data_command", required=True)

    build_parser = nested.add_parser(
        "build-manifest",
        help="Build manifest parquet from dagzoo packed shard outputs",
    )
    build_parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Input dagzoo data root",
    )
    build_parser.add_argument("--out-manifest", required=True, help="Output manifest parquet path")
    build_parser.add_argument(
        "--train-ratio",
        type=lambda raw: _finite_float(raw, flag="--train-ratio"),
        default=0.90,
    )
    build_parser.add_argument(
        "--val-ratio",
        type=lambda raw: _finite_float(raw, flag="--val-ratio"),
        default=0.05,
    )
    build_parser.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
        help="Dataset selection policy based on dagzoo filter metadata",
    )
    build_parser.add_argument(
        "--missing-value-policy",
        choices=("allow_any", "forbid_any"),
        default="allow_any",
        help="Dataset selection policy for NaN/Inf-containing inputs",
    )
    build_parser.set_defaults(func=_run_build_manifest)

    manifest_inspect_parser = nested.add_parser(
        "manifest-inspect",
        help="Inspect one manifest parquet and optionally preflight compatibility",
    )
    data_inspect_module.configure_parser(manifest_inspect_parser)
    manifest_inspect_parser.set_defaults(func=data_inspect_module.run_from_args)

    dagzoo_parser = nested.add_parser("dagzoo", help="dagzoo-backed data workflows")
    dagzoo_nested = dagzoo_parser.add_subparsers(dest="dagzoo_command", required=True)

    corpus_parser = nested.add_parser("corpus", help="First-class synthetic corpus workflows")
    corpus_nested = corpus_parser.add_subparsers(dest="corpus_command", required=True)

    list_recipes_parser = corpus_nested.add_parser(
        "list-recipes",
        help="List tracked corpus recipes",
    )
    list_recipes_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_recipes_parser.set_defaults(func=_run_corpus_list_recipes)

    materialize_parser = corpus_nested.add_parser(
        "materialize",
        help="Materialize one tracked corpus recipe under outputs/corpora/",
    )
    materialize_parser.add_argument("--recipe", required=True, help="Tracked corpus recipe id")
    materialize_parser.add_argument("--dagzoo-root", required=True, help="Local dagzoo checkout root")
    materialize_parser.add_argument("--force", action="store_true", help="Replace an existing local materialization")
    materialize_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    materialize_parser.set_defaults(func=_run_corpus_materialize)

    inspect_parser = corpus_nested.add_parser(
        "inspect",
        help="Inspect one materialized corpus record",
    )
    inspect_parser.add_argument("--corpus-ref", required=True, help="Corpus ref or recipe id")
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_parser.set_defaults(func=_run_corpus_inspect)

    compare_parser = corpus_nested.add_parser(
        "compare",
        help="Compare two materialized corpus records",
    )
    compare_parser.add_argument("--left", required=True, help="Left corpus ref or recipe id")
    compare_parser.add_argument("--right", required=True, help="Right corpus ref or recipe id")
    compare_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    compare_parser.set_defaults(func=_run_corpus_compare)

    results_parser = corpus_nested.add_parser(
        "results",
        help="List realized benchmark runs linked to one materialized corpus",
    )
    results_parser.add_argument("--corpus-ref", required=True, help="Corpus ref or recipe id")
    results_parser.add_argument(
        "--registry-path",
        default=None,
        help="Optional benchmark registry override",
    )
    results_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    results_parser.set_defaults(func=_run_corpus_results)

    generate_manifest_parser = dagzoo_nested.add_parser(
        "generate-manifest",
        help="Generate a dagzoo corpus and build a tab-foundry manifest",
    )
    generate_manifest_parser.add_argument(
        "--dagzoo-root",
        required=True,
        help="Local dagzoo checkout root",
    )
    generate_manifest_parser.add_argument(
        "--dagzoo-config",
        required=True,
        help="dagzoo config path (absolute or relative to --dagzoo-root)",
    )
    generate_manifest_parser.add_argument(
        "--handoff-root",
        required=True,
        help="dagzoo handoff root written by `dagzoo generate --handoff-root`",
    )
    generate_manifest_parser.add_argument(
        "--out-manifest",
        required=True,
        help="Output manifest parquet path",
    )
    generate_manifest_parser.add_argument(
        "--num-datasets",
        type=_positive_int,
        default=10,
        help="Number of dagzoo datasets to generate",
    )
    generate_manifest_parser.add_argument(
        "--seed",
        type=_seed_32bit_int,
        default=None,
        help="Optional 32-bit run seed override",
    )
    generate_manifest_parser.add_argument(
        "--rows",
        default=None,
        help="Optional dagzoo rows override",
    )
    generate_manifest_parser.add_argument(
        "--device",
        choices=_DEVICE_CHOICES,
        default=None,
        help="Optional dagzoo device override",
    )
    generate_manifest_parser.add_argument(
        "--hardware-policy",
        choices=_HARDWARE_POLICY_CHOICES,
        default="none",
        help="Explicit dagzoo hardware policy",
    )
    generate_manifest_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable dagzoo diagnostics coverage artifacts",
    )
    generate_manifest_parser.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional dagzoo diagnostics artifact directory",
    )
    generate_manifest_parser.add_argument(
        "--missing-rate",
        type=_missing_rate,
        default=None,
        help="Optional dagzoo missing-rate override in [0, 1]",
    )
    generate_manifest_parser.add_argument(
        "--missing-mechanism",
        choices=_MISSINGNESS_MECHANISM_CHOICES,
        default=None,
        help="Optional dagzoo missingness mechanism override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mar-observed-fraction",
        type=_missing_fraction,
        default=None,
        help="Optional dagzoo MAR observed-feature fraction override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mar-logit-scale",
        type=lambda raw: _positive_float(raw, flag="--missing-mar-logit-scale"),
        default=None,
        help="Optional dagzoo MAR logit scale override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mnar-logit-scale",
        type=lambda raw: _positive_float(raw, flag="--missing-mnar-logit-scale"),
        default=None,
        help="Optional dagzoo MNAR logit scale override",
    )
    generate_manifest_parser.add_argument(
        "--train-ratio",
        type=lambda raw: _finite_float(raw, flag="--train-ratio"),
        default=0.90,
    )
    generate_manifest_parser.add_argument(
        "--val-ratio",
        type=lambda raw: _finite_float(raw, flag="--val-ratio"),
        default=0.05,
    )
    generate_manifest_parser.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
        help="Dataset selection policy based on dagzoo filter metadata",
    )
    generate_manifest_parser.add_argument(
        "--missing-value-policy",
        choices=("allow_any", "forbid_any"),
        default="allow_any",
        help="Dataset selection policy for NaN/Inf-containing inputs",
    )
    generate_manifest_parser.set_defaults(func=_run_dagzoo_generate_manifest)
