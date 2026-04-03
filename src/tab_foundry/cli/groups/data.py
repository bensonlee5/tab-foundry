"""Data CLI group."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import click

import tab_foundry.cli.data_inspect as data_inspect_module
from tab_foundry.cli.click_utils import (
    apply_click_decorators,
    dagzoo_root_option,
    device_option,
    DEVICE_CHOICES,
    emit_payload,
    GROUP_KWARGS,
    json_output_option,
    MissingFractionType,
    MissingRateType,
    FiniteFloatType,
    materialize_worker_options,
    path_option,
    PositiveFloatType,
    POSITIVE_INT,
    UINT32,
)
from tab_foundry.data.corpus_loading import list_corpus_recipes
from tab_foundry.data.corpus_lookup import load_corpus_record
from tab_foundry.data.corpus_materialization import (
    finalize_staged_corpus_recipe,
    materialize_corpus_recipe,
)
from tab_foundry.data.corpus_reporting import corpus_compare_payload, corpus_results_payload
from tab_foundry.data.dagzoo_workflow import DagzooGenerateManifestConfig, run_dagzoo_generate_manifest
from tab_realdata_hub.manifest import build_manifest


_HARDWARE_POLICY_CHOICES = ("none", "cuda_tiered_v1")
_MISSINGNESS_MECHANISM_CHOICES = ("none", "mcar", "mar", "mnar")
_VERIFY_CHOICES = ("fast", "full")


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


def _manifest_selection_options(func):
    return apply_click_decorators(
        click.option(
            "--train-ratio",
            default=0.90,
            show_default=True,
            type=FiniteFloatType(flag_name="--train-ratio"),
        ),
        click.option(
            "--val-ratio",
            default=0.05,
            show_default=True,
            type=FiniteFloatType(flag_name="--val-ratio"),
        ),
        click.option(
            "--filter-policy",
            default="include_all",
            show_default=True,
            type=click.Choice(("include_all", "accepted_only")),
            help="Dataset selection policy based on dagzoo filter metadata",
        ),
        click.option(
            "--missing-value-policy",
            default="allow_any",
            show_default=True,
            type=click.Choice(("allow_any", "forbid_any")),
            help="Dataset selection policy for NaN/Inf-containing inputs",
        ),
    )(func)


def _run_build_manifest(
    *,
    data_root: tuple[Path, ...],
    out_manifest: Path,
    train_ratio: float,
    val_ratio: float,
    filter_policy: str,
    missing_value_policy: str,
) -> int:
    _validate_split_ratios(train_ratio=train_ratio, val_ratio=val_ratio)
    roots = [path.expanduser() for path in data_root]
    summary = build_manifest(
        data_roots=roots,
        out_path=out_manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        filter_policy=filter_policy,
        missing_value_policy=missing_value_policy,
    )
    _print_manifest_summary(summary)
    return 0


def _run_dagzoo_generate_manifest(
    *,
    dagzoo_root: Path,
    dagzoo_config: str,
    handoff_root: Path,
    out_manifest: Path,
    num_datasets: int,
    seed: int | None,
    rows: str | None,
    device: str | None,
    hardware_policy: str,
    diagnostics: bool,
    diagnostics_out_dir: Path | None,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    train_ratio: float,
    val_ratio: float,
    filter_policy: str,
    missing_value_policy: str,
) -> int:
    _validate_split_ratios(train_ratio=train_ratio, val_ratio=val_ratio)
    try:
        result = run_dagzoo_generate_manifest(
            DagzooGenerateManifestConfig(
                dagzoo_root=dagzoo_root,
                dagzoo_config=Path(dagzoo_config),
                handoff_root=handoff_root,
                out_manifest=out_manifest,
                num_datasets=num_datasets,
                seed=seed,
                rows=rows,
                device=device,
                hardware_policy=hardware_policy,
                diagnostics=diagnostics,
                diagnostics_out_dir=diagnostics_out_dir,
                missing_rate=missing_rate,
                missing_mechanism=missing_mechanism,
                missing_mar_observed_fraction=missing_mar_observed_fraction,
                missing_mar_logit_scale=missing_mar_logit_scale,
                missing_mnar_logit_scale=missing_mnar_logit_scale,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                filter_policy=filter_policy,
                missing_value_policy=missing_value_policy,
            )
        )
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    print(f"Dagzoo handoff manifest: {result.handoff.handoff_manifest_path}")
    print(f"Dagzoo generated dir: {result.handoff.generated_dir}")
    print(f"Output manifest: {result.summary.out_path}")
    _print_manifest_summary(result.summary)
    return 0


def _run_corpus_list_recipes(*, sweep_id: str | None, json_mode: bool) -> int:
    recipes = [
        recipe.to_dict()
        for recipe in list_corpus_recipes(sweep_id=None if sweep_id is None else str(sweep_id))
    ]
    if json_mode:
        emit_payload({"recipes": recipes}, json_mode=True)
        return 0
    for recipe in recipes:
        print(
            f"{recipe['recipe_id']}: "
            f"kind={recipe['kind']} "
            f"surface_label={recipe['surface_label']} "
            f"invocations={len(recipe['invocations'])}"
        )
    return 0


def _run_corpus_materialize(
    *,
    recipe: str,
    sweep_id: str | None,
    dagzoo_root: Path,
    force: bool,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    json_mode: bool,
) -> int:
    record = materialize_corpus_recipe(
        recipe_id=recipe,
        dagzoo_root=dagzoo_root,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        sweep_id=sweep_id,
    )
    if json_mode:
        emit_payload(record, json_mode=True)
        return 0
    print(f"Corpus materialized: {record['corpus_ref']}")
    print(f"Recipe: {record['recipe_id']}")
    print(f"Manifest: {record['manifest']['manifest_path']}")
    print(f"Surface label: {record['surface_label']}")
    return 0


def _run_corpus_finalize_staged(
    *,
    recipe: str,
    sweep_id: str | None,
    dagzoo_root: Path,
    stage_root: Path | None,
    verify: str,
    experiment: str | None,
    override: tuple[str, ...],
    force: bool,
    json_mode: bool,
) -> int:
    result = finalize_staged_corpus_recipe(
        recipe_id=recipe,
        dagzoo_root=dagzoo_root,
        verify=verify,
        stage_root=stage_root,
        force=force,
        sweep_id=sweep_id,
    )
    record = result["record"]
    manifest_preflight = None
    if experiment is not None or override:
        manifest_preflight = data_inspect_module.manifest_inspect_payload(
            Path(str(record["manifest"]["manifest_path"])),
            experiment=experiment,
            overrides=list(override),
        )
        result["manifest_preflight"] = manifest_preflight

    compatibility = (
        None
        if not isinstance(manifest_preflight, dict)
        else manifest_preflight.get("compatibility")
    )
    compatibility_verdict = (
        None
        if not isinstance(compatibility, dict)
        else str(compatibility.get("verdict", "")).strip().lower()
    )
    compatibility_summary = (
        None
        if not isinstance(compatibility, dict)
        else compatibility.get("summary")
    )
    exit_code = 1 if compatibility_verdict == "fail" else 0
    if json_mode:
        emit_payload(result, json_mode=True)
        return exit_code

    print(f"Corpus finalized from stage: {record['corpus_ref']}")
    print(f"Recipe: {record['recipe_id']}")
    print(f"Manifest: {record['manifest']['manifest_path']}")
    print(f"Surface label: {record['surface_label']}")
    print(f"Verification mode: {result['verification']['mode']}")
    if manifest_preflight is not None:
        if compatibility_verdict is None:
            print("Manifest preflight: unavailable")
        else:
            print(
                "Manifest preflight:",
                f"verdict={compatibility_verdict}",
                f"summary={compatibility_summary}",
            )
    return exit_code


def _run_corpus_inspect(*, corpus_ref: str, json_mode: bool) -> int:
    record = load_corpus_record(corpus_ref)
    if json_mode:
        emit_payload(record, json_mode=True)
        return 0
    manifest = record["manifest"]
    print(f"Corpus: {record['corpus_ref']}")
    print(f"Recipe: {record['recipe_id']}")
    print(f"Surface label: {record['surface_label']}")
    print(f"Manifest: {manifest['manifest_path']}")
    print(f"Records: {manifest['inspection']['total_records']}")
    print(f"Splits: {json.dumps(manifest['inspection']['split_counts'], sort_keys=True)}")
    return 0


def _run_corpus_compare(*, left: str, right: str, json_mode: bool) -> int:
    payload = corpus_compare_payload(left=left, right=right)
    if json_mode:
        emit_payload(payload, json_mode=True)
        return 0
    print(f"Left: {payload['left']['corpus_ref']}")
    print(f"Right: {payload['right']['corpus_ref']}")
    print(f"Differences: {payload['difference_count']}")
    for key in sorted(payload["differences"]):
        difference = payload["differences"][key]
        print(
            f"{key}: left={json.dumps(difference['left'], sort_keys=True)} "
            f"right={json.dumps(difference['right'], sort_keys=True)}"
        )
    return 0


def _run_corpus_results(
    *,
    corpus_ref: str,
    registry_path: Path | None,
    json_mode: bool,
) -> int:
    payload = corpus_results_payload(corpus_ref=corpus_ref, registry_path=registry_path)
    if json_mode:
        emit_payload(payload, json_mode=True)
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


@click.command(name="build-manifest", help="Build a manifest parquet from packed shard outputs")
@click.option(
    "--data-root",
    "data_root",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path),
    help="Input dagzoo data root",
)
@path_option("out-manifest", required=True, help="Output manifest parquet path")
@_manifest_selection_options
def BUILD_MANIFEST_COMMAND(
    data_root: tuple[Path, ...],
    out_manifest: Path,
    train_ratio: float,
    val_ratio: float,
    filter_policy: str,
    missing_value_policy: str,
) -> int:
    return _run_build_manifest(
        data_root=data_root,
        out_manifest=out_manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        filter_policy=filter_policy,
        missing_value_policy=missing_value_policy,
    )


@click.command(name="generate-manifest", help="Generate a dagzoo corpus and build a tab-foundry manifest")
@dagzoo_root_option()
@click.option("--dagzoo-config", required=True, help="dagzoo config path (absolute or relative to --dagzoo-root)")
@path_option("handoff-root", required=True, help="dagzoo handoff root written by `dagzoo generate --handoff-root`")
@path_option("out-manifest", required=True, help="Output manifest parquet path")
@click.option("--num-datasets", default=10, show_default=True, type=POSITIVE_INT, help="Number of dagzoo datasets to generate")
@click.option("--seed", default=None, type=UINT32, help="Optional 32-bit run seed override")
@click.option("--rows", default=None, help="Optional dagzoo rows override")
@device_option(default=None, choices=DEVICE_CHOICES, help="Optional dagzoo device override")
@click.option(
    "--hardware-policy",
    default="none",
    show_default=True,
    type=click.Choice(_HARDWARE_POLICY_CHOICES),
    help="Explicit dagzoo hardware policy",
)
@click.option("--diagnostics", is_flag=True, help="Enable dagzoo diagnostics coverage artifacts")
@path_option("diagnostics-out-dir", default=None, help="Optional dagzoo diagnostics artifact directory")
@click.option("--missing-rate", default=None, type=MissingRateType(flag_name="--missing-rate"), help="Optional dagzoo missing-rate override in [0, 1]")
@click.option(
    "--missing-mechanism",
    default=None,
    type=click.Choice(_MISSINGNESS_MECHANISM_CHOICES),
    help="Optional dagzoo missingness mechanism override",
)
@click.option(
    "--missing-mar-observed-fraction",
    default=None,
    type=MissingFractionType(flag_name="--missing-mar-observed-fraction"),
    help="Optional dagzoo MAR observed-feature fraction override",
)
@click.option(
    "--missing-mar-logit-scale",
    default=None,
    type=PositiveFloatType(flag_name="--missing-mar-logit-scale"),
    help="Optional dagzoo MAR logit scale override",
)
@click.option(
    "--missing-mnar-logit-scale",
    default=None,
    type=PositiveFloatType(flag_name="--missing-mnar-logit-scale"),
    help="Optional dagzoo MNAR logit scale override",
)
@_manifest_selection_options
def GENERATE_MANIFEST_COMMAND(
    dagzoo_root: Path,
    dagzoo_config: str,
    handoff_root: Path,
    out_manifest: Path,
    num_datasets: int,
    seed: int | None,
    rows: str | None,
    device: str | None,
    hardware_policy: str,
    diagnostics: bool,
    diagnostics_out_dir: Path | None,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    train_ratio: float,
    val_ratio: float,
    filter_policy: str,
    missing_value_policy: str,
) -> int:
    return _run_dagzoo_generate_manifest(
        dagzoo_root=dagzoo_root,
        dagzoo_config=dagzoo_config,
        handoff_root=handoff_root,
        out_manifest=out_manifest,
        num_datasets=num_datasets,
        seed=seed,
        rows=rows,
        device=device,
        hardware_policy=hardware_policy,
        diagnostics=diagnostics,
        diagnostics_out_dir=diagnostics_out_dir,
        missing_rate=missing_rate,
        missing_mechanism=missing_mechanism,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        filter_policy=filter_policy,
        missing_value_policy=missing_value_policy,
    )


@click.command(name="list-recipes", help="List tracked corpus recipes")
@click.option("--sweep-id", default=None, help="Optional sweep id to include sweep-local corpus recipes")
@json_output_option
def CORPUS_LIST_RECIPES_COMMAND(sweep_id: str | None, json_mode: bool) -> int:
    return _run_corpus_list_recipes(sweep_id=sweep_id, json_mode=json_mode)


@click.command(name="materialize", help="Materialize one tracked corpus recipe under outputs/corpora/")
@click.option("--recipe", required=True, help="Tracked corpus recipe id")
@click.option("--sweep-id", default=None, help="Optional sweep id to resolve sweep-local corpus recipes first")
@dagzoo_root_option()
@click.option("--force", is_flag=True, help="Replace an existing local materialization")
@materialize_worker_options(
    processes_help="Maximum concurrent invocation subprocesses to use while materializing the corpus",
)
@json_output_option
def CORPUS_MATERIALIZE_COMMAND(
    recipe: str,
    sweep_id: str | None,
    dagzoo_root: Path,
    force: bool,
    materialize_processes: int,
    materialize_worker_threads: int | None,
    json_mode: bool,
) -> int:
    return _run_corpus_materialize(
        recipe=recipe,
        sweep_id=sweep_id,
        dagzoo_root=dagzoo_root,
        force=force,
        materialize_processes=materialize_processes,
        materialize_worker_threads=materialize_worker_threads,
        json_mode=json_mode,
    )


@click.command(
    name="finalize-staged",
    help="Promote an already materialized .staging corpus into a first-class corpus record",
)
@click.option("--recipe", required=True, help="Tracked corpus recipe id")
@click.option("--sweep-id", default=None, help="Optional sweep id to resolve sweep-local corpus recipes first")
@dagzoo_root_option()
@path_option("stage-root", default=None, help="Optional staged corpus root override. Defaults to outputs/corpora/<recipe>/.staging")
@click.option("--verify", default="fast", show_default=True, type=click.Choice(_VERIFY_CHOICES), help="Staged corpus verification level before promotion")
@click.option("--experiment", default=None, help="Optional experiment name for manifest compatibility preflight after promotion")
@click.option("--override", "override_", multiple=True, help="Optional Hydra override applied on top of --experiment or repo defaults")
@click.option("--force", is_flag=True, help="Replace an existing local materialization")
@json_output_option
def CORPUS_FINALIZE_STAGED_COMMAND(
    recipe: str,
    sweep_id: str | None,
    dagzoo_root: Path,
    stage_root: Path | None,
    verify: str,
    experiment: str | None,
    override_: tuple[str, ...],
    force: bool,
    json_mode: bool,
) -> int:
    return _run_corpus_finalize_staged(
        recipe=recipe,
        sweep_id=sweep_id,
        dagzoo_root=dagzoo_root,
        stage_root=stage_root,
        verify=verify,
        experiment=experiment,
        override=override_,
        force=force,
        json_mode=json_mode,
    )


@click.command(name="inspect", help="Inspect one materialized corpus record")
@click.option("--corpus-ref", required=True, help="Corpus ref or recipe id")
@json_output_option
def CORPUS_INSPECT_COMMAND(corpus_ref: str, json_mode: bool) -> int:
    return _run_corpus_inspect(corpus_ref=corpus_ref, json_mode=json_mode)


@click.command(name="compare", help="Compare two materialized corpus records")
@click.option("--left", required=True, help="Left corpus ref or recipe id")
@click.option("--right", required=True, help="Right corpus ref or recipe id")
@json_output_option
def CORPUS_COMPARE_COMMAND(left: str, right: str, json_mode: bool) -> int:
    return _run_corpus_compare(left=left, right=right, json_mode=json_mode)


@click.command(name="results", help="List realized benchmark runs linked to one materialized corpus")
@click.option("--corpus-ref", required=True, help="Corpus ref or recipe id")
@path_option("registry-path", default=None, help="Optional benchmark registry override")
@json_output_option
def CORPUS_RESULTS_COMMAND(corpus_ref: str, registry_path: Path | None, json_mode: bool) -> int:
    return _run_corpus_results(
        corpus_ref=corpus_ref,
        registry_path=registry_path,
        json_mode=json_mode,
    )


@click.group(name="data", help="Data workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Data workflows."""


@click.group(name="corpus", help="First-class synthetic corpus workflows", **GROUP_KWARGS)
def CORPUS_GROUP() -> None:
    """Corpus workflows."""


CORPUS_GROUP.add_command(CORPUS_LIST_RECIPES_COMMAND)
CORPUS_GROUP.add_command(CORPUS_MATERIALIZE_COMMAND)
CORPUS_GROUP.add_command(CORPUS_FINALIZE_STAGED_COMMAND)
CORPUS_GROUP.add_command(CORPUS_INSPECT_COMMAND)
CORPUS_GROUP.add_command(CORPUS_COMPARE_COMMAND)
CORPUS_GROUP.add_command(CORPUS_RESULTS_COMMAND)


GROUP.add_command(data_inspect_module.COMMAND)
GROUP.add_command(CORPUS_GROUP)


@click.group(name="data", help="Internal data materialization helpers", **GROUP_KWARGS)
def DEV_GROUP() -> None:
    """Internal data materialization helpers."""


DEV_GROUP.add_command(BUILD_MANIFEST_COMMAND)
DEV_GROUP.add_command(GENERATE_MANIFEST_COMMAND)
