"""Canonical programmatic runtime for benchmark comparison execution."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence, cast

from tab_foundry.bench.artifacts import load_jsonl, write_json, write_jsonl
from tab_foundry.bench.comparison_contract import (
    DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
    DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION,
    BenchmarkComparisonConfig,
)
from tab_foundry.bench.comparison_reporting import benchmark_wandb_summary_payload
from tab_foundry.bench.external_runtime import (
    nanotabpfn_execution_metadata as _nanotabpfn_execution_metadata,
    nanotabpfn_helper_command as _nanotabpfn_helper_command,
    nanotabpfn_python as _nanotabpfn_python,
    resolve_reuse_curve_path as _resolve_reuse_curve_path,
    resolve_reuse_nanotabpfn_error as _resolve_reuse_nanotabpfn_error,
    reused_nanotabpfn_execution_metadata as _reused_nanotabpfn_execution_metadata,
    resolved_tab_realdata_hub_root as _resolved_tab_realdata_hub_root,
    tabiclv2_checkpoint_version as _tabiclv2_checkpoint_version,
    tabiclv2_execution_metadata as _tabiclv2_execution_metadata,
    tabiclv2_helper_command as _tabiclv2_helper_command,
    tabiclv2_python as _tabiclv2_python,
    validate_nanotabpfn_environment as _validate_nanotabpfn_environment,
    validate_tabiclv2_environment as _validate_tabiclv2_environment,
)
from tab_foundry.bench.run_registration import derive_benchmark_run_record
from tab_foundry.control_baseline_registry import load_control_baseline_entry
from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks,
)
from tab_foundry.bench.openml_benchmark import (
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
    benchmark_host_fingerprint,
    build_comparison_summary,
    default_benchmark_manifest_path,
    evaluate_tab_foundry_run,
    load_benchmark_manifest_datasets,
    plot_comparison_curve,
    resolve_device,
    summarize_checkpoint_curve,
)
from tab_foundry.repo_paths import repo_root
from tab_foundry.training.instability import gradient_history_path, telemetry_path
from tab_foundry.training.wandb import posthoc_update_wandb_summary

__all__ = [
    "BenchmarkComparisonConfig",
    "DEFAULT_NANOTABPFN_BATCH_SIZE",
    "DEFAULT_NANOTABPFN_EVAL_EVERY",
    "DEFAULT_NANOTABPFN_LR",
    "DEFAULT_NANOTABPFN_SEEDS",
    "DEFAULT_NANOTABPFN_STEPS",
    "DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION",
    "DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION",
    "EXTERNAL_BENCHMARK_NANOTABPFN",
    "EXTERNAL_BENCHMARK_TABICLV2",
    "derive_benchmark_run_record",
    "posthoc_update_wandb_summary",
    "run_nanotabpfn_benchmark",
]

_RUNTIME_BENCHMARK_SURFACE_TUPLE_SIZE = 3


def _helper_script_path() -> Path:
    return repo_root() / "scripts" / "bench" / "openml_benchmark_helper.py"


def _tabiclv2_helper_script_path() -> Path:
    return repo_root() / "scripts" / "bench" / "tabiclv2_helper.py"


def _src_root() -> Path:
    return repo_root() / "src"


def _resolve_primary_external_benchmark(
    requested_external_benchmarks: Sequence[str],
    *,
    nanotabpfn_records: Sequence[Mapping[str, Any]],
    tabiclv2_records: Sequence[Mapping[str, Any]],
) -> str | None:
    for external_benchmark in requested_external_benchmarks:
        if external_benchmark == EXTERNAL_BENCHMARK_NANOTABPFN and nanotabpfn_records:
            return external_benchmark
        if external_benchmark == EXTERNAL_BENCHMARK_TABICLV2 and tabiclv2_records:
            return external_benchmark
    return None


def _validate_tab_foundry_run_dir(path: Path) -> Path:
    tab_foundry_run_dir = path.expanduser().resolve()
    if not tab_foundry_run_dir.exists():
        raise RuntimeError(f"tab-foundry run dir does not exist: {tab_foundry_run_dir}")
    return tab_foundry_run_dir


def _load_runtime_benchmark_surface(
    benchmark_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    loaded = load_benchmark_manifest_datasets(
        benchmark_manifest_path=benchmark_manifest_path,
    )
    if not isinstance(loaded, tuple):
        raise RuntimeError(
            "load_benchmark_manifest_datasets must return a tuple of "
            "(datasets, benchmark_tasks, benchmark_surface)"
        )
    if len(loaded) != _RUNTIME_BENCHMARK_SURFACE_TUPLE_SIZE:
        raise RuntimeError(
            "load_benchmark_manifest_datasets returned an unexpected tuple shape: "
            f"{len(loaded)!r}"
        )
    datasets, benchmark_tasks, benchmark_surface = loaded
    return (
        cast(dict[str, Any], datasets),
        cast(list[dict[str, Any]], benchmark_tasks),
        cast(dict[str, Any], benchmark_surface),
    )


def _primary_external_curve_path(
    *,
    primary_external_benchmark: str | None,
    nanotabpfn_records: Sequence[Mapping[str, Any]],
    tabiclv2_records: Sequence[Mapping[str, Any]],
    nanotabpfn_curve_path: Path,
    tabiclv2_curve_path: Path,
) -> str | None:
    if primary_external_benchmark == EXTERNAL_BENCHMARK_NANOTABPFN and nanotabpfn_records:
        return str(nanotabpfn_curve_path)
    if primary_external_benchmark == EXTERNAL_BENCHMARK_TABICLV2 and tabiclv2_records:
        return str(tabiclv2_curve_path)
    return None


def _artifact_payload(
    *,
    requested_external_benchmarks: Sequence[str],
    primary_external_benchmark: str | None,
    nanotabpfn_records: Sequence[Mapping[str, Any]],
    tabiclv2_records: Sequence[Mapping[str, Any]],
    benchmark_tasks_path: Path,
    tab_foundry_curve_path: Path,
    nanotabpfn_curve_path: Path,
    tabiclv2_curve_path: Path,
    comparison_curve_path: Path,
    benchmark_manifest_path: Path,
    benchmark_run_record_path: Path,
    training_surface_record_path: Path,
    tab_foundry_run_dir: Path,
) -> dict[str, Any]:
    gradient_history_jsonl = gradient_history_path(tab_foundry_run_dir)
    telemetry_json = telemetry_path(tab_foundry_run_dir)
    return {
        "benchmark_tasks_json": str(benchmark_tasks_path),
        "tab_foundry_curve_jsonl": str(tab_foundry_curve_path),
        "primary_external_curve_jsonl": _primary_external_curve_path(
            primary_external_benchmark=primary_external_benchmark,
            nanotabpfn_records=nanotabpfn_records,
            tabiclv2_records=tabiclv2_records,
            nanotabpfn_curve_path=nanotabpfn_curve_path,
            tabiclv2_curve_path=tabiclv2_curve_path,
        ),
        "nanotabpfn_curve_jsonl": (
            str(nanotabpfn_curve_path)
            if EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks and nanotabpfn_records
            else None
        ),
        "tabiclv2_curve_jsonl": (
            str(tabiclv2_curve_path)
            if EXTERNAL_BENCHMARK_TABICLV2 in requested_external_benchmarks and tabiclv2_records
            else None
        ),
        "comparison_curve_png": str(comparison_curve_path),
        "benchmark_manifest": str(benchmark_manifest_path),
        "gradient_history_jsonl": (
            str(gradient_history_jsonl.resolve()) if gradient_history_jsonl.exists() else None
        ),
        "telemetry_json": str(telemetry_json.resolve()) if telemetry_json.exists() else None,
        "benchmark_run_record_json": str(benchmark_run_record_path),
        "training_surface_record_json": str(training_surface_record_path),
    }


def _finalize_benchmark_summary(
    *,
    summary: dict[str, Any],
    requested_external_benchmarks: Sequence[str],
    primary_external_benchmark: str | None,
    nanotabpfn_records: Sequence[Mapping[str, Any]],
    tabiclv2_records: Sequence[Mapping[str, Any]],
    benchmark_tasks_path: Path,
    tab_foundry_curve_path: Path,
    nanotabpfn_curve_path: Path,
    tabiclv2_curve_path: Path,
    comparison_curve_path: Path,
    benchmark_manifest_path: Path,
    comparison_summary_path: Path,
    benchmark_run_record_path: Path,
    training_surface_record_path: Path,
    tab_foundry_run_dir: Path,
) -> dict[str, Any]:
    summary["external_benchmarks"] = list(requested_external_benchmarks)
    if primary_external_benchmark is not None:
        summary["primary_external_benchmark"] = primary_external_benchmark
    summary["artifacts"] = _artifact_payload(
        requested_external_benchmarks=requested_external_benchmarks,
        primary_external_benchmark=primary_external_benchmark,
        nanotabpfn_records=nanotabpfn_records,
        tabiclv2_records=tabiclv2_records,
        benchmark_tasks_path=benchmark_tasks_path,
        tab_foundry_curve_path=tab_foundry_curve_path,
        nanotabpfn_curve_path=nanotabpfn_curve_path,
        tabiclv2_curve_path=tabiclv2_curve_path,
        comparison_curve_path=comparison_curve_path,
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_run_record_path=benchmark_run_record_path,
        training_surface_record_path=training_surface_record_path,
        tab_foundry_run_dir=tab_foundry_run_dir,
    )
    write_json(comparison_summary_path, summary)
    benchmark_run_record = derive_benchmark_run_record(
        run_dir=tab_foundry_run_dir,
        comparison_summary_path=comparison_summary_path,
        benchmark_run_record_path=benchmark_run_record_path,
    )
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    tab_foundry_summary["manifest_path"] = str(benchmark_run_record["manifest_path"])
    tab_foundry_summary["seed_set"] = list(benchmark_run_record["seed_set"])
    tab_foundry_summary["training_diagnostics"] = dict(benchmark_run_record["training_diagnostics"])
    tab_foundry_summary["model_size"] = dict(benchmark_run_record["model_size"])
    summary["artifacts"]["training_surface_record_json"] = cast(
        dict[str, Any],
        benchmark_run_record["artifacts"],
    ).get("training_surface_record_path")
    if benchmark_run_record.get("surface_labels") is not None:
        tab_foundry_summary["surface_labels"] = dict(benchmark_run_record["surface_labels"])
    write_json(comparison_summary_path, summary)
    write_json(benchmark_run_record_path, benchmark_run_record)
    _ = posthoc_update_wandb_summary(
        telemetry_path=telemetry_path(tab_foundry_run_dir),
        payload=benchmark_wandb_summary_payload(summary),
    )
    return summary


def run_nanotabpfn_benchmark(config: BenchmarkComparisonConfig) -> dict[str, Any]:
    """Run the manual tab-foundry benchmark comparison against external baselines."""

    benchmark_manifest_path = (
        default_benchmark_manifest_path()
        if config.benchmark_manifest_path is None
        else config.benchmark_manifest_path.expanduser().resolve()
    )
    requested_external_benchmarks = normalize_external_benchmarks(
        config.external_benchmarks,
        context="config.external_benchmarks",
        allow_empty=True,
    )
    datasets, benchmark_tasks, benchmark_surface = _load_runtime_benchmark_surface(
        benchmark_manifest_path,
    )
    task_type = str(benchmark_surface["task_type"])
    allow_missing_values = bool(benchmark_surface["allow_missing_values"])
    benchmark_bundle_summary = benchmark_surface.get("benchmark_bundle")
    if not isinstance(benchmark_bundle_summary, Mapping):
        raise RuntimeError(
            "benchmark comparison requires source bundle provenance in the manifest metadata: "
            f"{benchmark_manifest_path}"
        )
    reuse_curve_path = _resolve_reuse_curve_path(config)
    reuse_nanotabpfn_error = _resolve_reuse_nanotabpfn_error(config)
    if reuse_curve_path is not None and reuse_nanotabpfn_error is not None:
        raise RuntimeError(
            "reuse_nanotabpfn_curve_path and reuse_nanotabpfn_error are mutually exclusive"
        )
    if reuse_curve_path is not None and EXTERNAL_BENCHMARK_NANOTABPFN not in requested_external_benchmarks:
        raise RuntimeError(
            "reuse_nanotabpfn_curve_path requires config.external_benchmarks to include 'nanotabpfn'"
        )
    if (
        reuse_nanotabpfn_error is not None
        and EXTERNAL_BENCHMARK_NANOTABPFN not in requested_external_benchmarks
    ):
        raise RuntimeError(
            "reuse_nanotabpfn_error requires config.external_benchmarks to include 'nanotabpfn'"
        )
    if (
        config.reuse_nanotabpfn_metadata is not None
        and EXTERNAL_BENCHMARK_NANOTABPFN not in requested_external_benchmarks
    ):
        raise RuntimeError(
            "reuse_nanotabpfn_metadata requires config.external_benchmarks to include 'nanotabpfn'"
        )
    if (
        task_type == "supervised_regression"
        and EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks
    ):
        raise RuntimeError("nanoTabPFN external benchmark does not support regression bundles")
    tab_foundry_run_dir = _validate_tab_foundry_run_dir(config.tab_foundry_run_dir)
    require_nanotabpfn_environment = EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks and (
        (reuse_curve_path is None and reuse_nanotabpfn_error is None)
        or (
            reuse_curve_path is not None
            and config.reuse_nanotabpfn_metadata is None
        )
    )
    nanotabpfn_root: Path | None = None
    nanotabpfn_python: Path | None = None
    prior_dump: Path | None = None
    tabiclv2_root: Path | None = None
    tabiclv2_python: Path | None = None
    if require_nanotabpfn_environment:
        nanotabpfn_root, prior_dump = _validate_nanotabpfn_environment(config)
        nanotabpfn_python = _nanotabpfn_python(nanotabpfn_root)
    if EXTERNAL_BENCHMARK_TABICLV2 in requested_external_benchmarks:
        tabiclv2_root, tabiclv2_python = _validate_tabiclv2_environment(config)
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_tasks_path = out_root / "benchmark_tasks.json"
    tab_foundry_curve_path = out_root / "tab_foundry_curve.jsonl"
    nanotabpfn_curve_path = out_root / "nanotabpfn_curve.jsonl"
    tabiclv2_curve_path = out_root / "tabiclv2_curve.jsonl"
    comparison_curve_path = out_root / "comparison_curve.png"
    comparison_summary_path = out_root / "comparison_summary.json"
    benchmark_run_record_path = out_root / "benchmark_run_record.json"
    training_surface_record_path = out_root / "training_surface_record.json"
    control_baseline = None
    if config.control_baseline_id is not None:
        control_baseline = load_control_baseline_entry(
            str(config.control_baseline_id),
            registry_path=config.control_baseline_registry,
        )

    write_json(
        benchmark_tasks_path,
        {
            "manifest": benchmark_surface,
            "tasks": benchmark_tasks,
        },
    )

    tab_foundry_records = evaluate_tab_foundry_run(
        tab_foundry_run_dir,
        datasets=datasets,
        task_type=task_type,
        device=config.device,
        allow_checkpoint_failures=True,
        allow_missing_values=allow_missing_values,
        checkpoint_selection=config.tab_foundry_checkpoint_selection,
    )
    tab_foundry_records = cast(
        list[dict[str, Any]],
        summarize_checkpoint_curve(
            tab_foundry_records,
            bootstrap_samples=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
            bootstrap_confidence=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
            bootstrap_seed=DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
        )["records"],
    )
    write_jsonl(tab_foundry_curve_path, tab_foundry_records)

    nanotabpfn_records: list[dict[str, Any]] = []
    nanotabpfn_error: dict[str, Any] | None = None
    tabiclv2_records: list[dict[str, Any]] = []
    for external_benchmark in requested_external_benchmarks:
        if external_benchmark == EXTERNAL_BENCHMARK_NANOTABPFN:
            if reuse_curve_path is not None:
                nanotabpfn_records = load_jsonl(reuse_curve_path)
                if not nanotabpfn_records:
                    raise RuntimeError(f"reused nanoTabPFN curve is empty: {reuse_curve_path}")
                write_jsonl(nanotabpfn_curve_path, nanotabpfn_records)
                continue
            if reuse_nanotabpfn_error is not None:
                nanotabpfn_error = dict(reuse_nanotabpfn_error)
                continue
            if nanotabpfn_root is None:
                raise RuntimeError("nanoTabPFN environment validation did not resolve a root")
            helper_command = _nanotabpfn_helper_command(
                config=config,
                benchmark_manifest=benchmark_manifest_path,
                out_path=nanotabpfn_curve_path,
                allow_missing_values=allow_missing_values,
                helper_script_path=_helper_script_path(),
                src_root=_src_root(),
            )
            try:
                subprocess.run(
                    helper_command,
                    cwd=nanotabpfn_root,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                if not allow_missing_values:
                    raise
                nanotabpfn_error = {
                    "kind": "helper_failed_on_missing_bundle",
                    "message": str(exc),
                    "returncode": int(exc.returncode),
                }
            else:
                nanotabpfn_records = load_jsonl(nanotabpfn_curve_path)
                if not nanotabpfn_records:
                    raise RuntimeError("nanoTabPFN benchmark produced no curve records")
            continue

        if external_benchmark == EXTERNAL_BENCHMARK_TABICLV2:
            if tabiclv2_root is None or tabiclv2_python is None:
                raise RuntimeError("TabICLv2 environment validation did not resolve an interpreter")
            helper_command = _tabiclv2_helper_command(
                config=config,
                benchmark_manifest=benchmark_manifest_path,
                out_path=tabiclv2_curve_path,
                task_type=task_type,
                allow_missing_values=allow_missing_values,
                helper_script_path=_tabiclv2_helper_script_path(),
                src_root=_src_root(),
            )
            try:
                subprocess.run(
                    helper_command,
                    cwd=tabiclv2_root,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "TabICLv2 benchmark failed; ensure the sibling TabICLv2 environment exists and "
                    f"imports `tabicl`: {_tabiclv2_python(tabiclv2_root)}"
                ) from exc
            tabiclv2_records = load_jsonl(tabiclv2_curve_path)
            if not tabiclv2_records:
                raise RuntimeError("TabICLv2 benchmark produced no curve records")
            continue

        raise RuntimeError(f"unsupported external benchmark: {external_benchmark!r}")

    plot_comparison_curve(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        tabiclv2_records=tabiclv2_records,
        task_type=task_type,
        out_path=comparison_curve_path,
    )
    summary = build_comparison_summary(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        tabiclv2_records=tabiclv2_records,
        benchmark_tasks=benchmark_tasks,
        benchmark_bundle=cast(dict[str, Any], benchmark_bundle_summary),
        benchmark_bundle_path=Path(str(benchmark_bundle_summary["source_path"])),
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_manifest=benchmark_surface,
        tab_foundry_run_dir=tab_foundry_run_dir,
        task_type=task_type,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=nanotabpfn_python,
        tabiclv2_root=tabiclv2_root,
        tabiclv2_python=tabiclv2_python,
        control_baseline=control_baseline,
    )
    primary_external_benchmark = _resolve_primary_external_benchmark(
        requested_external_benchmarks,
        nanotabpfn_records=nanotabpfn_records,
        tabiclv2_records=tabiclv2_records,
    )
    nanotabpfn_summary = summary.get("nanotabpfn")
    if EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks and isinstance(nanotabpfn_summary, Mapping):
        if reuse_curve_path is not None and config.reuse_nanotabpfn_metadata is not None:
            execution_metadata = _reused_nanotabpfn_execution_metadata(
                metadata=config.reuse_nanotabpfn_metadata,
                reuse_curve_path=reuse_curve_path,
            )
        else:
            if nanotabpfn_root is None or nanotabpfn_python is None or prior_dump is None:
                raise RuntimeError(
                    "reuse_nanotabpfn_metadata is required to reuse a cached nanoTabPFN "
                    "curve without a local nanoTabPFN environment"
                )
            requested_device = str(config.device).strip()
            execution_metadata = _nanotabpfn_execution_metadata(
                requested_device=requested_device,
                resolved_device=resolve_device(requested_device),
                host_fingerprint=benchmark_host_fingerprint(),
                nanotabpfn_root=nanotabpfn_root,
                nanotabpfn_python_path=nanotabpfn_python,
                prior_dump=prior_dump,
                tab_realdata_hub_root=_resolved_tab_realdata_hub_root(config),
                steps=int(config.nanotabpfn_steps),
                eval_every=int(config.nanotabpfn_eval_every),
                seeds=int(config.nanotabpfn_seeds),
                batch_size=int(config.nanotabpfn_batch_size),
                lr=float(config.nanotabpfn_lr),
                reuse_curve_path=reuse_curve_path,
            )
        cast(dict[str, Any], nanotabpfn_summary).update(execution_metadata)
    if EXTERNAL_BENCHMARK_TABICLV2 in requested_external_benchmarks:
        tabiclv2_summary = summary.get("tabiclv2")
        if isinstance(tabiclv2_summary, Mapping):
            if tabiclv2_root is None or tabiclv2_python is None:
                raise RuntimeError("TabICLv2 environment validation did not resolve metadata paths")
            cast(dict[str, Any], tabiclv2_summary).update(
                _tabiclv2_execution_metadata(
                    requested_device=str(config.device).strip(),
                    resolved_device=resolve_device(str(config.device).strip()),
                    host_fingerprint=benchmark_host_fingerprint(),
                    tabicl_root=tabiclv2_root,
                    tabicl_python_path=tabiclv2_python,
                    checkpoint_version=_tabiclv2_checkpoint_version(
                        task_type=task_type,
                        config=config,
                    ),
                    tab_realdata_hub_root=_resolved_tab_realdata_hub_root(config),
                )
            )
    if nanotabpfn_error is not None:
        summary["nanotabpfn_error"] = nanotabpfn_error
    return _finalize_benchmark_summary(
        summary=summary,
        requested_external_benchmarks=requested_external_benchmarks,
        primary_external_benchmark=primary_external_benchmark,
        nanotabpfn_records=nanotabpfn_records,
        tabiclv2_records=tabiclv2_records,
        benchmark_tasks_path=benchmark_tasks_path,
        tab_foundry_curve_path=tab_foundry_curve_path,
        nanotabpfn_curve_path=nanotabpfn_curve_path,
        tabiclv2_curve_path=tabiclv2_curve_path,
        comparison_curve_path=comparison_curve_path,
        benchmark_manifest_path=benchmark_manifest_path,
        comparison_summary_path=comparison_summary_path,
        benchmark_run_record_path=benchmark_run_record_path,
        training_surface_record_path=training_surface_record_path,
        tab_foundry_run_dir=tab_foundry_run_dir,
    )
