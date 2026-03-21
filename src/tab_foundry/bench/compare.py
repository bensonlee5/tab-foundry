"""Manual nanoTabPFN comparison runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence, cast

from tab_foundry.bench.artifacts import load_jsonl, write_json, write_jsonl
from tab_foundry.bench.benchmark_run_registry import derive_benchmark_run_record
from tab_foundry.bench.control_baseline import (
    default_control_baseline_registry_path,
    load_control_baseline_entry,
)
from tab_foundry.bench.nanotabpfn import (
    benchmark_host_fingerprint,
    benchmark_bundle_task_type,
    default_benchmark_bundle_path,
    build_comparison_summary,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
    evaluate_tab_foundry_run,
    load_benchmark_bundle_for_execution,
    load_openml_benchmark_datasets,
    plot_comparison_curve,
    resolve_device,
    save_dataset_cache,
    summarize_checkpoint_curve,
)
from tab_foundry.training.instability import gradient_history_path, telemetry_path
from tab_foundry.training.wandb import posthoc_update_wandb_summary


DEFAULT_NANOTABPFN_STEPS = 2500
DEFAULT_NANOTABPFN_SEEDS = 2
DEFAULT_NANOTABPFN_EVAL_EVERY = 25
DEFAULT_NANOTABPFN_BATCH_SIZE = 32
DEFAULT_NANOTABPFN_LR = 4.0e-3
_BENCHMARK_METRIC_KEYS = (
    "best_step",
    "best_training_time",
    "final_step",
    "final_training_time",
    "best_roc_auc",
    "final_roc_auc",
    "best_log_loss",
    "final_log_loss",
    "best_brier_score",
    "final_brier_score",
    "best_crps",
    "final_crps",
    "best_avg_pinball_loss",
    "final_avg_pinball_loss",
    "best_picp_90",
    "final_picp_90",
    "best_to_final_roc_auc_delta",
    "best_to_final_log_loss_delta",
    "best_to_final_brier_score_delta",
    "best_to_final_crps_delta",
    "best_to_final_avg_pinball_loss_delta",
    "best_to_final_picp_90_delta",
)


@dataclass(slots=True)
class NanoTabPFNBenchmarkConfig:
    """Input configuration for the notebook-style nanoTabPFN comparison."""

    tab_foundry_run_dir: Path
    out_root: Path
    nanotabpfn_root: Path = Path("~/dev/nanoTabPFN")
    nanotab_prior_dump: Path | None = None
    device: str = "auto"
    nanotabpfn_steps: int = DEFAULT_NANOTABPFN_STEPS
    nanotabpfn_seeds: int = DEFAULT_NANOTABPFN_SEEDS
    nanotabpfn_eval_every: int = DEFAULT_NANOTABPFN_EVAL_EVERY
    nanotabpfn_batch_size: int = DEFAULT_NANOTABPFN_BATCH_SIZE
    nanotabpfn_lr: float = DEFAULT_NANOTABPFN_LR
    control_baseline_id: str | None = None
    control_baseline_registry: Path | None = None
    benchmark_bundle_path: Path | None = None
    reuse_nanotabpfn_curve_path: Path | None = None
    reuse_nanotabpfn_metadata: Mapping[str, Any] | None = None


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_nanotabpfn_benchmark_{stamp}"


def _nanotabpfn_python(root: Path) -> Path:
    return root.expanduser().resolve() / ".venv" / "bin" / "python"


def _nanotabpfn_prior_dump(root: Path, override: Path | None) -> Path:
    return (override or (root / "300k_150x5_2.h5")).expanduser().resolve()


def _helper_script_path() -> Path:
    return Path(__file__).resolve().with_name("nanotabpfn_helper.py")


def _src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_legacy_benchmark_record_compat_error(exc: Exception) -> bool:
    message = str(exc)
    return "persisted model.arch" in message or "omitted feature_group_size" in message


def _nanotabpfn_helper_command(
    *,
    config: NanoTabPFNBenchmarkConfig,
    dataset_cache: Path,
    out_path: Path,
    allow_missing_values: bool,
) -> list[str]:
    nanotab_root = config.nanotabpfn_root.expanduser().resolve()
    command = [
        str(_nanotabpfn_python(nanotab_root)),
        str(_helper_script_path()),
        "--tab-foundry-src",
        str(_src_root()),
        "--dataset-cache",
        str(dataset_cache),
        "--prior-dump",
        str(_nanotabpfn_prior_dump(nanotab_root, config.nanotab_prior_dump)),
        "--out-path",
        str(out_path),
        "--device",
        str(config.device),
        "--steps",
        str(int(config.nanotabpfn_steps)),
        "--eval-every",
        str(int(config.nanotabpfn_eval_every)),
        "--seeds",
        str(int(config.nanotabpfn_seeds)),
        "--batch-size",
        str(int(config.nanotabpfn_batch_size)),
        "--lr",
        str(float(config.nanotabpfn_lr)),
    ]
    if allow_missing_values:
        command.append("--allow-missing-values")
    return command


def _validate_tab_foundry_run_dir(path: Path) -> Path:
    tab_foundry_run_dir = path.expanduser().resolve()
    if not tab_foundry_run_dir.exists():
        raise RuntimeError(f"tab-foundry run dir does not exist: {tab_foundry_run_dir}")
    return tab_foundry_run_dir


def _validate_nanotabpfn_environment(
    config: NanoTabPFNBenchmarkConfig,
) -> tuple[Path, Path]:
    nanotabpfn_root = config.nanotabpfn_root.expanduser().resolve()
    nanotabpfn_python = _nanotabpfn_python(nanotabpfn_root)
    prior_dump = _nanotabpfn_prior_dump(nanotabpfn_root, config.nanotab_prior_dump)

    if not nanotabpfn_root.exists():
        raise RuntimeError(f"nanoTabPFN root does not exist: {nanotabpfn_root}")
    if not nanotabpfn_python.exists():
        raise RuntimeError(
            "missing nanoTabPFN interpreter at "
            f"{nanotabpfn_python}; run `tab-foundry bench env bootstrap` first"
        )
    if not prior_dump.exists():
        raise RuntimeError(f"nanoTabPFN prior dump does not exist: {prior_dump}")
    return nanotabpfn_root, prior_dump


def _resolve_reuse_curve_path(config: NanoTabPFNBenchmarkConfig) -> Path | None:
    if config.reuse_nanotabpfn_curve_path is None:
        return None
    return config.reuse_nanotabpfn_curve_path.expanduser().resolve()


def _nanotabpfn_execution_metadata(
    *,
    requested_device: str,
    resolved_device: str,
    host_fingerprint: str,
    nanotabpfn_root: Path | None,
    nanotabpfn_python: Path | None,
    prior_dump: Path | None,
    steps: int,
    eval_every: int,
    seeds: int,
    batch_size: int,
    lr: float,
    reuse_curve_path: Path | None,
) -> dict[str, Any]:
    return {
        "root": None if nanotabpfn_root is None else str(nanotabpfn_root.expanduser().resolve()),
        "python": None
        if nanotabpfn_python is None
        else str(nanotabpfn_python.expanduser().resolve()),
        "num_seeds": int(seeds),
        "device": str(requested_device),
        "resolved_device": str(resolved_device),
        "benchmark_host_fingerprint": str(host_fingerprint),
        "prior_dump_path": None if prior_dump is None else str(prior_dump.expanduser().resolve()),
        "steps": int(steps),
        "eval_every": int(eval_every),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "curve_source_mode": "reused" if reuse_curve_path is not None else "fresh",
        "reused_curve_path": (
            None
            if reuse_curve_path is None
            else str(reuse_curve_path.expanduser().resolve())
        ),
    }


def _fresh_nanotabpfn_execution_metadata(
    *,
    config: NanoTabPFNBenchmarkConfig,
    nanotabpfn_root: Path,
    nanotabpfn_python: Path,
    prior_dump: Path,
    reuse_curve_path: Path | None,
) -> dict[str, Any]:
    requested_device = str(config.device).strip()
    return _nanotabpfn_execution_metadata(
        requested_device=requested_device,
        resolved_device=resolve_device(requested_device),
        host_fingerprint=benchmark_host_fingerprint(),
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=nanotabpfn_python,
        prior_dump=prior_dump,
        steps=int(config.nanotabpfn_steps),
        eval_every=int(config.nanotabpfn_eval_every),
        seeds=int(config.nanotabpfn_seeds),
        batch_size=int(config.nanotabpfn_batch_size),
        lr=float(config.nanotabpfn_lr),
        reuse_curve_path=reuse_curve_path,
    )


def _required_reuse_metadata_string(
    metadata: Mapping[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"reuse_nanotabpfn_metadata.{key} must be a non-empty string")
    return str(value).strip()


def _optional_reuse_metadata_path(
    metadata: Mapping[str, Any],
    key: str,
) -> Path | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"reuse_nanotabpfn_metadata.{key} must be a non-empty string when provided")
    return Path(str(value)).expanduser().resolve()


def _reused_nanotabpfn_execution_metadata(
    *,
    metadata: Mapping[str, Any],
    reuse_curve_path: Path,
) -> dict[str, Any]:
    nanotabpfn_root = _optional_reuse_metadata_path(metadata, "root")
    nanotabpfn_python = _optional_reuse_metadata_path(metadata, "python")
    prior_dump = _optional_reuse_metadata_path(metadata, "prior_dump_path")
    seeds = metadata.get("num_seeds", metadata.get("seeds"))
    if not isinstance(seeds, int) or isinstance(seeds, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.num_seeds must be an integer")
    steps = metadata.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.steps must be an integer")
    eval_every = metadata.get("eval_every")
    if not isinstance(eval_every, int) or isinstance(eval_every, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.eval_every must be an integer")
    batch_size = metadata.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.batch_size must be an integer")
    lr = metadata.get("lr")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        raise RuntimeError("reuse_nanotabpfn_metadata.lr must be numeric")
    return _nanotabpfn_execution_metadata(
        requested_device=_required_reuse_metadata_string(metadata, "device"),
        resolved_device=_required_reuse_metadata_string(metadata, "resolved_device"),
        host_fingerprint=_required_reuse_metadata_string(metadata, "benchmark_host_fingerprint"),
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=nanotabpfn_python,
        prior_dump=prior_dump,
        steps=int(steps),
        eval_every=int(eval_every),
        seeds=int(seeds),
        batch_size=int(batch_size),
        lr=float(lr),
        reuse_curve_path=reuse_curve_path,
    )


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        return None
    return cast(Mapping[str, Any], raw_value)


def _compact_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in _BENCHMARK_METRIC_KEYS:
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _benchmark_wandb_summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_payload: dict[str, Any] = {"benchmark": {}}
    tab_foundry = _mapping_value(summary, "tab_foundry")
    if tab_foundry is not None:
        tab_foundry_payload = _compact_metric_payload(tab_foundry)
        training_diagnostics = _mapping_value(tab_foundry, "training_diagnostics")
        if training_diagnostics:
            tab_foundry_payload["training_diagnostics"] = dict(training_diagnostics)
        if tab_foundry_payload:
            benchmark_payload["benchmark"]["tab_foundry"] = tab_foundry_payload
        model_size = _mapping_value(tab_foundry, "model_size")
        if model_size:
            benchmark_payload["benchmark"]["model_size"] = dict(model_size)

    nanotabpfn = _mapping_value(summary, "nanotabpfn")
    if nanotabpfn is not None:
        nanotabpfn_payload = _compact_metric_payload(nanotabpfn)
        for key in ("num_seeds",):
            if key in nanotabpfn:
                nanotabpfn_payload[key] = nanotabpfn[key]
        if nanotabpfn_payload:
            benchmark_payload["benchmark"]["nanotabpfn"] = nanotabpfn_payload
    return benchmark_payload if benchmark_payload["benchmark"] else {}


def run_nanotabpfn_benchmark(config: NanoTabPFNBenchmarkConfig) -> dict[str, Any]:
    """Run the notebook-style tab-foundry vs nanoTabPFN comparison."""

    benchmark_bundle_path = (
        default_benchmark_bundle_path()
        if config.benchmark_bundle_path is None
        else config.benchmark_bundle_path.expanduser().resolve()
    )
    benchmark_bundle, allow_missing_values = load_benchmark_bundle_for_execution(
        benchmark_bundle_path
    )
    task_type = benchmark_bundle_task_type(benchmark_bundle)
    require_external_control = task_type != "supervised_regression"
    reuse_curve_path = _resolve_reuse_curve_path(config)
    tab_foundry_run_dir = _validate_tab_foundry_run_dir(config.tab_foundry_run_dir)
    require_nanotabpfn_environment = require_external_control and (
        reuse_curve_path is None or config.reuse_nanotabpfn_metadata is None
    )
    nanotabpfn_root: Path | None = None
    nanotabpfn_python: Path | None = None
    prior_dump: Path | None = None
    if require_nanotabpfn_environment:
        nanotabpfn_root, prior_dump = _validate_nanotabpfn_environment(config)
        nanotabpfn_python = _nanotabpfn_python(nanotabpfn_root)
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_tasks_path = out_root / "benchmark_tasks.json"
    dataset_cache_path = out_root / "benchmark_dataset_cache.npz"
    tab_foundry_curve_path = out_root / "tab_foundry_curve.jsonl"
    nanotabpfn_curve_path = out_root / "nanotabpfn_curve.jsonl"
    comparison_curve_path = out_root / "comparison_curve.png"
    comparison_summary_path = out_root / "comparison_summary.json"
    benchmark_run_record_path = out_root / "benchmark_run_record.json"
    training_surface_record_path = out_root / "training_surface_record.json"
    benchmark_selection = cast(dict[str, Any], benchmark_bundle["selection"])
    benchmark_new_instances = int(benchmark_selection["new_instances"])
    control_baseline = None
    if config.control_baseline_id is not None:
        control_baseline = load_control_baseline_entry(
            str(config.control_baseline_id),
            registry_path=config.control_baseline_registry,
        )

    datasets, benchmark_tasks = load_openml_benchmark_datasets(
        new_instances=benchmark_new_instances,
        benchmark_bundle_path=benchmark_bundle_path,
        allow_missing_values=allow_missing_values,
    )
    write_json(benchmark_tasks_path, benchmark_bundle)
    save_dataset_cache(dataset_cache_path, datasets)

    tab_foundry_records = evaluate_tab_foundry_run(
        tab_foundry_run_dir,
        datasets=datasets,
        task_type=task_type,
        device=config.device,
        allow_checkpoint_failures=True,
        allow_missing_values=allow_missing_values,
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
    if require_external_control:
        if reuse_curve_path is not None:
            nanotabpfn_records = load_jsonl(reuse_curve_path)
            if not nanotabpfn_records:
                raise RuntimeError(f"reused nanoTabPFN curve is empty: {reuse_curve_path}")
            write_jsonl(nanotabpfn_curve_path, nanotabpfn_records)
        else:
            assert nanotabpfn_root is not None
            helper_command = _nanotabpfn_helper_command(
                config=config,
                dataset_cache=dataset_cache_path,
                out_path=nanotabpfn_curve_path,
                allow_missing_values=allow_missing_values,
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

    plot_comparison_curve(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        task_type=task_type,
        out_path=comparison_curve_path,
    )
    summary = build_comparison_summary(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        benchmark_tasks=benchmark_tasks,
        benchmark_bundle=benchmark_bundle,
        benchmark_bundle_path=benchmark_bundle_path,
        tab_foundry_run_dir=tab_foundry_run_dir,
        task_type=task_type,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=nanotabpfn_python,
        control_baseline=control_baseline,
    )
    nanotabpfn_summary = summary.get("nanotabpfn")
    if require_external_control and isinstance(nanotabpfn_summary, Mapping):
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
            execution_metadata = _fresh_nanotabpfn_execution_metadata(
                config=config,
                nanotabpfn_root=nanotabpfn_root,
                nanotabpfn_python=nanotabpfn_python,
                prior_dump=prior_dump,
                reuse_curve_path=reuse_curve_path,
            )
        cast(dict[str, Any], nanotabpfn_summary).update(
            execution_metadata
        )
    if nanotabpfn_error is not None:
        summary["nanotabpfn_error"] = nanotabpfn_error
    gradient_history_jsonl = gradient_history_path(tab_foundry_run_dir)
    telemetry_json = telemetry_path(tab_foundry_run_dir)
    summary["artifacts"] = {
        "benchmark_tasks_json": str(benchmark_tasks_path),
        "tab_foundry_curve_jsonl": str(tab_foundry_curve_path),
        "nanotabpfn_curve_jsonl": (
            str(nanotabpfn_curve_path) if require_external_control and nanotabpfn_records else None
        ),
        "comparison_curve_png": str(comparison_curve_path),
        "benchmark_dataset_cache": str(dataset_cache_path),
        "gradient_history_jsonl": (
            str(gradient_history_jsonl.resolve()) if gradient_history_jsonl.exists() else None
        ),
        "telemetry_json": str(telemetry_json.resolve()) if telemetry_json.exists() else None,
        "benchmark_run_record_json": str(benchmark_run_record_path),
        "training_surface_record_json": str(training_surface_record_path),
    }
    write_json(comparison_summary_path, summary)
    try:
        benchmark_run_record = derive_benchmark_run_record(
            run_dir=tab_foundry_run_dir,
            comparison_summary_path=comparison_summary_path,
            benchmark_run_record_path=benchmark_run_record_path,
        )
    except (RuntimeError, ValueError) as exc:
        if not _is_legacy_benchmark_record_compat_error(exc):
            raise
        print(
            "Skipping benchmark_run_record.json derivation for legacy checkpoint metadata: "
            f"{exc}",
            file=sys.stderr,
        )
        summary["artifacts"]["benchmark_run_record_json"] = None
        summary["artifacts"]["training_surface_record_json"] = None
        cast(dict[str, Any], summary["tab_foundry"])[
            "benchmark_run_record_warning"
        ] = str(exc)
        write_json(comparison_summary_path, summary)
        return summary
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
        telemetry_path=telemetry_json,
        payload=_benchmark_wandb_summary_payload(summary),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a completed tab-foundry run against nanoTabPFN")
    parser.add_argument(
        "--tab-foundry-run-dir",
        required=True,
        help="Completed tab-foundry run directory with checkpoint snapshots",
    )
    parser.add_argument("--nanotabpfn-root", default="~/dev/nanoTabPFN", help="Local nanoTabPFN checkout")
    parser.add_argument("--nanotab-prior-dump", default=None, help="Path to nanoTabPFN prior dump (.h5)")
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Benchmark device",
    )
    parser.add_argument(
        "--nanotabpfn-steps",
        type=int,
        default=DEFAULT_NANOTABPFN_STEPS,
        help="nanoTabPFN training steps",
    )
    parser.add_argument(
        "--nanotabpfn-seeds",
        type=int,
        default=DEFAULT_NANOTABPFN_SEEDS,
        help="Number of nanoTabPFN random seeds",
    )
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id to copy into comparison_summary.json",
    )
    parser.add_argument(
        "--control-baseline-registry",
        default=str(default_control_baseline_registry_path()),
        help="Control baseline registry JSON path used with --control-baseline-id",
    )
    parser.add_argument(
        "--benchmark-bundle-path",
        default=None,
        help="Optional repo-tracked OpenML benchmark bundle JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=Path(str(args.tab_foundry_run_dir)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            nanotabpfn_root=Path(str(args.nanotabpfn_root)),
            nanotab_prior_dump=(Path(str(args.nanotab_prior_dump)) if args.nanotab_prior_dump else None),
            device=str(args.device),
            nanotabpfn_steps=int(args.nanotabpfn_steps),
            nanotabpfn_seeds=int(args.nanotabpfn_seeds),
            control_baseline_id=(
                str(args.control_baseline_id) if args.control_baseline_id else None
            ),
            control_baseline_registry=(
                Path(str(args.control_baseline_registry))
                if args.control_baseline_registry
                else None
            ),
            benchmark_bundle_path=(
                Path(str(args.benchmark_bundle_path))
                if args.benchmark_bundle_path
                else None
            ),
        )
    )
    print("nanoTabPFN comparison complete:")
    print(f"  dataset_count={summary['dataset_count']}")
    print(f"  tab_foundry={summary['tab_foundry']}")
    if "nanotabpfn" in summary:
        print(f"  nanotabpfn={summary['nanotabpfn']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
