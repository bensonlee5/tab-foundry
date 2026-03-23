"""Row-level execution orchestration for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import resolve_registry_path_value
from tab_foundry.bench.comparison_runtime import (
    DEFAULT_NANOTABPFN_BATCH_SIZE as _DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY as _DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR as _DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS as _DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS as _DEFAULT_NANOTABPFN_STEPS,
    NanoTabPFNBenchmarkConfig,
    run_nanotabpfn_benchmark,
)
from tab_foundry.bench.run_registration import register_benchmark_run
from tab_foundry.external_benchmarks import EXTERNAL_BENCHMARK_NANOTABPFN
from tab_foundry.research.lane_contract import (
    resolve_surface_role,
    resolve_training_config_profile,
    resolve_training_experiment,
)
from tab_foundry.training.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.surface import (
    TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_PRIOR_DUMP,
)
from tab_foundry.training.trainer import train as train_from_manifest_cfg
from tab_foundry.training.wandb import posthoc_update_wandb_summary

from . import curve_reuse as _curve_reuse
from . import row_dependencies as _row_dependencies
from . import training_state as _training_state
from .artifacts import ExecutionPaths, result_card_text, write_research_package
from .configuration import compose_cfg, resolve_training_backend, row_id_for_order
from .queue_updates import optional_metric, queue_metrics, update_queue_row, update_screened_queue_row
from .runtime_env import ensure_nanotabpfn_python
from .screening import screen_metrics
from .validation import resolve_sweep_external_benchmarks


DEFAULT_PRIOR_DUMP = Path("/workspace/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_NANOTABPFN_ROOT = Path("/workspace/nanoTabPFN")
DEFAULT_DEVICE = "cuda"
DEFAULT_NANOTABPFN_STEPS = _DEFAULT_NANOTABPFN_STEPS
DEFAULT_NANOTABPFN_SEEDS = _DEFAULT_NANOTABPFN_SEEDS
DEFAULT_NANOTABPFN_EVAL_EVERY = _DEFAULT_NANOTABPFN_EVAL_EVERY
DEFAULT_NANOTABPFN_BATCH_SIZE = _DEFAULT_NANOTABPFN_BATCH_SIZE
DEFAULT_NANOTABPFN_LR = _DEFAULT_NANOTABPFN_LR
DEFAULT_TRACK = "system_delta_binary_medium_v1"
DEFAULT_BUDGET_CLASS = "short-run"
DEFAULT_DECISION = "defer"
DEFAULT_CONCLUSION = (
    "Canonical benchmark comparison recorded against the locked sweep anchor; "
    "interpret this row in the full sweep context."
)
ALLOWED_DECISIONS = {"keep", "defer", "reject"}
SCREEN_ONLY_POLICY = "screen_only"
BENCHMARK_FULL_POLICY = "benchmark_full"


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        return None
    return cast(Mapping[str, Any], raw_value)


def _sweep_wandb_summary_payload(
    *,
    run_entry: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    sweep_payload = _mapping_value(run_entry, "sweep")
    if sweep_payload is not None:
        payload["sweep"] = dict(sweep_payload)

    comparison_payload: dict[str, Any] = {}
    comparisons = _mapping_value(run_entry, "comparisons")
    if comparisons is not None:
        for comparison_name in ("vs_anchor", "vs_parent"):
            comparison_values = _mapping_value(comparisons, comparison_name)
            if comparison_values:
                comparison_payload[comparison_name] = dict(comparison_values)

    stage_local_payload: dict[str, Any] = {}
    for stage_label, grad_key, activation_delta_key, activation_mean_key in (
        (
            "column",
            "column_encoder_final_window_mean_grad_norm",
            "column_activation_early_to_final_mean_delta",
            "column_activation_final_window_mean",
        ),
        (
            "row",
            "row_pool_final_window_mean_grad_norm",
            "row_activation_early_to_final_mean_delta",
            "row_activation_final_window_mean",
        ),
        (
            "context",
            "context_encoder_final_window_mean_grad_norm",
            "context_activation_early_to_final_mean_delta",
            "context_activation_final_window_mean",
        ),
    ):
        stage_payload: dict[str, Any] = {}
        for source_key, target_key in (
            (grad_key, "final_window_mean_grad_norm"),
            (activation_delta_key, "activation_early_to_final_mean_delta"),
            (activation_mean_key, "activation_final_window_mean"),
        ):
            if source_key in queue_metrics:
                stage_payload[target_key] = queue_metrics[source_key]
        if stage_payload:
            stage_local_payload[stage_label] = stage_payload
    if stage_local_payload:
        comparison_payload["stage_local_stability"] = stage_local_payload

    if comparison_payload:
        payload["comparison"] = comparison_payload
    return payload


def _normalize_execution_policy(queue_row: Mapping[str, Any]) -> str:
    policy = str(queue_row.get("execution_policy", BENCHMARK_FULL_POLICY)).strip().lower()
    if policy not in {SCREEN_ONLY_POLICY, BENCHMARK_FULL_POLICY}:
        raise RuntimeError(f"unsupported execution_policy {policy!r}")
    return policy


def run_row(
    *,
    sweep_id: str,
    sweep_meta: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
    anchor_run_id: str | None,
    parent_run_id: str | None,
    queue: Mapping[str, Any],
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
    decision: str,
    conclusion: str,
    paths: ExecutionPaths,
) -> str:
    execution_policy = _normalize_execution_policy(queue_row)
    _row_dependencies.resolve_dynamic_model_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    training_experiment = resolve_training_experiment(sweep_meta)
    training_config_profile = resolve_training_config_profile(sweep_meta)
    surface_role = resolve_surface_role(sweep_meta)
    external_benchmarks = tuple(resolve_sweep_external_benchmarks(sweep_meta))
    existing_run_id = queue_row.get("run_id")
    run_id = row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
    )
    delta_root = (
        paths.repo_root
        / "outputs"
        / "staged_ladder"
        / "research"
        / sweep_id
        / str(queue_row["delta_ref"])
    )
    run_root = delta_root / run_id
    train_dir = run_root / "train"
    benchmark_dir = run_root / "benchmark"

    write_research_package(
        delta_root=delta_root,
        materialized_row=materialized_row,
        queue_row=queue_row,
        sweep_meta=sweep_meta,
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        device=device,
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
    )
    cfg = compose_cfg(
        row=queue_row,
        run_dir=train_dir,
        device=device,
        training_experiment=training_experiment,
        sweep_id=sweep_id,
    )
    training_backend = resolve_training_backend(cfg)
    if _training_state.completed_train_artifacts_exist(
        train_dir,
        expected_backend=training_backend,
    ):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        existing_backend = _training_state.training_surface_record_backend(
            train_dir / "training_surface_record.json"
        )
        if (
            _training_state.completed_train_artifacts_exist(train_dir)
            and existing_backend != training_backend
        ):
            print(
                f"[row {int(queue_row['order']):02d}] existing train artifacts are not reusable",
                f"expected_backend={training_backend}",
                f"observed_backend={existing_backend or 'missing'}",
                flush=True,
            )
        archived_train_dir = _training_state.archive_incomplete_train_dir(train_dir)
        if archived_train_dir is not None:
            print(
                f"[row {int(queue_row['order']):02d}] archived incomplete train dir",
                f"run_id={run_id}",
                f"archived_dir={archived_train_dir}",
                flush=True,
            )
        print(
            f"[row {int(queue_row['order']):02d}] starting train",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            flush=True,
        )
        if training_backend == TRAINING_BACKEND_MANIFEST:
            train_result = train_from_manifest_cfg(cfg)
        elif training_backend == TRAINING_BACKEND_PRIOR_DUMP:
            train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        else:  # pragma: no cover - guarded by resolve_training_backend
            raise RuntimeError(f"unsupported training backend {training_backend!r}")
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            f"output_dir={train_result.output_dir}",
            flush=True,
        )

    if execution_policy == SCREEN_ONLY_POLICY:
        row_screen_metrics = screen_metrics(run_dir=train_dir)
        update_screened_queue_row(
            queue_row=queue_row,
            run_id=run_id,
            screen_metrics=row_screen_metrics,
            conclusion=conclusion,
        )
        final_window_mean = row_screen_metrics.get("upper_block_final_window_mean")
        final_window_text = (
            f"upper_block_final_window_mean={float(final_window_mean):.4f}"
            if final_window_mean is not None
            else "upper_block_final_window_mean=n/a"
        )
        print(
            f"[row {int(queue_row['order']):02d}] train-only screen complete",
            f"run_id={run_id}",
            final_window_text,
            flush=True,
        )
        return run_id

    reuse_selection = None
    reuse_curve_path = None
    reuse_nanotabpfn_error = None
    if EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks:
        reuse_selection = _curve_reuse.resolve_reusable_nanotabpfn_curve(
            sweep_meta=sweep_meta,
            anchor_run_id=anchor_run_id,
            nanotabpfn_root=nanotabpfn_root,
            prior_dump=prior_dump,
            requested_device=device,
            paths=paths,
            extra_candidates=_curve_reuse.prior_completed_row_curve_candidates(
                queue=queue,
                current_order=int(queue_row["order"]),
                anchor_run_id=anchor_run_id,
                parent_run_id=parent_run_id,
                registry_path=paths.registry_path,
            ),
        )
        reuse_curve_path = None if reuse_selection is None else reuse_selection.curve_path
        reuse_nanotabpfn_error = None if reuse_selection is None else reuse_selection.reusable_error
    if reuse_selection is not None and reuse_curve_path is not None:
        print(
            f"[row {int(queue_row['order']):02d}] reusing nanoTabPFN curve",
            f"source={reuse_selection.source_label}",
            f"path={reuse_curve_path}",
            flush=True,
        )
    elif reuse_selection is not None and reuse_nanotabpfn_error is not None:
        print(
            f"[row {int(queue_row['order']):02d}] reusing nanoTabPFN benchmark outcome",
            f"source={reuse_selection.source_label}",
            f"kind={reuse_nanotabpfn_error.get('kind', 'unknown')}",
            flush=True,
        )
    elif EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks:
        _ = ensure_nanotabpfn_python(
            nanotabpfn_root=nanotabpfn_root,
            fallback_python=fallback_python,
        )
        print(
            f"[row {int(queue_row['order']):02d}] running fresh nanoTabPFN helper",
            f"device={device}",
            flush=True,
        )
    else:
        print(
            f"[row {int(queue_row['order']):02d}] running external benchmarks",
            f"comparators={','.join(external_benchmarks)}",
            flush=True,
        )
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=train_dir,
            out_root=benchmark_dir,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=prior_dump,
            device=device,
            control_baseline_id=str(sweep_meta["control_baseline_id"]),
            control_baseline_registry=paths.control_baseline_registry_path,
            benchmark_bundle_path=resolve_registry_path_value(str(sweep_meta["benchmark_bundle_path"])),
            external_benchmarks=external_benchmarks,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
            reuse_nanotabpfn_error=reuse_nanotabpfn_error,
            reuse_nanotabpfn_metadata=(None if reuse_selection is None else reuse_selection.metadata),
        )
    )
    parent_sweep_id = sweep_meta.get("parent_sweep_id")
    registration = register_benchmark_run(
        run_id=run_id,
        track=DEFAULT_TRACK,
        experiment=training_experiment,
        config_profile=training_config_profile,
        budget_class=DEFAULT_BUDGET_CLASS,
        run_dir=train_dir,
        comparison_summary_path=benchmark_dir / "comparison_summary.json",
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=None,
        control_baseline_id=str(sweep_meta["control_baseline_id"]),
        sweep_id=sweep_id,
        delta_id=str(queue_row["delta_ref"]),
        parent_sweep_id=(
            None
            if not isinstance(parent_sweep_id, str) or not parent_sweep_id.strip()
            else str(parent_sweep_id)
        ),
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=paths.registry_path,
    )
    run_entry = cast(dict[str, Any], registration["run"])
    row_queue_metrics = queue_metrics(summary, run_dir=train_dir, run_entry=run_entry)
    _ = posthoc_update_wandb_summary(
        telemetry_path=train_dir / "telemetry.json",
        payload=_sweep_wandb_summary_payload(
            run_entry=run_entry,
            queue_metrics=row_queue_metrics,
        ),
    )
    (delta_root / "result_card.md").write_text(
        result_card_text(
            row=materialized_row,
            run_id=run_id,
            anchor_run_id=anchor_run_id,
            summary=summary,
            queue_metrics=row_queue_metrics,
            decision=decision,
            conclusion=conclusion,
        ),
        encoding="utf-8",
    )
    update_queue_row(
        queue_row=queue_row,
        run_id=run_id,
        queue_metrics=row_queue_metrics,
        decision=decision,
        conclusion=conclusion,
    )
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    final_metric_label = "final_log_loss"
    final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    if final_metric_value is None:
        final_metric_label = "final_crps"
        final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    if final_metric_value is None:
        final_metric_label = "final_roc_auc"
        final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    final_metric_text = (
        f"{final_metric_label}={final_metric_value:.4f}"
        if final_metric_value is not None
        else "final_metric=n/a"
    )
    print(
        f"[row {int(queue_row['order']):02d}] benchmark+registry complete",
        f"run_id={run_id}",
        final_metric_text,
        flush=True,
    )
    return run_id
