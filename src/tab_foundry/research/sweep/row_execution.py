"""Row-level execution orchestration for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf
from pydantic import ValidationError

from tab_foundry.benchmark_registry import resolve_registry_path_value
from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    normalize_external_benchmarks,
)
from tab_foundry.bench.comparison_contract import (
    DEFAULT_NANOTABPFN_BATCH_SIZE as _DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY as _DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR as _DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS as _DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS as _DEFAULT_NANOTABPFN_STEPS,
    BenchmarkComparisonConfig,
)
from tab_foundry.bench.comparison_runtime import (
    run_nanotabpfn_benchmark,
)
from tab_foundry.bench.run_registration import register_benchmark_run
from tab_foundry.research.lane_contract import resolve_sweep_semantics
from tab_foundry.training.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.surface import (
    TRAINING_BACKEND_LEGACY_PRIOR,
    TRAINING_BACKEND_MANIFEST,
)
from tab_foundry.training.trainer import train as train_from_manifest_cfg
from tab_foundry.training.wandb import posthoc_update_wandb_summary

from . import curve_reuse as _curve_reuse
from . import row_dependencies as _row_dependencies
from . import training_state as _training_state
from .artifacts import ExecutionPaths
from .configuration import (
    compose_cfg,
    resolve_training_backend,
    row_id_for_order,
    validate_one_epoch_contract,
)
from .models import DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS, SweepPayload
from .objective_metrics import (
    first_present_metric_key,
    objective_metric_from_run,
    preferred_final_metric_keys,
)
from .queue_updates import optional_metric, queue_metrics, update_queue_row, update_screened_queue_row
from .reporting import result_card_text, write_research_package
from .runtime_env import ensure_nanotabpfn_python
from .screening import screen_metrics
from .surface_resolution import build_lightweight_training_surface_record

DEFAULT_DEVICE = "cuda"
DEFAULT_NANOTABPFN_STEPS = _DEFAULT_NANOTABPFN_STEPS
DEFAULT_NANOTABPFN_SEEDS = _DEFAULT_NANOTABPFN_SEEDS
DEFAULT_NANOTABPFN_EVAL_EVERY = _DEFAULT_NANOTABPFN_EVAL_EVERY
DEFAULT_NANOTABPFN_BATCH_SIZE = _DEFAULT_NANOTABPFN_BATCH_SIZE
DEFAULT_NANOTABPFN_LR = _DEFAULT_NANOTABPFN_LR
DEFAULT_TRACK = "system_delta_binary_medium_v1"
CLASSIFICATION_SCALING_LAW_TRACK = "system_delta_classification_medium_v1"
DEFAULT_BUDGET_CLASS = "short-run"
DEFAULT_DECISION = "defer"
DEFAULT_BENCHMARK_CHECKPOINT_SELECTION = "all"
DEFAULT_CONCLUSION = (
    "Canonical benchmark comparison recorded against the locked sweep anchor; "
    "interpret this row in the full sweep context."
)
ALLOWED_DECISIONS = {"keep", "defer", "reject"}
SCREEN_ONLY_POLICY = "screen_only"
BENCHMARK_FULL_POLICY = "benchmark_full"
NANOTABPFN_REUSE_ONLY_MISSING_KIND = "reuse_only_missing"


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        return None
    return cast(Mapping[str, Any], raw_value)


def _append_unique_queue_note(queue_row: dict[str, Any], note: str) -> None:
    notes = queue_row.get("notes")
    normalized_notes = [str(item) for item in notes] if isinstance(notes, list) else []
    if note not in normalized_notes:
        normalized_notes.append(note)
    queue_row["notes"] = normalized_notes


def _reuse_train_artifact_payload(
    row: Mapping[str, Any],
) -> tuple[str, Path, str] | None:
    raw_payload = row.get("reuse_train_artifact")
    if raw_payload is None:
        return None
    if not isinstance(raw_payload, Mapping):
        raise RuntimeError("reuse_train_artifact must be a mapping when present")
    raw_run_dir = raw_payload.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        raise RuntimeError("reuse_train_artifact.run_dir must be a non-empty string")
    raw_fingerprint = raw_payload.get("training_surface_fingerprint")
    if not isinstance(raw_fingerprint, str) or not raw_fingerprint.strip():
        raise RuntimeError(
            "reuse_train_artifact.training_surface_fingerprint must be a non-empty string"
        )
    normalized_run_dir = str(raw_run_dir).strip()
    return (
        normalized_run_dir,
        resolve_registry_path_value(normalized_run_dir),
        str(raw_fingerprint).strip(),
    )


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


def _benchmark_checkpoint_selection(
    queue_row: Mapping[str, Any],
    materialized_row: Mapping[str, Any],
) -> str:
    for payload in (materialized_row, queue_row):
        raw_value = payload.get("benchmark_checkpoint_selection")
        if isinstance(raw_value, str) and raw_value.strip():
            return str(raw_value).strip().lower()
    return DEFAULT_BENCHMARK_CHECKPOINT_SELECTION


def _optional_typed_sweep(sweep_meta: Mapping[str, Any]) -> SweepPayload | None:
    try:
        return SweepPayload.model_validate(sweep_meta)
    except ValidationError:
        return None


def _sweep_external_benchmarks(
    sweep: SweepPayload | None,
    *,
    sweep_meta: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_values = sweep.external_benchmarks if sweep is not None else sweep_meta.get("external_benchmarks")
    return tuple(
        normalize_external_benchmarks(
            raw_values,
            default=DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
            context="sweep.external_benchmarks",
            allow_empty=True,
        )
    )


def _registration_track(
    sweep: SweepPayload | None,
    *,
    sweep_meta: Mapping[str, Any],
) -> str:
    surface_role_raw = sweep.surface_role if sweep is not None else sweep_meta.get("surface_role")
    surface_role = str(surface_role_raw).strip().lower() if surface_role_raw is not None else ""
    if surface_role == "classification_scaling_law":
        return CLASSIFICATION_SCALING_LAW_TRACK
    return DEFAULT_TRACK


def run_row(
    *,
    sweep_id: str,
    sweep_meta: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
    anchor_run_id: str | None,
    parent_run_id: str | None,
    queue: Mapping[str, Any],
    prior_dump: Path | None,
    nanotabpfn_root: Path | None,
    device: str,
    fallback_python: Path,
    decision: str,
    conclusion: str,
    paths: ExecutionPaths,
    reuse_nanotabpfn_only: bool = False,
) -> str:
    execution_policy = _normalize_execution_policy(queue_row)
    benchmark_checkpoint_selection = _benchmark_checkpoint_selection(queue_row, materialized_row)
    sweep = _optional_typed_sweep(sweep_meta)
    _row_dependencies.resolve_dynamic_model_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    _row_dependencies.resolve_dynamic_training_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    _row_dependencies.resolve_dynamic_reuse_train_artifact(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    resolved_sweep_meta = sweep if sweep is not None else sweep_meta
    sweep_semantics = resolve_sweep_semantics(resolved_sweep_meta)
    training_surface = sweep_semantics.training_surface
    external_benchmarks = _sweep_external_benchmarks(sweep, sweep_meta=sweep_meta)
    if EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks and nanotabpfn_root is None:
        raise RuntimeError(
            "--nanotabpfn-root is required when sweep external_benchmarks include 'nanotabpfn'"
        )
    delta_root = (
        paths.repo_root
        / "outputs"
        / "staged_ladder"
        / "research"
        / sweep_id
        / str(queue_row["delta_ref"])
    )
    existing_run_id = queue_row.get("run_id")
    row_status = str(queue_row.get("status", "")).strip().lower()
    run_id = row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
        delta_root=delta_root,
        registry_path=paths.registry_path,
        allow_existing_unregistered=row_status in {"running", "failed"},
    )
    run_root = delta_root / run_id
    train_dir = run_root / "train"
    benchmark_dir = run_root / "benchmark"
    validate_one_epoch_contract(
        materialized_row,
        repo_root=paths.repo_root,
        sweep_id=sweep_id,
        sweeps_root=paths.sweeps_root,
    )

    write_research_package(
        delta_root=delta_root,
        materialized_row=materialized_row,
        queue_row=queue_row,
        sweep_meta=sweep_meta,
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        device=device,
        training_surface=training_surface,
    )
    cfg = compose_cfg(
        row=materialized_row,
        run_dir=train_dir,
        device=device,
        training_surface=training_surface,
        sweep_id=sweep_id,
        sweeps_root=paths.sweeps_root,
    )
    expected_training_surface_record: Mapping[str, Any] | None = None
    if OmegaConf.is_config(cfg):
        raw_cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw_cfg, Mapping):
            raise RuntimeError("resolved sweep row cfg must be a mapping")
        expected_training_surface_record = build_lightweight_training_surface_record(
            raw_cfg=cast(Mapping[str, Any], raw_cfg),
            run_dir=train_dir,
        )
    expected_surface_fingerprint = (
        _training_state.training_surface_record_fingerprint(expected_training_surface_record)
        if expected_training_surface_record is not None
        else None
    )
    tracked_surface_fingerprint = (
        str(materialized_row["resolved_surface_fingerprint"])
        if isinstance(materialized_row.get("resolved_surface_fingerprint"), str)
        else None
    )
    if (
        expected_surface_fingerprint is not None
        and tracked_surface_fingerprint is not None
        and expected_surface_fingerprint != tracked_surface_fingerprint
    ):
        raise RuntimeError(
            "resolved_queue surface fingerprint mismatch for "
            f"sweep {sweep_id!r} row {int(queue_row['order']):02d}: "
            f"expected={expected_surface_fingerprint} tracked={tracked_surface_fingerprint}"
        )
    reuse_train_artifact = _reuse_train_artifact_payload(materialized_row)
    if reuse_train_artifact is None:
        reuse_train_artifact = _reuse_train_artifact_payload(queue_row)
    suppress_reused_artifact_wandb = reuse_train_artifact is not None
    training_backend = resolve_training_backend(
        cfg,
        allow_unresolved_corpus_ref=reuse_train_artifact is not None,
    )
    effective_train_dir = train_dir
    if reuse_train_artifact is not None:
        configured_reuse_run_dir, effective_train_dir, reuse_surface_fingerprint = reuse_train_artifact
        current_surface_fingerprint = expected_surface_fingerprint or tracked_surface_fingerprint
        if (
            current_surface_fingerprint is not None
            and reuse_surface_fingerprint != current_surface_fingerprint
        ):
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] pinned reusable train artifact does not match "
                "the resolved queue contract: "
                f"expected={current_surface_fingerprint} pinned={reuse_surface_fingerprint}"
            )
        if not _training_state.completed_train_artifacts_exist(
            effective_train_dir,
            expected_backend=training_backend,
        ):
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] pinned reusable train artifact is missing or incomplete: "
                f"{configured_reuse_run_dir}"
            )
        observed_training_surface_record = _training_state.load_training_surface_record(
            effective_train_dir / "training_surface_record.json"
        )
        if observed_training_surface_record is None:
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] reusable training surface record is missing at "
                f"{effective_train_dir / 'training_surface_record.json'}"
            )
        observed_surface_fingerprint = _training_state.training_surface_record_fingerprint(
            observed_training_surface_record
        )
        if observed_surface_fingerprint != reuse_surface_fingerprint:
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] pinned reusable train artifact fingerprint mismatch: "
                f"expected={reuse_surface_fingerprint} observed={observed_surface_fingerprint}"
            )
        _append_unique_queue_note(
            queue_row,
            f"Benchmarked pinned reusable training artifact `{configured_reuse_run_dir}`.",
        )
        print(
            f"[row {int(queue_row['order']):02d}] reusing pinned train artifacts",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            f"source_run_dir={effective_train_dir}",
            flush=True,
        )
    elif _training_state.completed_train_artifacts_exist(
        train_dir,
        expected_backend=training_backend,
        expected_training_surface_record=expected_training_surface_record,
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
            and (
                existing_backend != training_backend
                or not _training_state.completed_train_artifacts_exist(
                    train_dir,
                    expected_backend=training_backend,
                    expected_training_surface_record=expected_training_surface_record,
                )
            )
        ):
            print(
                f"[row {int(queue_row['order']):02d}] existing train artifacts are not reusable",
                f"expected_backend={training_backend}",
                f"observed_backend={existing_backend or 'missing'}",
                flush=True,
            )
            archived_train_dir = _training_state.archive_incompatible_train_dir(train_dir)
            if archived_train_dir is not None:
                print(
                    f"[row {int(queue_row['order']):02d}] archived incompatible train dir",
                    f"run_id={run_id}",
                    f"archived_dir={archived_train_dir}",
                    flush=True,
                )
        else:
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
        elif training_backend == TRAINING_BACKEND_LEGACY_PRIOR:
            if prior_dump is None:
                raise RuntimeError(
                    f"[row {int(queue_row['order']):02d}] "
                    "legacy-prior training requires --nanotabpfn-prior-dump"
                )
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
        effective_train_dir = train_dir

    if reuse_train_artifact is None:
        observed_training_surface_record = _training_state.load_training_surface_record(
            effective_train_dir / "training_surface_record.json"
        )
        if observed_training_surface_record is None:
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] training surface record is missing at "
                f"{effective_train_dir / 'training_surface_record.json'}"
            )
        observed_surface_fingerprint = _training_state.training_surface_record_fingerprint(
            observed_training_surface_record
        )
        if expected_surface_fingerprint is not None and observed_surface_fingerprint != expected_surface_fingerprint:
            raise RuntimeError(
                f"[row {int(queue_row['order']):02d}] executed training surface does not match resolved queue "
                f"contract: expected={expected_surface_fingerprint} observed={observed_surface_fingerprint}"
            )

    if execution_policy == SCREEN_ONLY_POLICY:
        row_screen_metrics = screen_metrics(run_dir=effective_train_dir)
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
        assert nanotabpfn_root is not None
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
        if reuse_nanotabpfn_only:
            reuse_nanotabpfn_error = {
                "kind": NANOTABPFN_REUSE_ONLY_MISSING_KIND,
                "message": (
                    "reuse-only execution requested but no reusable nanoTabPFN benchmark "
                    "artifact was available locally"
                ),
            }
            print(
                f"[row {int(queue_row['order']):02d}] skipping fresh nanoTabPFN helper",
                "reason=reuse_only_requested",
                flush=True,
            )
        else:
            assert nanotabpfn_root is not None
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
            f"[row {int(queue_row['order']):02d}] running benchmark evaluation",
            (
                f"comparators={','.join(external_benchmarks)}"
                if external_benchmarks
                else "comparators=none"
            ),
            flush=True,
        )
    summary = run_nanotabpfn_benchmark(
        BenchmarkComparisonConfig(
            tab_foundry_run_dir=effective_train_dir,
            out_root=benchmark_dir,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=prior_dump,
            device=device,
            tab_foundry_checkpoint_selection=benchmark_checkpoint_selection,
            control_baseline_id=str(
                sweep.control_baseline_id if sweep is not None else sweep_meta["control_baseline_id"]
            ),
            control_baseline_registry=paths.control_baseline_registry_path,
            benchmark_manifest_path=resolve_registry_path_value(
                str(sweep.benchmark_manifest_path if sweep is not None else sweep_meta["benchmark_manifest_path"])
            ),
            external_benchmarks=external_benchmarks,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
            reuse_nanotabpfn_error=reuse_nanotabpfn_error,
            reuse_nanotabpfn_metadata=(None if reuse_selection is None else reuse_selection.metadata),
            suppress_reused_artifact_wandb=suppress_reused_artifact_wandb,
        )
    )
    parent_sweep_id = sweep.parent_sweep_id if sweep is not None else sweep_meta.get("parent_sweep_id")
    registration = register_benchmark_run(
        run_id=run_id,
        track=_registration_track(sweep, sweep_meta=sweep_meta),
        experiment=training_surface.training_experiment,
        config_profile=training_surface.training_config_profile,
        budget_class=DEFAULT_BUDGET_CLASS,
        run_dir=effective_train_dir,
        comparison_summary_path=benchmark_dir / "comparison_summary.json",
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=None,
        control_baseline_id=str(
            sweep.control_baseline_id if sweep is not None else sweep_meta["control_baseline_id"]
        ),
        sweep_id=sweep_id,
        delta_id=str(queue_row["delta_ref"]),
        parent_sweep_id=(
            None
            if parent_sweep_id is None or not parent_sweep_id.strip()
            else str(parent_sweep_id)
        ),
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=paths.registry_path,
        suppress_reused_artifact_wandb=suppress_reused_artifact_wandb,
    )
    run_entry = cast(dict[str, Any], registration["run"])
    row_queue_metrics = queue_metrics(summary, run_dir=effective_train_dir, run_entry=run_entry)
    if not suppress_reused_artifact_wandb:
        _ = posthoc_update_wandb_summary(
            telemetry_path=effective_train_dir / "telemetry.json",
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
    objective_metric = objective_metric_from_run(run_entry)
    final_metric_label = first_present_metric_key(
        tab_foundry_summary,
        preferred_final_metric_keys(objective_metric),
    )
    final_metric_value = (
        None
        if final_metric_label is None
        else optional_metric(tab_foundry_summary, final_metric_label)
    )
    final_metric_text = (
        f"{final_metric_label}={final_metric_value:.4f}"
        if final_metric_label is not None and final_metric_value is not None
        else "final_metric=n/a"
    )
    print(
        f"[row {int(queue_row['order']):02d}] benchmark+registry complete",
        f"run_id={run_id}",
        final_metric_text,
        flush=True,
    )
    return run_id
