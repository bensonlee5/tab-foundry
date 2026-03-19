"""Execution orchestration helpers for system-delta sweeps."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.bench.benchmark_run_registry import register_benchmark_run
from tab_foundry.bench.compare import NanoTabPFNBenchmarkConfig, run_nanotabpfn_benchmark
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.config import compose_config
from tab_foundry.research import system_delta
from tab_foundry.research.system_delta_promote import promote_anchor

from .artifacts import ExecutionPaths, read_yaml, result_card_text, write_research_package, write_yaml
from .queue_updates import optional_metric, queue_metrics, update_queue_row
from .selection import select_queue_rows


DEFAULT_PRIOR_DUMP = Path("/workspace/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_NANOTABPFN_ROOT = Path("/workspace/nanoTabPFN")
DEFAULT_DEVICE = "cuda"
DEFAULT_TRACK = "system_delta_binary_medium_v1"
DEFAULT_EXPERIMENT = "cls_benchmark_staged_prior"
DEFAULT_CONFIG_PROFILE = "cls_benchmark_staged_prior"
DEFAULT_BUDGET_CLASS = "short-run"
DEFAULT_DECISION = "defer"
DEFAULT_CONCLUSION = (
    "Canonical benchmark comparison recorded against the locked sweep anchor; "
    "interpret this row in the full sweep context."
)
ALLOWED_DECISIONS = {"keep", "defer", "reject"}


def row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
    nanotab_python = nanotabpfn_root / ".venv" / "bin" / "python"
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    if nanotab_python.exists() and not nanotab_python.is_symlink():
        return nanotab_python
    if nanotab_python.exists() or nanotab_python.is_symlink():
        nanotab_python.unlink()
    nanotab_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(fallback_python.resolve()))} \"$@\"\n",
        encoding="utf-8",
    )
    nanotab_python.chmod(0o755)
    return nanotab_python


def completed_train_artifacts_exist(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "train_history.jsonl",
        run_dir / "gradient_history.jsonl",
        run_dir / "training_surface_record.json",
        run_dir / "checkpoints" / "latest.pt",
    )
    return all(path.exists() for path in required_paths)


def materialized_row_map(*, sweep_id: str, paths: ExecutionPaths) -> dict[str, dict[str, Any]]:
    materialized = system_delta.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["delta_id"]): row for row in rows}


def apply_mapping(cfg: DictConfig, prefix: str, payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        merge = not (
            prefix == "model"
            and key == "module_overrides"
            and isinstance(value, Mapping)
        )
        OmegaConf.update(cfg, f"{prefix}.{key}", value, merge=merge)


def _queue_aware_run_name(*, run_dir: Path) -> str:
    return str(run_dir.parent.name if run_dir.name == "train" else run_dir.name)


def compose_cfg(*, row: Mapping[str, Any], run_dir: Path, device: str) -> DictConfig:
    cfg = compose_config([f"experiment={DEFAULT_EXPERIMENT}"])
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.runtime.device = str(device)
    cfg.logging.run_name = _queue_aware_run_name(run_dir=run_dir)
    apply_mapping(cfg, "model", cast(Mapping[str, Any], row.get("model", {})))
    apply_mapping(cfg, "data", cast(Mapping[str, Any], row.get("data", {})))
    apply_mapping(cfg, "preprocessing", cast(Mapping[str, Any], row.get("preprocessing", {})))

    training_payload = cast(Mapping[str, Any], row.get("training", {}))
    for key in (
        "surface_label",
        "prior_dump_non_finite_policy",
        "prior_dump_batch_size",
        "prior_dump_lr_scale_rule",
        "prior_dump_batch_reference_size",
    ):
        if key in training_payload:
            OmegaConf.update(cfg, f"training.{key}", training_payload[key], merge=True)

    overrides = cast(Mapping[str, Any], training_payload.get("overrides", {}))
    if "apply_schedule" in overrides:
        OmegaConf.update(cfg, "training.apply_schedule", overrides["apply_schedule"], merge=True)
    for key in ("optimizer", "runtime", "schedule"):
        override_payload = overrides.get(key)
        if isinstance(override_payload, dict):
            apply_mapping(cfg, key, cast(Mapping[str, Any], override_payload))
    return cfg


def sync_sweep_matrix(*, sweep_id: str, paths: ExecutionPaths) -> None:
    _ = system_delta.render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )


def sync_active_aliases_if_active(*, sweep_id: str, paths: ExecutionPaths) -> None:
    index = system_delta.load_system_delta_index(paths.index_path)
    if str(index["active_sweep_id"]) != sweep_id:
        return
    _ = system_delta.sync_active_sweep_aliases(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        registry_path=paths.registry_path,
        sweeps_root=paths.sweeps_root,
    )


def run_row(
    *,
    sweep_id: str,
    sweep_meta: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: Mapping[str, Any],
    anchor_run_id: str | None,
    parent_run_id: str | None,
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
    decision: str,
    conclusion: str,
    paths: ExecutionPaths,
) -> str:
    existing_run_id = queue_row.get("run_id")
    run_id = row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
    )
    delta_root = paths.repo_root / "outputs" / "staged_ladder" / "research" / sweep_id / str(queue_row["delta_ref"])
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
        training_experiment=DEFAULT_EXPERIMENT,
    )
    if completed_train_artifacts_exist(train_dir):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        cfg = compose_cfg(row=queue_row, run_dir=train_dir, device=device)
        train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
            f"output_dir={train_result.output_dir}",
            flush=True,
        )

    _ = ensure_nanotabpfn_python(nanotabpfn_root=nanotabpfn_root, fallback_python=fallback_python)
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=train_dir,
            out_root=benchmark_dir,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=prior_dump,
            device=device,
            control_baseline_id=str(sweep_meta["control_baseline_id"]),
            control_baseline_registry=paths.control_baseline_registry_path,
            benchmark_bundle_path=paths.repo_root / str(sweep_meta["benchmark_bundle_path"]),
        )
    )
    parent_sweep_id = sweep_meta.get("parent_sweep_id")
    registration = register_benchmark_run(
        run_id=run_id,
        track=DEFAULT_TRACK,
        experiment=DEFAULT_EXPERIMENT,
        config_profile=DEFAULT_CONFIG_PROFILE,
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
            None if not isinstance(parent_sweep_id, str) or not parent_sweep_id.strip() else str(parent_sweep_id)
        ),
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=paths.registry_path,
    )
    run_entry = cast(dict[str, Any], registration["run"])
    row_queue_metrics = queue_metrics(summary, run_dir=train_dir, run_entry=run_entry)
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


def execute_sweep(
    *,
    sweep_id: str | None,
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
    orders: list[int] | None = None,
    start_order: int | None = None,
    stop_after_order: int | None = None,
    include_completed: bool = False,
    decision_default: str = DEFAULT_DECISION,
    conclusion_default: str = DEFAULT_CONCLUSION,
    decision_overrides: Mapping[int, str] | None = None,
    conclusion_overrides: Mapping[int, str] | None = None,
    promote_first_executed_row_to_anchor: bool = False,
    paths: ExecutionPaths | None = None,
    run_row_fn: Any | None = None,
    sync_sweep_matrix_fn: Any | None = None,
    sync_active_aliases_if_active_fn: Any | None = None,
    promote_anchor_fn: Any | None = None,
) -> list[str]:
    resolved_paths = ExecutionPaths.default() if paths is None else paths
    run_row_impl = run_row if run_row_fn is None else run_row_fn
    sync_matrix_impl = sync_sweep_matrix if sync_sweep_matrix_fn is None else sync_sweep_matrix_fn
    sync_aliases_impl = sync_active_aliases_if_active if sync_active_aliases_if_active_fn is None else sync_active_aliases_if_active_fn
    promote_anchor_impl = promote_anchor if promote_anchor_fn is None else promote_anchor_fn

    sweep_meta = system_delta.load_system_delta_sweep(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_sweep_id = str(sweep_meta["sweep_id"])
    queue_path = system_delta.sweep_queue_path(resolved_sweep_id, sweeps_root=resolved_paths.sweeps_root)
    queue = read_yaml(queue_path)
    materialized_rows = materialized_row_map(sweep_id=resolved_sweep_id, paths=resolved_paths)
    selected_rows = select_queue_rows(
        queue,
        orders=orders,
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
    )
    if not selected_rows:
        print("No rows selected for execution.", f"sweep_id={resolved_sweep_id}", flush=True)
        return []

    current_anchor_run_id = sweep_meta.get("anchor_run_id")
    active_anchor = str(current_anchor_run_id) if isinstance(current_anchor_run_id, str) and current_anchor_run_id.strip() else None
    executed_run_ids: list[str] = []
    decision_map = dict(decision_overrides or {})
    conclusion_map = dict(conclusion_overrides or {})

    for index, queue_row in enumerate(selected_rows):
        order = int(queue_row["order"])
        decision = str(decision_map.get(order, decision_default)).strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
        conclusion = str(conclusion_map.get(order, conclusion_default)).strip()
        if not conclusion:
            raise RuntimeError("conclusion must be non-empty")

        promote_now = bool(promote_first_executed_row_to_anchor and index == 0)
        materialized_row = materialized_rows[str(queue_row["delta_ref"])]
        run_id = run_row_impl(
            sweep_id=resolved_sweep_id,
            sweep_meta=sweep_meta,
            queue_row=queue_row,
            materialized_row=materialized_row,
            anchor_run_id=None if promote_now else active_anchor,
            parent_run_id=None if promote_now else active_anchor,
            prior_dump=prior_dump,
            nanotabpfn_root=nanotabpfn_root,
            device=device,
            fallback_python=fallback_python,
            decision=decision,
            conclusion=conclusion,
            paths=resolved_paths,
        )
        write_yaml(queue_path, queue)
        if promote_now:
            _ = promote_anchor_impl(
                sweep_id=resolved_sweep_id,
                anchor_run_id=run_id,
                set_active=False,
                paths=resolved_paths.promotion_paths(),
            )
            active_anchor = run_id
            sweep_meta = system_delta.load_system_delta_sweep(
                resolved_sweep_id,
                index_path=resolved_paths.index_path,
                sweeps_root=resolved_paths.sweeps_root,
            )
        sync_matrix_impl(sweep_id=resolved_sweep_id, paths=resolved_paths)
        sync_aliases_impl(sweep_id=resolved_sweep_id, paths=resolved_paths)
        executed_run_ids.append(run_id)

    return executed_run_ids
