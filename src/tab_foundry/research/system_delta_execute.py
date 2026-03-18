"""Execute system-delta sweep rows through train, benchmark, and registration."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf
import yaml

from tab_foundry.bench.benchmark_run_registry import register_benchmark_run
from tab_foundry.bench.compare import NanoTabPFNBenchmarkConfig, run_nanotabpfn_benchmark
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.config import compose_config
from tab_foundry.research import system_delta
from tab_foundry.research.system_delta_promote import PromotionPaths, promote_anchor
from tab_foundry.training.instability import diagnostics_summary


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
_ALLOWED_DECISIONS = {"keep", "defer", "reject"}


@dataclass(frozen=True)
class ExecutionPaths:
    repo_root: Path
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path
    program_path: Path
    control_baseline_registry_path: Path

    @classmethod
    def default(cls) -> "ExecutionPaths":
        repo_root = system_delta.repo_root()
        return cls(
            repo_root=repo_root,
            index_path=system_delta.default_sweep_index_path(),
            catalog_path=system_delta.default_catalog_path(),
            sweeps_root=system_delta.default_sweeps_root(),
            registry_path=system_delta.default_registry_path(),
            program_path=repo_root / "program.md",
            control_baseline_registry_path=repo_root / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json",
        )

    def promotion_paths(self) -> PromotionPaths:
        return PromotionPaths(
            index_path=self.index_path,
            catalog_path=self.catalog_path,
            sweeps_root=self.sweeps_root,
            registry_path=self.registry_path,
            program_path=self.program_path,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping at {path}")
    return cast(dict[str, Any], payload)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    system_delta._write_yaml(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast(dict[str, Any], payload))
    return records


def _row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def _ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
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


def _sorted_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("queue rows must be a list")
    return sorted(cast(list[dict[str, Any]], rows), key=lambda row: int(row["order"]))


def _completed_train_artifacts_exist(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "train_history.jsonl",
        run_dir / "gradient_history.jsonl",
        run_dir / "training_surface_record.json",
        run_dir / "checkpoints" / "latest.pt",
    )
    return all(path.exists() for path in required_paths)


def _materialized_row_map(*, sweep_id: str, paths: ExecutionPaths) -> dict[str, dict[str, Any]]:
    materialized = system_delta.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["delta_id"]): row for row in rows}


def _apply_mapping(cfg: DictConfig, prefix: str, payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        OmegaConf.update(cfg, f"{prefix}.{key}", value, merge=True)


def _compose_cfg(*, row: Mapping[str, Any], run_dir: Path, device: str) -> DictConfig:
    cfg = compose_config([f"experiment={DEFAULT_EXPERIMENT}"])
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.runtime.device = str(device)
    _apply_mapping(cfg, "model", cast(Mapping[str, Any], row.get("model", {})))
    _apply_mapping(cfg, "data", cast(Mapping[str, Any], row.get("data", {})))
    _apply_mapping(cfg, "preprocessing", cast(Mapping[str, Any], row.get("preprocessing", {})))

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
            _apply_mapping(cfg, key, cast(Mapping[str, Any], override_payload))
    return cfg


def _queue_metrics(summary: Mapping[str, Any], *, run_dir: Path) -> dict[str, Any]:
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    nanotabpfn = cast(dict[str, Any], summary["nanotabpfn"])
    gradient_records = _read_jsonl(run_dir / "gradient_history.jsonl")
    training_surface_record = _read_json(run_dir / "training_surface_record.json")
    diagnostics = diagnostics_summary(gradient_records, training_surface_record=training_surface_record)
    return {
        "best_roc_auc": float(tab_foundry["best_roc_auc"]),
        "best_step": int(float(tab_foundry["best_step"])),
        "final_roc_auc": float(tab_foundry["final_roc_auc"]),
        "drift": float(tab_foundry["best_to_final_roc_auc_delta"]),
        "nanotabpfn_best": float(nanotabpfn["best_roc_auc"]),
        "nanotabpfn_final": float(nanotabpfn["final_roc_auc"]),
        "max_grad_norm": float(cast(dict[str, Any], tab_foundry["training_diagnostics"])["max_grad_norm"]),
        "clipped_step_fraction": float(cast(dict[str, Any], diagnostics["grad_clip"])["clipped_step_fraction"]),
    }


def _research_card_text(*, row: Mapping[str, Any], sweep_id: str, anchor_run_id: str | None) -> str:
    plan = cast(list[str], row.get("parameter_adequacy_plan", []))
    plan_lines = "\n".join(f"- {item}" for item in plan) if plan else "- No extra adequacy plan recorded."
    anchor_display = anchor_run_id or "none"
    return "\n".join(
        [
            "# Research Card",
            "",
            "## Delta",
            "",
            f"- `delta_id`: `{row['delta_id']}`",
            f"- `sweep_id`: `{sweep_id}`",
            f"- `dimension_family`: `{row['dimension_family']}`",
            f"- `family`: `{row['family']}`",
            f"- `anchor_run_id`: `{anchor_display}`",
            "- `comparison_policy`: `anchor_only`",
            "",
            "## What Changes",
            "",
            f"- {row['description']}",
            f"- Anchor delta: {row['anchor_delta']}",
            "",
            "## Why This Row Is Informative",
            "",
            f"- {row['rationale']}",
            f"- Hypothesis: {row['hypothesis']}",
            "",
            "## Adequacy Plan",
            "",
            plan_lines,
            "",
        ]
    )


def _campaign_payload(
    *,
    queue_row: Mapping[str, Any],
    materialized_row: Mapping[str, Any],
    sweep_meta: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    device: str,
) -> dict[str, Any]:
    changed_settings: dict[str, Any] = {
        "model": cast(dict[str, Any], queue_row.get("model", {})),
        "data": cast(dict[str, Any], queue_row.get("data", {})),
        "preprocessing": cast(dict[str, Any], queue_row.get("preprocessing", {})),
        "training": cast(dict[str, Any], queue_row.get("training", {})),
    }
    return {
        "sweep_id": sweep_id,
        "delta_id": materialized_row["delta_id"],
        "dimension_family": materialized_row["dimension_family"],
        "family": materialized_row["family"],
        "comparison_policy": str(sweep_meta.get("comparison_policy", "anchor_only")),
        "anchor_run_id": anchor_run_id,
        "locked_bundle_path": str(sweep_meta["benchmark_bundle_path"]),
        "locked_control_baseline_id": str(sweep_meta["control_baseline_id"]),
        "training_experiment": DEFAULT_EXPERIMENT,
        "preserved_settings": {
            "queue_ref": f"reference/system_delta_sweeps/{sweep_id}/queue.yaml",
            "runtime.device": str(device),
            "logging.use_wandb": True,
        },
        "changed_settings": changed_settings,
        "adequacy_knobs": cast(list[str], materialized_row.get("adequacy_knobs", [])),
        "decision_hypothesis": "needs_followup",
    }


def _write_research_package(
    *,
    delta_root: Path,
    materialized_row: Mapping[str, Any],
    queue_row: Mapping[str, Any],
    sweep_meta: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    device: str,
) -> None:
    delta_root.mkdir(parents=True, exist_ok=True)
    (delta_root / "research_card.md").write_text(
        _research_card_text(row=materialized_row, sweep_id=sweep_id, anchor_run_id=anchor_run_id),
        encoding="utf-8",
    )
    (delta_root / "campaign.yaml").write_text(
        yaml.safe_dump(
            _campaign_payload(
                queue_row=queue_row,
                materialized_row=materialized_row,
                sweep_meta=sweep_meta,
                sweep_id=sweep_id,
                anchor_run_id=anchor_run_id,
                device=device,
            ),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )


def _result_card_text(
    *,
    row: Mapping[str, Any],
    run_id: str,
    anchor_run_id: str | None,
    summary: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
) -> str:
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    nanotabpfn = cast(dict[str, Any], summary["nanotabpfn"])
    anchor_display = anchor_run_id or "none"
    return "\n".join(
        [
            "# Result Card",
            "",
            "## What changed",
            "",
            f"- `delta_id`: `{row['delta_id']}`",
            f"- `run_id`: `{run_id}`",
            f"- `anchor_run_id`: `{anchor_display}`",
            f"- `description`: {row['description']}",
            f"- `anchor_delta`: {row['anchor_delta']}",
            "",
            "## Measured metrics versus the anchor",
            "",
            f"- Best ROC AUC: `{float(tab_foundry['best_roc_auc']):.4f}` at step `{int(float(tab_foundry['best_step']))}`",
            f"- Final ROC AUC: `{float(tab_foundry['final_roc_auc']):.4f}`",
            f"- Final minus best: `{float(tab_foundry['best_to_final_roc_auc_delta']):+.4f}`",
            f"- nanoTabPFN best ROC AUC: `{float(nanotabpfn['best_roc_auc']):.4f}`",
            f"- nanoTabPFN final ROC AUC: `{float(nanotabpfn['final_roc_auc']):.4f}`",
            f"- max_grad_norm: `{float(queue_metrics['max_grad_norm']):.4f}`",
            f"- clipped_step_fraction: `{float(queue_metrics['clipped_step_fraction']):.4f}`",
            "",
            "## Was the change actually isolated?",
            "",
            "- The run used the queue row as the only source of model/data/preprocessing/training overrides.",
            "- Bundle, control baseline, experiment family, and schedule budget stayed on the locked sweep surface.",
            "",
            "## Hyperparameter adequacy",
            "",
            "- The row preserved the queue-declared short-run budget unless the row explicitly changed it.",
            "- No extra tuning beyond the queue row was introduced during this execution.",
            "",
            "## Why this may have helped or hurt",
            "",
            f"- Decision recorded in the registry: `{decision}`.",
            f"- Conclusion: {conclusion}",
            "",
            "## Remaining confounders",
            "",
            "- This auto-generated card is intentionally conservative; deeper interpretation still belongs in the sweep review.",
            "",
        ]
    )


def _append_note(notes: list[str], note: str) -> list[str]:
    updated = list(notes)
    if note not in updated:
        updated.append(note)
    return updated


def _update_queue_row(
    *,
    queue_row: dict[str, Any],
    run_id: str,
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
) -> None:
    original_run_id = queue_row.get("run_id")
    queue_row["status"] = "completed"
    queue_row["run_id"] = run_id
    queue_row["followup_run_ids"] = []
    queue_row["decision"] = decision
    queue_row["interpretation_status"] = "completed"
    queue_row["benchmark_metrics"] = dict(queue_metrics)
    queue_row["confounders"] = []
    notes = cast(list[str], queue_row.get("notes", []))
    if isinstance(original_run_id, str) and original_run_id.strip() and original_run_id != run_id:
        notes = _append_note(
            notes,
            f"Supersedes historical queue run `{original_run_id}`; that registry entry is retained as history only.",
        )
    notes = _append_note(notes, f"Canonical rerun registered as `{run_id}`.")
    notes = _append_note(notes, conclusion)
    queue_row["notes"] = notes


def parse_order_overrides(values: list[str] | None, *, arg_name: str) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for raw in values or []:
        left, separator, right = str(raw).partition("=")
        if separator != "=" or not left.strip() or not right.strip():
            raise RuntimeError(f"{arg_name} values must look like <order>=<value>, got {raw!r}")
        try:
            order = int(left)
        except ValueError as exc:
            raise RuntimeError(f"{arg_name} order must be an integer, got {left!r}") from exc
        overrides[order] = right
    return overrides


def select_queue_rows(
    queue: Mapping[str, Any],
    *,
    orders: list[int] | None = None,
    start_order: int | None = None,
    stop_after_order: int | None = None,
    include_completed: bool = False,
) -> list[dict[str, Any]]:
    rows = _sorted_rows(queue)
    explicit_orders = list(orders or [])
    explicit_selection = bool(explicit_orders) or start_order is not None or stop_after_order is not None
    if not explicit_selection:
        return [row for row in rows if str(row.get("status", "")).strip().lower() == "ready"]

    known_orders = {int(row["order"]) for row in rows}
    selected_orders = set(explicit_orders)
    if start_order is not None or stop_after_order is not None:
        min_order = min(known_orders)
        max_order = max(known_orders)
        lower = min_order if start_order is None else int(start_order)
        upper = max_order if stop_after_order is None else int(stop_after_order)
        if lower > upper:
            raise RuntimeError("start_order must be less than or equal to stop_after_order")
        selected_orders.update(range(lower, upper + 1))
    missing = sorted(order for order in selected_orders if order not in known_orders)
    if missing:
        raise RuntimeError(f"unknown queue orders for selection: {missing}")

    selected = [row for row in rows if int(row["order"]) in selected_orders]
    if not include_completed:
        completed_orders = [
            int(row["order"])
            for row in selected
            if str(row.get("status", "")).strip().lower() == "completed"
        ]
        if completed_orders:
            raise RuntimeError(
                "explicitly selected completed rows require --include-completed; "
                f"got completed orders {completed_orders}"
            )
    return selected


def _sync_sweep_matrix(*, sweep_id: str, paths: ExecutionPaths) -> None:
    queue = system_delta.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    matrix = system_delta.render_system_delta_matrix(queue, registry_path=paths.registry_path)
    system_delta._write_text(
        system_delta.sweep_matrix_path(sweep_id, sweeps_root=paths.sweeps_root),
        matrix,
    )


def _sync_active_aliases_if_active(*, sweep_id: str, paths: ExecutionPaths) -> None:
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


def _run_row(
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
    run_id = _row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
    )
    delta_root = paths.repo_root / "outputs" / "staged_ladder" / "research" / sweep_id / str(queue_row["delta_ref"])
    run_root = delta_root / run_id
    train_dir = run_root / "train"
    benchmark_dir = run_root / "benchmark"

    _write_research_package(
        delta_root=delta_root,
        materialized_row=materialized_row,
        queue_row=queue_row,
        sweep_meta=sweep_meta,
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        device=device,
    )
    if _completed_train_artifacts_exist(train_dir):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        cfg = _compose_cfg(row=queue_row, run_dir=train_dir, device=device)
        train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
            f"output_dir={train_result.output_dir}",
            flush=True,
        )

    _ = _ensure_nanotabpfn_python(nanotabpfn_root=nanotabpfn_root, fallback_python=fallback_python)
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
    queue_metrics = _queue_metrics(summary, run_dir=train_dir)
    parent_sweep_id = sweep_meta.get("parent_sweep_id")
    _ = register_benchmark_run(
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
        parent_sweep_id=(None if not isinstance(parent_sweep_id, str) or not parent_sweep_id.strip() else str(parent_sweep_id)),
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=paths.registry_path,
    )
    (delta_root / "result_card.md").write_text(
        _result_card_text(
            row=materialized_row,
            run_id=run_id,
            anchor_run_id=anchor_run_id,
            summary=summary,
            queue_metrics=queue_metrics,
            decision=decision,
            conclusion=conclusion,
        ),
        encoding="utf-8",
    )
    _update_queue_row(
        queue_row=queue_row,
        run_id=run_id,
        queue_metrics=queue_metrics,
        decision=decision,
        conclusion=conclusion,
    )
    print(
        f"[row {int(queue_row['order']):02d}] benchmark+registry complete",
        f"run_id={run_id}",
        f"final_roc_auc={float(cast(dict[str, Any], summary['tab_foundry'])['final_roc_auc']):.4f}",
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
) -> list[str]:
    resolved_paths = ExecutionPaths.default() if paths is None else paths
    sweep_meta = system_delta.load_system_delta_sweep(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_sweep_id = str(sweep_meta["sweep_id"])
    queue_path = system_delta.sweep_queue_path(resolved_sweep_id, sweeps_root=resolved_paths.sweeps_root)
    queue = _read_yaml(queue_path)
    materialized_rows = _materialized_row_map(sweep_id=resolved_sweep_id, paths=resolved_paths)
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
    active_anchor = (
        str(current_anchor_run_id)
        if isinstance(current_anchor_run_id, str) and current_anchor_run_id.strip()
        else None
    )
    executed_run_ids: list[str] = []
    decision_map = dict(decision_overrides or {})
    conclusion_map = dict(conclusion_overrides or {})

    for index, queue_row in enumerate(selected_rows):
        order = int(queue_row["order"])
        decision = str(decision_map.get(order, decision_default)).strip().lower()
        if decision not in _ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(_ALLOWED_DECISIONS)}, got {decision!r}")
        conclusion = str(conclusion_map.get(order, conclusion_default)).strip()
        if not conclusion:
            raise RuntimeError("conclusion must be non-empty")

        promote_now = bool(promote_first_executed_row_to_anchor and index == 0)
        materialized_row = materialized_rows[str(queue_row["delta_ref"])]
        run_id = _run_row(
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
        _write_yaml(queue_path, queue)
        if promote_now:
            _ = promote_anchor(
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
        _sync_sweep_matrix(sweep_id=resolved_sweep_id, paths=resolved_paths)
        _sync_active_aliases_if_active(sweep_id=resolved_sweep_id, paths=resolved_paths)
        executed_run_ids.append(run_id)

    return executed_run_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute system-delta sweep rows")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to execute; defaults to the active sweep")
    parser.add_argument("--order", type=int, action="append", default=[], help="Explicit queue order to execute; repeatable")
    parser.add_argument("--start-order", type=int, default=None, help="Optional starting queue order for a contiguous range")
    parser.add_argument("--stop-after-order", type=int, default=None, help="Optional inclusive last queue order for a contiguous range")
    parser.add_argument("--include-completed", action="store_true", help="Allow explicitly selected completed rows to run again")
    parser.add_argument("--promote-first-executed-row-to-anchor", action="store_true", help="Promote the first executed row to the sweep anchor after it completes")
    parser.add_argument("--prior-dump", default=str(DEFAULT_PRIOR_DUMP), help="Path to the nanoTabPFN prior dump")
    parser.add_argument("--nanotabpfn-root", default=str(DEFAULT_NANOTABPFN_ROOT), help="Path to the nanoTabPFN checkout")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument(
        "--tab-foundry-python",
        default=str(system_delta.repo_root() / ".venv" / "bin" / "python"),
        help="Interpreter to expose under nanoTabPFN/.venv/bin/python",
    )
    parser.add_argument("--decision-default", default=DEFAULT_DECISION, choices=sorted(_ALLOWED_DECISIONS))
    parser.add_argument("--conclusion-default", default=DEFAULT_CONCLUSION, help="Default conclusion recorded for executed rows")
    parser.add_argument("--decision-override", action="append", default=[], help="Per-order override like 7=keep")
    parser.add_argument("--conclusion-override", action="append", default=[], help="Per-order override like 7=Promote this surface.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prior_dump = Path(str(args.prior_dump)).expanduser().resolve()
    nanotabpfn_root = Path(str(args.nanotabpfn_root)).expanduser().resolve()
    fallback_python = Path(str(args.tab_foundry_python)).expanduser().resolve()
    if not prior_dump.exists():
        raise RuntimeError(f"prior dump does not exist: {prior_dump}")
    if not fallback_python.exists():
        raise RuntimeError(f"tab-foundry interpreter does not exist: {fallback_python}")

    decision_overrides = parse_order_overrides(list(args.decision_override), arg_name="--decision-override")
    conclusion_overrides = parse_order_overrides(list(args.conclusion_override), arg_name="--conclusion-override")
    for decision in decision_overrides.values():
        if decision not in _ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(_ALLOWED_DECISIONS)}, got {decision!r}")

    executed = execute_sweep(
        sweep_id=(None if args.sweep_id is None else str(args.sweep_id)),
        prior_dump=prior_dump,
        nanotabpfn_root=nanotabpfn_root,
        device=str(args.device),
        fallback_python=fallback_python,
        orders=list(args.order),
        start_order=(None if args.start_order is None else int(args.start_order)),
        stop_after_order=(None if args.stop_after_order is None else int(args.stop_after_order)),
        include_completed=bool(args.include_completed),
        decision_default=str(args.decision_default),
        conclusion_default=str(args.conclusion_default),
        decision_overrides=decision_overrides,
        conclusion_overrides=conclusion_overrides,
        promote_first_executed_row_to_anchor=bool(args.promote_first_executed_row_to_anchor),
    )
    target_sweep = "active" if args.sweep_id is None else str(args.sweep_id)
    print(
        "Queue execution complete.",
        f"sweep_id={target_sweep}",
        f"executed_rows={len(executed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
