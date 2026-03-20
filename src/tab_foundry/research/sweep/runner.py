"""Execution orchestration helpers for system-delta sweeps."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.bench.benchmark_run_registry import (
    register_benchmark_run,
    resolve_registry_path_value,
)
from tab_foundry.bench.compare import (
    DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
    NanoTabPFNBenchmarkConfig,
    run_nanotabpfn_benchmark,
)
from tab_foundry.bench.nanotabpfn import benchmark_host_fingerprint, resolve_device
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.config import compose_config
from tab_foundry.research.lane_contract import (
    resolve_surface_role,
    resolve_training_config_profile,
    resolve_training_experiment,
)
from tab_foundry.research import system_delta
from tab_foundry.research.system_delta_promote import promote_anchor
from tab_foundry.training.artifacts import resolve_latest_checkpoint_path

from .artifacts import ExecutionPaths, read_yaml, result_card_text, write_research_package, write_yaml
from .queue_updates import append_note, optional_metric, queue_metrics, update_queue_row, update_screened_queue_row
from .screening import pick_screen_winner, screen_metrics
from .selection import select_queue_rows, sorted_rows


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
SCREEN_ONLY_POLICY = "screen_only"
BENCHMARK_FULL_POLICY = "benchmark_full"


@dataclass(frozen=True)
class NanoTabPFNCurveCandidate:
    source_label: str
    comparison_summary_path: Path
    declared_control_baseline_id: str | None = None


@dataclass(frozen=True)
class NanoTabPFNCurveReuseSelection:
    curve_path: Path
    source_label: str
    metadata: dict[str, Any]
    signature: dict[str, Any]


def row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def _python_can_import_torch(python_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import torch"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _absolute_path_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
    nanotab_python = nanotabpfn_root / ".venv" / "bin" / "python"
    fallback_executable = _absolute_path_without_resolving_symlinks(fallback_python)
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    if nanotab_python.exists() and _python_can_import_torch(nanotab_python):
        return nanotab_python
    if nanotab_python.exists() or nanotab_python.is_symlink():
        nanotab_python.unlink()
    if not _python_can_import_torch(fallback_executable):
        raise RuntimeError(
            "fallback interpreter cannot import torch: "
            f"{fallback_executable}"
        )
    nanotab_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(fallback_executable))} \"$@\"\n",
        encoding="utf-8",
    )
    nanotab_python.chmod(0o755)
    return nanotab_python


def completed_train_artifacts_exist(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "train_history.jsonl",
        run_dir / "gradient_history.jsonl",
        run_dir / "telemetry.json",
        run_dir / "training_surface_record.json",
    )
    return all(path.exists() for path in required_paths) and (
        resolve_latest_checkpoint_path(run_dir) is not None
    )


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


def compose_cfg(
    *,
    row: Mapping[str, Any],
    run_dir: Path,
    device: str,
    training_experiment: str = DEFAULT_EXPERIMENT,
) -> DictConfig:
    cfg = compose_config([f"experiment={training_experiment}"])
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


def _normalize_execution_policy(queue_row: Mapping[str, Any]) -> str:
    policy = str(queue_row.get("execution_policy", BENCHMARK_FULL_POLICY)).strip().lower()
    if policy not in {SCREEN_ONLY_POLICY, BENCHMARK_FULL_POLICY}:
        raise RuntimeError(f"unsupported execution_policy {policy!r}")
    return policy


def _optional_non_empty_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string when provided")
    return str(value.strip())


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def _candidate_curve_path(summary: Mapping[str, Any], *, summary_path: Path) -> Path | None:
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, Mapping):
        curve_value = artifacts.get("nanotabpfn_curve_jsonl")
        if isinstance(curve_value, str) and curve_value.strip():
            candidate = resolve_registry_path_value(curve_value)
            if candidate.exists():
                return candidate
    fallback = summary_path.parent / "nanotabpfn_curve.jsonl"
    return fallback if fallback.exists() else None


def _planned_nanotabpfn_python_path(nanotabpfn_root: Path) -> Path:
    return nanotabpfn_root.expanduser().resolve() / ".venv" / "bin" / "python"


def _float_matches(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-12


def _resolved_nanotabpfn_signature(
    *,
    benchmark_bundle_path: Path,
    control_baseline_id: str,
    nanotabpfn_root: Path,
    nanotabpfn_python: Path,
    prior_dump: Path,
    requested_device: str,
) -> dict[str, Any]:
    normalized_requested_device = str(requested_device).strip()
    return {
        "benchmark_bundle_path": benchmark_bundle_path.expanduser().resolve(),
        "control_baseline_id": str(control_baseline_id).strip(),
        "nanotabpfn_root": nanotabpfn_root.expanduser().resolve(),
        "nanotabpfn_python": nanotabpfn_python.expanduser().resolve(),
        "prior_dump_path": prior_dump.expanduser().resolve(),
        "device": normalized_requested_device,
        "resolved_device": resolve_device(normalized_requested_device),
        "benchmark_host_fingerprint": benchmark_host_fingerprint(),
        "steps": int(DEFAULT_NANOTABPFN_STEPS),
        "eval_every": int(DEFAULT_NANOTABPFN_EVAL_EVERY),
        "seeds": int(DEFAULT_NANOTABPFN_SEEDS),
        "batch_size": int(DEFAULT_NANOTABPFN_BATCH_SIZE),
        "lr": float(DEFAULT_NANOTABPFN_LR),
    }


def _candidate_signature(
    *,
    candidate: NanoTabPFNCurveCandidate,
) -> tuple[dict[str, Any], dict[str, Any], Path] | None:
    if not candidate.comparison_summary_path.exists():
        return None
    try:
        summary = _read_json_mapping(candidate.comparison_summary_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None

    bundle = summary.get("benchmark_bundle")
    nanotabpfn = summary.get("nanotabpfn")
    if not isinstance(bundle, Mapping) or not isinstance(nanotabpfn, Mapping):
        return None

    curve_path = _candidate_curve_path(summary, summary_path=candidate.comparison_summary_path)
    if curve_path is None:
        return None

    bundle_source = bundle.get("source_path")
    if not isinstance(bundle_source, str) or not bundle_source.strip():
        return None

    control_baseline_id = candidate.declared_control_baseline_id
    if control_baseline_id is None:
        baseline = summary.get("control_baseline")
        if not isinstance(baseline, Mapping):
            return None
        baseline_id = baseline.get("baseline_id")
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            return None
        control_baseline_id = str(baseline_id).strip()

    requested_device = nanotabpfn.get("device")
    if not isinstance(requested_device, str) or not requested_device.strip():
        return None
    resolved_device = nanotabpfn.get("resolved_device")
    if not isinstance(resolved_device, str) or not resolved_device.strip():
        return None
    host_fingerprint = nanotabpfn.get("benchmark_host_fingerprint")
    if not isinstance(host_fingerprint, str) or not host_fingerprint.strip():
        return None

    root_value = nanotabpfn.get("root")
    root = (
        Path(str(root_value)).expanduser().resolve()
        if isinstance(root_value, str) and root_value.strip()
        else None
    )
    python_value = nanotabpfn.get("python")
    nanotabpfn_python = (
        Path(str(python_value)).expanduser().resolve()
        if isinstance(python_value, str) and python_value.strip()
        else None
    )

    raw_prior_dump_path = nanotabpfn.get("prior_dump_path")
    prior_dump_path = (
        Path(str(raw_prior_dump_path)).expanduser().resolve()
        if isinstance(raw_prior_dump_path, str) and raw_prior_dump_path.strip()
        else None
    )

    signature = {
        "benchmark_bundle_path": resolve_registry_path_value(str(bundle_source)),
        "control_baseline_id": control_baseline_id,
        "nanotabpfn_root": root,
        "nanotabpfn_python": nanotabpfn_python,
        "prior_dump_path": prior_dump_path,
        "device": str(requested_device).strip(),
        "resolved_device": str(resolved_device).strip().lower(),
        "benchmark_host_fingerprint": str(host_fingerprint).strip(),
        "steps": int(nanotabpfn.get("steps", DEFAULT_NANOTABPFN_STEPS)),
        "eval_every": int(nanotabpfn.get("eval_every", DEFAULT_NANOTABPFN_EVAL_EVERY)),
        "seeds": int(nanotabpfn.get("num_seeds", nanotabpfn.get("seeds", DEFAULT_NANOTABPFN_SEEDS))),
        "batch_size": int(nanotabpfn.get("batch_size", DEFAULT_NANOTABPFN_BATCH_SIZE)),
        "lr": float(nanotabpfn.get("lr", DEFAULT_NANOTABPFN_LR)),
    }
    metadata = {
        "root": None if root is None else str(root),
        "python": None if nanotabpfn_python is None else str(nanotabpfn_python),
        "device": signature["device"],
        "resolved_device": signature["resolved_device"],
        "benchmark_host_fingerprint": signature["benchmark_host_fingerprint"],
        "prior_dump_path": None if prior_dump_path is None else str(prior_dump_path),
        "num_seeds": signature["seeds"],
        "steps": signature["steps"],
        "eval_every": signature["eval_every"],
        "batch_size": signature["batch_size"],
        "lr": signature["lr"],
    }
    return signature, metadata, curve_path


def _matching_nanotabpfn_curve(
    *,
    current_signature: Mapping[str, Any],
    candidate: NanoTabPFNCurveCandidate,
) -> NanoTabPFNCurveReuseSelection | None:
    candidate_payload = _candidate_signature(candidate=candidate)
    if candidate_payload is None:
        return None
    candidate_signature, metadata, curve_path = candidate_payload
    comparable_keys = (
        "benchmark_bundle_path",
        "control_baseline_id",
        "resolved_device",
        "benchmark_host_fingerprint",
        "steps",
        "eval_every",
        "seeds",
        "batch_size",
    )
    for key in comparable_keys:
        if candidate_signature[key] != current_signature[key]:
            return None
    for key in ("nanotabpfn_root", "nanotabpfn_python", "prior_dump_path"):
        candidate_value = candidate_signature[key]
        current_value = current_signature[key]
        if candidate_value is not None and candidate_value != current_value:
            return None
    if not _float_matches(float(candidate_signature["lr"]), float(current_signature["lr"])):
        return None
    return NanoTabPFNCurveReuseSelection(
        curve_path=curve_path,
        source_label=candidate.source_label,
        metadata=metadata,
        signature=candidate_signature,
    )


def _anchor_curve_candidate(
    *,
    anchor_run_id: str | None,
    registry_path: Path,
) -> NanoTabPFNCurveCandidate | None:
    if anchor_run_id is None:
        return None
    try:
        payload = _read_json_mapping(registry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        return None
    run = runs.get(anchor_run_id)
    if not isinstance(run, Mapping):
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    summary_value = artifacts.get("comparison_summary_path")
    if not isinstance(summary_value, str) or not summary_value.strip():
        return None

    return NanoTabPFNCurveCandidate(
        source_label="anchor",
        comparison_summary_path=resolve_registry_path_value(summary_value),
    )


def _control_baseline_curve_candidate(
    *,
    control_baseline_id: str,
    control_baseline_registry_path: Path,
) -> NanoTabPFNCurveCandidate | None:
    try:
        payload = _read_json_mapping(control_baseline_registry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    baselines = payload.get("baselines")
    if not isinstance(baselines, Mapping):
        return None
    entry = baselines.get(control_baseline_id)
    if not isinstance(entry, Mapping):
        return None
    summary_value = entry.get("comparison_summary_path")
    if not isinstance(summary_value, str) or not summary_value.strip():
        return None
    return NanoTabPFNCurveCandidate(
        source_label="control baseline",
        comparison_summary_path=resolve_registry_path_value(summary_value),
        declared_control_baseline_id=control_baseline_id,
    )


def resolve_reusable_nanotabpfn_curve(
    *,
    sweep_meta: Mapping[str, Any],
    anchor_run_id: str | None,
    nanotabpfn_root: Path,
    prior_dump: Path,
    requested_device: str,
    paths: ExecutionPaths,
) -> NanoTabPFNCurveReuseSelection | None:
    control_baseline_id = str(sweep_meta["control_baseline_id"]).strip()
    current_signature = _resolved_nanotabpfn_signature(
        benchmark_bundle_path=resolve_registry_path_value(str(sweep_meta["benchmark_bundle_path"])),
        control_baseline_id=control_baseline_id,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=_planned_nanotabpfn_python_path(nanotabpfn_root),
        prior_dump=prior_dump,
        requested_device=requested_device,
    )
    candidates = [
        _anchor_curve_candidate(anchor_run_id=anchor_run_id, registry_path=paths.registry_path),
        _control_baseline_curve_candidate(
            control_baseline_id=control_baseline_id,
            control_baseline_registry_path=paths.control_baseline_registry_path,
        ),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        selection = _matching_nanotabpfn_curve(
            current_signature=current_signature,
            candidate=candidate,
        )
        if selection is not None:
            return selection
    return None


def _resolve_parent_row(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_order = int(queue_row["order"])
    current_delta_ref = str(queue_row["delta_ref"])
    parent_delta_ref = _optional_non_empty_string(
        queue_row.get("parent_delta_ref"),
        context=f"queue row {current_order}.parent_delta_ref",
    )
    if parent_delta_ref is None:
        return None

    matching_rows = [
        row for row in queue_rows if str(row.get("delta_ref", "")).strip() == parent_delta_ref
    ]
    if not matching_rows:
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} is missing from sweep {queue_row.get('sweep_id', '<unknown>')!r}"
        )

    earlier_rows = [row for row in matching_rows if int(row["order"]) < current_order]
    if earlier_rows:
        return max(earlier_rows, key=lambda row: int(row["order"]))

    matching_orders = [int(row["order"]) for row in matching_rows]
    if any(order == current_order for order in matching_orders):
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} must reference an earlier row, not itself; "
            f"matching orders={matching_orders}"
        )
    raise RuntimeError(
        f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
        f"{parent_delta_ref!r} must reference an earlier row; matching orders={matching_orders}"
    )


def _resolve_parent_run_id(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
    active_anchor: str | None,
) -> str | None:
    parent_row = _resolve_parent_row(queue_row=queue_row, queue_rows=queue_rows)
    if parent_row is None:
        return active_anchor

    parent_run_id = parent_row.get("run_id")
    if not isinstance(parent_run_id, str) or not parent_run_id.strip():
        raise RuntimeError(
            f"queue row {int(queue_row['order'])} ({queue_row['delta_ref']}) parent_delta_ref "
            f"{parent_row['delta_ref']!r} resolved to row {int(parent_row['order'])}, "
            "but that row does not have a completed run_id"
        )
    return str(parent_run_id)


def _resolve_dynamic_model_overrides(
    *,
    queue: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
) -> None:
    dynamic_overrides = queue_row.get("dynamic_model_overrides")
    if not isinstance(dynamic_overrides, Mapping):
        return
    queue_rows_raw = queue.get("rows")
    if not isinstance(queue_rows_raw, list):
        raise RuntimeError("queue rows must be a list")
    rows_by_order = {
        int(raw_row["order"]): cast(dict[str, Any], raw_row)
        for raw_row in queue_rows_raw
        if isinstance(raw_row, dict)
    }
    queue_model = cast(dict[str, Any], queue_row.setdefault("model", {}))
    queue_module_overrides = cast(dict[str, Any], queue_model.setdefault("module_overrides", {}))
    materialized_model = cast(dict[str, Any], materialized_row.setdefault("model", {}))
    materialized_module_overrides = cast(
        dict[str, Any],
        materialized_model.setdefault("module_overrides", {}),
    )
    queue_notes = cast(list[str], queue_row.setdefault("notes", []))
    materialized_notes = cast(list[str], materialized_row.setdefault("notes", []))

    for override_key, policy_raw in dynamic_overrides.items():
        if not isinstance(policy_raw, dict):
            raise RuntimeError(f"dynamic_model_overrides.{override_key} must be a mapping")
        policy = cast(dict[str, Any], policy_raw)
        if str(policy.get("kind")) != "screen_winner":
            raise RuntimeError(f"unsupported dynamic override policy kind: {policy.get('kind')!r}")
        resolved_value = policy.get("resolved_value")
        if isinstance(resolved_value, str) and resolved_value.strip():
            queue_module_overrides[str(override_key)] = resolved_value
            materialized_module_overrides[str(override_key)] = resolved_value
            continue
        compare_orders = policy.get("compare_orders")
        if not isinstance(compare_orders, list) or not compare_orders:
            raise RuntimeError(
                f"dynamic_model_overrides.{override_key}.compare_orders must be a non-empty list"
            )
        candidates: list[dict[str, Any]] = []
        for candidate_raw in compare_orders:
            if not isinstance(candidate_raw, Mapping):
                raise RuntimeError("dynamic compare_orders entries must be mappings")
            order = int(candidate_raw["order"])
            value = str(candidate_raw["value"])
            candidate_row = rows_by_order.get(order)
            if candidate_row is None:
                raise RuntimeError(f"dynamic compare order {order} is missing from the queue")
            candidates.append(
                {
                    "order": order,
                    "value": value,
                    "screen_metrics": candidate_row.get("screen_metrics"),
                }
            )
        resolution = pick_screen_winner(
            candidates=candidates,
            tie_break_preference=str(policy.get("tie_break_preference", "rmsnorm")),
        )
        winning_value = str(resolution["winning_value"])
        policy["resolved_value"] = winning_value
        policy["resolved_from_order"] = int(resolution["winning_order"])
        policy["resolution_reason"] = str(resolution["reason"])
        queue_module_overrides[str(override_key)] = winning_value
        materialized_module_overrides[str(override_key)] = winning_value
        resolution_note = (
            f"Resolved `{override_key}` to `{winning_value}` from screen row "
            f"`{int(resolution['winning_order'])}` ({resolution['reason']})."
        )
        queue_row["notes"] = append_note(queue_notes, resolution_note)
        materialized_row["notes"] = append_note(materialized_notes, resolution_note)


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
    _resolve_dynamic_model_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    training_experiment = resolve_training_experiment(sweep_meta)
    training_config_profile = resolve_training_config_profile(sweep_meta)
    surface_role = resolve_surface_role(sweep_meta)
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
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
    )
    if completed_train_artifacts_exist(train_dir):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        cfg = compose_cfg(
            row=queue_row,
            run_dir=train_dir,
            device=device,
            training_experiment=training_experiment,
        )
        train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
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

    reuse_selection = resolve_reusable_nanotabpfn_curve(
        sweep_meta=sweep_meta,
        anchor_run_id=anchor_run_id,
        nanotabpfn_root=nanotabpfn_root,
        prior_dump=prior_dump,
        requested_device=device,
        paths=paths,
    )
    reuse_curve_path = None if reuse_selection is None else reuse_selection.curve_path
    if reuse_selection is not None:
        print(
            f"[row {int(queue_row['order']):02d}] reusing nanoTabPFN curve",
            f"source={reuse_selection.source_label}",
            f"path={reuse_curve_path}",
            flush=True,
        )
    else:
        _ = ensure_nanotabpfn_python(
            nanotabpfn_root=nanotabpfn_root,
            fallback_python=fallback_python,
        )
        print(
            f"[row {int(queue_row['order']):02d}] running fresh nanoTabPFN helper",
            f"device={device}",
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
            reuse_nanotabpfn_curve_path=reuse_curve_path,
            reuse_nanotabpfn_metadata=(
                None if reuse_selection is None else reuse_selection.metadata
            ),
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
    queue_rows = sorted_rows(queue)
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
            parent_run_id=(
                None
                if promote_now
                else _resolve_parent_run_id(
                    queue_row=queue_row,
                    queue_rows=queue_rows,
                    active_anchor=active_anchor,
                )
            ),
            queue=queue,
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
