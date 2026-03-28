"""Artifact and filesystem helpers for sweep inspection targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)

from .paths_io import repo_root


def load_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def artifact_entry(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": bool(resolved.exists()),
    }


def optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value)


def registry_run_entry(
    run_id: str | None,
    *,
    registry_path: Path,
) -> dict[str, Any] | None:
    if run_id is None:
        return None
    registry = load_benchmark_run_registry(registry_path)
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        return None
    raw_run = runs.get(run_id)
    if not isinstance(raw_run, Mapping):
        return None
    return dict(cast(Mapping[str, Any], raw_run))


def registry_artifact_path(run_entry: Mapping[str, Any] | None, key: str) -> Path | None:
    if not isinstance(run_entry, Mapping):
        return None
    artifacts = run_entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    raw_value = artifacts.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    return resolve_registry_path_value(raw_value)


def canonical_row_run_root(*, sweep_id: str, delta_id: str, run_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / delta_id / run_id


def inspection_run_dir(
    *,
    sweep_id: str,
    target_kind: str,
    target_id: str,
) -> Path:
    return repo_root() / "outputs" / ".inspection" / "research" / sweep_id / target_kind / target_id / "train"


def row_artifacts(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    run_id = optional_string(row.get("run_id"))
    delta_id = str(row["delta_id"])
    sweep_id = str(queue["sweep_id"])
    expected_root = (
        None if run_id is None else canonical_row_run_root(sweep_id=sweep_id, delta_id=delta_id, run_id=run_id)
    )
    registry_run = registry_run_entry(run_id, registry_path=registry_path)

    resolved_run_dir = registry_artifact_path(registry_run, "run_dir")
    if resolved_run_dir is None:
        resolved_run_dir = None if expected_root is None else expected_root / "train"
    resolved_benchmark_dir = registry_artifact_path(registry_run, "benchmark_dir")
    if resolved_benchmark_dir is None:
        resolved_benchmark_dir = None if expected_root is None else expected_root / "benchmark"
    resolved_training_surface = registry_artifact_path(registry_run, "training_surface_record_path")
    if resolved_training_surface is None and resolved_run_dir is not None:
        resolved_training_surface = resolved_run_dir / "training_surface_record.json"
    resolved_best_checkpoint = registry_artifact_path(registry_run, "best_checkpoint_path")
    if resolved_best_checkpoint is None and resolved_run_dir is not None:
        resolved_best_checkpoint = resolved_run_dir / "checkpoints" / "best.pt"
    resolved_comparison_summary = registry_artifact_path(registry_run, "comparison_summary_path")
    if resolved_comparison_summary is None and resolved_benchmark_dir is not None:
        resolved_comparison_summary = resolved_benchmark_dir / "comparison_summary.json"
    resolved_benchmark_record = registry_artifact_path(registry_run, "benchmark_run_record_path")
    if resolved_benchmark_record is None and resolved_benchmark_dir is not None:
        resolved_benchmark_record = resolved_benchmark_dir / "benchmark_run_record.json"

    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        "expected_research_root": None if expected_root is None else artifact_entry(expected_root),
        "run_dir": None if resolved_run_dir is None else artifact_entry(resolved_run_dir),
        "benchmark_dir": None if resolved_benchmark_dir is None else artifact_entry(resolved_benchmark_dir),
        "training_surface_record_json": (
            None if resolved_training_surface is None else artifact_entry(resolved_training_surface)
        ),
        "best_checkpoint_path": (
            None if resolved_best_checkpoint is None else artifact_entry(resolved_best_checkpoint)
        ),
        "comparison_summary_json": (
            None if resolved_comparison_summary is None else artifact_entry(resolved_comparison_summary)
        ),
        "benchmark_run_record_json": (
            None if resolved_benchmark_record is None else artifact_entry(resolved_benchmark_record)
        ),
    }
    if resolved_run_dir is not None:
        artifacts["train_history_jsonl"] = artifact_entry(resolved_run_dir / "train_history.jsonl")
        artifacts["gradient_history_jsonl"] = artifact_entry(resolved_run_dir / "gradient_history.jsonl")
        artifacts["telemetry_json"] = artifact_entry(resolved_run_dir / "telemetry.json")
    return artifacts


def anchor_run_artifacts(
    *,
    queue: Mapping[str, Any],
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    anchor_run_id = optional_string(queue.get("anchor_run_id"))
    registry_run = registry_run_entry(anchor_run_id, registry_path=registry_path)
    run_dir = registry_artifact_path(registry_run, "run_dir")
    comparison_summary = registry_artifact_path(registry_run, "comparison_summary_path")
    benchmark_record = registry_artifact_path(registry_run, "benchmark_run_record_path")
    training_surface_record = registry_artifact_path(registry_run, "training_surface_record_path")
    best_checkpoint = registry_artifact_path(registry_run, "best_checkpoint_path")
    benchmark_dir = registry_artifact_path(registry_run, "benchmark_dir")
    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        "run_dir": None if run_dir is None else artifact_entry(run_dir),
        "benchmark_dir": None if benchmark_dir is None else artifact_entry(benchmark_dir),
        "training_surface_record_json": (
            None if training_surface_record is None else artifact_entry(training_surface_record)
        ),
        "best_checkpoint_path": None if best_checkpoint is None else artifact_entry(best_checkpoint),
        "comparison_summary_json": (
            None if comparison_summary is None else artifact_entry(comparison_summary)
        ),
        "benchmark_run_record_json": None if benchmark_record is None else artifact_entry(benchmark_record),
    }
    if run_dir is not None:
        artifacts["train_history_jsonl"] = artifact_entry(run_dir / "train_history.jsonl")
        artifacts["gradient_history_jsonl"] = artifact_entry(run_dir / "gradient_history.jsonl")
        artifacts["telemetry_json"] = artifact_entry(run_dir / "telemetry.json")
    return artifacts, registry_run


def anchor_has_training_artifacts(artifacts: Mapping[str, Any]) -> bool:
    training_surface_entry = artifacts.get("training_surface_record_json")
    if isinstance(training_surface_entry, Mapping) and bool(training_surface_entry.get("exists")):
        return True
    run_dir_entry = artifacts.get("run_dir")
    checkpoint_entry = artifacts.get("best_checkpoint_path")
    return (
        isinstance(run_dir_entry, Mapping)
        and bool(run_dir_entry.get("exists"))
        and isinstance(checkpoint_entry, Mapping)
        and bool(checkpoint_entry.get("exists"))
    )
