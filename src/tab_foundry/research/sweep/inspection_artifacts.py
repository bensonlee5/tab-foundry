"""Artifact and filesystem helpers for sweep inspection targets."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.research.lane_contract import resolve_sweep_semantics

from .materialize import load_system_delta_queue_for_inspection, ordered_rows
from .paths_io import repo_root

PathResolver = Callable[[str], Path]


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


def load_inspection_queue(
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        load_system_delta_queue_for_inspection(
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        ),
    )


def queue_metadata_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    semantics = resolve_sweep_semantics(queue)
    return {
        "sweep_id": str(queue["sweep_id"]),
        "anchor_run_id": optional_string(queue.get("anchor_run_id")),
        **semantics.to_payload_dict(),
        "benchmark_manifest_path": str(queue["benchmark_manifest_path"]),
        "control_baseline_id": str(queue["control_baseline_id"]),
        "external_benchmarks": list(cast(list[Any], queue.get("external_benchmarks", []))),
        "canonical_sweep_path": str(queue["canonical_sweep_path"]),
        "canonical_queue_path": str(queue["canonical_queue_path"]),
        "canonical_matrix_path": str(queue["canonical_matrix_path"]),
    }


def find_row(queue: Mapping[str, Any], *, order: int) -> dict[str, Any]:
    for row in ordered_rows(queue):
        if int(row["order"]) == int(order):
            return row
    raise RuntimeError(f"unknown sweep order: {order}")


def queue_anchor_row(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    anchor_run_id = optional_string(queue.get("anchor_run_id"))
    if anchor_run_id is None:
        return None
    for row in ordered_rows(queue):
        row_run_id = optional_string(row.get("run_id"))
        if row_run_id == anchor_run_id:
            return row
    return None


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


def registry_artifact_path(
    run_entry: Mapping[str, Any] | None,
    key: str,
    *,
    resolve_registry_path: PathResolver | None = None,
) -> Path | None:
    if not isinstance(run_entry, Mapping):
        return None
    artifacts = run_entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    raw_value = artifacts.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    resolver = resolve_registry_path or resolve_registry_path_value
    return resolver(raw_value)


def canonical_row_run_root(*, sweep_id: str, delta_id: str, run_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / delta_id / run_id


def inspection_run_dir(
    *,
    sweep_id: str,
    target_kind: str,
    target_id: str,
) -> Path:
    return repo_root() / "outputs" / ".inspection" / "research" / sweep_id / target_kind / target_id / "train"


def result_card_path(*, sweep_id: str, delta_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / delta_id / "result_card.md"


def _resolved_run_artifact_paths(
    *,
    registry_run: Mapping[str, Any] | None,
    expected_root: Path | None,
    resolve_registry_path: PathResolver | None = None,
) -> dict[str, Path | None]:
    run_dir = registry_artifact_path(
        registry_run,
        "run_dir",
        resolve_registry_path=resolve_registry_path,
    )
    if run_dir is None and expected_root is not None:
        run_dir = expected_root / "train"
    benchmark_dir = registry_artifact_path(
        registry_run,
        "benchmark_dir",
        resolve_registry_path=resolve_registry_path,
    )
    if benchmark_dir is None and expected_root is not None:
        benchmark_dir = expected_root / "benchmark"
    training_surface_record = registry_artifact_path(
        registry_run,
        "training_surface_record_path",
        resolve_registry_path=resolve_registry_path,
    )
    if training_surface_record is None and run_dir is not None:
        training_surface_record = run_dir / "training_surface_record.json"
    best_checkpoint = registry_artifact_path(
        registry_run,
        "best_checkpoint_path",
        resolve_registry_path=resolve_registry_path,
    )
    if best_checkpoint is None and run_dir is not None:
        best_checkpoint = run_dir / "checkpoints" / "best.pt"
    comparison_summary = registry_artifact_path(
        registry_run,
        "comparison_summary_path",
        resolve_registry_path=resolve_registry_path,
    )
    if comparison_summary is None and benchmark_dir is not None:
        comparison_summary = benchmark_dir / "comparison_summary.json"
    benchmark_record = registry_artifact_path(
        registry_run,
        "benchmark_run_record_path",
        resolve_registry_path=resolve_registry_path,
    )
    if benchmark_record is None and benchmark_dir is not None:
        benchmark_record = benchmark_dir / "benchmark_run_record.json"
    return {
        "run_dir": run_dir,
        "benchmark_dir": benchmark_dir,
        "training_surface_record_json": training_surface_record,
        "best_checkpoint_path": best_checkpoint,
        "comparison_summary_json": comparison_summary,
        "benchmark_run_record_json": benchmark_record,
    }


def resolved_row_artifact_paths(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_run: Mapping[str, Any] | None,
    resolve_registry_path: PathResolver | None = None,
) -> dict[str, Path | None]:
    run_id = optional_string(row.get("run_id"))
    expected_root = (
        None
        if run_id is None
        else canonical_row_run_root(
            sweep_id=str(queue["sweep_id"]),
            delta_id=str(row["delta_id"]),
            run_id=run_id,
        )
    )
    return {
        "expected_research_root": expected_root,
        "result_card_md": result_card_path(
            sweep_id=str(queue["sweep_id"]),
            delta_id=str(row["delta_id"]),
        ),
        **_resolved_run_artifact_paths(
            registry_run=registry_run,
            expected_root=expected_root,
            resolve_registry_path=resolve_registry_path,
        ),
    }


def resolved_anchor_artifact_paths(
    *,
    registry_run: Mapping[str, Any] | None,
    resolve_registry_path: PathResolver | None = None,
) -> dict[str, Path | None]:
    return _resolved_run_artifact_paths(
        registry_run=registry_run,
        expected_root=None,
        resolve_registry_path=resolve_registry_path,
    )


def _artifact_entries_from_paths(paths: Mapping[str, Path | None]) -> dict[str, Any]:
    artifacts = {
        key: None if path is None else artifact_entry(path)
        for key, path in paths.items()
    }
    run_dir = paths.get("run_dir")
    if run_dir is not None:
        artifacts["train_history_jsonl"] = artifact_entry(run_dir / "train_history.jsonl")
        artifacts["gradient_history_jsonl"] = artifact_entry(run_dir / "gradient_history.jsonl")
        artifacts["telemetry_json"] = artifact_entry(run_dir / "telemetry.json")
    return artifacts


def row_artifacts(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    run_id = optional_string(row.get("run_id"))
    registry_run = registry_run_entry(run_id, registry_path=registry_path)
    paths = resolved_row_artifact_paths(
        queue=queue,
        row=row,
        registry_run=registry_run,
    )
    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        **_artifact_entries_from_paths(paths),
    }
    return artifacts


def anchor_run_artifacts(
    *,
    queue: Mapping[str, Any],
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    anchor_run_id = optional_string(queue.get("anchor_run_id"))
    registry_run = registry_run_entry(anchor_run_id, registry_path=registry_path)
    paths = resolved_anchor_artifact_paths(registry_run=registry_run)
    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        **_artifact_entries_from_paths(paths),
    }
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
