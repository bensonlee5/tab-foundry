"""Inspect one system-delta sweep row and its resolved surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf
import torch

from tab_foundry.bench.benchmark_run_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.model.inspection import model_surface_payload, parameter_counts_from_model_spec
from tab_foundry.training.surface import build_training_surface_record

from .graph import (
    _training_surface_record_model_spec,
    resolve_anchor_model_spec,
    resolve_queue_row_model_spec,
)
from .materialize import load_system_delta_queue, ordered_rows
from .paths_io import (
    _copy_jsonable,
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
    repo_root,
)
from .runner import compose_cfg


def _load_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _artifact_entry(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": bool(resolved.exists()),
    }


def _queue_metadata_payload(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sweep_id": str(queue["sweep_id"]),
        "anchor_run_id": str(queue["anchor_run_id"]),
        "training_experiment": str(queue["training_experiment"]),
        "training_config_profile": str(queue["training_config_profile"]),
        "surface_role": str(queue["surface_role"]),
        "comparison_policy": str(queue["comparison_policy"]),
        "benchmark_bundle_path": str(queue["benchmark_bundle_path"]),
        "control_baseline_id": str(queue["control_baseline_id"]),
        "canonical_sweep_path": str(queue["canonical_sweep_path"]),
        "canonical_queue_path": str(queue["canonical_queue_path"]),
        "canonical_matrix_path": str(queue["canonical_matrix_path"]),
    }


def _find_row(queue: Mapping[str, Any], *, order: int) -> dict[str, Any]:
    for row in ordered_rows(queue):
        if int(row["order"]) == int(order):
            return row
    raise RuntimeError(f"unknown sweep order: {order}")


def _registry_run_entry(
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


def _registry_artifact_path(run_entry: Mapping[str, Any] | None, key: str) -> Path | None:
    if not isinstance(run_entry, Mapping):
        return None
    artifacts = run_entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    raw_value = artifacts.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    return resolve_registry_path_value(raw_value)


def _canonical_row_run_root(*, sweep_id: str, delta_id: str, run_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / delta_id / run_id


def _row_artifacts(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    run_id = row.get("run_id")
    delta_id = str(row["delta_id"])
    sweep_id = str(queue["sweep_id"])
    expected_root = (
        None
        if not isinstance(run_id, str) or not run_id.strip()
        else _canonical_row_run_root(sweep_id=sweep_id, delta_id=delta_id, run_id=run_id)
    )
    registry_run = _registry_run_entry(
        None if not isinstance(run_id, str) else run_id,
        registry_path=registry_path,
    )

    resolved_run_dir = _registry_artifact_path(registry_run, "run_dir")
    if resolved_run_dir is None:
        resolved_run_dir = None if expected_root is None else expected_root / "train"
    resolved_benchmark_dir = _registry_artifact_path(registry_run, "benchmark_dir")
    if resolved_benchmark_dir is None:
        resolved_benchmark_dir = None if expected_root is None else expected_root / "benchmark"
    resolved_training_surface = _registry_artifact_path(registry_run, "training_surface_record_path")
    if resolved_training_surface is None and resolved_run_dir is not None:
        resolved_training_surface = resolved_run_dir / "training_surface_record.json"
    resolved_best_checkpoint = _registry_artifact_path(registry_run, "best_checkpoint_path")
    if resolved_best_checkpoint is None and resolved_run_dir is not None:
        resolved_best_checkpoint = resolved_run_dir / "checkpoints" / "best.pt"
    resolved_comparison_summary = _registry_artifact_path(registry_run, "comparison_summary_path")
    if resolved_comparison_summary is None and resolved_benchmark_dir is not None:
        resolved_comparison_summary = resolved_benchmark_dir / "comparison_summary.json"
    resolved_benchmark_record = _registry_artifact_path(registry_run, "benchmark_run_record_path")
    if resolved_benchmark_record is None and resolved_benchmark_dir is not None:
        resolved_benchmark_record = resolved_benchmark_dir / "benchmark_run_record.json"

    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        "expected_research_root": None if expected_root is None else _artifact_entry(expected_root),
        "run_dir": None if resolved_run_dir is None else _artifact_entry(resolved_run_dir),
        "benchmark_dir": None if resolved_benchmark_dir is None else _artifact_entry(resolved_benchmark_dir),
        "training_surface_record_json": (
            None if resolved_training_surface is None else _artifact_entry(resolved_training_surface)
        ),
        "best_checkpoint_path": (
            None if resolved_best_checkpoint is None else _artifact_entry(resolved_best_checkpoint)
        ),
        "comparison_summary_json": (
            None if resolved_comparison_summary is None else _artifact_entry(resolved_comparison_summary)
        ),
        "benchmark_run_record_json": (
            None if resolved_benchmark_record is None else _artifact_entry(resolved_benchmark_record)
        ),
    }
    if resolved_run_dir is not None:
        artifacts["train_history_jsonl"] = _artifact_entry(resolved_run_dir / "train_history.jsonl")
        artifacts["gradient_history_jsonl"] = _artifact_entry(resolved_run_dir / "gradient_history.jsonl")
        artifacts["telemetry_json"] = _artifact_entry(resolved_run_dir / "telemetry.json")
    return artifacts


def _surface_payload(
    *,
    spec: Any,
    training_surface_record: Mapping[str, Any],
) -> dict[str, Any]:
    record_payload = dict(cast(dict[str, Any], _copy_jsonable(training_surface_record)))
    labels = record_payload.get("labels")
    return {
        "surface_labels": (
            dict(cast(Mapping[str, Any], labels))
            if isinstance(labels, Mapping)
            else None
        ),
        "model": {
            **model_surface_payload(spec),
            "parameter_counts": parameter_counts_from_model_spec(spec),
        },
        "data": (
            dict(cast(Mapping[str, Any], record_payload["data"]))
            if isinstance(record_payload.get("data"), Mapping)
            else None
        ),
        "preprocessing": (
            dict(cast(Mapping[str, Any], record_payload["preprocessing"]))
            if isinstance(record_payload.get("preprocessing"), Mapping)
            else None
        ),
        "training": (
            dict(cast(Mapping[str, Any], record_payload["training"]))
            if isinstance(record_payload.get("training"), Mapping)
            else None
        ),
    }


def _cfg_mapping(cfg: Any) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError("resolved config must be a mapping")
    return {str(key): value for key, value in payload.items()}


def _build_lightweight_training_surface_record(
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=state_dict,
        include_manifest_characteristics=False,
    )


def resolve_row_target(
    *,
    queue: Mapping[str, Any],
    row: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    artifacts = _row_artifacts(queue=queue, row=row, registry_path=registry_path)
    training_surface_entry = artifacts.get("training_surface_record_json")
    if isinstance(training_surface_entry, Mapping) and bool(training_surface_entry.get("exists")):
        training_surface_path = Path(str(training_surface_entry["path"]))
        spec = _training_surface_record_model_spec(training_surface_path)
        training_surface_record = _load_json_mapping(
            training_surface_path,
            context=f"row {int(row['order']):02d} training surface record",
        )
    else:
        run_dir_entry = artifacts.get("run_dir")
        if not isinstance(run_dir_entry, Mapping):
            raise RuntimeError(f"row {int(row['order']):02d} has no resolvable run directory")
        run_dir = Path(str(run_dir_entry["path"]))
        spec = resolve_queue_row_model_spec(
            row,
            training_experiment=str(queue["training_experiment"]),
        )
        cfg = compose_cfg(
            row=row,
            run_dir=run_dir,
            device="cpu",
            training_experiment=str(queue["training_experiment"]),
        )
        training_surface_record = _build_lightweight_training_surface_record(
            raw_cfg=_cfg_mapping(cfg),
            run_dir=run_dir,
        )
    metrics = row.get("benchmark_metrics")
    if not isinstance(metrics, Mapping):
        metrics = row.get("screen_metrics")
    return {
        "kind": "row",
        "identity": {
            "order": int(row["order"]),
            "delta_id": str(row["delta_id"]),
            "status": str(row["status"]),
            "decision": None if row.get("decision") is None else str(row["decision"]),
            "run_id": None if row.get("run_id") is None else str(row["run_id"]),
        },
        "artifacts": artifacts,
        "resolved": _surface_payload(spec=spec, training_surface_record=training_surface_record),
        "metrics": None if not isinstance(metrics, Mapping) else dict(cast(Mapping[str, Any], metrics)),
    }


def _anchor_run_artifacts(
    *,
    queue: Mapping[str, Any],
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    anchor_run_id = str(queue["anchor_run_id"])
    registry_run = _registry_run_entry(anchor_run_id, registry_path=registry_path)
    run_dir = _registry_artifact_path(registry_run, "run_dir")
    comparison_summary = _registry_artifact_path(registry_run, "comparison_summary_path")
    benchmark_record = _registry_artifact_path(registry_run, "benchmark_run_record_path")
    training_surface_record = _registry_artifact_path(registry_run, "training_surface_record_path")
    best_checkpoint = _registry_artifact_path(registry_run, "best_checkpoint_path")
    benchmark_dir = _registry_artifact_path(registry_run, "benchmark_dir")
    artifacts = {
        "registry_run_present": bool(registry_run is not None),
        "run_dir": None if run_dir is None else _artifact_entry(run_dir),
        "benchmark_dir": None if benchmark_dir is None else _artifact_entry(benchmark_dir),
        "training_surface_record_json": (
            None if training_surface_record is None else _artifact_entry(training_surface_record)
        ),
        "best_checkpoint_path": None if best_checkpoint is None else _artifact_entry(best_checkpoint),
        "comparison_summary_json": (
            None if comparison_summary is None else _artifact_entry(comparison_summary)
        ),
        "benchmark_run_record_json": None if benchmark_record is None else _artifact_entry(benchmark_record),
    }
    if run_dir is not None:
        artifacts["train_history_jsonl"] = _artifact_entry(run_dir / "train_history.jsonl")
        artifacts["gradient_history_jsonl"] = _artifact_entry(run_dir / "gradient_history.jsonl")
        artifacts["telemetry_json"] = _artifact_entry(run_dir / "telemetry.json")
    return artifacts, registry_run


def _anchor_training_surface_record(
    *,
    queue: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_context = queue.get("anchor_context")
    if isinstance(anchor_context, Mapping):
        raw_surface_labels = anchor_context.get("surface_labels")
        surface_labels = (
            dict(cast(Mapping[str, Any], raw_surface_labels))
            if isinstance(raw_surface_labels, Mapping)
            else {}
        )
    else:
        surface_labels = {}

    training_surface_entry = artifacts.get("training_surface_record_json")
    if isinstance(training_surface_entry, Mapping) and bool(training_surface_entry.get("exists")):
        return _load_json_mapping(
            Path(str(training_surface_entry["path"])),
            context="anchor training surface record",
        )

    run_dir_entry = artifacts.get("run_dir")
    checkpoint_entry = artifacts.get("best_checkpoint_path")
    if isinstance(run_dir_entry, Mapping) and isinstance(checkpoint_entry, Mapping) and bool(
        checkpoint_entry.get("exists")
    ):
        checkpoint_payload = torch.load(
            Path(str(checkpoint_entry["path"])),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise RuntimeError("anchor checkpoint payload must be a mapping")
        raw_cfg = checkpoint_payload.get("config")
        if not isinstance(raw_cfg, Mapping):
            raise RuntimeError("anchor checkpoint payload omitted config")
        raw_state_dict = checkpoint_payload.get("model")
        state_dict = raw_state_dict if isinstance(raw_state_dict, Mapping) else None
        record = _build_lightweight_training_surface_record(
            raw_cfg={str(key): value for key, value in raw_cfg.items()},
            run_dir=Path(str(run_dir_entry["path"])),
            state_dict=state_dict,
        )
    else:
        record = {
            "labels": dict(surface_labels),
            "data": {
                "surface_label": surface_labels.get("data"),
            },
            "preprocessing": {
                "surface_label": surface_labels.get("preprocessing"),
            },
            "training": {
                "surface_label": surface_labels.get("training"),
            },
        }
    labels = record.get("labels")
    if not isinstance(labels, Mapping):
        record["labels"] = dict(surface_labels)
    else:
        merged_labels = dict(surface_labels)
        merged_labels.update(dict(cast(Mapping[str, Any], labels)))
        record["labels"] = merged_labels

    if not isinstance(record.get("training"), Mapping) and surface_labels.get("training") is not None:
        record["training"] = {"surface_label": surface_labels.get("training")}
    return record


def _anchor_metrics_payload(registry_run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(registry_run, Mapping):
        return None
    payload: dict[str, Any] = {}
    tab_foundry_metrics = registry_run.get("tab_foundry_metrics")
    if isinstance(tab_foundry_metrics, Mapping):
        payload.update(dict(cast(Mapping[str, Any], tab_foundry_metrics)))
    training_diagnostics = registry_run.get("training_diagnostics")
    if isinstance(training_diagnostics, Mapping):
        payload.update(dict(cast(Mapping[str, Any], training_diagnostics)))
    return payload or None


def resolve_anchor_target(
    *,
    queue: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    spec, metadata = resolve_anchor_model_spec(queue=queue, registry_path=registry_path)
    artifacts, registry_run = _anchor_run_artifacts(queue=queue, registry_path=registry_path)
    training_surface_record = _anchor_training_surface_record(queue=queue, artifacts=artifacts)
    return {
        "kind": "anchor",
        "identity": {
            "run_id": str(queue["anchor_run_id"]),
            "source": str(metadata["source"]),
        },
        "artifacts": artifacts,
        "resolved": _surface_payload(spec=spec, training_surface_record=training_surface_record),
        "metrics": _anchor_metrics_payload(registry_run),
    }


def inspect_sweep_row(
    *,
    order: int,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    row = _find_row(queue, order=int(order))
    resolved_registry_path = registry_path or default_registry_path()
    target = resolve_row_target(queue=queue, row=row, registry_path=resolved_registry_path)
    return {
        "queue": _queue_metadata_payload(queue),
        "row": dict(cast(dict[str, Any], _copy_jsonable(row))),
        "target": target,
    }


def render_sweep_row_text(payload: Mapping[str, Any]) -> str:
    queue = cast(Mapping[str, Any], payload["queue"])
    row = cast(Mapping[str, Any], payload["row"])
    target = cast(Mapping[str, Any], payload["target"])
    resolved = cast(Mapping[str, Any], target["resolved"])
    model = cast(Mapping[str, Any], resolved["model"])
    data = cast(Mapping[str, Any] | None, resolved.get("data"))
    preprocessing = cast(Mapping[str, Any] | None, resolved.get("preprocessing"))
    training = cast(Mapping[str, Any] | None, resolved.get("training"))
    parameter_counts = cast(Mapping[str, Any], model["parameter_counts"])
    lines = [
        "Sweep row inspection.",
        f"sweep_id={queue['sweep_id']}",
        f"order={int(row['order']):02d}",
        f"delta_id={row['delta_id']}",
        f"status={row['status']}",
        f"decision={row.get('decision') or 'n/a'}",
        f"run_id={row.get('run_id') or 'n/a'}",
        f"training_experiment={queue['training_experiment']}",
        f"training_config_profile={queue['training_config_profile']}",
        f"surface_role={queue['surface_role']}",
        f"model.stage_label={model.get('stage_label')}",
        f"model.arch={model.get('arch')}",
        f"model.parameters.total={parameter_counts['total_params']}",
        f"model.parameters.trainable={parameter_counts['trainable_params']}",
    ]
    if data is not None:
        lines.append(f"data.surface_label={data.get('surface_label')}")
    if preprocessing is not None:
        lines.append(f"preprocessing.surface_label={preprocessing.get('surface_label')}")
    if training is not None:
        lines.append(f"training.surface_label={training.get('surface_label')}")
    module_selection = model.get("module_selection")
    if isinstance(module_selection, Mapping):
        lines.append(f"model.module_selection={json.dumps(module_selection, sort_keys=True)}")
    module_hyperparameters = model.get("module_hyperparameters")
    if isinstance(module_hyperparameters, Mapping):
        lines.append(
            f"model.module_hyperparameters={json.dumps(module_hyperparameters, sort_keys=True)}"
        )
    metrics = target.get("metrics")
    if isinstance(metrics, Mapping):
        lines.append(f"metrics={json.dumps(dict(metrics), sort_keys=True)}")
    artifacts = cast(Mapping[str, Any], target["artifacts"])
    for key in (
        "run_dir",
        "benchmark_dir",
        "training_surface_record_json",
        "comparison_summary_json",
    ):
        entry = artifacts.get(key)
        if isinstance(entry, Mapping):
            lines.append(f"{key}={entry['path']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one system-delta sweep row")
    parser.add_argument("--order", type=int, required=True, help="Row order to inspect")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to the benchmark run registry",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = inspect_sweep_row(
        order=int(args.order),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        index_path=Path(str(args.index_path)).expanduser().resolve(),
        catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
        sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
        registry_path=Path(str(args.registry_path)).expanduser().resolve(),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_sweep_row_text(payload))
    return 0
