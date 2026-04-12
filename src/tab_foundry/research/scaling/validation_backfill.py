"""Posthoc validation-loss backfill for completed scaling-study checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import torch

from tab_foundry.bench.artifacts import write_json
from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.config import compose_config
from tab_foundry.repo_paths import normalize_repo_relative_path, repo_root
from tab_foundry.research.scaling.fit import (
    _completed_benchmark_backed_row,
    _registry_root,
)
from tab_foundry.research.scaling.study import (
    ScalingStudyConfig,
    load_scaling_study_config,
)
from tab_foundry.research.scaling.validation_backfill_schema import (
    VALIDATION_BACKFILL_FILENAME,
    VALIDATION_BACKFILL_SCHEMA,
    VALIDATION_BACKFILL_VERSION,
)
from tab_foundry.research.sweep.materialize import load_system_delta_queue, ordered_rows
from tab_foundry.training.evaluate import evaluate_checkpoint


_REQUIRED_TRAIN_FILES = (
    "checkpoints/latest.pt",
    "train_history.jsonl",
    "telemetry.json",
    "training_surface_record.json",
)


@dataclass(frozen=True, slots=True)
class _SourceRoot:
    value: str
    mode: str


@dataclass(frozen=True, slots=True)
class _BackfillCandidate:
    family: str
    sweep_id: str
    row_order: int
    row_label: str
    run_id: str
    run_dir: str
    source_uri: str | None
    missing_artifacts: tuple[str, ...]


def _is_gcs_uri(value: str) -> bool:
    return str(value).startswith("gs://")


def _join_uri(root: str, *parts: str) -> str:
    normalized_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if _is_gcs_uri(root):
        return "/".join([root.rstrip("/"), *normalized_parts])
    path = Path(root).expanduser().resolve()
    for part in normalized_parts:
        path = path / part
    return str(path)


def _uri_exists(uri: str) -> bool:
    if not _is_gcs_uri(uri):
        return Path(uri).exists()
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _copy_uri_to_local(uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not _is_gcs_uri(uri):
        shutil.copy2(Path(uri), local_path)
        return
    result = subprocess.run(
        ["gcloud", "storage", "cp", uri, str(local_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to copy validation artifact from GCS: "
            f"uri={uri!r}, stderr={result.stderr.strip()}"
        )


def _upload_local_to_gcs(local_path: Path, uri: str) -> None:
    result = subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), uri],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to upload validation backfill artifact to GCS: "
            f"uri={uri!r}, stderr={result.stderr.strip()}"
        )


def _source_train_uri(*, source: _SourceRoot, run_dir: str) -> str:
    if source.mode == "launch":
        return _join_uri(source.value, "artifacts", run_dir)
    if source.mode == "preseed":
        return _join_uri(source.value, run_dir)
    raise RuntimeError(f"unsupported validation backfill source mode: {source.mode!r}")


def _source_missing_artifacts(source_uri: str) -> tuple[str, ...]:
    missing = [
        relative_path
        for relative_path in _REQUIRED_TRAIN_FILES
        if not _uri_exists(_join_uri(source_uri, relative_path))
    ]
    return tuple(missing)


def _resolve_source(
    *,
    run_dir: str,
    source_roots: Sequence[_SourceRoot],
) -> tuple[str | None, tuple[str, ...]]:
    last_missing: tuple[str, ...] = _REQUIRED_TRAIN_FILES
    for source in source_roots:
        source_uri = _source_train_uri(source=source, run_dir=run_dir)
        missing = _source_missing_artifacts(source_uri)
        if not missing:
            return source_uri, ()
        last_missing = missing
    return None, last_missing


def _row_label_from_entry(entry: Mapping[str, Any]) -> str:
    model = entry.get("model")
    if not isinstance(model, Mapping):
        return "unknown"
    d_icl = model.get("d_icl")
    layers = model.get("sandwich_layers")
    build_spec = model.get("build_spec")
    if layers is None and isinstance(build_spec, Mapping):
        layers = build_spec.get("sandwich_layers")
    if d_icl is None or layers is None:
        return "unknown"
    return f"{int(d_icl)}x{int(layers)}"


def _registry_run_dir(
    entry: Mapping[str, Any],
    *,
    registry_path: Path,
) -> str:
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("benchmark registry run entry missing artifacts mapping")
    run_dir_raw = artifacts.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        raise RuntimeError("benchmark registry run entry missing artifacts.run_dir")
    root = _registry_root(registry_path)
    return normalize_repo_relative_path(
        resolve_registry_path_value(run_dir_raw, root=root),
        root=root,
    )


def _completed_candidates(
    *,
    config: ScalingStudyConfig,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
    source_roots: Sequence[_SourceRoot],
    start_order: int | None,
    stop_after_order: int | None,
) -> tuple[_BackfillCandidate, ...]:
    registry = load_benchmark_run_registry(registry_path)
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        raise RuntimeError("benchmark run registry missing runs mapping")
    candidates: list[_BackfillCandidate] = []
    for sweep_ref in config.sweeps:
        queue = load_system_delta_queue(
            sweep_id=sweep_ref.sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
        for row in ordered_rows(queue):
            if not _completed_benchmark_backed_row(row):
                continue
            row_order = int(row.get("order") or 0)
            if start_order is not None and row_order < int(start_order):
                continue
            if stop_after_order is not None and row_order > int(stop_after_order):
                continue
            run_id = str(row["run_id"])
            entry = runs.get(run_id)
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"completed queue row references unknown run_id: {run_id!r}")
            run_dir = _registry_run_dir(entry, registry_path=registry_path)
            source_uri, missing_artifacts = _resolve_source(
                run_dir=run_dir,
                source_roots=source_roots,
            )
            candidates.append(
                _BackfillCandidate(
                    family=sweep_ref.family,
                    sweep_id=sweep_ref.sweep_id,
                    row_order=row_order,
                    row_label=_row_label_from_entry(entry),
                    run_id=run_id,
                    run_dir=run_dir,
                    source_uri=source_uri,
                    missing_artifacts=missing_artifacts,
                )
            )
    return tuple(candidates)


def _copy_required_train_artifacts(*, source_uri: str, local_run_dir: Path) -> None:
    for relative_path in _REQUIRED_TRAIN_FILES:
        _copy_uri_to_local(_join_uri(source_uri, relative_path), local_run_dir / relative_path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sidecar_is_current(
    sidecar_path: Path,
    *,
    source_checkpoint_uri: str,
    val_batches: int,
    device: str,
) -> bool:
    payload = _read_json(sidecar_path)
    if payload is None:
        return False
    if payload.get("schema") != VALIDATION_BACKFILL_SCHEMA:
        return False
    checkpoint = payload.get("checkpoint")
    evaluation = payload.get("evaluation")
    metrics = payload.get("metrics")
    return (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("source_uri") == source_checkpoint_uri
        and isinstance(evaluation, Mapping)
        and int(evaluation.get("max_batches") or 0) == int(val_batches)
        and str(evaluation.get("device") or "") == str(device)
        and isinstance(metrics, Mapping)
        and metrics.get("val_loss") is not None
    )


def _checkpoint_global_step(checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return 0
    raw_step = payload.get("global_step")
    if raw_step is None:
        return 0
    try:
        return max(0, int(raw_step))
    except (TypeError, ValueError):
        return 0


def _evaluate_local_checkpoint(
    *,
    checkpoint_path: Path,
    val_batches: int,
    device: str,
) -> tuple[int, dict[str, float]]:
    cfg = compose_config(
        [
            f"eval.checkpoint={checkpoint_path}",
            "eval.split=val",
            f"eval.max_batches={int(val_batches)}",
            f"runtime.device={device}",
            "runtime.mixed_precision=no",
            "logging.use_wandb=false",
        ]
    )
    result = evaluate_checkpoint(cfg)
    global_step = _checkpoint_global_step(checkpoint_path)
    val_loss = float(result.metrics["loss"])
    if not math.isfinite(val_loss):
        raise RuntimeError(f"validation backfill produced non-finite val_loss: {val_loss!r}")
    metrics: dict[str, float] = {"val_loss": val_loss}
    if "acc" in result.metrics:
        metrics["val_acc"] = float(result.metrics["acc"])
    return global_step, metrics


def _sidecar_payload(
    *,
    config: ScalingStudyConfig,
    candidate: _BackfillCandidate,
    source_checkpoint_uri: str,
    local_checkpoint_path: Path,
    global_step: int,
    val_batches: int,
    device: str,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_BACKFILL_SCHEMA,
        "version": VALIDATION_BACKFILL_VERSION,
        "study_id": config.study_id,
        "family": candidate.family,
        "sweep_id": candidate.sweep_id,
        "row_order": candidate.row_order,
        "row_label": candidate.row_label,
        "run_id": candidate.run_id,
        "checkpoint": {
            "path": normalize_repo_relative_path(local_checkpoint_path, root=repo_root()),
            "source_uri": source_checkpoint_uri,
            "global_step": int(global_step),
        },
        "evaluation": {
            "split": "val",
            "max_batches": int(val_batches),
            "device": str(device),
        },
        "metrics": {str(key): float(value) for key, value in metrics.items()},
    }


def _default_cache_root(study_id: str) -> Path:
    return repo_root() / "outputs" / "validation_backfill" / "cache" / study_id


def _default_out_root(study_id: str) -> Path:
    return repo_root() / "outputs" / "validation_backfill" / study_id


def _default_upload_root(
    *,
    launch_roots: Sequence[str],
    study_id: str,
) -> str | None:
    for root in launch_roots:
        if _is_gcs_uri(root):
            return _join_uri(root, "validation_backfill", study_id)
    return None


def _candidate_payload(candidate: _BackfillCandidate) -> dict[str, Any]:
    return {
        "family": candidate.family,
        "sweep_id": candidate.sweep_id,
        "row_order": candidate.row_order,
        "row_label": candidate.row_label,
        "run_id": candidate.run_id,
        "run_dir": candidate.run_dir,
        "source_uri": candidate.source_uri,
        "missing_artifacts": list(candidate.missing_artifacts),
    }


def _write_registry_overlay(
    *,
    registry_path: Path,
    out_root: Path,
    local_run_dirs_by_run_id: Mapping[str, Path],
) -> Path:
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry_payload, dict):
        raise RuntimeError(f"benchmark registry overlay source must be a mapping: {registry_path}")
    runs = registry_payload.get("runs")
    if not isinstance(runs, dict):
        raise RuntimeError(f"benchmark registry overlay source missing runs mapping: {registry_path}")
    for run_id, local_run_dir in local_run_dirs_by_run_id.items():
        entry = runs.get(run_id)
        if not isinstance(entry, dict):
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        normalized_run_dir = str(local_run_dir.expanduser().resolve())
        artifacts["run_dir"] = normalized_run_dir
        artifacts["history_path"] = str(
            (local_run_dir / "train_history.jsonl").expanduser().resolve()
        )
    overlay_path = out_root / "benchmark_run_registry_v1.json"
    write_json(overlay_path, registry_payload)
    return overlay_path


def backfill_validation_study(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
    launch_gcs_roots: Sequence[str] = (),
    preseed_gcs_root: str | None = None,
    cache_root: Path | None = None,
    out_root: Path | None = None,
    val_batches: int = 16,
    device: str = "cpu",
    start_order: int | None = None,
    stop_after_order: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill validation loss sidecars from completed scaling-study checkpoints."""

    config = load_scaling_study_config(
        study_id=study_id,
        study_path=study_path,
        studies_root=studies_root,
    )
    resolved_registry_path = registry_path.expanduser().resolve()
    resolved_index_path = index_path.expanduser().resolve()
    resolved_catalog_path = catalog_path.expanduser().resolve()
    resolved_sweeps_root = sweeps_root.expanduser().resolve()
    resolved_cache_root = (
        _default_cache_root(config.study_id)
        if cache_root is None
        else cache_root.expanduser().resolve()
    )
    resolved_out_root = (
        _default_out_root(config.study_id)
        if out_root is None
        else out_root.expanduser().resolve()
    )
    launch_roots = tuple(str(root).rstrip("/") for root in launch_gcs_roots)
    preseed_roots = () if preseed_gcs_root is None else (str(preseed_gcs_root).rstrip("/"),)
    source_roots = tuple(_SourceRoot(root, "launch") for root in launch_roots) + tuple(
        _SourceRoot(root, "preseed") for root in preseed_roots
    )
    if not source_roots:
        raise RuntimeError("validation backfill requires at least one launch or preseed source root")
    candidates = _completed_candidates(
        config=config,
        registry_path=resolved_registry_path,
        index_path=resolved_index_path,
        catalog_path=resolved_catalog_path,
        sweeps_root=resolved_sweeps_root,
        source_roots=source_roots,
        start_order=start_order,
        stop_after_order=stop_after_order,
    )
    upload_root = _default_upload_root(
        launch_roots=launch_roots,
        study_id=config.study_id,
    )
    row_payloads: list[dict[str, Any]] = []
    uploaded_files: list[str] = []
    local_run_dirs_by_run_id: dict[str, Path] = {}
    counts = {
        "candidate_rows": len(candidates),
        "validated_rows": 0,
        "skipped_existing": 0,
        "incomplete_artifacts": 0,
        "dry_run_ready": 0,
    }

    for candidate in candidates:
        row_payload = _candidate_payload(candidate)
        if candidate.source_uri is None:
            row_payload["status"] = "incomplete_artifacts"
            counts["incomplete_artifacts"] += 1
            row_payloads.append(row_payload)
            continue
        row_payload["status"] = "ready" if dry_run else "pending"
        if dry_run:
            counts["dry_run_ready"] += 1
            row_payloads.append(row_payload)
            continue
        local_run_dir = resolved_cache_root / candidate.run_dir
        source_checkpoint_uri = _join_uri(candidate.source_uri, "checkpoints/latest.pt")
        sidecar_path = local_run_dir / VALIDATION_BACKFILL_FILENAME
        if (
            not force
            and sidecar_path.exists()
            and _sidecar_is_current(
                sidecar_path,
                source_checkpoint_uri=source_checkpoint_uri,
                val_batches=val_batches,
                device=device,
            )
        ):
            row_payload["status"] = "skipped_existing"
            row_payload["sidecar_path"] = str(sidecar_path)
            counts["skipped_existing"] += 1
            local_run_dirs_by_run_id[candidate.run_id] = local_run_dir
            row_payloads.append(row_payload)
            continue
        _copy_required_train_artifacts(
            source_uri=candidate.source_uri,
            local_run_dir=local_run_dir,
        )
        local_checkpoint_path = local_run_dir / "checkpoints" / "latest.pt"
        global_step, metrics = _evaluate_local_checkpoint(
            checkpoint_path=local_checkpoint_path,
            val_batches=val_batches,
            device=device,
        )
        sidecar = _sidecar_payload(
            config=config,
            candidate=candidate,
            source_checkpoint_uri=source_checkpoint_uri,
            local_checkpoint_path=local_checkpoint_path,
            global_step=global_step,
            val_batches=val_batches,
            device=device,
            metrics=metrics,
        )
        write_json(sidecar_path, sidecar)
        local_run_dirs_by_run_id[candidate.run_id] = local_run_dir
        row_payload["status"] = "validated"
        row_payload["sidecar_path"] = str(sidecar_path)
        row_payload["metrics"] = dict(sidecar["metrics"])
        counts["validated_rows"] += 1
        if upload_root is not None:
            sidecar_uri = _join_uri(
                upload_root,
                candidate.run_dir,
                VALIDATION_BACKFILL_FILENAME,
            )
            _upload_local_to_gcs(sidecar_path, sidecar_uri)
            uploaded_files.append(sidecar_uri)
            row_payload["sidecar_uri"] = sidecar_uri
        row_payloads.append(row_payload)

    payload = {
        "study": config.as_dict(),
        "config": {
            "cache_root": str(resolved_cache_root),
            "out_root": str(resolved_out_root),
            "val_batches": int(val_batches),
            "device": str(device),
            "dry_run": bool(dry_run),
            "upload_root": upload_root,
        },
        "counts": counts,
        "rows": row_payloads,
        "uploaded_files": uploaded_files,
    }
    if not dry_run:
        registry_overlay_path = _write_registry_overlay(
            registry_path=resolved_registry_path,
            out_root=resolved_out_root,
            local_run_dirs_by_run_id=local_run_dirs_by_run_id,
        )
        manifest_path = resolved_out_root / "validation_backfill_manifest.json"
        payload["registry_overlay_path"] = str(registry_overlay_path)
        payload["manifest_path"] = str(manifest_path)
        if upload_root is not None:
            registry_overlay_uri = _join_uri(upload_root, "benchmark_run_registry_v1.json")
            _upload_local_to_gcs(registry_overlay_path, registry_overlay_uri)
            payload["registry_overlay_uri"] = registry_overlay_uri
            manifest_uri = _join_uri(upload_root, "validation_backfill_manifest.json")
            payload["manifest_uri"] = manifest_uri
        write_json(manifest_path, payload)
        if upload_root is not None:
            _upload_local_to_gcs(manifest_path, str(payload["manifest_uri"]))
    return payload


def render_validation_backfill_text(payload: Mapping[str, Any]) -> str:
    counts = payload.get("counts")
    counts_payload = counts if isinstance(counts, Mapping) else {}
    lines = [
        f"Candidate rows: {counts_payload.get('candidate_rows', 0)}",
        f"Validated rows: {counts_payload.get('validated_rows', 0)}",
        f"Skipped existing: {counts_payload.get('skipped_existing', 0)}",
        f"Incomplete artifacts: {counts_payload.get('incomplete_artifacts', 0)}",
    ]
    config = payload.get("config")
    if isinstance(config, Mapping):
        lines.append(f"Cache root: {config.get('cache_root')}")
        upload_root = config.get("upload_root")
        if upload_root:
            lines.append(f"Upload root: {upload_root}")
    registry_overlay_path = payload.get("registry_overlay_path")
    if registry_overlay_path:
        lines.append(f"Registry overlay: {registry_overlay_path}")
    return "\n".join(lines)
