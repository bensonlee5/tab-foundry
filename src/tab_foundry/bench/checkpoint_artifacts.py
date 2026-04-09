"""Benchmark checkpoint publication and resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from tab_foundry.bench.openml_benchmark import resolve_tab_foundry_best_checkpoint
from tab_foundry.benchmark_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.bench.artifacts import write_json
from tab_foundry.repo_paths import repo_root
from tab_foundry.training.instability import telemetry_path
from tab_foundry.training.wandb import (
    download_checkpoint_artifact,
    publish_checkpoint_artifact,
)


CheckpointSource = Literal["local_registry", "wandb_artifact"]

_WANDB_REQUIRED_FIELDS = ("project", "run_id", "run_name")


@dataclass(frozen=True, slots=True)
class CheckpointResolution:
    """Resolved checkpoint source for one benchmark run."""

    run_id: str
    source: CheckpointSource
    checkpoint_path: Path
    artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class PublishedBenchmarkCheckpoint:
    """Result of publishing one benchmark run checkpoint to W&B."""

    run_id: str
    checkpoint_path: Path
    artifact_ref: str
    wandb: dict[str, str]


def default_checkpoint_cache_root() -> Path:
    """Return the repo-local checkpoint artifact cache root."""

    return repo_root() / ".artifacts" / "checkpoints"


def artifact_name_for_run_id(run_id: str) -> str:
    """Return the stable W&B artifact name for one benchmark run."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RuntimeError("run_id must be a non-empty string")
    return f"benchmark-checkpoint-{normalized_run_id}"


def normalized_wandb_registry_payload(value: Any) -> dict[str, str] | None:
    """Normalize persisted W&B identity metadata from a registry entry or telemetry payload."""

    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str] = {}
    for field in ("entity", "project", "run_id", "run_name"):
        raw_value = value.get(field)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            raise RuntimeError(f"wandb metadata field {field!r} must be a string when present")
        stripped = raw_value.strip()
        if stripped:
            normalized[field] = stripped
    missing = [field for field in _WANDB_REQUIRED_FIELDS if field not in normalized]
    if missing:
        return None
    return normalized


def wandb_registry_payload_from_telemetry(telemetry_payload: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Extract registry-safe W&B identity fields from one telemetry payload."""

    if not isinstance(telemetry_payload, Mapping):
        return None
    return normalized_wandb_registry_payload(telemetry_payload.get("wandb"))


def remote_artifacts_payload(*, artifact_ref: str) -> dict[str, str]:
    """Build the persisted remote-artifact payload for one checkpoint artifact."""

    normalized_ref = str(artifact_ref).strip()
    if not normalized_ref:
        raise RuntimeError("artifact_ref must be a non-empty string")
    return {
        "best_checkpoint_wandb_artifact": normalized_ref,
    }


def load_training_telemetry_payload(run_dir: Path) -> dict[str, Any] | None:
    """Load one run's telemetry payload when present."""

    resolved_path = telemetry_path(run_dir)
    if not resolved_path.exists():
        return None
    payload = cast(dict[str, Any], json.loads(resolved_path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise RuntimeError(f"telemetry.json must be a JSON object: {resolved_path}")
    return payload


def resolve_registry_best_checkpoint_path(
    entry: Mapping[str, Any],
    *,
    registry_root: Path,
    run_dir_override: Path | None = None,
) -> Path:
    """Resolve the best checkpoint path for one registry entry."""

    if run_dir_override is not None:
        return resolve_tab_foundry_best_checkpoint(run_dir_override.expanduser().resolve())
    raw_artifacts = entry.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise RuntimeError("benchmark run entry is missing artifacts metadata")
    raw_checkpoint_path = raw_artifacts.get("best_checkpoint_path")
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path.strip():
        raise RuntimeError("benchmark run entry artifacts.best_checkpoint_path must be a non-empty string")
    return resolve_registry_path_value(raw_checkpoint_path, root=registry_root)


def resolve_registry_run_dir(
    entry: Mapping[str, Any],
    *,
    registry_root: Path,
    run_dir_override: Path | None = None,
) -> Path:
    """Resolve the canonical run directory for one registry entry."""

    if run_dir_override is not None:
        return run_dir_override.expanduser().resolve()
    raw_artifacts = entry.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise RuntimeError("benchmark run entry is missing artifacts metadata")
    raw_run_dir = raw_artifacts.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        raise RuntimeError("benchmark run entry artifacts.run_dir must be a non-empty string")
    return resolve_registry_path_value(raw_run_dir, root=registry_root)


def publish_benchmark_checkpoint_artifact(
    *,
    run_id: str,
    entry: Mapping[str, Any],
    registry_root: Path,
    run_dir_override: Path | None = None,
) -> PublishedBenchmarkCheckpoint:
    """Publish one benchmark run checkpoint to W&B."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RuntimeError("run_id must be a non-empty string")
    checkpoint_path = resolve_registry_best_checkpoint_path(
        entry,
        registry_root=registry_root,
        run_dir_override=run_dir_override,
    )
    if not checkpoint_path.exists():
        raise RuntimeError(f"benchmark checkpoint does not exist: {checkpoint_path}")
    wandb_payload = normalized_wandb_registry_payload(entry.get("wandb"))
    if wandb_payload is None:
        telemetry_payload = load_training_telemetry_payload(
            resolve_registry_run_dir(
                entry,
                registry_root=registry_root,
                run_dir_override=run_dir_override,
            )
        )
        wandb_payload = wandb_registry_payload_from_telemetry(telemetry_payload)
    if wandb_payload is None:
        raise RuntimeError(
            "benchmark run is missing W&B identity metadata and cannot publish a checkpoint artifact"
        )

    sweep_payload = entry.get("sweep")
    metadata: dict[str, Any] = {"benchmark_run_id": normalized_run_id}
    if isinstance(sweep_payload, Mapping):
        for key in ("sweep_id", "delta_id", "parent_sweep_id", "queue_order", "run_kind"):
            if key in sweep_payload:
                metadata[key] = sweep_payload[key]
    artifact = publish_checkpoint_artifact(
        checkpoint_path=checkpoint_path,
        artifact_name=artifact_name_for_run_id(normalized_run_id),
        entity=wandb_payload.get("entity"),
        project=wandb_payload["project"],
        run_id=wandb_payload["run_id"],
        run_name=wandb_payload.get("run_name"),
        metadata=metadata,
        aliases=["best"],
    )
    return PublishedBenchmarkCheckpoint(
        run_id=normalized_run_id,
        checkpoint_path=artifact.local_path,
        artifact_ref=artifact.artifact_ref,
        wandb=wandb_payload,
    )


def resolve_benchmark_checkpoint(
    *,
    run_id: str,
    registry_path: Path | None = None,
    allow_remote: bool,
    cache_root: Path | None = None,
) -> CheckpointResolution:
    """Resolve one benchmark checkpoint from the registry or W&B artifact cache."""

    resolved_registry_path = (registry_path or default_benchmark_run_registry_path()).expanduser().resolve()
    registry_payload = load_benchmark_run_registry(resolved_registry_path)
    runs = cast(dict[str, Any], registry_payload["runs"])
    try:
        entry = cast(dict[str, Any], runs[str(run_id)])
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark registry run_id: {run_id!r}") from exc
    local_path = resolve_registry_best_checkpoint_path(
        entry,
        registry_root=repo_root(),
    )
    if local_path.exists():
        return CheckpointResolution(
            run_id=str(run_id),
            source="local_registry",
            checkpoint_path=local_path,
        )
    if not allow_remote:
        raise RuntimeError(
            "benchmark checkpoint is not available locally; rerun with --allow-remote to use W&B "
            f"artifact recovery for run_id={run_id}"
        )
    remote_payload = entry.get("remote_artifacts")
    if not isinstance(remote_payload, Mapping):
        raise RuntimeError(
            "benchmark run entry does not include remote checkpoint metadata: "
            f"run_id={run_id!r}"
        )
    artifact_ref = remote_payload.get("best_checkpoint_wandb_artifact")
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        raise RuntimeError(
            "benchmark run entry remote_artifacts.best_checkpoint_wandb_artifact must be a non-empty "
            f"string: run_id={run_id!r}"
        )
    resolved_cache_root = (
        default_checkpoint_cache_root()
        if cache_root is None
        else cache_root.expanduser().resolve()
    )
    downloaded = download_checkpoint_artifact(
        artifact_ref=artifact_ref,
        out_dir=resolved_cache_root / str(run_id),
    )
    return CheckpointResolution(
        run_id=str(run_id),
        source="wandb_artifact",
        checkpoint_path=downloaded.local_path,
        artifact_ref=downloaded.artifact_ref,
    )


def backfill_benchmark_checkpoint_artifact(
    *,
    run_id: str,
    registry_path: Path | None = None,
    run_dir_override: Path | None = None,
) -> PublishedBenchmarkCheckpoint:
    """Publish one historical checkpoint artifact and persist its registry metadata."""

    resolved_registry_path = (registry_path or default_benchmark_run_registry_path()).expanduser().resolve()
    registry_payload = load_benchmark_run_registry(resolved_registry_path)
    runs = cast(dict[str, Any], registry_payload["runs"])
    try:
        entry = cast(dict[str, Any], runs[str(run_id)])
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark registry run_id: {run_id!r}") from exc
    published = publish_benchmark_checkpoint_artifact(
        run_id=str(run_id),
        entry=entry,
        registry_root=repo_root(),
        run_dir_override=run_dir_override,
    )
    updated_entry = {str(key): value for key, value in entry.items()}
    updated_entry["wandb"] = dict(published.wandb)
    updated_entry["remote_artifacts"] = remote_artifacts_payload(artifact_ref=published.artifact_ref)
    runs[str(run_id)] = updated_entry
    write_json(resolved_registry_path, registry_payload)
    return published
