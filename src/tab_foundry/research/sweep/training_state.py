"""Training artifact reuse helpers for sweep execution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.hashing import sha256_text
from tab_foundry.training.checkpoint_paths import resolve_latest_checkpoint_path
from tab_foundry.training.surface import (
    normalize_training_backend,
)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def _copy_jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    if not isinstance(copied, dict):
        raise RuntimeError("expected copied JSON mapping")
    return cast(dict[str, Any], copied)


def load_training_surface_record(record_path: Path) -> dict[str, Any] | None:
    if not record_path.exists():
        return None
    try:
        return _read_json_mapping(record_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def _training_surface_backend(payload: Mapping[str, Any]) -> str | None:
    training = payload.get("training")
    if not isinstance(training, Mapping):
        return None
    raw_backend = training.get("backend")
    if not isinstance(raw_backend, str) or not raw_backend.strip():
        return None
    try:
        return normalize_training_backend(raw_backend)
    except ValueError:
        return None


def training_surface_record_backend(record_path: Path) -> str | None:
    payload = load_training_surface_record(record_path)
    if payload is None:
        return None
    return _training_surface_backend(payload)


def _training_telemetry_succeeded(telemetry_path: Path) -> bool:
    if not telemetry_path.exists():
        return False
    try:
        payload = _read_json_mapping(telemetry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(payload.get("success") is True)


def _archive_train_dir(train_dir: Path, *, reason: str) -> Path | None:
    if not train_dir.exists() or not train_dir.is_dir():
        return None
    try:
        has_entries = any(train_dir.iterdir())
    except OSError:
        return None
    if not has_entries:
        return None
    suffix = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    candidate = train_dir.with_name(f"{train_dir.name}_{reason}_{suffix}")
    counter = 1
    while candidate.exists():
        candidate = train_dir.with_name(f"{train_dir.name}_{reason}_{suffix}_{counter:02d}")
        counter += 1
    train_dir.rename(candidate)
    return candidate


def archive_incomplete_train_dir(train_dir: Path) -> Path | None:
    return _archive_train_dir(train_dir, reason="incomplete")


def archive_incompatible_train_dir(train_dir: Path) -> Path | None:
    return _archive_train_dir(train_dir, reason="incompatible")


def normalize_training_surface_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ("labels", "model", "data", "preprocessing", "training", "runtime"):
        raw_value = payload.get(key)
        if not isinstance(raw_value, Mapping):
            continue
        normalized[key] = _copy_jsonable_mapping(cast(Mapping[str, Any], raw_value))

    data_payload = normalized.get("data")
    if isinstance(data_payload, dict):
        manifest = data_payload.get("manifest")
        if isinstance(manifest, dict):
            manifest.pop("characteristics", None)
            manifest.pop("characteristics_error", None)

    training_payload = normalized.get("training")
    if isinstance(training_payload, dict):
        raw_backend = training_payload.get("backend")
        if raw_backend is not None:
            try:
                normalized_backend = normalize_training_backend(raw_backend)
            except ValueError:
                normalized_backend = None
            if normalized_backend is None:
                training_payload.pop("backend", None)
            else:
                training_payload["backend"] = normalized_backend

    return normalized


def training_surface_record_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            normalize_training_surface_record(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def completed_train_artifacts_exist(
    run_dir: Path,
    *,
    expected_backend: str | None = None,
    expected_training_surface_record: Mapping[str, Any] | None = None,
) -> bool:
    required_paths = (
        run_dir / "train_history.jsonl",
        run_dir / "gradient_history.jsonl",
        run_dir / "telemetry.json",
        run_dir / "training_surface_record.json",
    )
    if not all(path.exists() for path in required_paths):
        return False
    if resolve_latest_checkpoint_path(run_dir) is None:
        return False
    if not _training_telemetry_succeeded(run_dir / "telemetry.json"):
        return False
    if expected_backend is None and expected_training_surface_record is None:
        return True

    observed_record = load_training_surface_record(run_dir / "training_surface_record.json")
    if observed_record is None:
        return False
    if expected_backend is not None and _training_surface_backend(observed_record) != expected_backend:
        return False
    if expected_training_surface_record is None:
        return True
    return normalize_training_surface_record(observed_record) == normalize_training_surface_record(
        expected_training_surface_record
    )
