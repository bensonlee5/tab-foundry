"""Training artifact reuse helpers for sweep execution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.training.artifacts import resolve_latest_checkpoint_path
from tab_foundry.training.surface import (
    normalize_training_backend,
)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def training_surface_record_backend(record_path: Path) -> str | None:
    if not record_path.exists():
        return None
    try:
        payload = _read_json_mapping(record_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
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


def _training_telemetry_succeeded(telemetry_path: Path) -> bool:
    if not telemetry_path.exists():
        return False
    try:
        payload = _read_json_mapping(telemetry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(payload.get("success") is True)


def archive_incomplete_train_dir(train_dir: Path) -> Path | None:
    if not train_dir.exists() or not train_dir.is_dir():
        return None
    try:
        has_entries = any(train_dir.iterdir())
    except OSError:
        return None
    if not has_entries:
        return None
    suffix = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    candidate = train_dir.with_name(f"{train_dir.name}_incomplete_{suffix}")
    counter = 1
    while candidate.exists():
        candidate = train_dir.with_name(f"{train_dir.name}_incomplete_{suffix}_{counter:02d}")
        counter += 1
    train_dir.rename(candidate)
    return candidate


def completed_train_artifacts_exist(run_dir: Path, *, expected_backend: str | None = None) -> bool:
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
    if expected_backend is None:
        return True
    return training_surface_record_backend(run_dir / "training_surface_record.json") == expected_backend
