"""Dependency-light read-only helpers for the control-baseline registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from .repo_paths import normalize_repo_relative_path, repo_root, resolve_repo_relative_path


REGISTRY_SCHEMA = "tab-foundry-control-baselines-v1"
REGISTRY_VERSION = 1
_TOP_LEVEL_KEYS = {"schema", "version", "baselines"}
_ENTRY_KEYS = {
    "baseline_id",
    "experiment",
    "config_profile",
    "budget_class",
    "manifest_path",
    "seed_set",
    "run_dir",
    "comparison_summary_path",
    "benchmark_bundle",
    "tab_foundry_metrics",
}
_BENCHMARK_BUNDLE_KEYS = {"name", "version", "source_path", "task_count", "task_ids"}
_REQUIRED_TAB_FOUNDRY_METRIC_KEYS = {
    "best_step",
    "best_training_time",
    "final_step",
    "final_training_time",
}
_OPTIONAL_TAB_FOUNDRY_METRIC_KEYS = {
    "best_roc_auc",
    "best_log_loss",
    "best_brier_score",
    "best_crps",
    "best_avg_pinball_loss",
    "best_picp_90",
    "final_roc_auc",
    "final_log_loss",
    "final_brier_score",
    "final_crps",
    "final_avg_pinball_loss",
    "final_picp_90",
}
_TAB_FOUNDRY_METRIC_KEYS = _REQUIRED_TAB_FOUNDRY_METRIC_KEYS | _OPTIONAL_TAB_FOUNDRY_METRIC_KEYS


def default_control_baseline_registry_path() -> Path:
    """Return the repo-tracked control-baseline registry path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"


def resolve_registry_path_value(
    value: str,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve one registry-stored path value."""

    return resolve_repo_relative_path(value, root=root)

def normalize_registry_path_value(
    path: Path,
    *,
    root: Path | None = None,
) -> str:
    """Normalize one absolute path into the repo-relative registry form when possible."""

    return normalize_repo_relative_path(path, root=root)


def _copy_jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _validate_baseline_entry(entry: Any, *, baseline_id: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"control baseline entry must be an object: baseline_id={baseline_id}")
    entry_payload = {str(key): value for key, value in entry.items()}
    actual_keys = set(entry_payload.keys())
    if actual_keys != _ENTRY_KEYS:
        raise RuntimeError(
            f"control baseline entry keys mismatch for {baseline_id}: "
            f"missing={sorted(_ENTRY_KEYS - actual_keys)}, extra={sorted(actual_keys - _ENTRY_KEYS)}"
        )
    actual_baseline_id = entry_payload.get("baseline_id")
    if not isinstance(actual_baseline_id, str) or actual_baseline_id != baseline_id:
        raise RuntimeError(
            "control baseline entry baseline_id mismatch: "
            f"expected={baseline_id!r}, actual={actual_baseline_id!r}"
        )
    seed_set = entry_payload.get("seed_set")
    if not isinstance(seed_set, list) or not seed_set:
        raise RuntimeError(f"control baseline entry seed_set must be a non-empty list: {baseline_id}")
    benchmark_bundle = entry_payload.get("benchmark_bundle")
    if not isinstance(benchmark_bundle, Mapping):
        raise RuntimeError(
            f"control baseline entry benchmark_bundle must match expected schema: {baseline_id}"
        )
    actual_bundle_keys = {str(key) for key in benchmark_bundle.keys()}
    if actual_bundle_keys != _BENCHMARK_BUNDLE_KEYS:
        raise RuntimeError(
            f"control baseline entry benchmark_bundle must match expected schema: {baseline_id}"
        )
    tab_foundry_metrics = entry_payload.get("tab_foundry_metrics")
    if not isinstance(tab_foundry_metrics, Mapping):
        raise RuntimeError(
            f"control baseline entry tab_foundry_metrics must match expected schema: {baseline_id}"
        )
    actual_metric_keys = {str(key) for key in tab_foundry_metrics.keys()}
    if not _REQUIRED_TAB_FOUNDRY_METRIC_KEYS.issubset(actual_metric_keys) or not actual_metric_keys.issubset(
        _TAB_FOUNDRY_METRIC_KEYS
    ):
        raise RuntimeError(
            f"control baseline entry tab_foundry_metrics must match expected schema: {baseline_id}"
        )
    return entry_payload


def load_control_baseline_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the control-baseline registry."""

    registry_path = (path or default_control_baseline_registry_path()).expanduser().resolve()
    if not registry_path.exists():
        raise RuntimeError(f"control baseline registry does not exist: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"control baseline registry must be a JSON object: {registry_path}")

    actual_keys = set(payload.keys())
    if actual_keys != _TOP_LEVEL_KEYS:
        raise RuntimeError(
            "control baseline registry keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}"
        )
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError(
            "control baseline registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload.get('schema')!r}"
        )
    if int(payload.get("version", -1)) != REGISTRY_VERSION:
        raise RuntimeError(
            "control baseline registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload.get('version')!r}"
        )

    baselines = payload.get("baselines")
    if not isinstance(baselines, Mapping):
        raise RuntimeError("control baseline registry baselines must be an object")
    normalized_baselines: dict[str, Any] = {}
    for baseline_id, entry in baselines.items():
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise RuntimeError("control baseline registry baseline ids must be non-empty strings")
        normalized_baselines[str(baseline_id)] = _validate_baseline_entry(entry, baseline_id=str(baseline_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": cast(dict[str, Any], normalized_baselines),
    }


def load_control_baseline_entry(
    baseline_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load one control-baseline entry by id."""

    registry = load_control_baseline_registry(registry_path)
    baselines = cast(dict[str, dict[str, Any]], registry["baselines"])
    entry = baselines.get(str(baseline_id))
    if entry is None:
        raise RuntimeError(f"unknown control baseline id: {baseline_id}")
    return cast(dict[str, Any], _copy_jsonable(entry))
