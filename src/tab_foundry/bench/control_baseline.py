"""Canonical control-baseline registry helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.nanotabpfn import resolve_tab_foundry_best_checkpoint
from tab_foundry.bench.registry_common import (
    copy_jsonable as _copy_jsonable,
    load_comparison_summary as _load_comparison_summary,
    normalize_path_value as _common_normalize_path_value,
    project_root as _project_root,
    resolve_config_path as _common_resolve_config_path,
    resolve_registry_path_value as _common_resolve_registry_path_value,
)
from tab_foundry.bench.registry.storage import (
    ensure_registry_payload as _ensure_registry_payload_common,
    load_versioned_registry_payload as _load_versioned_registry_payload,
    upsert_registry_entry as _upsert_registry_entry_common,
)
from tab_foundry.bench.registry.summary_metrics import (
    benchmark_bundle_payload_from_summary as _benchmark_bundle_payload_from_summary,
    tab_foundry_metrics_from_summary as _tab_foundry_metrics_from_summary,
)


REGISTRY_SCHEMA = "tab-foundry-control-baselines-v1"
REGISTRY_VERSION = 1
DEFAULT_BASELINE_ID = "cls_benchmark_linear_v2"
DEFAULT_CONFIG_PROFILE = "cls_benchmark_linear"
DEFAULT_BUDGET_CLASS = "short-run"

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


def project_root() -> Path:
    """Return the repository root for repo-relative artifact paths."""

    return _project_root()


def _normalize_path_value(path: Path) -> str:
    return _common_normalize_path_value(path, root=project_root())


def resolve_registry_path_value(value: str) -> Path:
    """Resolve a registry path value to an absolute path."""

    return _common_resolve_registry_path_value(value, root=project_root())


def _resolve_config_path(raw_value: Any) -> Path:
    return _common_resolve_config_path(raw_value, root=project_root())


def default_control_baseline_registry_path() -> Path:
    """Return the repo-tracked control baseline registry path."""

    return Path(__file__).resolve().with_name("control_baselines_v1.json")


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {},
    }


def _load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    return _load_versioned_registry_payload(
        path,
        allow_missing=allow_missing,
        empty_payload=_empty_registry(),
        top_level_keys=_TOP_LEVEL_KEYS,
        schema=REGISTRY_SCHEMA,
        version=REGISTRY_VERSION,
        entries_key="baselines",
        registry_label="control baseline registry",
        validate_entry_fn=_validate_baseline_entry,
        entry_label="baseline_id",
    )


def load_control_baseline_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the control baseline registry."""

    return _load_registry_payload(path or default_control_baseline_registry_path(), allow_missing=False)


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    return _ensure_registry_payload_common(
        path,
        default_path=default_control_baseline_registry_path(),
        load_registry_payload_fn=_load_registry_payload,
    )


def _validate_baseline_entry(entry: Any, *, baseline_id: str) -> None:
    if not isinstance(entry, dict):
        raise RuntimeError(f"control baseline entry must be an object: baseline_id={baseline_id}")
    actual_keys = set(entry.keys())
    if actual_keys != _ENTRY_KEYS:
        raise RuntimeError(
            f"control baseline entry keys mismatch for {baseline_id}: "
            f"missing={sorted(_ENTRY_KEYS - actual_keys)}, extra={sorted(actual_keys - _ENTRY_KEYS)}"
        )
    if str(entry["baseline_id"]) != baseline_id:
        raise RuntimeError(
            "control baseline entry baseline_id mismatch: "
            f"expected={baseline_id!r}, actual={entry['baseline_id']!r}"
        )
    if not isinstance(entry["seed_set"], list) or not entry["seed_set"]:
        raise RuntimeError(f"control baseline entry seed_set must be a non-empty list: {baseline_id}")
    benchmark_bundle = entry["benchmark_bundle"]
    if not isinstance(benchmark_bundle, dict) or set(benchmark_bundle.keys()) != _BENCHMARK_BUNDLE_KEYS:
        raise RuntimeError(
            f"control baseline entry benchmark_bundle must match expected schema: {baseline_id}"
        )
    tab_foundry_metrics = entry["tab_foundry_metrics"]
    if not isinstance(tab_foundry_metrics, dict):
        raise RuntimeError(
            f"control baseline entry tab_foundry_metrics must match expected schema: {baseline_id}"
        )
    actual_metric_keys = set(tab_foundry_metrics.keys())
    if not _REQUIRED_TAB_FOUNDRY_METRIC_KEYS.issubset(actual_metric_keys) or not actual_metric_keys.issubset(
        _TAB_FOUNDRY_METRIC_KEYS
    ):
        raise RuntimeError(
            f"control baseline entry tab_foundry_metrics must match expected schema: {baseline_id}"
        )


def load_control_baseline_entry(
    baseline_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load one control baseline entry by id."""

    registry = load_control_baseline_registry(registry_path)
    baselines = cast(dict[str, dict[str, Any]], registry["baselines"])
    entry = baselines.get(str(baseline_id))
    if entry is None:
        raise RuntimeError(f"unknown control baseline id: {baseline_id}")
    return _copy_jsonable(entry)


def derive_control_baseline_entry(
    *,
    baseline_id: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
) -> dict[str, Any]:
    """Derive one control baseline entry from a completed run and comparison summary."""

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_summary_path = comparison_summary_path.expanduser().resolve()
    summary = _load_comparison_summary(resolved_summary_path)
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    summary_run_dir_raw = tab_foundry.get("run_dir")
    if not isinstance(summary_run_dir_raw, str) or not summary_run_dir_raw.strip():
        raise RuntimeError("comparison summary tab_foundry.run_dir must be a non-empty string")
    summary_run_dir = Path(summary_run_dir_raw).expanduser().resolve()
    if summary_run_dir != resolved_run_dir:
        raise RuntimeError(
            "comparison summary run_dir does not match requested run dir: "
            f"summary={summary_run_dir}, requested={resolved_run_dir}"
        )

    best_checkpoint = resolve_tab_foundry_best_checkpoint(resolved_run_dir)
    checkpoint_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {best_checkpoint}")
    raw_cfg = checkpoint_payload.get("config")
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"checkpoint config must be a mapping: {best_checkpoint}")
    data_cfg = raw_cfg.get("data")
    runtime_cfg = raw_cfg.get("runtime")
    if not isinstance(data_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise RuntimeError(f"checkpoint config must include data/runtime mappings: {best_checkpoint}")
    manifest_path = _resolve_config_path(data_cfg.get("manifest_path"))
    seed_raw = runtime_cfg.get("seed")
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        raise RuntimeError(f"checkpoint runtime.seed must be an int: {best_checkpoint}")

    benchmark_bundle = cast(dict[str, Any], summary["benchmark_bundle"])
    benchmark_bundle_payload = _benchmark_bundle_payload_from_summary(
        benchmark_bundle,
        source_context="comparison summary benchmark_bundle.source_path",
        normalize_path_value_fn=_normalize_path_value,
        resolve_registry_path_value_fn=resolve_registry_path_value,
    )
    tab_foundry_metrics = _tab_foundry_metrics_from_summary(tab_foundry)
    entry = {
        "baseline_id": str(baseline_id),
        "experiment": str(experiment),
        "config_profile": str(config_profile),
        "budget_class": str(budget_class),
        "manifest_path": _normalize_path_value(manifest_path),
        "seed_set": [int(seed_raw)],
        "run_dir": _normalize_path_value(resolved_run_dir),
        "comparison_summary_path": _normalize_path_value(resolved_summary_path),
        "benchmark_bundle": benchmark_bundle_payload,
        "tab_foundry_metrics": tab_foundry_metrics,
    }
    _validate_baseline_entry(entry, baseline_id=str(baseline_id))
    return entry


def upsert_control_baseline_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> Path:
    """Insert or replace one control baseline entry in the registry."""

    return _upsert_registry_entry_common(
        entry,
        entry_id_key="baseline_id",
        validate_entry_fn=_validate_baseline_entry,
        registry_path=registry_path,
        default_path=default_control_baseline_registry_path(),
        load_registry_payload_fn=_load_registry_payload,
        entries_key="baselines",
        write_json_fn=write_json,
        copy_jsonable_fn=_copy_jsonable,
    )


def freeze_control_baseline(
    *,
    baseline_id: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Promote a completed run and comparison summary into the baseline registry."""

    entry = derive_control_baseline_entry(
        baseline_id=baseline_id,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary_path,
    )
    resolved_registry_path = upsert_control_baseline_entry(entry, registry_path=registry_path)
    return {
        "registry_path": str(resolved_registry_path),
        "baseline": entry,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze one canonical tab-foundry control baseline")
    parser.add_argument("--run-dir", required=True, help="Completed tab-foundry run directory")
    parser.add_argument(
        "--comparison-summary",
        required=True,
        help="Benchmark comparison_summary.json for the same run",
    )
    parser.add_argument(
        "--baseline-id",
        default=DEFAULT_BASELINE_ID,
        help="Registry id for the frozen baseline",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_CONFIG_PROFILE,
        help="Logical experiment name stored in the registry entry",
    )
    parser.add_argument(
        "--config-profile",
        default=DEFAULT_CONFIG_PROFILE,
        help="Config profile name stored in the registry entry",
    )
    parser.add_argument(
        "--budget-class",
        default=DEFAULT_BUDGET_CLASS,
        help="Budget class label stored in the registry entry",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_control_baseline_registry_path()),
        help="Control baseline registry JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = freeze_control_baseline(
        baseline_id=str(args.baseline_id),
        experiment=str(args.experiment),
        config_profile=str(args.config_profile),
        budget_class=str(args.budget_class),
        run_dir=Path(str(args.run_dir)),
        comparison_summary_path=Path(str(args.comparison_summary)),
        registry_path=Path(str(args.registry_path)),
    )
    print("Control baseline frozen:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  baseline={result['baseline']}")
    return 0
