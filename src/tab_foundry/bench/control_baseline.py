"""Canonical control-baseline registry helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.nanotabpfn import resolve_tab_foundry_best_checkpoint


REGISTRY_SCHEMA = "tab-foundry-control-baselines-v1"
REGISTRY_VERSION = 1
DEFAULT_BASELINE_ID = "cls_benchmark_linear_v1"
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
_TAB_FOUNDRY_METRIC_KEYS = {
    "best_step",
    "best_training_time",
    "best_roc_auc",
    "final_step",
    "final_training_time",
    "final_roc_auc",
}


def project_root() -> Path:
    """Return the repository root for repo-relative artifact paths."""

    return Path(__file__).resolve().parents[3]


def default_control_baseline_registry_path() -> Path:
    """Return the repo-tracked control baseline registry path."""

    return Path(__file__).resolve().with_name("control_baselines_v1.json")


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {},
    }


def _copy_jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))


def _normalize_path_value(path: Path) -> str:
    resolved = path.expanduser().resolve()
    root = project_root()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def resolve_registry_path_value(value: str) -> Path:
    """Resolve a registry path value to an absolute path."""

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def _resolve_config_path(raw_value: Any) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise RuntimeError("checkpoint config must include a non-empty data.manifest_path")
    return resolve_registry_path_value(str(raw_value))


def _load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    registry_path = path.expanduser().resolve()
    if not registry_path.exists():
        if allow_missing:
            return _empty_registry()
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
    if payload["schema"] != REGISTRY_SCHEMA:
        raise RuntimeError(
            "control baseline registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload['schema']!r}"
        )
    if int(payload["version"]) != REGISTRY_VERSION:
        raise RuntimeError(
            "control baseline registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload['version']}"
        )
    baselines = payload["baselines"]
    if not isinstance(baselines, dict):
        raise RuntimeError("control baseline registry baselines must be an object")
    for baseline_id, entry in baselines.items():
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise RuntimeError("control baseline registry baseline ids must be non-empty strings")
        _validate_baseline_entry(entry, baseline_id=str(baseline_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {str(key): value for key, value in baselines.items()},
    }


def load_control_baseline_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the control baseline registry."""

    return _load_registry_payload(path or default_control_baseline_registry_path(), allow_missing=False)


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    registry_path = (path or default_control_baseline_registry_path()).expanduser().resolve()
    payload = _load_registry_payload(registry_path, allow_missing=True)
    return registry_path, payload


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
    if not isinstance(tab_foundry_metrics, dict) or set(tab_foundry_metrics.keys()) != _TAB_FOUNDRY_METRIC_KEYS:
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


def _load_comparison_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"comparison summary must be a JSON object: {path}")
    benchmark_bundle = payload.get("benchmark_bundle")
    tab_foundry = payload.get("tab_foundry")
    if not isinstance(benchmark_bundle, dict):
        raise RuntimeError(f"comparison summary missing benchmark_bundle: {path}")
    if not isinstance(tab_foundry, dict):
        raise RuntimeError(f"comparison summary missing tab_foundry section: {path}")
    return cast(dict[str, Any], payload)


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
    benchmark_bundle_source = benchmark_bundle.get("source_path")
    if not isinstance(benchmark_bundle_source, str) or not benchmark_bundle_source.strip():
        raise RuntimeError("comparison summary benchmark_bundle.source_path must be a non-empty string")
    benchmark_bundle_payload = {
        "name": str(benchmark_bundle["name"]),
        "version": int(benchmark_bundle["version"]),
        "source_path": _normalize_path_value(resolve_registry_path_value(benchmark_bundle_source)),
        "task_count": int(benchmark_bundle["task_count"]),
        "task_ids": [int(task_id) for task_id in cast(list[Any], benchmark_bundle["task_ids"])],
    }
    tab_foundry_metrics = {
        "best_step": float(tab_foundry["best_step"]),
        "best_training_time": float(tab_foundry["best_training_time"]),
        "best_roc_auc": float(tab_foundry["best_roc_auc"]),
        "final_step": float(tab_foundry["final_step"]),
        "final_training_time": float(tab_foundry["final_training_time"]),
        "final_roc_auc": float(tab_foundry["final_roc_auc"]),
    }
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

    baseline_id = str(entry["baseline_id"])
    _validate_baseline_entry(entry, baseline_id=baseline_id)
    resolved_registry_path, payload = _ensure_registry_payload(registry_path)
    baselines = cast(dict[str, Any], payload["baselines"])
    baselines[baseline_id] = _copy_jsonable(entry)
    write_json(resolved_registry_path, payload)
    return resolved_registry_path


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
