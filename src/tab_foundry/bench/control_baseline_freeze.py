"""Canonical programmatic surface for control-baseline freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import torch

import tab_foundry.control_baseline_registry as read_control_baseline_registry
from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.openml_benchmark import collect_checkpoint_snapshots, resolve_tab_foundry_best_checkpoint
from tab_foundry.registry.common import (
    copy_jsonable as _copy_jsonable,
    load_comparison_summary as _load_comparison_summary,
    resolve_config_path as _resolve_config_path_common,
)
from tab_foundry.registry.storage import load_versioned_registry_payload as _load_versioned_registry_payload
from tab_foundry.bench.registry.summary_metrics import (
    tab_foundry_metrics_from_summary as _tab_foundry_metrics_from_summary,
)
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.repo_paths import repo_root


REGISTRY_SCHEMA = read_control_baseline_registry.REGISTRY_SCHEMA
REGISTRY_VERSION = read_control_baseline_registry.REGISTRY_VERSION
DEFAULT_BASELINE_ID = "cls_benchmark_linear_v2"
DEFAULT_EXPERIMENT = "cls_benchmark_staged_prior"
DEFAULT_CONFIG_PROFILE = DEFAULT_EXPERIMENT
DEFAULT_BUDGET_CLASS = "short-run"


def _canonical_registry_path() -> Path:
    return read_control_baseline_registry.default_control_baseline_registry_path().expanduser().resolve()


def _ensure_repo_local_path_value(path_value: str, *, field_name: str) -> None:
    resolved_path = read_control_baseline_registry.resolve_registry_path_value(
        path_value,
        root=repo_root(),
    )
    resolved_root = repo_root().expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            "canonical control baseline registry requires repo-local artifact paths: "
            f"field={field_name}, value={path_value!r}, resolved={resolved_path}"
        ) from exc


def _is_repo_local_path_value(path_value: str) -> bool:
    resolved_path = read_control_baseline_registry.resolve_registry_path_value(
        path_value,
        root=repo_root(),
    )
    resolved_root = repo_root().expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def _enforce_canonical_registry_entry_paths(entry: Mapping[str, Any]) -> None:
    _ensure_repo_local_path_value(str(entry["manifest_path"]), field_name="manifest_path")
    _ensure_repo_local_path_value(str(entry["run_dir"]), field_name="run_dir")
    _ensure_repo_local_path_value(
        str(entry["comparison_summary_path"]),
        field_name="comparison_summary_path",
    )
    benchmark_bundle = cast(dict[str, Any], entry["benchmark_bundle"])
    _ensure_repo_local_path_value(
        str(benchmark_bundle["source_path"]),
        field_name="benchmark_bundle.source_path",
    )


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
        top_level_keys=read_control_baseline_registry._TOP_LEVEL_KEYS,
        schema=REGISTRY_SCHEMA,
        version=REGISTRY_VERSION,
        entries_key="baselines",
        registry_label="control baseline registry",
        validate_entry_fn=read_control_baseline_registry._validate_baseline_entry,
        entry_label="baseline_id",
    )


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    registry_path = (
        path or read_control_baseline_registry.default_control_baseline_registry_path()
    ).expanduser().resolve()
    payload = _load_registry_payload(registry_path, allow_missing=True)
    return registry_path, payload


def _normalize_registry_path(path: Path) -> str:
    return read_control_baseline_registry.normalize_registry_path_value(
        path.expanduser().resolve(),
        root=repo_root(),
    )


def _benchmark_bundle_payload(benchmark_bundle: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_bundle_source = str(
        benchmark_bundle.get("source_path")
        if benchmark_bundle.get("source_path") is not None
        else ""
    ).strip()
    if not benchmark_bundle_source:
        raise RuntimeError("comparison summary benchmark_bundle.source_path must be a non-empty string")
    resolved_source_path = read_control_baseline_registry.resolve_registry_path_value(
        benchmark_bundle_source,
        root=repo_root(),
    )
    return {
        "name": str(benchmark_bundle["name"]),
        "version": int(benchmark_bundle["version"]),
        "source_path": _normalize_registry_path(resolved_source_path),
        "task_count": int(benchmark_bundle["task_count"]),
        "task_ids": [int(task_id) for task_id in cast(list[Any], benchmark_bundle["task_ids"])],
    }


def _resolve_config_path(raw_value: Any) -> Path:
    return _resolve_config_path_common(raw_value, root=repo_root())


def _resolve_baseline_checkpoint(run_dir: Path, *, summary_tab_foundry: Mapping[str, Any]) -> Path:
    try:
        return resolve_tab_foundry_best_checkpoint(run_dir)
    except RuntimeError as exc:
        best_step_raw = summary_tab_foundry.get("best_step")
        if not isinstance(best_step_raw, (int, float)) or isinstance(best_step_raw, bool):
            raise
        best_step = int(best_step_raw)
        for snapshot in collect_checkpoint_snapshots(run_dir):
            if int(snapshot["step"]) == best_step:
                return Path(str(snapshot["path"])).expanduser().resolve()
        raise RuntimeError(
            "missing best checkpoint under "
            f"{run_dir.expanduser().resolve()}; no checkpoint snapshot matched summary best_step={best_step}"
        ) from exc


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

    best_checkpoint = _resolve_baseline_checkpoint(
        resolved_run_dir,
        summary_tab_foundry=tab_foundry,
    )
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
    direct_manifest = data_cfg.get("manifest_path")
    manifest_path = (
        None
        if direct_manifest is None
        else _resolve_config_path(direct_manifest)
    )
    if manifest_path is None:
        resolved_data_surface = resolve_data_surface(data_cfg)
        manifest_path = resolved_data_surface.manifest_path
    if manifest_path is None:
        raise RuntimeError(
            "checkpoint config must resolve a manifest-backed training surface via "
            "data.manifest_path or data.corpus_ref"
        )
    seed_raw = runtime_cfg.get("seed")
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        raise RuntimeError(f"checkpoint runtime.seed must be an int: {best_checkpoint}")

    benchmark_bundle = cast(dict[str, Any], summary["benchmark_bundle"])
    benchmark_bundle_for_registry = dict(benchmark_bundle)
    benchmark_bundle_source = benchmark_bundle_for_registry.get("source_path")
    summary_artifacts = summary.get("artifacts")
    if (
        isinstance(benchmark_bundle_source, str)
        and benchmark_bundle_source.strip()
        and not _is_repo_local_path_value(benchmark_bundle_source)
        and isinstance(summary_artifacts, Mapping)
    ):
        benchmark_manifest_path = summary_artifacts.get("benchmark_manifest")
        if isinstance(benchmark_manifest_path, str) and benchmark_manifest_path.strip():
            benchmark_manifest_value = read_control_baseline_registry.normalize_registry_path_value(
                Path(benchmark_manifest_path),
                root=repo_root(),
            )
            if _is_repo_local_path_value(benchmark_manifest_value):
                benchmark_bundle_for_registry["source_path"] = benchmark_manifest_value
    benchmark_bundle_payload = _benchmark_bundle_payload(benchmark_bundle_for_registry)
    tab_foundry_metrics = _tab_foundry_metrics_from_summary(tab_foundry)
    entry = {
        "baseline_id": str(baseline_id),
        "experiment": str(experiment),
        "config_profile": str(config_profile),
        "budget_class": str(budget_class),
        "manifest_path": read_control_baseline_registry.normalize_registry_path_value(
            manifest_path,
            root=repo_root(),
        ),
        "seed_set": [int(seed_raw)],
        "run_dir": read_control_baseline_registry.normalize_registry_path_value(
            resolved_run_dir,
            root=repo_root(),
        ),
        "comparison_summary_path": read_control_baseline_registry.normalize_registry_path_value(
            resolved_summary_path,
            root=repo_root(),
        ),
        "benchmark_bundle": benchmark_bundle_payload,
        "tab_foundry_metrics": tab_foundry_metrics,
    }
    _ = read_control_baseline_registry._validate_baseline_entry(entry, baseline_id=str(baseline_id))
    return entry


def upsert_control_baseline_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> Path:
    """Insert or replace one control baseline entry in the registry."""

    baseline_id = str(entry["baseline_id"])
    _ = read_control_baseline_registry._validate_baseline_entry(entry, baseline_id=baseline_id)
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
    requested_registry_path = (
        read_control_baseline_registry.default_control_baseline_registry_path()
        if registry_path is None
        else registry_path
    )
    resolved_registry_path = requested_registry_path.expanduser().resolve()
    if resolved_registry_path == _canonical_registry_path():
        _enforce_canonical_registry_entry_paths(entry)
    resolved_registry_path = upsert_control_baseline_entry(entry, registry_path=resolved_registry_path)
    return {
        "registry_path": str(resolved_registry_path),
        "baseline": entry,
    }
