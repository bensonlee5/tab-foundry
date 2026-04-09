"""Canonical programmatic surface for hardware architecture baseline freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import tab_foundry.benchmark_registry as read_benchmark_registry
import tab_foundry.hardware_architecture_registry as read_hardware_architecture_registry
from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.registry.summary_metrics import (
    ensure_mapping,
    ensure_non_empty_string,
    ensure_optional_positive_int,
)
from tab_foundry.hardware_profiles import build_hardware_profile_id
from tab_foundry.registry.common import copy_jsonable as _copy_jsonable


DEFAULT_SELECTION_RULE = "best_loss_healthy_only"


def _canonical_registry_path() -> Path:
    return (
        read_hardware_architecture_registry.default_hardware_architecture_registry_path()
        .expanduser()
        .resolve()
    )


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": read_hardware_architecture_registry.REGISTRY_SCHEMA,
        "version": read_hardware_architecture_registry.REGISTRY_VERSION,
        "baselines": {},
    }


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    registry_path = (
        path or read_hardware_architecture_registry.default_hardware_architecture_registry_path()
    ).expanduser().resolve()
    if not registry_path.exists():
        return registry_path, _empty_registry()
    payload = read_hardware_architecture_registry.load_hardware_architecture_registry(registry_path)
    return registry_path, payload


def _resolve_registry_run(run_id: str, *, registry_path: Path | None = None) -> dict[str, Any]:
    return read_benchmark_registry.load_benchmark_run_entry(run_id, path=registry_path)


def _runtime_summary_excerpt(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {
        "peak_vram_allocated": payload.get("peak_vram_allocated"),
        "peak_vram_reserved": payload.get("peak_vram_reserved"),
        "throughput_examples_per_second": payload.get("throughput_examples_per_second"),
        "throughput_tokens_per_second": payload.get("throughput_tokens_per_second"),
        "non_train_overhead_seconds": payload.get("non_train_overhead_seconds"),
    }


def _preferred_architecture_payload(run_entry: Mapping[str, Any]) -> dict[str, Any]:
    model = ensure_mapping(run_entry.get("model"), context="run_entry.model")
    payload = {
        "arch": ensure_non_empty_string(model.get("arch"), context="run_entry.model.arch"),
        "d_icl": int(model["d_icl"]),
        "head_hidden_dim": int(model["head_hidden_dim"]),
        "tficl_n_heads": int(model["tficl_n_heads"]),
        "tficl_n_layers": int(model["tficl_n_layers"]),
        "architecture": (
            dict(cast(Mapping[str, Any], model.get("architecture")))
            if isinstance(model.get("architecture"), Mapping)
            else None
        ),
        "build_spec": (
            dict(cast(Mapping[str, Any], model.get("build_spec")))
            if isinstance(model.get("build_spec"), Mapping)
            else None
        ),
    }
    return payload


def _hardware_identity(run_entry: Mapping[str, Any]) -> tuple[str, str, int]:
    hardware_summary = ensure_mapping(run_entry.get("hardware_summary"), context="run_entry.hardware_summary")
    gpu_class = ensure_non_empty_string(
        hardware_summary.get("gpu_class"),
        context="run_entry.hardware_summary.gpu_class",
    )
    vram_class_gb = ensure_optional_positive_int(
        hardware_summary.get("vram_class_gb"),
        context="run_entry.hardware_summary.vram_class_gb",
    )
    if vram_class_gb is None:
        raise RuntimeError("run_entry.hardware_summary.vram_class_gb must be present")
    hardware_profile_id = (
        ensure_non_empty_string(
            hardware_summary.get("hardware_profile_id"),
            context="run_entry.hardware_summary.hardware_profile_id",
        )
        if hardware_summary.get("hardware_profile_id") is not None
        else build_hardware_profile_id(gpu_class=gpu_class, vram_class_gb=vram_class_gb)
    )
    if hardware_profile_id is None:
        raise RuntimeError("failed to derive hardware_profile_id")
    return hardware_profile_id, gpu_class, int(vram_class_gb)


def derive_hardware_architecture_baseline_entry(
    *,
    baseline_id: str,
    preferred_run_id: str,
    formal_anchor_run_id: str,
    baseline_run_id: str,
    evidence_run_ids: Sequence[str],
    rationale: str,
    decision: str,
    surface_role: str,
    runtime_profile: str | None = None,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    benchmark_registry_path: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Derive one hardware architecture baseline entry from benchmark-backed runs."""

    preferred_entry = _resolve_registry_run(
        preferred_run_id,
        registry_path=benchmark_registry_path,
    )
    anchor_entry = _resolve_registry_run(
        formal_anchor_run_id,
        registry_path=benchmark_registry_path,
    )
    baseline_entry = _resolve_registry_run(
        baseline_run_id,
        registry_path=benchmark_registry_path,
    )
    evidence_entries = [
        _resolve_registry_run(run_id, registry_path=benchmark_registry_path)
        for run_id in evidence_run_ids
    ]

    hardware_profile_id, gpu_class, vram_class_gb = _hardware_identity(preferred_entry)
    for run_id, entry in (
        [(formal_anchor_run_id, anchor_entry), (baseline_run_id, baseline_entry)]
        + list(zip(evidence_run_ids, evidence_entries, strict=False))
    ):
        other_profile_id, other_gpu_class, other_vram_class_gb = _hardware_identity(entry)
        if (
            other_profile_id != hardware_profile_id
            or other_gpu_class != gpu_class
            or other_vram_class_gb != vram_class_gb
        ):
            raise RuntimeError(
                "hardware architecture baseline evidence must share one hardware profile: "
                f"expected={hardware_profile_id}, run_id={run_id}, actual={other_profile_id}"
            )

    track = ensure_non_empty_string(preferred_entry.get("track"), context="preferred_entry.track")
    config_profile = ensure_non_empty_string(
        preferred_entry.get("config_profile"),
        context="preferred_entry.config_profile",
    )
    resolved_runtime_profile = ensure_non_empty_string(
        runtime_profile if runtime_profile is not None else config_profile,
        context="runtime_profile",
    )
    regime_budget = ensure_mapping(
        preferred_entry.get("regime_budget"),
        context="preferred_entry.regime_budget",
    )
    objective_metric = ensure_non_empty_string(
        regime_budget.get("objective_metric"),
        context="preferred_entry.regime_budget.objective_metric",
    )
    surface_labels = (
        dict(cast(Mapping[str, Any], preferred_entry.get("surface_labels")))
        if isinstance(preferred_entry.get("surface_labels"), Mapping)
        else None
    )
    sweep = ensure_mapping(preferred_entry.get("sweep"), context="preferred_entry.sweep")
    preferred_delta_ref = sweep.get("delta_id")
    benchmark_manifest_path = ensure_non_empty_string(
        preferred_entry.get("manifest_path"),
        context="preferred_entry.manifest_path",
    )
    lineage = ensure_mapping(preferred_entry.get("lineage"), context="preferred_entry.lineage")
    control_baseline_id = lineage.get("control_baseline_id")
    sweep_id = sweep.get("sweep_id")

    entry = {
        "baseline_id": ensure_non_empty_string(baseline_id, context="baseline_id"),
        "hardware_profile_id": hardware_profile_id,
        "gpu_class": gpu_class,
        "vram_class_gb": int(vram_class_gb),
        "track": track,
        "surface_role": ensure_non_empty_string(surface_role, context="surface_role"),
        "runtime_profile": resolved_runtime_profile,
        "config_profile": config_profile,
        "benchmark_manifest_path": benchmark_manifest_path,
        "control_baseline_id": None
        if control_baseline_id is None
        else ensure_non_empty_string(control_baseline_id, context="control_baseline_id"),
        "sweep_id": None if sweep_id is None else ensure_non_empty_string(sweep_id, context="sweep_id"),
        "surface_labels": surface_labels,
        "formal_anchor_run_id": ensure_non_empty_string(
            formal_anchor_run_id,
            context="formal_anchor_run_id",
        ),
        "baseline_run_id": ensure_non_empty_string(baseline_run_id, context="baseline_run_id"),
        "preferred_run_id": ensure_non_empty_string(preferred_run_id, context="preferred_run_id"),
        "preferred_delta_ref": None
        if preferred_delta_ref is None
        else ensure_non_empty_string(preferred_delta_ref, context="preferred_delta_ref"),
        "preferred_architecture": _preferred_architecture_payload(preferred_entry),
        "objective_metric": objective_metric,
        "selection_rule": ensure_non_empty_string(selection_rule, context="selection_rule"),
        "evidence_run_ids": [
            ensure_non_empty_string(run_id, context="evidence_run_ids")
            for run_id in evidence_run_ids
        ],
        "decision": ensure_non_empty_string(decision, context="decision"),
        "rationale": ensure_non_empty_string(rationale, context="rationale"),
        "preferred_runtime_summary": _runtime_summary_excerpt(
            cast(Mapping[str, Any] | None, preferred_entry.get("runtime_summary"))
        ),
    }
    _ = read_hardware_architecture_registry._validate_baseline_entry(
        entry,
        baseline_id=str(entry["baseline_id"]),
    )
    return entry


def upsert_hardware_architecture_baseline_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> Path:
    """Insert or replace one hardware architecture baseline entry in the registry."""

    baseline_id = str(entry["baseline_id"])
    _ = read_hardware_architecture_registry._validate_baseline_entry(entry, baseline_id=baseline_id)
    resolved_registry_path, payload = _ensure_registry_payload(registry_path)
    baselines = cast(dict[str, Any], payload["baselines"])
    baselines[baseline_id] = _copy_jsonable(entry)
    write_json(resolved_registry_path, payload)
    return resolved_registry_path


def freeze_hardware_architecture_baseline(
    *,
    baseline_id: str,
    preferred_run_id: str,
    formal_anchor_run_id: str,
    baseline_run_id: str,
    evidence_run_ids: Sequence[str],
    rationale: str,
    decision: str,
    surface_role: str,
    runtime_profile: str | None = None,
    selection_rule: str = DEFAULT_SELECTION_RULE,
    benchmark_registry_path: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Promote benchmark-backed evidence into the hardware architecture registry."""

    entry = derive_hardware_architecture_baseline_entry(
        baseline_id=baseline_id,
        preferred_run_id=preferred_run_id,
        formal_anchor_run_id=formal_anchor_run_id,
        baseline_run_id=baseline_run_id,
        evidence_run_ids=evidence_run_ids,
        rationale=rationale,
        decision=decision,
        surface_role=surface_role,
        runtime_profile=runtime_profile,
        selection_rule=selection_rule,
        benchmark_registry_path=benchmark_registry_path,
        registry_path=registry_path,
    )
    requested_registry_path = (
        read_hardware_architecture_registry.default_hardware_architecture_registry_path()
        if registry_path is None
        else registry_path
    )
    resolved_registry_path = requested_registry_path.expanduser().resolve()
    resolved_registry_path = upsert_hardware_architecture_baseline_entry(
        entry,
        registry_path=resolved_registry_path,
    )
    return {
        "registry_path": str(resolved_registry_path),
        "baseline": entry,
    }
