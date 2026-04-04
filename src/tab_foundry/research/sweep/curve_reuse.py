"""Reusable nanoTabPFN curve resolution for sweep execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from tab_realdata_hub.manifest import manifest_sha256
from tab_foundry.benchmark_registry import resolve_registry_path_value
import tab_foundry.control_baseline_registry as control_baseline_registry
from tab_foundry.bench.comparison_contract import (
    DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
)
from tab_foundry.bench.openml_benchmark import benchmark_host_fingerprint

from .artifacts import ExecutionPaths
from . import device_policy as device_policy_module
from .runtime_env import planned_nanotabpfn_python_path


@dataclass(frozen=True)
class NanoTabPFNCurveCandidate:
    source_label: str
    comparison_summary_path: Path
    declared_control_baseline_id: str | None = None


@dataclass(frozen=True)
class NanoTabPFNCurveReuseSelection:
    curve_path: Path | None
    source_label: str
    metadata: dict[str, Any]
    signature: dict[str, Any]
    reusable_error: dict[str, Any] | None = None


@dataclass(frozen=True)
class NanoTabPFNCandidatePayload:
    benchmark_manifest_path: str
    benchmark_manifest_sha256: str
    control_baseline_id: str
    signature: dict[str, Any] | None
    metadata: dict[str, Any] | None
    curve_path: Path | None
    reusable_error: dict[str, Any] | None


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def _candidate_curve_path(summary: Mapping[str, Any], *, summary_path: Path) -> Path | None:
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, Mapping):
        curve_value = artifacts.get("nanotabpfn_curve_jsonl")
        if isinstance(curve_value, str) and curve_value.strip():
            candidate = resolve_registry_path_value(curve_value)
            if candidate.exists():
                return candidate
    fallback = summary_path.parent / "nanotabpfn_curve.jsonl"
    return fallback if fallback.exists() else None


def _float_matches(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-12


def _normalized_manifest_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _resolved_nanotabpfn_signature(
    *,
    benchmark_manifest_path: Path,
    control_baseline_id: str,
    nanotabpfn_root: Path,
    prior_dump: Path | None,
    requested_device: str,
) -> dict[str, Any]:
    normalized_requested_device, resolved_device = device_policy_module.resolve_sweep_metadata_device(
        requested_device
    )
    return {
        "benchmark_manifest_path": _normalized_manifest_path(benchmark_manifest_path),
        "benchmark_manifest_sha256": manifest_sha256(benchmark_manifest_path),
        "control_baseline_id": str(control_baseline_id).strip(),
        "nanotabpfn_root": nanotabpfn_root.expanduser().resolve(),
        "nanotabpfn_python": planned_nanotabpfn_python_path(nanotabpfn_root),
        "prior_dump_path": None if prior_dump is None else prior_dump.expanduser().resolve(),
        "device": normalized_requested_device,
        "resolved_device": resolved_device,
        "benchmark_host_fingerprint": benchmark_host_fingerprint(),
        "steps": int(DEFAULT_NANOTABPFN_STEPS),
        "eval_every": int(DEFAULT_NANOTABPFN_EVAL_EVERY),
        "seeds": int(DEFAULT_NANOTABPFN_SEEDS),
        "batch_size": int(DEFAULT_NANOTABPFN_BATCH_SIZE),
        "lr": float(DEFAULT_NANOTABPFN_LR),
    }


def _signature_metadata(signature: Mapping[str, Any]) -> dict[str, Any]:
    nanotabpfn_root = signature.get("nanotabpfn_root")
    nanotabpfn_python = signature.get("nanotabpfn_python")
    prior_dump_path = signature.get("prior_dump_path")
    return {
        "benchmark_manifest_path": str(signature["benchmark_manifest_path"]),
        "benchmark_manifest_sha256": str(signature["benchmark_manifest_sha256"]),
        "root": None if nanotabpfn_root is None else str(cast(Path, nanotabpfn_root)),
        "python": None if nanotabpfn_python is None else str(cast(Path, nanotabpfn_python)),
        "device": str(signature["device"]),
        "resolved_device": str(signature["resolved_device"]),
        "benchmark_host_fingerprint": str(signature["benchmark_host_fingerprint"]),
        "prior_dump_path": None if prior_dump_path is None else str(cast(Path, prior_dump_path)),
        "num_seeds": int(signature["seeds"]),
        "steps": int(signature["steps"]),
        "eval_every": int(signature["eval_every"]),
        "batch_size": int(signature["batch_size"]),
        "lr": float(signature["lr"]),
    }


def _candidate_control_baseline_id(
    *,
    summary: Mapping[str, Any],
    candidate: NanoTabPFNCurveCandidate,
) -> str | None:
    control_baseline_id = candidate.declared_control_baseline_id
    if control_baseline_id is not None:
        return str(control_baseline_id).strip()
    baseline = summary.get("control_baseline")
    if not isinstance(baseline, Mapping):
        return None
    baseline_id = baseline.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        return None
    return str(baseline_id).strip()


def _candidate_signature(
    *,
    candidate: NanoTabPFNCurveCandidate,
) -> NanoTabPFNCandidatePayload | None:
    if not candidate.comparison_summary_path.exists():
        return None
    try:
        summary = _read_json_mapping(candidate.comparison_summary_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None

    benchmark_manifest = summary.get("benchmark_manifest")
    nanotabpfn = summary.get("nanotabpfn")
    nanotabpfn_error = summary.get("nanotabpfn_error")
    if not isinstance(benchmark_manifest, Mapping):
        return None

    manifest_path_value = benchmark_manifest.get("path")
    if not isinstance(manifest_path_value, str) or not manifest_path_value.strip():
        return None
    manifest_path = resolve_registry_path_value(manifest_path_value)
    manifest_digest_value = benchmark_manifest.get("sha256")
    if isinstance(manifest_digest_value, str) and manifest_digest_value.strip():
        manifest_digest = str(manifest_digest_value).strip()
    elif manifest_path.exists():
        manifest_digest = manifest_sha256(manifest_path)
    else:
        return None

    control_baseline_id = _candidate_control_baseline_id(summary=summary, candidate=candidate)
    if control_baseline_id is None:
        return None

    curve_path = None
    signature = None
    metadata = None
    if isinstance(nanotabpfn, Mapping):
        curve_path = _candidate_curve_path(summary, summary_path=candidate.comparison_summary_path)
        requested_device = nanotabpfn.get("device")
        resolved_device = nanotabpfn.get("resolved_device")
        host_fingerprint = nanotabpfn.get("benchmark_host_fingerprint")
        if (
            isinstance(requested_device, str)
            and requested_device.strip()
            and isinstance(resolved_device, str)
            and resolved_device.strip()
            and isinstance(host_fingerprint, str)
            and host_fingerprint.strip()
        ):
            root_value = nanotabpfn.get("root")
            root = (
                Path(str(root_value)).expanduser().resolve()
                if isinstance(root_value, str) and root_value.strip()
                else None
            )
            python_value = nanotabpfn.get("python")
            nanotabpfn_python = (
                Path(str(python_value)).expanduser().resolve()
                if isinstance(python_value, str) and python_value.strip()
                else None
            )
            raw_prior_dump_path = nanotabpfn.get("prior_dump_path")
            prior_dump_path = (
                Path(str(raw_prior_dump_path)).expanduser().resolve()
                if isinstance(raw_prior_dump_path, str) and raw_prior_dump_path.strip()
                else None
            )
            signature = {
                "benchmark_manifest_path": _normalized_manifest_path(manifest_path),
                "benchmark_manifest_sha256": manifest_digest,
                "control_baseline_id": control_baseline_id,
                "nanotabpfn_root": root,
                "nanotabpfn_python": nanotabpfn_python,
                "prior_dump_path": prior_dump_path,
                "device": str(requested_device).strip(),
                "resolved_device": str(resolved_device).strip().lower(),
                "benchmark_host_fingerprint": str(host_fingerprint).strip(),
                "steps": int(nanotabpfn.get("steps", DEFAULT_NANOTABPFN_STEPS)),
                "eval_every": int(nanotabpfn.get("eval_every", DEFAULT_NANOTABPFN_EVAL_EVERY)),
                "seeds": int(
                    nanotabpfn.get("num_seeds", nanotabpfn.get("seeds", DEFAULT_NANOTABPFN_SEEDS))
                ),
                "batch_size": int(nanotabpfn.get("batch_size", DEFAULT_NANOTABPFN_BATCH_SIZE)),
                "lr": float(nanotabpfn.get("lr", DEFAULT_NANOTABPFN_LR)),
            }
            metadata = _signature_metadata(signature)

    reusable_error = (
        dict(cast(Mapping[str, Any], nanotabpfn_error))
        if isinstance(nanotabpfn_error, Mapping)
        else None
    )
    if curve_path is None and reusable_error is None:
        return None
    return NanoTabPFNCandidatePayload(
        benchmark_manifest_path=_normalized_manifest_path(manifest_path),
        benchmark_manifest_sha256=manifest_digest,
        control_baseline_id=control_baseline_id,
        signature=signature,
        metadata=metadata,
        curve_path=curve_path,
        reusable_error=reusable_error,
    )


def _signatures_match(
    *,
    current_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
) -> bool:
    comparable_keys = (
        "benchmark_manifest_path",
        "benchmark_manifest_sha256",
        "control_baseline_id",
        "benchmark_host_fingerprint",
        "steps",
        "eval_every",
        "seeds",
        "batch_size",
    )
    for key in comparable_keys:
        if candidate_signature[key] != current_signature[key]:
            return False
    for key in ("nanotabpfn_root", "nanotabpfn_python"):
        candidate_value = candidate_signature[key]
        current_value = current_signature[key]
        if candidate_value is not None and candidate_value != current_value:
            return False
    candidate_prior_dump = candidate_signature["prior_dump_path"]
    current_prior_dump = current_signature["prior_dump_path"]
    if current_prior_dump is not None and candidate_prior_dump is not None:
        if candidate_prior_dump != current_prior_dump:
            return False
    if not _float_matches(float(candidate_signature["lr"]), float(current_signature["lr"])):
        return False
    return True


def _allows_legacy_error_reuse(source_label: str) -> bool:
    normalized = str(source_label).strip().lower()
    return normalized == "parent row" or normalized.startswith("sweep row ")


def _matching_nanotabpfn_curve(
    *,
    current_signature: Mapping[str, Any],
    candidate: NanoTabPFNCurveCandidate,
) -> NanoTabPFNCurveReuseSelection | None:
    payload = _candidate_signature(candidate=candidate)
    if payload is None:
        return None
    if payload.signature is not None and _signatures_match(
        current_signature=current_signature,
        candidate_signature=payload.signature,
    ):
        if payload.curve_path is not None and payload.metadata is not None:
            return NanoTabPFNCurveReuseSelection(
                curve_path=payload.curve_path,
                source_label=candidate.source_label,
                metadata=payload.metadata,
                signature=payload.signature,
            )
        if payload.reusable_error is not None:
            return NanoTabPFNCurveReuseSelection(
                curve_path=None,
                source_label=candidate.source_label,
                metadata=(
                    payload.metadata
                    if payload.metadata is not None
                    else _signature_metadata(current_signature)
                ),
                signature=payload.signature,
                reusable_error=payload.reusable_error,
            )
        return None

    if payload.reusable_error is None or not _allows_legacy_error_reuse(candidate.source_label):
        return None
    if payload.benchmark_manifest_path != current_signature["benchmark_manifest_path"]:
        return None
    if payload.benchmark_manifest_sha256 != current_signature["benchmark_manifest_sha256"]:
        return None
    if payload.control_baseline_id != current_signature["control_baseline_id"]:
        return None
    return NanoTabPFNCurveReuseSelection(
        curve_path=None,
        source_label=candidate.source_label,
        metadata=_signature_metadata(current_signature),
        signature=dict(current_signature),
        reusable_error=payload.reusable_error,
    )


def _registry_curve_candidate(
    *,
    run_id: str | None,
    source_label: str,
    registry_path: Path,
) -> NanoTabPFNCurveCandidate | None:
    if run_id is None:
        return None
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        return None
    try:
        payload = _read_json_mapping(registry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        return None
    run = runs.get(normalized_run_id)
    if not isinstance(run, Mapping):
        return None
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    summary_value = artifacts.get("comparison_summary_path")
    if not isinstance(summary_value, str) or not summary_value.strip():
        return None
    return NanoTabPFNCurveCandidate(
        source_label=source_label,
        comparison_summary_path=resolve_registry_path_value(summary_value),
    )


def _anchor_curve_candidate(
    *,
    anchor_run_id: str | None,
    registry_path: Path,
) -> NanoTabPFNCurveCandidate | None:
    return _registry_curve_candidate(
        run_id=anchor_run_id,
        source_label="anchor",
        registry_path=registry_path,
    )


def _control_baseline_curve_candidate(
    *,
    control_baseline_id: str,
    control_baseline_registry_path: Path,
) -> NanoTabPFNCurveCandidate | None:
    try:
        entry = control_baseline_registry.load_control_baseline_entry(
            control_baseline_id,
            registry_path=control_baseline_registry_path,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    summary_value = entry.get("comparison_summary_path")
    if not isinstance(summary_value, str) or not summary_value.strip():
        return None
    return NanoTabPFNCurveCandidate(
        source_label="control baseline",
        comparison_summary_path=control_baseline_registry.resolve_registry_path_value(summary_value),
        declared_control_baseline_id=control_baseline_id,
    )


def resolve_reusable_nanotabpfn_curve(
    *,
    sweep_meta: Mapping[str, Any],
    anchor_run_id: str | None,
    nanotabpfn_root: Path,
    prior_dump: Path | None,
    requested_device: str,
    paths: ExecutionPaths,
    extra_candidates: Sequence[NanoTabPFNCurveCandidate] | None = None,
) -> NanoTabPFNCurveReuseSelection | None:
    control_baseline_id = str(sweep_meta["control_baseline_id"]).strip()
    current_signature = _resolved_nanotabpfn_signature(
        benchmark_manifest_path=resolve_registry_path_value(str(sweep_meta["benchmark_manifest_path"])),
        control_baseline_id=control_baseline_id,
        nanotabpfn_root=nanotabpfn_root,
        prior_dump=prior_dump,
        requested_device=requested_device,
    )
    candidates = [
        *(list(extra_candidates) if extra_candidates is not None else []),
        _anchor_curve_candidate(anchor_run_id=anchor_run_id, registry_path=paths.registry_path),
        _control_baseline_curve_candidate(
            control_baseline_id=control_baseline_id,
            control_baseline_registry_path=paths.control_baseline_registry_path,
        ),
    ]
    error_selection = None
    for candidate in candidates:
        if candidate is None:
            continue
        selection = _matching_nanotabpfn_curve(
            current_signature=current_signature,
            candidate=candidate,
        )
        if selection is not None:
            if selection.curve_path is not None:
                return selection
            if selection.reusable_error is not None and error_selection is None:
                error_selection = selection
    return error_selection


def prior_completed_row_curve_candidates(
    *,
    queue: Mapping[str, Any],
    current_order: int,
    anchor_run_id: str | None,
    parent_run_id: str | None,
    registry_path: Path,
) -> list[NanoTabPFNCurveCandidate]:
    queue_rows_raw = queue.get("rows")
    if not isinstance(queue_rows_raw, list):
        return []

    normalized_anchor_run_id = None if anchor_run_id is None else str(anchor_run_id).strip()
    candidates: list[NanoTabPFNCurveCandidate] = []
    seen_run_ids: set[str] = set()

    def _append_candidate(run_id: str | None, source_label: str) -> None:
        if run_id is None:
            return
        normalized_run_id = str(run_id).strip()
        if (
            not normalized_run_id
            or normalized_run_id in seen_run_ids
            or normalized_run_id == normalized_anchor_run_id
        ):
            return
        candidate = _registry_curve_candidate(
            run_id=normalized_run_id,
            source_label=source_label,
            registry_path=registry_path,
        )
        if candidate is None:
            return
        candidates.append(candidate)
        seen_run_ids.add(normalized_run_id)

    _append_candidate(parent_run_id, "parent row")

    earlier_rows = sorted(
        (
            row
            for row in queue_rows_raw
            if isinstance(row, Mapping) and int(row.get("order", 0)) < int(current_order)
        ),
        key=lambda row: int(row["order"]),
        reverse=True,
    )
    for row in earlier_rows:
        row_run_id = row.get("run_id")
        if not isinstance(row_run_id, str) or not row_run_id.strip():
            continue
        row_order = int(row["order"])
        row_delta_ref = str(row.get("delta_ref", "")).strip() or f"order {row_order:02d}"
        _append_candidate(str(row_run_id), f"sweep row {row_order:02d} ({row_delta_ref})")

    return candidates
