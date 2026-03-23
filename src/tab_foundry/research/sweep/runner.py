"""Execution orchestration helpers for system-delta sweeps."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.bench.benchmark_run_registry import (
    register_benchmark_run,
    resolve_registry_path_value,
)
from tab_foundry.bench.compare import (
    DEFAULT_NANOTABPFN_BATCH_SIZE,
    DEFAULT_NANOTABPFN_EVAL_EVERY,
    DEFAULT_NANOTABPFN_LR,
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
    EXTERNAL_BENCHMARK_NANOTABPFN,
    NanoTabPFNBenchmarkConfig,
    run_nanotabpfn_benchmark,
)
from tab_foundry.bench.nanotabpfn import benchmark_host_fingerprint, resolve_device
from tab_foundry.bench.nanotabpfn.bundle import canonical_benchmark_bundle_source_path
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.config import compose_config
from tab_foundry.research.lane_contract import (
    resolve_surface_role,
    resolve_training_config_profile,
    resolve_training_experiment,
)
from tab_foundry.research import system_delta
from tab_foundry.research.system_delta_promote import promote_anchor
from tab_foundry.training.artifacts import resolve_latest_checkpoint_path
from tab_foundry.training.surface import (
    TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_PRIOR_DUMP,
    resolve_training_backend_from_data_cfg,
)
from tab_foundry.training.trainer import train as train_from_manifest_cfg
from tab_foundry.training.wandb import posthoc_update_wandb_summary

from .artifacts import ExecutionPaths, read_yaml, result_card_text, write_research_package, write_yaml
from .queue_updates import append_note, optional_metric, queue_metrics, update_queue_row, update_screened_queue_row
from .screening import pick_screen_winner, screen_metrics
from .selection import select_queue_rows, sorted_rows
from .validation import resolve_sweep_external_benchmarks


DEFAULT_PRIOR_DUMP = Path("/workspace/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_NANOTABPFN_ROOT = Path("/workspace/nanoTabPFN")
DEFAULT_DEVICE = "cuda"
DEFAULT_TRACK = "system_delta_binary_medium_v1"
DEFAULT_EXPERIMENT = "cls_benchmark_staged_prior"
DEFAULT_CONFIG_PROFILE = "cls_benchmark_staged_prior"
DEFAULT_BUDGET_CLASS = "short-run"
DEFAULT_DECISION = "defer"
DEFAULT_CONCLUSION = (
    "Canonical benchmark comparison recorded against the locked sweep anchor; "
    "interpret this row in the full sweep context."
)
ALLOWED_DECISIONS = {"keep", "defer", "reject"}
SCREEN_ONLY_POLICY = "screen_only"
BENCHMARK_FULL_POLICY = "benchmark_full"
_ALLOWED_TRAINING_BACKENDS = {
    TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_PRIOR_DUMP,
}


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
    benchmark_bundle_path: str
    control_baseline_id: str
    signature: dict[str, Any] | None
    metadata: dict[str, Any] | None
    curve_path: Path | None
    reusable_error: dict[str, Any] | None


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        return None
    return cast(Mapping[str, Any], raw_value)


def _sweep_wandb_summary_payload(
    *,
    run_entry: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    sweep_payload = _mapping_value(run_entry, "sweep")
    if sweep_payload is not None:
        payload["sweep"] = dict(sweep_payload)

    comparison_payload: dict[str, Any] = {}
    comparisons = _mapping_value(run_entry, "comparisons")
    if comparisons is not None:
        for comparison_name in ("vs_anchor", "vs_parent"):
            comparison_values = _mapping_value(comparisons, comparison_name)
            if comparison_values:
                comparison_payload[comparison_name] = dict(comparison_values)

    stage_local_payload: dict[str, Any] = {}
    for stage_label, grad_key, activation_delta_key, activation_mean_key in (
        (
            "column",
            "column_encoder_final_window_mean_grad_norm",
            "column_activation_early_to_final_mean_delta",
            "column_activation_final_window_mean",
        ),
        (
            "row",
            "row_pool_final_window_mean_grad_norm",
            "row_activation_early_to_final_mean_delta",
            "row_activation_final_window_mean",
        ),
        (
            "context",
            "context_encoder_final_window_mean_grad_norm",
            "context_activation_early_to_final_mean_delta",
            "context_activation_final_window_mean",
        ),
    ):
        stage_payload: dict[str, Any] = {}
        for source_key, target_key in (
            (grad_key, "final_window_mean_grad_norm"),
            (activation_delta_key, "activation_early_to_final_mean_delta"),
            (activation_mean_key, "activation_final_window_mean"),
        ):
            if source_key in queue_metrics:
                stage_payload[target_key] = queue_metrics[source_key]
        if stage_payload:
            stage_local_payload[stage_label] = stage_payload
    if stage_local_payload:
        comparison_payload["stage_local_stability"] = stage_local_payload

    if comparison_payload:
        payload["comparison"] = comparison_payload
    return payload


def row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def _python_can_import_torch(python_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import torch"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _absolute_path_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def ensure_nanotabpfn_python(*, nanotabpfn_root: Path, fallback_python: Path) -> Path:
    nanotab_python = nanotabpfn_root / ".venv" / "bin" / "python"
    fallback_executable = _absolute_path_without_resolving_symlinks(fallback_python)
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    if nanotab_python.exists() and _python_can_import_torch(nanotab_python):
        return nanotab_python
    if nanotab_python.exists() or nanotab_python.is_symlink():
        nanotab_python.unlink()
    if not _python_can_import_torch(fallback_executable):
        raise RuntimeError(
            "fallback interpreter cannot import torch: "
            f"{fallback_executable}"
        )
    nanotab_python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(str(fallback_executable))} \"$@\"\n",
        encoding="utf-8",
    )
    nanotab_python.chmod(0o755)
    return nanotab_python


def materialized_row_map(*, sweep_id: str, paths: ExecutionPaths) -> dict[str, dict[str, Any]]:
    materialized = system_delta.load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )
    rows = cast(list[dict[str, Any]], materialized["rows"])
    return {str(row["delta_id"]): row for row in rows}


def apply_mapping(cfg: DictConfig, prefix: str, payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if prefix == "data" and key == "corpus_ref":
            OmegaConf.update(cfg, f"{prefix}.surface_overrides.{key}", value, merge=True)
            continue
        merge = not (
            prefix == "model"
            and key == "module_overrides"
            and isinstance(value, Mapping)
        )
        OmegaConf.update(cfg, f"{prefix}.{key}", value, merge=merge)


def _queue_aware_run_name(*, run_dir: Path) -> str:
    return str(run_dir.parent.name if run_dir.name == "train" else run_dir.name)


def compose_cfg(
    *,
    row: Mapping[str, Any],
    run_dir: Path,
    device: str,
    training_experiment: str = DEFAULT_EXPERIMENT,
    sweep_id: str | None = None,
) -> DictConfig:
    cfg = compose_config([f"experiment={training_experiment}"])
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.runtime.device = str(device)
    cfg.logging.run_name = _queue_aware_run_name(run_dir=run_dir)
    if sweep_id is not None:
        cfg.logging.group = str(sweep_id)
    apply_mapping(cfg, "model", cast(Mapping[str, Any], row.get("model", {})))
    apply_mapping(cfg, "data", cast(Mapping[str, Any], row.get("data", {})))
    apply_mapping(cfg, "preprocessing", cast(Mapping[str, Any], row.get("preprocessing", {})))

    training_payload = cast(Mapping[str, Any], row.get("training", {}))
    for key in (
        "surface_label",
        "prior_dump_non_finite_policy",
        "prior_dump_batch_size",
        "prior_dump_lr_scale_rule",
        "prior_dump_batch_reference_size",
    ):
        if key in training_payload:
            OmegaConf.update(cfg, f"training.{key}", training_payload[key], merge=True)

    overrides = cast(Mapping[str, Any], training_payload.get("overrides", {}))
    if "apply_schedule" in overrides:
        OmegaConf.update(cfg, "training.apply_schedule", overrides["apply_schedule"], merge=True)
    for key in ("optimizer", "runtime", "schedule"):
        override_payload = overrides.get(key)
        if isinstance(override_payload, dict):
            apply_mapping(cfg, key, cast(Mapping[str, Any], override_payload))
    return cfg


def sync_sweep_matrix(*, sweep_id: str, paths: ExecutionPaths) -> None:
    _ = system_delta.render_and_write_system_delta_matrix(
        sweep_id=sweep_id,
        registry_path=paths.registry_path,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        sweeps_root=paths.sweeps_root,
    )


def sync_active_aliases_if_active(*, sweep_id: str, paths: ExecutionPaths) -> None:
    index = system_delta.load_system_delta_index(paths.index_path)
    if str(index["active_sweep_id"]) != sweep_id:
        return
    _ = system_delta.sync_active_sweep_aliases(
        sweep_id=sweep_id,
        index_path=paths.index_path,
        catalog_path=paths.catalog_path,
        registry_path=paths.registry_path,
        sweeps_root=paths.sweeps_root,
    )


def _normalize_execution_policy(queue_row: Mapping[str, Any]) -> str:
    policy = str(queue_row.get("execution_policy", BENCHMARK_FULL_POLICY)).strip().lower()
    if policy not in {SCREEN_ONLY_POLICY, BENCHMARK_FULL_POLICY}:
        raise RuntimeError(f"unsupported execution_policy {policy!r}")
    return policy


def _optional_non_empty_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string when provided")
    return str(value.strip())


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def _cfg_data_mapping(cfg: Any) -> Mapping[str, Any] | None:
    raw_data_cfg = getattr(cfg, "data", None)
    if raw_data_cfg is None and isinstance(cfg, Mapping):
        raw_data_cfg = cfg.get("data")
    if raw_data_cfg is None:
        return None
    if isinstance(raw_data_cfg, DictConfig):
        raw_data_cfg = OmegaConf.to_container(raw_data_cfg, resolve=True)
    if not isinstance(raw_data_cfg, Mapping):
        raise RuntimeError("cfg.data must be a mapping when present")
    return cast(Mapping[str, Any], raw_data_cfg)


def resolve_training_backend(cfg: Any) -> str:
    return resolve_training_backend_from_data_cfg(_cfg_data_mapping(cfg))


def _training_surface_record_backend(record_path: Path) -> str | None:
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
    backend = str(raw_backend).strip().lower()
    return backend if backend in _ALLOWED_TRAINING_BACKENDS else None


def _training_telemetry_succeeded(telemetry_path: Path) -> bool:
    if not telemetry_path.exists():
        return False
    try:
        payload = _read_json_mapping(telemetry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(payload.get("success") is True)


def _archive_incomplete_train_dir(train_dir: Path) -> Path | None:
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
    return _training_surface_record_backend(run_dir / "training_surface_record.json") == expected_backend


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


def _planned_nanotabpfn_python_path(nanotabpfn_root: Path) -> Path:
    return nanotabpfn_root.expanduser().resolve() / ".venv" / "bin" / "python"


def _float_matches(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-12


def _resolved_nanotabpfn_signature(
    *,
    benchmark_bundle_path: Path,
    control_baseline_id: str,
    nanotabpfn_root: Path,
    nanotabpfn_python: Path,
    prior_dump: Path,
    requested_device: str,
) -> dict[str, Any]:
    normalized_requested_device = str(requested_device).strip()
    return {
        "benchmark_bundle_path": canonical_benchmark_bundle_source_path(benchmark_bundle_path),
        "control_baseline_id": str(control_baseline_id).strip(),
        "nanotabpfn_root": nanotabpfn_root.expanduser().resolve(),
        "nanotabpfn_python": nanotabpfn_python.expanduser().resolve(),
        "prior_dump_path": prior_dump.expanduser().resolve(),
        "device": normalized_requested_device,
        "resolved_device": resolve_device(normalized_requested_device),
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

    bundle = summary.get("benchmark_bundle")
    nanotabpfn = summary.get("nanotabpfn")
    nanotabpfn_error = summary.get("nanotabpfn_error")
    if not isinstance(bundle, Mapping):
        return None

    bundle_source = bundle.get("source_path")
    if not isinstance(bundle_source, str) or not bundle_source.strip():
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
                "benchmark_bundle_path": canonical_benchmark_bundle_source_path(str(bundle_source)),
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
        benchmark_bundle_path=canonical_benchmark_bundle_source_path(str(bundle_source)),
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
        "benchmark_bundle_path",
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
    # The cached nanoTabPFN benchmark outcome is independent of the tab-foundry
    # training row, so reuse survives a device change on the same host when the
    # benchmark contract is otherwise identical.
    for key in ("nanotabpfn_root", "nanotabpfn_python", "prior_dump_path"):
        candidate_value = candidate_signature[key]
        current_value = current_signature[key]
        if candidate_value is not None and candidate_value != current_value:
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
    if payload.benchmark_bundle_path != current_signature["benchmark_bundle_path"]:
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
        payload = _read_json_mapping(control_baseline_registry_path)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    baselines = payload.get("baselines")
    if not isinstance(baselines, Mapping):
        return None
    entry = baselines.get(control_baseline_id)
    if not isinstance(entry, Mapping):
        return None
    summary_value = entry.get("comparison_summary_path")
    if not isinstance(summary_value, str) or not summary_value.strip():
        return None
    return NanoTabPFNCurveCandidate(
        source_label="control baseline",
        comparison_summary_path=resolve_registry_path_value(summary_value),
        declared_control_baseline_id=control_baseline_id,
    )


def resolve_reusable_nanotabpfn_curve(
    *,
    sweep_meta: Mapping[str, Any],
    anchor_run_id: str | None,
    nanotabpfn_root: Path,
    prior_dump: Path,
    requested_device: str,
    paths: ExecutionPaths,
    extra_candidates: Sequence[NanoTabPFNCurveCandidate] | None = None,
) -> NanoTabPFNCurveReuseSelection | None:
    control_baseline_id = str(sweep_meta["control_baseline_id"]).strip()
    current_signature = _resolved_nanotabpfn_signature(
        benchmark_bundle_path=resolve_registry_path_value(str(sweep_meta["benchmark_bundle_path"])),
        control_baseline_id=control_baseline_id,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=_planned_nanotabpfn_python_path(nanotabpfn_root),
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


def _prior_completed_row_curve_candidates(
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


def _resolve_parent_row(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_order = int(queue_row["order"])
    current_delta_ref = str(queue_row["delta_ref"])
    parent_delta_ref = _optional_non_empty_string(
        queue_row.get("parent_delta_ref"),
        context=f"queue row {current_order}.parent_delta_ref",
    )
    if parent_delta_ref is None:
        return None

    matching_rows = [
        row for row in queue_rows if str(row.get("delta_ref", "")).strip() == parent_delta_ref
    ]
    if not matching_rows:
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} is missing from sweep {queue_row.get('sweep_id', '<unknown>')!r}"
        )

    earlier_rows = [row for row in matching_rows if int(row["order"]) < current_order]
    if earlier_rows:
        return max(earlier_rows, key=lambda row: int(row["order"]))

    matching_orders = [int(row["order"]) for row in matching_rows]
    if any(order == current_order for order in matching_orders):
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} must reference an earlier row, not itself; "
            f"matching orders={matching_orders}"
        )
    raise RuntimeError(
        f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
        f"{parent_delta_ref!r} must reference an earlier row; matching orders={matching_orders}"
    )


def _resolve_parent_run_id(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
    active_anchor: str | None,
) -> str | None:
    parent_row = _resolve_parent_row(queue_row=queue_row, queue_rows=queue_rows)
    if parent_row is None:
        return active_anchor

    parent_run_id = parent_row.get("run_id")
    if not isinstance(parent_run_id, str) or not parent_run_id.strip():
        raise RuntimeError(
            f"queue row {int(queue_row['order'])} ({queue_row['delta_ref']}) parent_delta_ref "
            f"{parent_row['delta_ref']!r} resolved to row {int(parent_row['order'])}, "
            "but that row does not have a completed run_id"
        )
    return str(parent_run_id)


def _resolve_dynamic_model_overrides(
    *,
    queue: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
) -> None:
    dynamic_overrides = queue_row.get("dynamic_model_overrides")
    if not isinstance(dynamic_overrides, Mapping):
        return
    queue_rows_raw = queue.get("rows")
    if not isinstance(queue_rows_raw, list):
        raise RuntimeError("queue rows must be a list")
    rows_by_order = {
        int(raw_row["order"]): cast(dict[str, Any], raw_row)
        for raw_row in queue_rows_raw
        if isinstance(raw_row, dict)
    }
    queue_model = cast(dict[str, Any], queue_row.setdefault("model", {}))
    queue_module_overrides = cast(dict[str, Any], queue_model.setdefault("module_overrides", {}))
    materialized_model = cast(dict[str, Any], materialized_row.setdefault("model", {}))
    materialized_module_overrides = cast(
        dict[str, Any],
        materialized_model.setdefault("module_overrides", {}),
    )
    queue_notes = cast(list[str], queue_row.setdefault("notes", []))
    materialized_notes = cast(list[str], materialized_row.setdefault("notes", []))

    for override_key, policy_raw in dynamic_overrides.items():
        if not isinstance(policy_raw, dict):
            raise RuntimeError(f"dynamic_model_overrides.{override_key} must be a mapping")
        policy = cast(dict[str, Any], policy_raw)
        if str(policy.get("kind")) != "screen_winner":
            raise RuntimeError(f"unsupported dynamic override policy kind: {policy.get('kind')!r}")
        resolved_value = policy.get("resolved_value")
        if isinstance(resolved_value, str) and resolved_value.strip():
            queue_module_overrides[str(override_key)] = resolved_value
            materialized_module_overrides[str(override_key)] = resolved_value
            continue
        compare_orders = policy.get("compare_orders")
        if not isinstance(compare_orders, list) or not compare_orders:
            raise RuntimeError(
                f"dynamic_model_overrides.{override_key}.compare_orders must be a non-empty list"
            )
        candidates: list[dict[str, Any]] = []
        for candidate_raw in compare_orders:
            if not isinstance(candidate_raw, Mapping):
                raise RuntimeError("dynamic compare_orders entries must be mappings")
            order = int(candidate_raw["order"])
            value = str(candidate_raw["value"])
            candidate_row = rows_by_order.get(order)
            if candidate_row is None:
                raise RuntimeError(f"dynamic compare order {order} is missing from the queue")
            candidates.append(
                {
                    "order": order,
                    "value": value,
                    "screen_metrics": candidate_row.get("screen_metrics"),
                }
            )
        resolution = pick_screen_winner(
            candidates=candidates,
            tie_break_preference=str(policy.get("tie_break_preference", "rmsnorm")),
        )
        winning_value = str(resolution["winning_value"])
        policy["resolved_value"] = winning_value
        policy["resolved_from_order"] = int(resolution["winning_order"])
        policy["resolution_reason"] = str(resolution["reason"])
        queue_module_overrides[str(override_key)] = winning_value
        materialized_module_overrides[str(override_key)] = winning_value
        resolution_note = (
            f"Resolved `{override_key}` to `{winning_value}` from screen row "
            f"`{int(resolution['winning_order'])}` ({resolution['reason']})."
        )
        queue_row["notes"] = append_note(queue_notes, resolution_note)
        materialized_row["notes"] = append_note(materialized_notes, resolution_note)


def run_row(
    *,
    sweep_id: str,
    sweep_meta: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
    anchor_run_id: str | None,
    parent_run_id: str | None,
    queue: Mapping[str, Any],
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
    decision: str,
    conclusion: str,
    paths: ExecutionPaths,
) -> str:
    execution_policy = _normalize_execution_policy(queue_row)
    _resolve_dynamic_model_overrides(
        queue=queue,
        queue_row=queue_row,
        materialized_row=materialized_row,
    )
    training_experiment = resolve_training_experiment(sweep_meta)
    training_config_profile = resolve_training_config_profile(sweep_meta)
    surface_role = resolve_surface_role(sweep_meta)
    external_benchmarks = tuple(resolve_sweep_external_benchmarks(sweep_meta))
    existing_run_id = queue_row.get("run_id")
    run_id = row_id_for_order(
        sweep_id,
        int(queue_row["order"]),
        str(queue_row["delta_ref"]),
        str(existing_run_id) if isinstance(existing_run_id, str) else None,
    )
    delta_root = paths.repo_root / "outputs" / "staged_ladder" / "research" / sweep_id / str(queue_row["delta_ref"])
    run_root = delta_root / run_id
    train_dir = run_root / "train"
    benchmark_dir = run_root / "benchmark"

    write_research_package(
        delta_root=delta_root,
        materialized_row=materialized_row,
        queue_row=queue_row,
        sweep_meta=sweep_meta,
        sweep_id=sweep_id,
        anchor_run_id=anchor_run_id,
        device=device,
        training_experiment=training_experiment,
        training_config_profile=training_config_profile,
        surface_role=surface_role,
    )
    cfg = compose_cfg(
        row=queue_row,
        run_dir=train_dir,
        device=device,
        training_experiment=training_experiment,
        sweep_id=sweep_id,
    )
    training_backend = resolve_training_backend(cfg)
    if completed_train_artifacts_exist(train_dir, expected_backend=training_backend):
        print(
            f"[row {int(queue_row['order']):02d}] reusing existing train artifacts",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            f"output_dir={train_dir}",
            flush=True,
        )
    else:
        existing_backend = _training_surface_record_backend(train_dir / "training_surface_record.json")
        if completed_train_artifacts_exist(train_dir) and existing_backend != training_backend:
            print(
                f"[row {int(queue_row['order']):02d}] existing train artifacts are not reusable",
                f"expected_backend={training_backend}",
                f"observed_backend={existing_backend or 'missing'}",
                flush=True,
            )
        archived_train_dir = _archive_incomplete_train_dir(train_dir)
        if archived_train_dir is not None:
            print(
                f"[row {int(queue_row['order']):02d}] archived incomplete train dir",
                f"run_id={run_id}",
                f"archived_dir={archived_train_dir}",
                flush=True,
            )
        print(
            f"[row {int(queue_row['order']):02d}] starting train",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            flush=True,
        )
        if training_backend == TRAINING_BACKEND_MANIFEST:
            train_result = train_from_manifest_cfg(cfg)
        elif training_backend == TRAINING_BACKEND_PRIOR_DUMP:
            train_result = train_tabfoundry_simple_prior(cfg, prior_dump_path=prior_dump)
        else:  # pragma: no cover - guarded by resolve_training_backend
            raise RuntimeError(f"unsupported training backend {training_backend!r}")
        print(
            f"[row {int(queue_row['order']):02d}] train complete",
            f"run_id={run_id}",
            f"training_backend={training_backend}",
            f"output_dir={train_result.output_dir}",
            flush=True,
        )

    if execution_policy == SCREEN_ONLY_POLICY:
        row_screen_metrics = screen_metrics(run_dir=train_dir)
        update_screened_queue_row(
            queue_row=queue_row,
            run_id=run_id,
            screen_metrics=row_screen_metrics,
            conclusion=conclusion,
        )
        final_window_mean = row_screen_metrics.get("upper_block_final_window_mean")
        final_window_text = (
            f"upper_block_final_window_mean={float(final_window_mean):.4f}"
            if final_window_mean is not None
            else "upper_block_final_window_mean=n/a"
        )
        print(
            f"[row {int(queue_row['order']):02d}] train-only screen complete",
            f"run_id={run_id}",
            final_window_text,
            flush=True,
        )
        return run_id

    reuse_selection = None
    reuse_curve_path = None
    reuse_nanotabpfn_error = None
    if EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks:
        reuse_selection = resolve_reusable_nanotabpfn_curve(
            sweep_meta=sweep_meta,
            anchor_run_id=anchor_run_id,
            nanotabpfn_root=nanotabpfn_root,
            prior_dump=prior_dump,
            requested_device=device,
            paths=paths,
            extra_candidates=_prior_completed_row_curve_candidates(
                queue=queue,
                current_order=int(queue_row["order"]),
                anchor_run_id=anchor_run_id,
                parent_run_id=parent_run_id,
                registry_path=paths.registry_path,
            ),
        )
        reuse_curve_path = None if reuse_selection is None else reuse_selection.curve_path
        reuse_nanotabpfn_error = (
            None if reuse_selection is None else reuse_selection.reusable_error
        )
    if reuse_selection is not None and reuse_curve_path is not None:
        print(
            f"[row {int(queue_row['order']):02d}] reusing nanoTabPFN curve",
            f"source={reuse_selection.source_label}",
            f"path={reuse_curve_path}",
            flush=True,
        )
    elif reuse_selection is not None and reuse_nanotabpfn_error is not None:
        print(
            f"[row {int(queue_row['order']):02d}] reusing nanoTabPFN benchmark outcome",
            f"source={reuse_selection.source_label}",
            f"kind={reuse_nanotabpfn_error.get('kind', 'unknown')}",
            flush=True,
        )
    elif EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks:
        _ = ensure_nanotabpfn_python(
            nanotabpfn_root=nanotabpfn_root,
            fallback_python=fallback_python,
        )
        print(
            f"[row {int(queue_row['order']):02d}] running fresh nanoTabPFN helper",
            f"device={device}",
            flush=True,
        )
    else:
        print(
            f"[row {int(queue_row['order']):02d}] running external benchmarks",
            f"comparators={','.join(external_benchmarks)}",
            flush=True,
        )
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=train_dir,
            out_root=benchmark_dir,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=prior_dump,
            device=device,
            control_baseline_id=str(sweep_meta["control_baseline_id"]),
            control_baseline_registry=paths.control_baseline_registry_path,
            benchmark_bundle_path=resolve_registry_path_value(str(sweep_meta["benchmark_bundle_path"])),
            external_benchmarks=external_benchmarks,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
            reuse_nanotabpfn_error=reuse_nanotabpfn_error,
            reuse_nanotabpfn_metadata=(
                None if reuse_selection is None else reuse_selection.metadata
            ),
        )
    )
    parent_sweep_id = sweep_meta.get("parent_sweep_id")
    registration = register_benchmark_run(
        run_id=run_id,
        track=DEFAULT_TRACK,
        experiment=training_experiment,
        config_profile=training_config_profile,
        budget_class=DEFAULT_BUDGET_CLASS,
        run_dir=train_dir,
        comparison_summary_path=benchmark_dir / "comparison_summary.json",
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=None,
        control_baseline_id=str(sweep_meta["control_baseline_id"]),
        sweep_id=sweep_id,
        delta_id=str(queue_row["delta_ref"]),
        parent_sweep_id=(
            None if not isinstance(parent_sweep_id, str) or not parent_sweep_id.strip() else str(parent_sweep_id)
        ),
        queue_order=int(queue_row["order"]),
        run_kind="primary",
        registry_path=paths.registry_path,
    )
    run_entry = cast(dict[str, Any], registration["run"])
    row_queue_metrics = queue_metrics(summary, run_dir=train_dir, run_entry=run_entry)
    _ = posthoc_update_wandb_summary(
        telemetry_path=train_dir / "telemetry.json",
        payload=_sweep_wandb_summary_payload(
            run_entry=run_entry,
            queue_metrics=row_queue_metrics,
        ),
    )
    (delta_root / "result_card.md").write_text(
        result_card_text(
            row=materialized_row,
            run_id=run_id,
            anchor_run_id=anchor_run_id,
            summary=summary,
            queue_metrics=row_queue_metrics,
            decision=decision,
            conclusion=conclusion,
        ),
        encoding="utf-8",
    )
    update_queue_row(
        queue_row=queue_row,
        run_id=run_id,
        queue_metrics=row_queue_metrics,
        decision=decision,
        conclusion=conclusion,
    )
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    final_metric_label = "final_log_loss"
    final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    if final_metric_value is None:
        final_metric_label = "final_crps"
        final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    if final_metric_value is None:
        final_metric_label = "final_roc_auc"
        final_metric_value = optional_metric(tab_foundry_summary, final_metric_label)
    final_metric_text = (
        f"{final_metric_label}={final_metric_value:.4f}"
        if final_metric_value is not None
        else "final_metric=n/a"
    )
    print(
        f"[row {int(queue_row['order']):02d}] benchmark+registry complete",
        f"run_id={run_id}",
        final_metric_text,
        flush=True,
    )
    return run_id


def execute_sweep(
    *,
    sweep_id: str | None,
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
    orders: list[int] | None = None,
    start_order: int | None = None,
    stop_after_order: int | None = None,
    include_completed: bool = False,
    decision_default: str = DEFAULT_DECISION,
    conclusion_default: str = DEFAULT_CONCLUSION,
    decision_overrides: Mapping[int, str] | None = None,
    conclusion_overrides: Mapping[int, str] | None = None,
    promote_first_executed_row_to_anchor: bool = False,
    paths: ExecutionPaths | None = None,
    run_row_fn: Any | None = None,
    sync_sweep_matrix_fn: Any | None = None,
    sync_active_aliases_if_active_fn: Any | None = None,
    promote_anchor_fn: Any | None = None,
) -> list[str]:
    resolved_paths = ExecutionPaths.default() if paths is None else paths
    run_row_impl = run_row if run_row_fn is None else run_row_fn
    sync_matrix_impl = sync_sweep_matrix if sync_sweep_matrix_fn is None else sync_sweep_matrix_fn
    sync_aliases_impl = sync_active_aliases_if_active if sync_active_aliases_if_active_fn is None else sync_active_aliases_if_active_fn
    promote_anchor_impl = promote_anchor if promote_anchor_fn is None else promote_anchor_fn

    sweep_meta = system_delta.load_system_delta_sweep(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_sweep_id = str(sweep_meta["sweep_id"])
    queue_path = system_delta.sweep_queue_path(resolved_sweep_id, sweeps_root=resolved_paths.sweeps_root)
    queue = read_yaml(queue_path)
    queue_rows = sorted_rows(queue)
    materialized_rows = materialized_row_map(sweep_id=resolved_sweep_id, paths=resolved_paths)
    selected_rows = select_queue_rows(
        queue,
        orders=orders,
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
    )
    if not selected_rows:
        print("No rows selected for execution.", f"sweep_id={resolved_sweep_id}", flush=True)
        return []

    current_anchor_run_id = sweep_meta.get("anchor_run_id")
    active_anchor = str(current_anchor_run_id) if isinstance(current_anchor_run_id, str) and current_anchor_run_id.strip() else None
    executed_run_ids: list[str] = []
    decision_map = dict(decision_overrides or {})
    conclusion_map = dict(conclusion_overrides or {})

    for index, queue_row in enumerate(selected_rows):
        order = int(queue_row["order"])
        decision = str(decision_map.get(order, decision_default)).strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
        conclusion = str(conclusion_map.get(order, conclusion_default)).strip()
        if not conclusion:
            raise RuntimeError("conclusion must be non-empty")

        promote_now = bool(promote_first_executed_row_to_anchor and index == 0)
        materialized_row = materialized_rows[str(queue_row["delta_ref"])]
        run_id = run_row_impl(
            sweep_id=resolved_sweep_id,
            sweep_meta=sweep_meta,
            queue_row=queue_row,
            materialized_row=materialized_row,
            anchor_run_id=None if promote_now else active_anchor,
            parent_run_id=(
                None
                if promote_now
                else _resolve_parent_run_id(
                    queue_row=queue_row,
                    queue_rows=queue_rows,
                    active_anchor=active_anchor,
                )
            ),
            queue=queue,
            prior_dump=prior_dump,
            nanotabpfn_root=nanotabpfn_root,
            device=device,
            fallback_python=fallback_python,
            decision=decision,
            conclusion=conclusion,
            paths=resolved_paths,
        )
        write_yaml(queue_path, queue)
        if promote_now:
            _ = promote_anchor_impl(
                sweep_id=resolved_sweep_id,
                anchor_run_id=run_id,
                set_active=False,
                paths=resolved_paths.promotion_paths(),
            )
            active_anchor = run_id
            sweep_meta = system_delta.load_system_delta_sweep(
                resolved_sweep_id,
                index_path=resolved_paths.index_path,
                sweeps_root=resolved_paths.sweeps_root,
            )
        sync_matrix_impl(sweep_id=resolved_sweep_id, paths=resolved_paths)
        sync_aliases_impl(sweep_id=resolved_sweep_id, paths=resolved_paths)
        executed_run_ids.append(run_id)

    return executed_run_ids
