"""Canonical benchmark-run registry helpers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch

from tab_foundry.bench.artifacts import load_history, write_json
from tab_foundry.bench.nanotabpfn import (
    resolve_tab_foundry_best_checkpoint,
    resolve_tab_foundry_run_artifact_paths,
)
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)
from tab_foundry.training.schedule import build_stage_configs, warmup_steps_for_stage


REGISTRY_SCHEMA = "tab-foundry-benchmark-runs-v1"
REGISTRY_VERSION = 1
DEFAULT_BUDGET_CLASS = "short-run"
ALLOWED_DECISIONS = ("keep", "reject", "defer")

_TOP_LEVEL_KEYS = {"schema", "version", "runs"}
_ENTRY_KEYS = {
    "run_id",
    "track",
    "experiment",
    "config_profile",
    "budget_class",
    "model",
    "lineage",
    "manifest_path",
    "seed_set",
    "benchmark_bundle",
    "artifacts",
    "tab_foundry_metrics",
    "training_diagnostics",
    "model_size",
    "comparisons",
    "decision",
    "conclusion",
    "registered_at_utc",
}
_MODEL_KEYS = {
    "arch",
    "stage",
    "benchmark_profile",
    "d_icl",
    "tficl_n_heads",
    "tficl_n_layers",
    "head_hidden_dim",
    "input_normalization",
    "many_class_base",
}
_LINEAGE_KEYS = {"parent_run_id", "anchor_run_id", "control_baseline_id"}
_BENCHMARK_BUNDLE_KEYS = {"name", "version", "source_path", "task_count", "task_ids"}
_ARTIFACT_KEYS = {
    "run_dir",
    "benchmark_dir",
    "prior_dir",
    "history_path",
    "best_checkpoint_path",
    "comparison_summary_path",
    "comparison_curve_path",
    "benchmark_run_record_path",
}
_TAB_FOUNDRY_METRIC_KEYS = {
    "best_step",
    "best_training_time",
    "best_roc_auc",
    "final_step",
    "final_training_time",
    "final_roc_auc",
}
_TRAINING_DIAGNOSTIC_KEYS = {
    "best_val_loss",
    "final_val_loss",
    "best_val_step",
    "post_warmup_train_loss_var",
    "mean_grad_norm",
    "max_grad_norm",
    "final_grad_norm",
    "train_elapsed_seconds",
    "wall_elapsed_seconds",
}
_MODEL_SIZE_KEYS = {"total_params", "trainable_params"}
_COMPARISON_KEYS = {
    "reference_run_id",
    "best_roc_auc_delta",
    "final_roc_auc_delta",
    "best_training_time_delta",
    "final_training_time_delta",
}
_COMPARISONS_KEYS = {"vs_parent", "vs_anchor"}


def project_root() -> Path:
    """Return the repository root for repo-relative artifact paths."""

    return Path(__file__).resolve().parents[3]


def default_benchmark_run_registry_path() -> Path:
    """Return the repo-tracked benchmark-run registry path."""

    return Path(__file__).resolve().with_name("benchmark_run_registry_v1.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


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


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": {},
    }


def _ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _ensure_optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return _ensure_non_empty_string(value, context=context)


def _ensure_optional_finite_number(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{context} must be a number or null")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise RuntimeError(f"{context} must be finite when present")
    return value_f


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be an object")
    return cast(dict[str, Any], value)


def _load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    registry_path = path.expanduser().resolve()
    if not registry_path.exists():
        if allow_missing:
            return _empty_registry()
        raise RuntimeError(f"benchmark run registry does not exist: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"benchmark run registry must be a JSON object: {registry_path}")
    actual_keys = set(payload.keys())
    if actual_keys != _TOP_LEVEL_KEYS:
        raise RuntimeError(
            "benchmark run registry keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}"
        )
    if payload["schema"] != REGISTRY_SCHEMA:
        raise RuntimeError(
            "benchmark run registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload['schema']!r}"
        )
    if int(payload["version"]) != REGISTRY_VERSION:
        raise RuntimeError(
            "benchmark run registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload['version']}"
        )
    runs = payload["runs"]
    if not isinstance(runs, dict):
        raise RuntimeError("benchmark run registry runs must be an object")
    for run_id, entry in runs.items():
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError("benchmark run registry ids must be non-empty strings")
        _validate_run_entry(entry, run_id=str(run_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "runs": {str(key): value for key, value in runs.items()},
    }


def load_benchmark_run_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the benchmark run registry."""

    return _load_registry_payload(path or default_benchmark_run_registry_path(), allow_missing=False)


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    registry_path = (path or default_benchmark_run_registry_path()).expanduser().resolve()
    payload = _load_registry_payload(registry_path, allow_missing=True)
    return registry_path, payload


def load_benchmark_run_entry(
    run_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load one benchmark run entry by id."""

    registry = load_benchmark_run_registry(registry_path)
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    entry = runs.get(str(run_id))
    if entry is None:
        raise RuntimeError(f"unknown benchmark run id: {run_id}")
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


def _resolve_config_path(raw_value: Any) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise RuntimeError("checkpoint config must include a non-empty data.manifest_path")
    return resolve_registry_path_value(str(raw_value))


def _history_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def _finite_history_values(history: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in history:
        raw = record.get(key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def _training_diagnostics_from_history(
    history: list[dict[str, Any]],
    *,
    raw_cfg: dict[str, Any],
) -> dict[str, float | None]:
    val_records: list[tuple[float, float]] = []
    for record in history:
        raw_val_loss = record.get("val_loss")
        if raw_val_loss is None:
            continue
        value = float(raw_val_loss)
        if math.isfinite(value):
            val_records.append((float(record["step"]), value))

    best_val_loss: float | None = None
    best_val_step: float | None = None
    final_val_loss: float | None = None
    if val_records:
        best_val_step, best_val_loss = min(val_records, key=lambda item: (item[1], item[0]))
        final_val_loss = float(val_records[-1][1])

    grad_norms = _finite_history_values(history, "grad_norm")
    raw_schedule = raw_cfg.get("schedule")
    warmup_steps = 0
    if isinstance(raw_schedule, dict):
        raw_stages = raw_schedule.get("stages")
        if isinstance(raw_stages, list):
            normalized_stages = [
                {str(key): value for key, value in stage.items()}
                for stage in raw_stages
                if isinstance(stage, dict)
            ]
            if normalized_stages:
                stage_configs = build_stage_configs(normalized_stages)
                if stage_configs:
                    warmup_steps = warmup_steps_for_stage(stage_configs[0])
    post_warmup_losses = [
        float(record["train_loss"])
        for record in history
        if int(record["step"]) > warmup_steps
        and record.get("train_loss") is not None
        and math.isfinite(float(record["train_loss"]))
    ]
    last_record = history[-1]
    train_elapsed = float(last_record.get("train_elapsed_seconds", 0.0))
    wall_elapsed = float(last_record.get("elapsed_seconds", train_elapsed))
    return {
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "final_val_loss": None if final_val_loss is None else float(final_val_loss),
        "best_val_step": None if best_val_step is None else float(best_val_step),
        "post_warmup_train_loss_var": _history_variance(post_warmup_losses),
        "mean_grad_norm": None
        if not grad_norms
        else float(sum(grad_norms) / float(len(grad_norms))),
        "max_grad_norm": None if not grad_norms else float(max(grad_norms)),
        "final_grad_norm": None if not grad_norms else float(grad_norms[-1]),
        "train_elapsed_seconds": train_elapsed if math.isfinite(train_elapsed) else None,
        "wall_elapsed_seconds": wall_elapsed if math.isfinite(wall_elapsed) else None,
    }


def _count_parameters_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
) -> dict[str, int]:
    task = _ensure_non_empty_string(raw_cfg.get("task"), context="checkpoint config.task")
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("checkpoint config must include a model mapping")
    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    model_spec = checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=model_cfg,
        state_dict=state_dict,
    )
    model = build_model_from_spec(model_spec)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def _model_payload_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    summary_tab_foundry: Mapping[str, Any],
) -> dict[str, Any]:
    task = _ensure_non_empty_string(raw_cfg.get("task"), context="checkpoint config.task")
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("checkpoint config must include a model mapping")
    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    benchmark_profile_raw = summary_tab_foundry.get("benchmark_profile")
    return {
        "arch": str(model_spec.arch),
        "stage": None if model_spec.stage is None else str(model_spec.stage),
        "benchmark_profile": None if benchmark_profile_raw is None else str(benchmark_profile_raw),
        "d_icl": int(model_spec.d_icl),
        "tficl_n_heads": int(model_spec.tficl_n_heads),
        "tficl_n_layers": int(model_spec.tficl_n_layers),
        "head_hidden_dim": int(model_spec.head_hidden_dim),
        "input_normalization": str(model_spec.input_normalization),
        "many_class_base": int(model_spec.many_class_base),
    }


def _coerce_integral_step(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value_f = float(value)
    if not math.isfinite(value_f) or not value_f.is_integer():
        return None
    return int(value_f)


def _resolve_record_checkpoint_path(
    run_dir: Path,
    *,
    summary_tab_foundry: Mapping[str, Any],
) -> Path:
    try:
        return resolve_tab_foundry_best_checkpoint(run_dir)
    except RuntimeError as best_exc:
        _history_path, checkpoint_dir = resolve_tab_foundry_run_artifact_paths(run_dir)
        candidate_steps: list[int] = []
        for key in ("best_step", "final_step"):
            step = _coerce_integral_step(summary_tab_foundry.get(key))
            if step is not None and step > 0 and step not in candidate_steps:
                candidate_steps.append(step)
        for step in candidate_steps:
            step_path = checkpoint_dir / f"step_{step:06d}.pt"
            if step_path.exists():
                return step_path.resolve()
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return latest_path.resolve()
        step_paths = sorted(checkpoint_dir.glob("step_*.pt"))
        if step_paths:
            return step_paths[-1].resolve()
        raise RuntimeError(
            "missing checkpoint artifact suitable for benchmark run record under "
            f"{run_dir.expanduser().resolve()}; best.pt lookup failed with: {best_exc}"
        ) from best_exc


def derive_benchmark_run_record(
    *,
    run_dir: Path,
    comparison_summary_path: Path,
    prior_dir: Path | None = None,
    benchmark_run_record_path: Path | None = None,
) -> dict[str, Any]:
    """Derive one machine-readable benchmark run record from current artifacts."""

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_summary_path = comparison_summary_path.expanduser().resolve()
    summary = _load_comparison_summary(resolved_summary_path)
    tab_foundry = _ensure_mapping(summary["tab_foundry"], context="comparison_summary.tab_foundry")
    summary_run_dir = resolve_registry_path_value(
        _ensure_non_empty_string(
            tab_foundry.get("run_dir"),
            context="comparison_summary.tab_foundry.run_dir",
        )
    )
    if summary_run_dir != resolved_run_dir:
        raise RuntimeError(
            "comparison summary run_dir does not match requested run dir: "
            f"summary={summary_run_dir}, requested={resolved_run_dir}"
        )

    history_path, _checkpoint_dir = resolve_tab_foundry_run_artifact_paths(resolved_run_dir)
    best_checkpoint_path = _resolve_record_checkpoint_path(
        resolved_run_dir,
        summary_tab_foundry=tab_foundry,
    )
    checkpoint_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {best_checkpoint_path}")
    raw_cfg = checkpoint_payload.get("config")
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"checkpoint config must be a mapping: {best_checkpoint_path}")
    raw_state_dict = checkpoint_payload.get("model")
    if raw_state_dict is not None and not isinstance(raw_state_dict, dict):
        raise RuntimeError(f"checkpoint model state_dict must be a mapping: {best_checkpoint_path}")
    data_cfg = raw_cfg.get("data")
    runtime_cfg = raw_cfg.get("runtime")
    if not isinstance(data_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise RuntimeError(f"checkpoint config must include data/runtime mappings: {best_checkpoint_path}")
    manifest_path = _resolve_config_path(data_cfg.get("manifest_path"))
    seed_raw = runtime_cfg.get("seed")
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        raise RuntimeError(f"checkpoint runtime.seed must be an int: {best_checkpoint_path}")

    history = load_history(history_path)
    benchmark_bundle = _ensure_mapping(summary["benchmark_bundle"], context="comparison_summary.benchmark_bundle")
    benchmark_bundle_source = _ensure_non_empty_string(
        benchmark_bundle.get("source_path"),
        context="comparison_summary.benchmark_bundle.source_path",
    )
    raw_artifacts = summary.get("artifacts")
    comparison_curve_path: Path | None = None
    if isinstance(raw_artifacts, dict):
        raw_curve = raw_artifacts.get("comparison_curve_png")
        if isinstance(raw_curve, str) and raw_curve.strip():
            comparison_curve_path = Path(str(raw_curve)).expanduser().resolve()
    if comparison_curve_path is None:
        comparison_curve_path = resolved_summary_path.parent / "comparison_curve.png"

    record = {
        "manifest_path": _normalize_path_value(manifest_path),
        "seed_set": [int(seed_raw)],
        "model": _model_payload_from_cfg(raw_cfg, summary_tab_foundry=tab_foundry),
        "benchmark_bundle": {
            "name": str(benchmark_bundle["name"]),
            "version": int(benchmark_bundle["version"]),
            "source_path": _normalize_path_value(resolve_registry_path_value(benchmark_bundle_source)),
            "task_count": int(benchmark_bundle["task_count"]),
            "task_ids": [
                int(task_id) for task_id in cast(list[Any], benchmark_bundle["task_ids"])
            ],
        },
        "artifacts": {
            "run_dir": _normalize_path_value(resolved_run_dir),
            "benchmark_dir": _normalize_path_value(resolved_summary_path.parent),
            "prior_dir": None if prior_dir is None else _normalize_path_value(prior_dir),
            "history_path": _normalize_path_value(history_path),
            "best_checkpoint_path": _normalize_path_value(best_checkpoint_path),
            "comparison_summary_path": _normalize_path_value(resolved_summary_path),
            "comparison_curve_path": _normalize_path_value(comparison_curve_path),
            "benchmark_run_record_path": None
            if benchmark_run_record_path is None
            else _normalize_path_value(benchmark_run_record_path),
        },
        "tab_foundry_metrics": {
            key: float(tab_foundry[key])
            for key in sorted(_TAB_FOUNDRY_METRIC_KEYS)
        },
        "training_diagnostics": _training_diagnostics_from_history(history, raw_cfg=raw_cfg),
        "model_size": _count_parameters_from_cfg(raw_cfg, state_dict=raw_state_dict),
        "generated_at_utc": _utc_now(),
    }
    _validate_record_payload(record)
    return record


def _validate_record_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("benchmark run record must be an object")
    required_keys = {
        "manifest_path",
        "seed_set",
        "model",
        "benchmark_bundle",
        "artifacts",
        "tab_foundry_metrics",
        "training_diagnostics",
        "model_size",
        "generated_at_utc",
    }
    actual_keys = set(payload.keys())
    if actual_keys != required_keys:
        raise RuntimeError(
            "benchmark run record keys mismatch: "
            f"missing={sorted(required_keys - actual_keys)}, extra={sorted(actual_keys - required_keys)}"
        )
    if not isinstance(payload["seed_set"], list) or not payload["seed_set"]:
        raise RuntimeError("benchmark run record seed_set must be a non-empty list")
    _validate_model_payload(payload["model"])
    _validate_benchmark_bundle_payload(payload["benchmark_bundle"])
    _validate_artifacts_payload(payload["artifacts"])
    _validate_tab_foundry_metrics_payload(payload["tab_foundry_metrics"])
    _validate_training_diagnostics_payload(payload["training_diagnostics"])
    _validate_model_size_payload(payload["model_size"])
    _ensure_non_empty_string(payload["manifest_path"], context="benchmark run record manifest_path")
    _ensure_non_empty_string(payload["generated_at_utc"], context="benchmark run record generated_at_utc")


def _validate_model_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run model")
    actual_keys = set(mapping.keys())
    if actual_keys != _MODEL_KEYS:
        raise RuntimeError(
            "benchmark run model keys mismatch: "
            f"missing={sorted(_MODEL_KEYS - actual_keys)}, extra={sorted(actual_keys - _MODEL_KEYS)}"
        )
    _ensure_non_empty_string(mapping["arch"], context="benchmark run model.arch")
    _ensure_optional_string(mapping["stage"], context="benchmark run model.stage")
    _ensure_optional_string(mapping["benchmark_profile"], context="benchmark run model.benchmark_profile")
    for key in ("d_icl", "tficl_n_heads", "tficl_n_layers", "head_hidden_dim", "many_class_base"):
        if not isinstance(mapping[key], int) or isinstance(mapping[key], bool):
            raise RuntimeError(f"benchmark run model.{key} must be an int")
    _ensure_non_empty_string(mapping["input_normalization"], context="benchmark run model.input_normalization")


def _validate_benchmark_bundle_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run benchmark_bundle")
    actual_keys = set(mapping.keys())
    if actual_keys != _BENCHMARK_BUNDLE_KEYS:
        raise RuntimeError(
            "benchmark run benchmark_bundle keys mismatch: "
            f"missing={sorted(_BENCHMARK_BUNDLE_KEYS - actual_keys)}, extra={sorted(actual_keys - _BENCHMARK_BUNDLE_KEYS)}"
        )
    _ensure_non_empty_string(mapping["name"], context="benchmark run benchmark_bundle.name")
    _ensure_non_empty_string(mapping["source_path"], context="benchmark run benchmark_bundle.source_path")
    for key in ("version", "task_count"):
        if not isinstance(mapping[key], int) or isinstance(mapping[key], bool):
            raise RuntimeError(f"benchmark run benchmark_bundle.{key} must be an int")
    if not isinstance(mapping["task_ids"], list) or not mapping["task_ids"]:
        raise RuntimeError("benchmark run benchmark_bundle.task_ids must be a non-empty list")


def _validate_artifacts_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run artifacts")
    actual_keys = set(mapping.keys())
    if actual_keys != _ARTIFACT_KEYS:
        raise RuntimeError(
            "benchmark run artifacts keys mismatch: "
            f"missing={sorted(_ARTIFACT_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _ARTIFACT_KEYS)}"
        )
    for key in (
        "run_dir",
        "benchmark_dir",
        "history_path",
        "best_checkpoint_path",
        "comparison_summary_path",
        "comparison_curve_path",
    ):
        _ensure_non_empty_string(mapping[key], context=f"benchmark run artifacts.{key}")
    _ensure_optional_string(mapping["prior_dir"], context="benchmark run artifacts.prior_dir")
    _ensure_optional_string(
        mapping["benchmark_run_record_path"],
        context="benchmark run artifacts.benchmark_run_record_path",
    )


def _validate_tab_foundry_metrics_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run tab_foundry_metrics")
    actual_keys = set(mapping.keys())
    if actual_keys != _TAB_FOUNDRY_METRIC_KEYS:
        raise RuntimeError(
            "benchmark run tab_foundry_metrics keys mismatch: "
            f"missing={sorted(_TAB_FOUNDRY_METRIC_KEYS - actual_keys)}, extra={sorted(actual_keys - _TAB_FOUNDRY_METRIC_KEYS)}"
        )
    for key in _TAB_FOUNDRY_METRIC_KEYS:
        _ensure_optional_finite_number(mapping[key], context=f"benchmark run tab_foundry_metrics.{key}")


def _validate_training_diagnostics_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run training_diagnostics")
    actual_keys = set(mapping.keys())
    if actual_keys != _TRAINING_DIAGNOSTIC_KEYS:
        raise RuntimeError(
            "benchmark run training_diagnostics keys mismatch: "
            f"missing={sorted(_TRAINING_DIAGNOSTIC_KEYS - actual_keys)}, extra={sorted(actual_keys - _TRAINING_DIAGNOSTIC_KEYS)}"
        )
    for key in _TRAINING_DIAGNOSTIC_KEYS:
        _ensure_optional_finite_number(mapping[key], context=f"benchmark run training_diagnostics.{key}")


def _validate_model_size_payload(payload: Any) -> None:
    mapping = _ensure_mapping(payload, context="benchmark run model_size")
    actual_keys = set(mapping.keys())
    if actual_keys != _MODEL_SIZE_KEYS:
        raise RuntimeError(
            "benchmark run model_size keys mismatch: "
            f"missing={sorted(_MODEL_SIZE_KEYS - actual_keys)}, extra={sorted(_MODEL_SIZE_KEYS - actual_keys)}"
        )
    for key in _MODEL_SIZE_KEYS:
        if not isinstance(mapping[key], int) or isinstance(mapping[key], bool):
            raise RuntimeError(f"benchmark run model_size.{key} must be an int")


def _validate_comparison_payload(payload: Any, *, context: str) -> None:
    mapping = _ensure_mapping(payload, context=context)
    actual_keys = set(mapping.keys())
    if actual_keys != _COMPARISON_KEYS:
        raise RuntimeError(
            f"{context} keys mismatch: "
            f"missing={sorted(_COMPARISON_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _COMPARISON_KEYS)}"
        )
    _ensure_non_empty_string(mapping["reference_run_id"], context=f"{context}.reference_run_id")
    for key in (
        "best_roc_auc_delta",
        "final_roc_auc_delta",
        "best_training_time_delta",
        "final_training_time_delta",
    ):
        _ensure_optional_finite_number(mapping[key], context=f"{context}.{key}")


def _validate_run_entry(entry: Any, *, run_id: str) -> None:
    mapping = _ensure_mapping(entry, context=f"benchmark run entry {run_id}")
    actual_keys = set(mapping.keys())
    if actual_keys != _ENTRY_KEYS:
        raise RuntimeError(
            f"benchmark run entry keys mismatch for {run_id}: "
            f"missing={sorted(_ENTRY_KEYS - actual_keys)}, extra={sorted(actual_keys - _ENTRY_KEYS)}"
        )
    if _ensure_non_empty_string(mapping["run_id"], context="benchmark run entry run_id") != run_id:
        raise RuntimeError(
            "benchmark run entry run_id mismatch: "
            f"expected={run_id!r}, actual={mapping['run_id']!r}"
        )
    _ensure_non_empty_string(mapping["track"], context="benchmark run entry track")
    _ensure_non_empty_string(mapping["experiment"], context="benchmark run entry experiment")
    _ensure_non_empty_string(mapping["config_profile"], context="benchmark run entry config_profile")
    _ensure_non_empty_string(mapping["budget_class"], context="benchmark run entry budget_class")
    _ensure_non_empty_string(mapping["manifest_path"], context="benchmark run entry manifest_path")
    if not isinstance(mapping["seed_set"], list) or not mapping["seed_set"]:
        raise RuntimeError(f"benchmark run entry seed_set must be a non-empty list: {run_id}")
    _validate_model_payload(mapping["model"])
    lineage = _ensure_mapping(mapping["lineage"], context="benchmark run entry lineage")
    actual_lineage_keys = set(lineage.keys())
    if actual_lineage_keys != _LINEAGE_KEYS:
        raise RuntimeError(
            "benchmark run entry lineage keys mismatch: "
            f"missing={sorted(_LINEAGE_KEYS - actual_lineage_keys)}, extra={sorted(actual_lineage_keys - _LINEAGE_KEYS)}"
        )
    for key in _LINEAGE_KEYS:
        _ensure_optional_string(lineage[key], context=f"benchmark run entry lineage.{key}")
    _validate_benchmark_bundle_payload(mapping["benchmark_bundle"])
    _validate_artifacts_payload(mapping["artifacts"])
    _validate_tab_foundry_metrics_payload(mapping["tab_foundry_metrics"])
    _validate_training_diagnostics_payload(mapping["training_diagnostics"])
    _validate_model_size_payload(mapping["model_size"])
    comparisons = _ensure_mapping(mapping["comparisons"], context="benchmark run entry comparisons")
    actual_comparisons_keys = set(comparisons.keys())
    if actual_comparisons_keys != _COMPARISONS_KEYS:
        raise RuntimeError(
            "benchmark run entry comparisons keys mismatch: "
            f"missing={sorted(_COMPARISONS_KEYS - actual_comparisons_keys)}, extra={sorted(actual_comparisons_keys - _COMPARISONS_KEYS)}"
        )
    for key in _COMPARISONS_KEYS:
        value = comparisons[key]
        if value is not None:
            _validate_comparison_payload(value, context=f"benchmark run entry comparisons.{key}")
    decision = _ensure_non_empty_string(mapping["decision"], context="benchmark run entry decision").lower()
    if decision not in ALLOWED_DECISIONS:
        raise RuntimeError(
            "benchmark run entry decision must be one of "
            f"{sorted(ALLOWED_DECISIONS)}, got {mapping['decision']!r}"
        )
    _ensure_non_empty_string(mapping["conclusion"], context="benchmark run entry conclusion")
    _ensure_non_empty_string(mapping["registered_at_utc"], context="benchmark run entry registered_at_utc")


def _comparison_delta(
    *,
    reference_run_id: str,
    current_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "reference_run_id": str(reference_run_id),
        "best_roc_auc_delta": float(current_metrics["best_roc_auc"]) - float(reference_metrics["best_roc_auc"]),
        "final_roc_auc_delta": float(current_metrics["final_roc_auc"]) - float(reference_metrics["final_roc_auc"]),
        "best_training_time_delta": float(current_metrics["best_training_time"])
        - float(reference_metrics["best_training_time"]),
        "final_training_time_delta": float(current_metrics["final_training_time"])
        - float(reference_metrics["final_training_time"]),
    }


def derive_benchmark_run_entry(
    *,
    run_id: str,
    track: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    decision: str,
    conclusion: str,
    parent_run_id: str | None = None,
    anchor_run_id: str | None = None,
    prior_dir: Path | None = None,
    control_baseline_id: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Derive one benchmark registry entry from benchmark artifacts and lineage."""

    normalized_run_id = _ensure_non_empty_string(run_id, context="run_id")
    normalized_track = _ensure_non_empty_string(track, context="track")
    normalized_experiment = _ensure_non_empty_string(experiment, context="experiment")
    normalized_config_profile = _ensure_non_empty_string(config_profile, context="config_profile")
    normalized_budget_class = _ensure_non_empty_string(budget_class, context="budget_class")
    normalized_decision = _ensure_non_empty_string(decision, context="decision").lower()
    if normalized_decision not in ALLOWED_DECISIONS:
        raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
    normalized_conclusion = _ensure_non_empty_string(conclusion, context="conclusion")

    resolved_summary_path = comparison_summary_path.expanduser().resolve()
    resolved_record_path = resolved_summary_path.parent / "benchmark_run_record.json"
    record = derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=resolved_summary_path,
        prior_dir=prior_dir,
        benchmark_run_record_path=resolved_record_path,
    )
    write_json(resolved_record_path, record)

    _resolved_registry_path, payload = _ensure_registry_payload(registry_path)
    runs = cast(dict[str, Any], payload["runs"])
    if normalized_run_id in {str(parent_run_id), str(anchor_run_id)}:
        raise RuntimeError("run_id must not match parent_run_id or anchor_run_id")

    parent_entry = None
    if parent_run_id is not None:
        parent_entry = runs.get(str(parent_run_id))
        if parent_entry is None:
            raise RuntimeError(f"unknown parent_run_id: {parent_run_id}")
    anchor_entry = None
    if anchor_run_id is not None:
        anchor_entry = runs.get(str(anchor_run_id))
        if anchor_entry is None:
            raise RuntimeError(f"unknown anchor_run_id: {anchor_run_id}")

    entry = {
        "run_id": normalized_run_id,
        "track": normalized_track,
        "experiment": normalized_experiment,
        "config_profile": normalized_config_profile,
        "budget_class": normalized_budget_class,
        "model": record["model"],
        "lineage": {
            "parent_run_id": None if parent_run_id is None else str(parent_run_id),
            "anchor_run_id": None if anchor_run_id is None else str(anchor_run_id),
            "control_baseline_id": None if control_baseline_id is None else str(control_baseline_id),
        },
        "manifest_path": str(record["manifest_path"]),
        "seed_set": list(record["seed_set"]),
        "benchmark_bundle": record["benchmark_bundle"],
        "artifacts": record["artifacts"],
        "tab_foundry_metrics": record["tab_foundry_metrics"],
        "training_diagnostics": record["training_diagnostics"],
        "model_size": record["model_size"],
        "comparisons": {
            "vs_parent": None
            if parent_entry is None
            else _comparison_delta(
                reference_run_id=str(parent_run_id),
                current_metrics=cast(dict[str, Any], record["tab_foundry_metrics"]),
                reference_metrics=cast(dict[str, Any], parent_entry["tab_foundry_metrics"]),
            ),
            "vs_anchor": None
            if anchor_entry is None
            else _comparison_delta(
                reference_run_id=str(anchor_run_id),
                current_metrics=cast(dict[str, Any], record["tab_foundry_metrics"]),
                reference_metrics=cast(dict[str, Any], anchor_entry["tab_foundry_metrics"]),
            ),
        },
        "decision": normalized_decision,
        "conclusion": normalized_conclusion,
        "registered_at_utc": _utc_now(),
    }
    _validate_run_entry(entry, run_id=normalized_run_id)
    return entry


def upsert_benchmark_run_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> Path:
    """Insert or replace one benchmark run entry in the registry."""

    run_id = str(entry["run_id"])
    _validate_run_entry(entry, run_id=run_id)
    resolved_registry_path, payload = _ensure_registry_payload(registry_path)
    runs = cast(dict[str, Any], payload["runs"])
    runs[run_id] = _copy_jsonable(entry)
    write_json(resolved_registry_path, payload)
    return resolved_registry_path


def register_benchmark_run(
    *,
    run_id: str,
    track: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    decision: str,
    conclusion: str,
    parent_run_id: str | None = None,
    anchor_run_id: str | None = None,
    prior_dir: Path | None = None,
    control_baseline_id: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Register one completed benchmark-facing run in the canonical registry."""

    entry = derive_benchmark_run_entry(
        run_id=run_id,
        track=track,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary_path,
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=prior_dir,
        control_baseline_id=control_baseline_id,
        registry_path=registry_path,
    )
    resolved_registry_path = upsert_benchmark_run_entry(entry, registry_path=registry_path)
    return {
        "registry_path": str(resolved_registry_path),
        "run": entry,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one completed benchmark-facing tab-foundry run"
    )
    parser.add_argument("--run-id", required=True, help="Canonical registry id for the run")
    parser.add_argument(
        "--track",
        required=True,
        help="Logical track label, e.g. binary_ladder or many_class_branch",
    )
    parser.add_argument("--run-dir", required=True, help="Completed tab-foundry run directory")
    parser.add_argument(
        "--comparison-summary",
        required=True,
        help="Benchmark comparison_summary.json for the same run",
    )
    parser.add_argument("--experiment", required=True, help="Logical experiment name stored in the registry")
    parser.add_argument(
        "--config-profile",
        default=None,
        help="Config profile stored in the registry entry; defaults to --experiment",
    )
    parser.add_argument(
        "--budget-class",
        default=DEFAULT_BUDGET_CLASS,
        help="Budget class label stored in the registry entry",
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=ALLOWED_DECISIONS,
        help="Human review decision stored with the run",
    )
    parser.add_argument("--conclusion", required=True, help="One-line keep/reject/defer conclusion")
    parser.add_argument("--parent-run-id", default=None, help="Optional previous-stage benchmark run id")
    parser.add_argument("--anchor-run-id", default=None, help="Optional frozen anchor run id")
    parser.add_argument("--prior-dir", default=None, help="Optional prior-training artifact directory")
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id associated with the run",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = register_benchmark_run(
        run_id=str(args.run_id),
        track=str(args.track),
        experiment=str(args.experiment),
        config_profile=str(args.config_profile or args.experiment),
        budget_class=str(args.budget_class),
        run_dir=Path(str(args.run_dir)),
        comparison_summary_path=Path(str(args.comparison_summary)),
        decision=str(args.decision),
        conclusion=str(args.conclusion),
        parent_run_id=None if args.parent_run_id is None else str(args.parent_run_id),
        anchor_run_id=None if args.anchor_run_id is None else str(args.anchor_run_id),
        prior_dir=None if args.prior_dir is None else Path(str(args.prior_dir)),
        control_baseline_id=(
            None if args.control_baseline_id is None else str(args.control_baseline_id)
        ),
        registry_path=Path(str(args.registry_path)),
    )
    print("Benchmark run registered:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  run={result['run']}")
    return 0
