"""Canonical benchmark-run registry helpers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)
import torch

from tab_foundry.bench.artifacts import load_history, write_json
from tab_foundry.bench.nanotabpfn import (
    resolve_tab_foundry_best_checkpoint,
    resolve_tab_foundry_run_artifact_paths,
)
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    ModelBuildSpec,
)
from tab_foundry.training.surface import write_training_surface_record
from tab_foundry.training.schedule import build_stage_configs, warmup_steps_for_stage


REGISTRY_SCHEMA = "tab-foundry-benchmark-runs-v1"
REGISTRY_VERSION = 1
DEFAULT_BUDGET_CLASS = "short-run"
ALLOWED_DECISIONS = ("keep", "reject", "defer")

_TOP_LEVEL_KEYS = {"schema", "version", "runs"}
_TAB_FOUNDRY_METRIC_KEYS = {
    "best_step",
    "best_training_time",
    "best_roc_auc",
    "best_log_loss",
    "best_brier_score",
    "best_crps",
    "best_avg_pinball_loss",
    "best_picp_90",
    "final_step",
    "final_training_time",
    "final_roc_auc",
    "final_log_loss",
    "final_brier_score",
    "final_crps",
    "final_avg_pinball_loss",
    "final_picp_90",
}

_RegistryPayloadT = TypeVar("_RegistryPayloadT", bound="_RegistryPayloadModel")


class _RegistryPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _BenchmarkRunModelPayload(_RegistryPayloadModel):
    arch: StrictStr
    stage: StrictStr | None = None
    stage_label: StrictStr | None = None
    benchmark_profile: StrictStr | None = None
    d_icl: StrictInt
    tficl_n_heads: StrictInt
    tficl_n_layers: StrictInt
    head_hidden_dim: StrictInt
    input_normalization: StrictStr
    many_class_base: StrictInt
    module_selection: dict[StrictStr, Any] | None = None
    module_hyperparameters: dict[StrictStr, Any] | None = None


class _BenchmarkBundlePayload(_RegistryPayloadModel):
    name: StrictStr
    version: StrictInt
    source_path: StrictStr
    task_count: StrictInt
    task_ids: list[StrictInt] = Field(min_length=1)


class _BenchmarkArtifactsPayload(_RegistryPayloadModel):
    run_dir: StrictStr
    benchmark_dir: StrictStr
    prior_dir: StrictStr | None = None
    history_path: StrictStr
    best_checkpoint_path: StrictStr
    comparison_summary_path: StrictStr
    comparison_curve_path: StrictStr
    benchmark_run_record_path: StrictStr | None = None
    training_surface_record_path: StrictStr | None = None


class _TabFoundryMetricsPayload(_RegistryPayloadModel):
    best_step: FiniteFloat | None = None
    best_training_time: FiniteFloat | None = None
    best_roc_auc: FiniteFloat | None = None
    best_log_loss: FiniteFloat | None = None
    best_brier_score: FiniteFloat | None = None
    best_crps: FiniteFloat | None = None
    best_avg_pinball_loss: FiniteFloat | None = None
    best_picp_90: FiniteFloat | None = None
    final_step: FiniteFloat | None = None
    final_training_time: FiniteFloat | None = None
    final_roc_auc: FiniteFloat | None = None
    final_log_loss: FiniteFloat | None = None
    final_brier_score: FiniteFloat | None = None
    final_crps: FiniteFloat | None = None
    final_avg_pinball_loss: FiniteFloat | None = None
    final_picp_90: FiniteFloat | None = None


class _TrainingDiagnosticsPayload(_RegistryPayloadModel):
    best_val_loss: FiniteFloat | None = None
    final_val_loss: FiniteFloat | None = None
    best_val_step: FiniteFloat | None = None
    post_warmup_train_loss_var: FiniteFloat | None = None
    mean_grad_norm: FiniteFloat | None = None
    max_grad_norm: FiniteFloat | None = None
    final_grad_norm: FiniteFloat | None = None
    train_elapsed_seconds: FiniteFloat | None = None
    wall_elapsed_seconds: FiniteFloat | None = None


class _ModelSizePayload(_RegistryPayloadModel):
    total_params: StrictInt
    trainable_params: StrictInt


class _ComparisonPayload(_RegistryPayloadModel):
    reference_run_id: StrictStr
    best_roc_auc_delta: FiniteFloat | None = None
    final_roc_auc_delta: FiniteFloat | None = None
    final_log_loss_delta: FiniteFloat | None = None
    final_brier_score_delta: FiniteFloat | None = None
    final_crps_delta: FiniteFloat | None = None
    final_avg_pinball_loss_delta: FiniteFloat | None = None
    final_picp_90_delta: FiniteFloat | None = None
    best_training_time_delta: FiniteFloat | None = None
    final_training_time_delta: FiniteFloat | None = None


class _ComparisonsPayload(_RegistryPayloadModel):
    vs_parent: _ComparisonPayload | None = None
    vs_anchor: _ComparisonPayload | None = None


class _LineagePayload(_RegistryPayloadModel):
    parent_run_id: StrictStr | None = None
    anchor_run_id: StrictStr | None = None
    control_baseline_id: StrictStr | None = None


class _SurfaceLabelsPayload(_RegistryPayloadModel):
    model: StrictStr
    data: StrictStr
    preprocessing: StrictStr
    training: StrictStr | None = None


class _SweepPayload(_RegistryPayloadModel):
    sweep_id: StrictStr
    delta_id: StrictStr
    parent_sweep_id: StrictStr | None = None
    queue_order: StrictInt | None = None
    run_kind: Literal["primary", "followup"]


class _BenchmarkRunRecordPayload(_RegistryPayloadModel):
    manifest_path: StrictStr
    seed_set: list[StrictInt] = Field(min_length=1)
    model: _BenchmarkRunModelPayload
    benchmark_bundle: _BenchmarkBundlePayload
    artifacts: _BenchmarkArtifactsPayload
    tab_foundry_metrics: _TabFoundryMetricsPayload
    training_diagnostics: _TrainingDiagnosticsPayload
    model_size: _ModelSizePayload
    surface_labels: _SurfaceLabelsPayload | None = None
    sweep: _SweepPayload | None = None
    generated_at_utc: StrictStr


class _BenchmarkRunEntryPayload(_RegistryPayloadModel):
    run_id: StrictStr
    track: StrictStr
    experiment: StrictStr
    config_profile: StrictStr
    budget_class: StrictStr
    model: _BenchmarkRunModelPayload
    lineage: _LineagePayload
    manifest_path: StrictStr
    seed_set: list[StrictInt] = Field(min_length=1)
    benchmark_bundle: _BenchmarkBundlePayload
    artifacts: _BenchmarkArtifactsPayload
    tab_foundry_metrics: _TabFoundryMetricsPayload
    training_diagnostics: _TrainingDiagnosticsPayload
    model_size: _ModelSizePayload
    surface_labels: _SurfaceLabelsPayload | None = None
    sweep: _SweepPayload | None = None
    comparisons: _ComparisonsPayload
    decision: Literal["keep", "reject", "defer"]
    conclusion: StrictStr
    registered_at_utc: StrictStr


def _validate_payload_model(
    payload_model: type[_RegistryPayloadT],
    payload: Any,
    *,
    context: str,
) -> _RegistryPayloadT:
    try:
        return payload_model.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"{context} is invalid: {exc}") from exc


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


def _ensure_optional_positive_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"{context} must be a positive int or null")
    return int(value)


def _sweep_payload(
    *,
    sweep_id: str | None,
    delta_id: str | None,
    parent_sweep_id: str | None,
    queue_order: int | None,
    run_kind: str | None,
) -> dict[str, Any] | None:
    raw_values = (sweep_id, delta_id, parent_sweep_id, queue_order, run_kind)
    if all(value is None for value in raw_values):
        return None
    normalized_sweep_id = _ensure_non_empty_string(sweep_id, context="sweep_id")
    normalized_delta_id = _ensure_non_empty_string(delta_id, context="delta_id")
    normalized_parent_sweep_id = _ensure_optional_string(
        parent_sweep_id,
        context="parent_sweep_id",
    )
    normalized_queue_order = _ensure_optional_positive_int(queue_order, context="queue_order")
    normalized_run_kind = _ensure_non_empty_string(run_kind, context="run_kind").lower()
    if normalized_run_kind not in {"primary", "followup"}:
        raise RuntimeError("run_kind must be 'primary' or 'followup'")
    return {
        "sweep_id": normalized_sweep_id,
        "delta_id": normalized_delta_id,
        "parent_sweep_id": normalized_parent_sweep_id,
        "queue_order": normalized_queue_order,
        "run_kind": normalized_run_kind,
    }


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


def _tab_foundry_metrics_from_summary(tab_foundry: Mapping[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "best_step": float(tab_foundry["best_step"]),
        "best_training_time": float(tab_foundry["best_training_time"]),
        "best_roc_auc": _ensure_optional_finite_number(
            tab_foundry.get("best_roc_auc"),
            context="comparison_summary.tab_foundry.best_roc_auc",
        ),
        "best_log_loss": _ensure_optional_finite_number(
            tab_foundry.get("best_log_loss"),
            context="comparison_summary.tab_foundry.best_log_loss",
        ),
        "best_brier_score": _ensure_optional_finite_number(
            tab_foundry.get("best_brier_score"),
            context="comparison_summary.tab_foundry.best_brier_score",
        ),
        "best_crps": _ensure_optional_finite_number(
            tab_foundry.get("best_crps"),
            context="comparison_summary.tab_foundry.best_crps",
        ),
        "best_avg_pinball_loss": _ensure_optional_finite_number(
            tab_foundry.get("best_avg_pinball_loss"),
            context="comparison_summary.tab_foundry.best_avg_pinball_loss",
        ),
        "best_picp_90": _ensure_optional_finite_number(
            tab_foundry.get("best_picp_90"),
            context="comparison_summary.tab_foundry.best_picp_90",
        ),
        "final_step": float(tab_foundry["final_step"]),
        "final_training_time": float(tab_foundry["final_training_time"]),
        "final_roc_auc": _ensure_optional_finite_number(
            tab_foundry.get("final_roc_auc"),
            context="comparison_summary.tab_foundry.final_roc_auc",
        ),
        "final_log_loss": _ensure_optional_finite_number(
            tab_foundry.get("final_log_loss"),
            context="comparison_summary.tab_foundry.final_log_loss",
        ),
        "final_brier_score": _ensure_optional_finite_number(
            tab_foundry.get("final_brier_score"),
            context="comparison_summary.tab_foundry.final_brier_score",
        ),
        "final_crps": _ensure_optional_finite_number(
            tab_foundry.get("final_crps"),
            context="comparison_summary.tab_foundry.final_crps",
        ),
        "final_avg_pinball_loss": _ensure_optional_finite_number(
            tab_foundry.get("final_avg_pinball_loss"),
            context="comparison_summary.tab_foundry.final_avg_pinball_loss",
        ),
        "final_picp_90": _ensure_optional_finite_number(
            tab_foundry.get("final_picp_90"),
            context="comparison_summary.tab_foundry.final_picp_90",
        ),
    }
    return metrics


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


def _checkpoint_model_spec_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
) -> ModelBuildSpec:
    task = _ensure_non_empty_string(raw_cfg.get("task"), context="checkpoint config.task")
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("checkpoint config must include a model mapping")
    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    arch = model_cfg.get("arch")
    if not isinstance(arch, str) or not arch.strip():
        raise RuntimeError(
            "checkpoint config must include explicit model.arch metadata for benchmark registration; "
            "legacy checkpoints without persisted model.arch cannot be registered"
        )
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=model_cfg,
        state_dict=state_dict,
    )


def _count_parameters_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
) -> dict[str, int]:
    model_spec = _checkpoint_model_spec_from_cfg(raw_cfg, state_dict=state_dict)
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
    state_dict: dict[str, Any] | None,
    summary_tab_foundry: Mapping[str, Any],
) -> dict[str, Any]:
    model_spec = _checkpoint_model_spec_from_cfg(raw_cfg, state_dict=state_dict)
    benchmark_profile_raw = summary_tab_foundry.get("benchmark_profile")
    payload: dict[str, Any] = {
        "arch": str(model_spec.arch),
        "stage": None if model_spec.stage is None else str(model_spec.stage),
        "stage_label": None if model_spec.stage_label is None else str(model_spec.stage_label),
        "benchmark_profile": None if benchmark_profile_raw is None else str(benchmark_profile_raw),
        "d_icl": int(model_spec.d_icl),
        "tficl_n_heads": int(model_spec.tficl_n_heads),
        "tficl_n_layers": int(model_spec.tficl_n_layers),
        "head_hidden_dim": int(model_spec.head_hidden_dim),
        "input_normalization": str(model_spec.input_normalization),
        "many_class_base": int(model_spec.many_class_base),
    }
    if model_spec.arch == "tabfoundry_staged":
        surface = resolve_staged_surface(model_spec)
        if payload["stage_label"] is None:
            payload["stage_label"] = str(surface.stage_label)
        if payload["benchmark_profile"] is None:
            payload["benchmark_profile"] = str(surface.benchmark_profile)
        payload["module_selection"] = surface.module_selection()
        payload["module_hyperparameters"] = surface.component_hyperparameters()
    return payload


def _training_surface_record(
    *,
    run_dir: Path,
    raw_cfg: dict[str, Any],
    raw_state_dict: dict[str, Any] | None,
    benchmark_run_record_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    run_record_path = run_dir.expanduser().resolve() / "training_surface_record.json"
    if run_record_path.exists():
        with run_record_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError(f"training_surface_record.json must be a JSON object: {run_record_path}")
        return cast(dict[str, Any], payload), run_record_path
    if benchmark_run_record_path is None:
        return None, None
    derived_path = benchmark_run_record_path.parent / "training_surface_record.json"
    payload = write_training_surface_record(
        derived_path,
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=raw_state_dict,
    )
    return payload, derived_path


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
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
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
    data_surface = resolve_data_surface(data_cfg)
    manifest_path_raw = (
        data_surface.overrides["manifest_path"]
        if "manifest_path" in data_surface.overrides
        else data_cfg.get("manifest_path")
    )
    if manifest_path_raw is None:
        raise RuntimeError(
            "checkpoint config must include a non-empty effective data.manifest_path"
        )
    manifest_path = _resolve_config_path(manifest_path_raw)
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
    training_surface_payload, training_surface_path = _training_surface_record(
        run_dir=resolved_run_dir,
        raw_cfg=raw_cfg,
        raw_state_dict=raw_state_dict,
        benchmark_run_record_path=benchmark_run_record_path,
    )

    record = {
        "manifest_path": _normalize_path_value(manifest_path),
        "seed_set": [int(seed_raw)],
        "model": _model_payload_from_cfg(
            raw_cfg,
            state_dict=raw_state_dict,
            summary_tab_foundry=tab_foundry,
        ),
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
            "training_surface_record_path": None
            if training_surface_path is None
            else _normalize_path_value(training_surface_path),
        },
        "tab_foundry_metrics": _tab_foundry_metrics_from_summary(tab_foundry),
        "training_diagnostics": _training_diagnostics_from_history(history, raw_cfg=raw_cfg),
        "model_size": _count_parameters_from_cfg(raw_cfg, state_dict=raw_state_dict),
        "surface_labels": None
        if training_surface_payload is None
        else dict(cast(dict[str, Any], training_surface_payload["labels"])),
        "sweep": _sweep_payload(
            sweep_id=sweep_id,
            delta_id=delta_id,
            parent_sweep_id=parent_sweep_id,
            queue_order=queue_order,
            run_kind=run_kind,
        ),
        "generated_at_utc": _utc_now(),
    }
    _validate_record_payload(record)
    return record


def _validate_record_payload(payload: Any) -> None:
    _ = _validate_payload_model(
        _BenchmarkRunRecordPayload,
        payload,
        context="benchmark run record",
    )


def _validate_run_entry(entry: Any, *, run_id: str) -> None:
    validated = _validate_payload_model(
        _BenchmarkRunEntryPayload,
        entry,
        context=f"benchmark run entry {run_id}",
    )
    if str(validated.run_id) != run_id:
        raise RuntimeError(
            "benchmark run entry run_id mismatch: "
            f"expected={run_id!r}, actual={validated.run_id!r}"
        )


def _comparison_delta(
    *,
    reference_run_id: str,
    current_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    def _metric_delta(metric_name: str) -> float | None:
        current_value = _ensure_optional_finite_number(
            current_metrics.get(metric_name),
            context=f"current_metrics.{metric_name}",
        )
        reference_value = _ensure_optional_finite_number(
            reference_metrics.get(metric_name),
            context=f"reference_metrics.{metric_name}",
        )
        if current_value is None or reference_value is None:
            return None
        return float(current_value) - float(reference_value)

    current_final_log_loss = _ensure_optional_finite_number(
        current_metrics.get("final_log_loss"),
        context="current_metrics.final_log_loss",
    )
    reference_final_log_loss = _ensure_optional_finite_number(
        reference_metrics.get("final_log_loss"),
        context="reference_metrics.final_log_loss",
    )
    return {
        "reference_run_id": str(reference_run_id),
        "best_roc_auc_delta": _metric_delta("best_roc_auc"),
        "final_roc_auc_delta": _metric_delta("final_roc_auc"),
        "final_log_loss_delta": None
        if current_final_log_loss is None or reference_final_log_loss is None
        else float(current_final_log_loss) - float(reference_final_log_loss),
        "final_brier_score_delta": _metric_delta("final_brier_score"),
        "final_crps_delta": _metric_delta("final_crps"),
        "final_avg_pinball_loss_delta": _metric_delta("final_avg_pinball_loss"),
        "final_picp_90_delta": _metric_delta("final_picp_90"),
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
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
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
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
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
        "surface_labels": record.get("surface_labels"),
        "sweep": record.get("sweep"),
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
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
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
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
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
    parser.add_argument("--sweep-id", default=None, help="Optional sweep id associated with the run")
    parser.add_argument("--delta-id", default=None, help="Optional delta id associated with the run")
    parser.add_argument(
        "--parent-sweep-id",
        default=None,
        help="Optional parent sweep id associated with the run",
    )
    parser.add_argument(
        "--queue-order",
        default=None,
        type=int,
        help="Optional positive queue order within the sweep",
    )
    parser.add_argument(
        "--run-kind",
        default=None,
        choices=("primary", "followup"),
        help="Optional sweep-local run kind",
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
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        delta_id=None if args.delta_id is None else str(args.delta_id),
        parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
        queue_order=None if args.queue_order is None else int(args.queue_order),
        run_kind=None if args.run_kind is None else str(args.run_kind),
        registry_path=Path(str(args.registry_path)),
    )
    print("Benchmark run registered:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  run={result['run']}")
    return 0
