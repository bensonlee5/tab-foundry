"""Phase-two scaling-study point collection, fitting, and artifact rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import numpy as np

from tab_foundry.bench.artifacts import load_history, write_json
from tab_foundry.bench.width_depth_scaling import fit_power_law
from tab_foundry.benchmark_registry import (
    load_benchmark_run_entry,
    resolve_registry_path_value,
)
from tab_foundry.research.navigation import build_scaling_navigation_payload
from tab_foundry.research.scaling.study import ScalingStudyConfig, load_scaling_study_config
from tab_foundry.research.scaling.validation_backfill_schema import (
    VALIDATION_BACKFILL_FILENAME,
    VALIDATION_BACKFILL_SCHEMA,
)
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tab_foundry.repo_paths import normalize_repo_relative_path, repo_root
from tab_foundry.training.wandb import posthoc_update_wandb_summary


_FLOAT_TOLERANCE = 1.0e-9
_MIN_POSITIVE = 1.0e-12
_MAX_ITERATIONS = 200
_STEP_TOLERANCE = 1.0e-8
_MIN_SLICE_POINTS = 2
_MIN_LINE_SEARCH_SCALE = 1.0e-4
_MIN_LOG_POSITIVE = math.log(_MIN_POSITIVE)
_MAX_LOG_PARAMETER_SEED = 80.0
_C_AXIS_MONOTONIC_RELATIVE_TOLERANCE = 1.0e-6
_C_AXIS_ERROR_EXAMPLE_LIMIT = 5
VALIDATION_OVERLAY_SCHEMA = "tab-foundry-scaling-validation-overlay-v1"
FIT_SCOPE_ALL = "all"
FIT_SCOPE_NS_ONLY = "ns-only"
FIT_SCOPE_CHOICES = (FIT_SCOPE_ALL, FIT_SCOPE_NS_ONLY)
FitScope = Literal["all", "ns-only"]


@dataclass(frozen=True, slots=True)
class ScalingStudyRunPoint:
    """One completed benchmark-backed run admitted into a scaling study."""

    family: str
    sweep_id: str
    row_order: int
    row_label: str
    run_id: str
    d_icl: int
    layers: int
    max_steps: int
    grad_accum_steps: int
    task_batch_size: int
    strict_embedding_params: int
    strict_non_embedding_params: int
    expanded_embedding_like_params: int
    expanded_non_embedding_params: int
    canonical_non_embedding_params: int
    benchmark_log_loss: float
    validation_loss: float | None
    validation_loss_source: str | None
    validation_loss_missing_reason: str | None
    steps: int
    tokens_seen: float
    tokens_per_step: float
    train_flops_per_token: float
    train_flops_per_step: float
    total_train_flops: float
    run_dir: str
    history_path: str
    telemetry_path: str
    throughput_tokens_per_second: float | None = None
    end_to_end_wall_seconds: float | None = None
    compute_training_shape_summary_present: bool = True
    compute_training_shape_signature_count: int = 1
    uses_reuse_train_artifact: bool = False

    @property
    def n(self) -> float:
        return float(self.canonical_non_embedding_params)

    @property
    def d(self) -> float:
        return float(self.tokens_seen)

    @property
    def s(self) -> float:
        return float(self.steps)

    @property
    def b_eff(self) -> float:
        return float(self.tokens_per_step)

    @property
    def c(self) -> float:
        return float(self.total_train_flops)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_mapping(payload: Mapping[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} missing mapping payload {key!r}")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _required_str(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} missing string payload {key!r}")
    return str(value)


def _required_float(payload: Mapping[str, Any], key: str, *, context: str) -> float:
    value = payload.get(key)
    if value is None:
        raise RuntimeError(f"{context} missing numeric payload {key!r}")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise RuntimeError(f"{context} payload {key!r} must be finite")
    return value_f


def _required_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if value is None:
        raise RuntimeError(f"{context} missing integer payload {key!r}")
    try:
        value_i = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} payload {key!r} must be an integer") from exc
    return value_i


def _training_shape_signature_count(payload: Any) -> int:
    if not isinstance(payload, Mapping):
        return 0
    signature_task_counts = payload.get("signature_task_counts")
    if not isinstance(signature_task_counts, Mapping):
        return 0
    count = 0
    for signature_text, task_count in signature_task_counts.items():
        if not isinstance(signature_text, str) or not signature_text.strip():
            continue
        try:
            resolved_count = int(task_count)
        except (TypeError, ValueError):
            continue
        if resolved_count > 0:
            count += 1
    return count


def _normalize_fit_scope(value: str) -> FitScope:
    normalized = str(value).strip().lower()
    if normalized not in FIT_SCOPE_CHOICES:
        raise RuntimeError(
            f"fit_scope must be one of {', '.join(FIT_SCOPE_CHOICES)}; got {value!r}"
        )
    return normalized  # type: ignore[return-value]


def _load_validation_overlay(config: ScalingStudyConfig) -> dict[str, dict[str, Any]]:
    overlay_path = config.validation_overlay_resolved_path(root=repo_root())
    if overlay_path is None:
        return {}
    try:
        payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read scaling validation overlay: {overlay_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"scaling validation overlay is not valid JSON: {overlay_path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"scaling validation overlay must be a mapping: {overlay_path}")
    schema = payload.get("schema")
    if schema != VALIDATION_OVERLAY_SCHEMA:
        raise RuntimeError(
            "scaling validation overlay schema mismatch: "
            f"expected={VALIDATION_OVERLAY_SCHEMA!r}, actual={schema!r}, path={overlay_path}"
        )
    study_id = payload.get("study_id")
    if study_id != config.study_id:
        raise RuntimeError(
            "scaling validation overlay study_id mismatch: "
            f"expected={config.study_id!r}, actual={study_id!r}, path={overlay_path}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"scaling validation overlay rows must be a list: {overlay_path}")
    overlay: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"scaling validation overlay row {index} must be a mapping")
        row_payload = {str(key): value for key, value in row.items()}
        run_id = _required_str(
            row_payload,
            "run_id",
            context=f"scaling validation overlay row {index}",
        )
        if run_id in overlay:
            raise RuntimeError(f"scaling validation overlay has duplicate run_id={run_id!r}")
        _ = _required_float(
            row_payload,
            "validation_loss",
            context=f"scaling validation overlay row {index}",
        )
        global_step = _required_float(
            row_payload,
            "global_step",
            context=f"scaling validation overlay row {index}",
        )
        if global_step <= 0.0:
            raise RuntimeError(f"scaling validation overlay row {index} global_step must be positive")
        overlay[run_id] = row_payload
    return overlay


def _row_label_from_model_payload(model_payload: Mapping[str, Any]) -> str:
    raw_d_icl = model_payload.get("d_icl")
    if raw_d_icl is None:
        raise RuntimeError("row model payload missing d_icl")
    raw_layers = model_payload.get("sandwich_layers")
    if raw_layers is None:
        build_spec = model_payload.get("build_spec")
        if isinstance(build_spec, Mapping):
            raw_layers = build_spec.get("sandwich_layers")
    if raw_layers is None:
        raise RuntimeError("row model payload missing sandwich_layers")
    return f"{int(raw_d_icl)}x{int(raw_layers)}"


def _completed_benchmark_backed_row(row: Mapping[str, Any]) -> bool:
    if str(row.get("interpretation_status") or "").strip().lower() != "completed":
        return False
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return False
    benchmark_metrics = row.get("benchmark_metrics")
    if not isinstance(benchmark_metrics, Mapping):
        return False
    return benchmark_metrics.get("final_log_loss") is not None


def _read_validation_backfill_loss(
    sidecar_path: Path,
) -> tuple[float | None, str | None, str | None]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except OSError:
        return None, None, "validation_backfill_missing"
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"validation backfill sidecar is not valid JSON: {sidecar_path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"validation backfill sidecar must be a mapping: {sidecar_path}")
    schema = payload.get("schema")
    if schema != VALIDATION_BACKFILL_SCHEMA:
        raise RuntimeError(
            "validation backfill sidecar schema mismatch: "
            f"expected={VALIDATION_BACKFILL_SCHEMA!r}, actual={schema!r}, path={sidecar_path}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"validation backfill sidecar missing metrics mapping: {sidecar_path}")
    if metrics.get("val_loss") is None:
        return None, None, "validation_backfill_missing_val_loss"
    value = float(metrics["val_loss"])
    if not math.isfinite(value):
        raise RuntimeError(f"validation backfill sidecar recorded non-finite val_loss: {sidecar_path}")
    return value, "validation_backfill_v1", None


def _read_validation_overlay_loss(
    overlay_entry: Mapping[str, Any] | None,
    *,
    context: str,
) -> tuple[float | None, str | None, str | None]:
    if overlay_entry is None:
        return None, None, "validation_overlay_missing"
    value = _required_float(overlay_entry, "validation_loss", context=context)
    source = overlay_entry.get("validation_loss_source")
    if not isinstance(source, str) or not source.strip():
        source = "validation_overlay_v1"
    return value, source, None


def _read_final_validation_loss(
    history_path: Path,
    *,
    validation_overlay_entry: Mapping[str, Any] | None = None,
    context: str,
) -> tuple[float | None, str | None, str | None]:
    try:
        records = load_history(history_path)
    except OSError:
        backfilled = _read_validation_backfill_loss(history_path.parent / VALIDATION_BACKFILL_FILENAME)
        if backfilled[0] is not None:
            return backfilled
        overlay = _read_validation_overlay_loss(validation_overlay_entry, context=context)
        if overlay[0] is not None:
            return overlay
        return None, None, "history_file_missing"
    validation_records = [record for record in records if record.get("val_loss") is not None]
    if not validation_records:
        backfilled = _read_validation_backfill_loss(history_path.parent / VALIDATION_BACKFILL_FILENAME)
        if backfilled[0] is not None:
            return backfilled
        overlay = _read_validation_overlay_loss(validation_overlay_entry, context=context)
        if overlay[0] is not None:
            return overlay
        return None, None, "history_missing_validation_records"
    value = float(validation_records[-1]["val_loss"])
    if not math.isfinite(value):
        raise RuntimeError(f"history file recorded a non-finite validation loss: {history_path}")
    return value, "train_history", None


def _registry_root(registry_path: Path) -> Path:
    resolved = registry_path.expanduser().resolve()
    try:
        if (
            resolved.name == "benchmark_run_registry_v1.json"
            and resolved.parent.name == "bench"
            and resolved.parent.parent.name == "tab_foundry"
            and resolved.parent.parent.parent.name == "src"
        ):
            return resolved.parent.parent.parent.parent
    except IndexError:  # pragma: no cover - defensive path guard
        pass
    return resolved.parent


def _read_history_final_step(history_path: Path) -> int | None:
    try:
        records = load_history(history_path)
    except OSError:
        return None
    steps: list[int] = []
    for record in records:
        if record.get("step") is None:
            continue
        step = int(record["step"])
        if step > 0:
            steps.append(step)
    if not steps:
        return None
    return max(steps)


def _read_validation_backfill_step(sidecar_path: Path, *, context: str) -> int | None:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"validation backfill sidecar is not valid JSON: {sidecar_path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"validation backfill sidecar must be a mapping: {sidecar_path}")
    schema = payload.get("schema")
    if schema != VALIDATION_BACKFILL_SCHEMA:
        raise RuntimeError(
            "validation backfill sidecar schema mismatch: "
            f"expected={VALIDATION_BACKFILL_SCHEMA!r}, actual={schema!r}, path={sidecar_path}"
        )
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("global_step") is None:
        return None
    value_f = float(checkpoint["global_step"])
    if not math.isfinite(value_f) or value_f <= 0:
        raise RuntimeError(f"{context} validation backfill global_step must be positive and finite")
    return int(round(value_f))


def _read_validation_overlay_step(
    overlay_entry: Mapping[str, Any] | None,
    *,
    context: str,
) -> int | None:
    if overlay_entry is None:
        return None
    value_f = _required_float(overlay_entry, "global_step", context=context)
    if value_f <= 0.0:
        raise RuntimeError(f"{context} validation overlay global_step must be positive")
    return int(round(value_f))


def _resolve_steps(
    entry: Mapping[str, Any],
    *,
    history_path: Path,
    validation_overlay_entry: Mapping[str, Any] | None = None,
    context: str,
) -> int:
    history_step = _read_history_final_step(history_path)
    if history_step is not None:
        return history_step
    sidecar_step = _read_validation_backfill_step(
        history_path.parent / VALIDATION_BACKFILL_FILENAME,
        context=context,
    )
    if sidecar_step is not None:
        return sidecar_step
    overlay_step = _read_validation_overlay_step(validation_overlay_entry, context=context)
    if overlay_step is not None:
        return overlay_step
    metrics = _required_mapping(entry, "tab_foundry_metrics", context=context)
    final_step = metrics.get("final_step")
    if final_step is not None:
        value_f = float(final_step)
        if not math.isfinite(value_f):
            raise RuntimeError(f"{context} final_step must be finite")
        return int(round(value_f))
    raise RuntimeError(f"{context} is missing a resolvable training step count")


def _resolve_runtime_hyperparameters(row: Mapping[str, Any]) -> tuple[int, int, int]:
    training_payload = row.get("training")
    if not isinstance(training_payload, Mapping):
        raise RuntimeError("scaling-study queue row missing training payload")
    task_batch_size = int(training_payload.get("task_batch_size") or 1)
    overrides = training_payload.get("overrides")
    runtime_payload = (
        {}
        if not isinstance(overrides, Mapping) or not isinstance(overrides.get("runtime"), Mapping)
        else {str(key): value for key, value in overrides["runtime"].items()}
    )
    max_steps = int(runtime_payload.get("max_steps") or 0)
    grad_accum_steps = int(runtime_payload.get("grad_accum_steps") or 1)
    return max_steps, grad_accum_steps, task_batch_size


def _assert_tokens_match_steps(
    *,
    steps: int,
    tokens_seen: float,
    tokens_per_step: float,
    context: str,
) -> None:
    expected_tokens_seen = float(steps) * float(tokens_per_step)
    tolerance = max(1.0, 0.01 * max(abs(tokens_seen), abs(expected_tokens_seen)))
    if abs(float(tokens_seen) - expected_tokens_seen) > tolerance:
        raise RuntimeError(
            f"{context} violates D = B_eff * S: "
            f"tokens_seen={tokens_seen}, tokens_per_step={tokens_per_step}, steps={steps}"
        )


def _collect_run_point(
    *,
    family: str,
    sweep_id: str,
    row: Mapping[str, Any],
    registry_path: Path,
    validation_overlay: Mapping[str, Mapping[str, Any]],
) -> ScalingStudyRunPoint:
    run_id = _required_str(row, "run_id", context=f"{sweep_id} row")
    validation_overlay_entry = validation_overlay.get(run_id)
    entry = load_benchmark_run_entry(run_id, path=registry_path)
    entry_context = f"benchmark registry run {run_id!r}"
    model_payload = _required_mapping(entry, "model", context=entry_context)
    row_label = _row_label_from_model_payload(model_payload)
    parameter_accounting = _required_mapping(entry, "parameter_accounting", context=entry_context)
    strict_partition = _required_mapping(parameter_accounting, "strict", context=entry_context)
    expanded_partition = _required_mapping(parameter_accounting, "expanded", context=entry_context)
    compute_accounting = _required_mapping(entry, "compute_accounting", context=entry_context)
    training_shape_summary = compute_accounting.get("training_shape_summary")
    training_shape_signature_count = _training_shape_signature_count(training_shape_summary)
    uses_reuse_train_artifact = isinstance(row.get("reuse_train_artifact"), Mapping)
    artifacts = _required_mapping(entry, "artifacts", context=entry_context)
    regime_budget = _required_mapping(entry, "regime_budget", context=entry_context)
    registry_root = _registry_root(registry_path)
    run_dir = resolve_registry_path_value(
        _required_str(artifacts, "run_dir", context=entry_context),
        root=registry_root,
    )
    history_path = resolve_registry_path_value(
        _required_str(artifacts, "history_path", context=entry_context),
        root=registry_root,
    )
    telemetry_path = run_dir / "telemetry.json"
    steps = _resolve_steps(
        entry,
        history_path=history_path,
        validation_overlay_entry=validation_overlay_entry,
        context=entry_context,
    )
    tokens_seen = _required_float(regime_budget, "tokens_seen", context=entry_context)
    tokens_per_step = _required_float(regime_budget, "tokens_per_step", context=entry_context)
    _assert_tokens_match_steps(
        steps=steps,
        tokens_seen=tokens_seen,
        tokens_per_step=tokens_per_step,
        context=entry_context,
    )
    max_steps, grad_accum_steps, task_batch_size = _resolve_runtime_hyperparameters(row)
    metrics = _required_mapping(entry, "tab_foundry_metrics", context=entry_context)
    build_spec = model_payload.get("build_spec")
    layers = model_payload.get("sandwich_layers")
    if layers is None and isinstance(build_spec, Mapping):
        layers = build_spec.get("sandwich_layers")
    if layers is None:
        raise RuntimeError(f"{entry_context} missing sandwich_layers")
    validation_loss, validation_loss_source, validation_loss_missing_reason = (
        _read_final_validation_loss(
            history_path,
            validation_overlay_entry=validation_overlay_entry,
            context=entry_context,
        )
    )
    return ScalingStudyRunPoint(
        family=family,
        sweep_id=sweep_id,
        row_order=int(row.get("order") or 0),
        row_label=row_label,
        run_id=run_id,
        d_icl=int(model_payload["d_icl"]),
        layers=int(layers),
        max_steps=max_steps if max_steps > 0 else steps,
        grad_accum_steps=grad_accum_steps,
        task_batch_size=task_batch_size,
        strict_embedding_params=_required_int(
            strict_partition, "embedding_params", context=entry_context
        ),
        strict_non_embedding_params=_required_int(
            strict_partition, "non_embedding_params", context=entry_context
        ),
        expanded_embedding_like_params=_required_int(
            expanded_partition,
            "embedding_like_params",
            context=entry_context,
        ),
        expanded_non_embedding_params=_required_int(
            expanded_partition,
            "non_embedding_params",
            context=entry_context,
        ),
        canonical_non_embedding_params=_required_int(
            parameter_accounting,
            "canonical_non_embedding_params",
            context=entry_context,
        ),
        benchmark_log_loss=_required_float(metrics, "final_log_loss", context=entry_context),
        validation_loss=validation_loss,
        validation_loss_source=validation_loss_source,
        validation_loss_missing_reason=validation_loss_missing_reason,
        steps=steps,
        tokens_seen=tokens_seen,
        tokens_per_step=tokens_per_step,
        throughput_tokens_per_second=(
            None
            if metrics.get("throughput_tokens_per_second") is None
            else float(metrics["throughput_tokens_per_second"])
        ),
        end_to_end_wall_seconds=(
            None
            if metrics.get("end_to_end_wall_seconds") is None
            else float(metrics["end_to_end_wall_seconds"])
        ),
        train_flops_per_token=_required_float(
            compute_accounting,
            "train_flops_per_token",
            context=entry_context,
        ),
        train_flops_per_step=_required_float(
            compute_accounting,
            "train_flops_per_step",
            context=entry_context,
        ),
        total_train_flops=_required_float(
            compute_accounting,
            "total_train_flops",
            context=entry_context,
        ),
        run_dir=str(run_dir),
        history_path=str(history_path),
        telemetry_path=str(telemetry_path),
        compute_training_shape_summary_present=isinstance(training_shape_summary, Mapping),
        compute_training_shape_signature_count=training_shape_signature_count,
        uses_reuse_train_artifact=uses_reuse_train_artifact,
    )


def collect_completed_scaling_points(
    *,
    config: ScalingStudyConfig,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
) -> tuple[ScalingStudyRunPoint, ...]:
    """Collect completed study points from the configured sweeps."""

    points: list[ScalingStudyRunPoint] = []
    validation_overlay = _load_validation_overlay(config)
    for sweep_ref in config.sweeps:
        queue = load_system_delta_queue(
            sweep_id=sweep_ref.sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
        raw_rows = queue.get("rows")
        if not isinstance(raw_rows, list):
            raise RuntimeError(f"sweep {sweep_ref.sweep_id!r} queue is missing rows")
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise RuntimeError(f"sweep {sweep_ref.sweep_id!r} queue row must be a mapping")
            if not _completed_benchmark_backed_row(row):
                continue
            points.append(
                _collect_run_point(
                    family=sweep_ref.family,
                    sweep_id=sweep_ref.sweep_id,
                    row=row,
                    registry_path=registry_path,
                    validation_overlay=validation_overlay,
                )
            )
    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.family,
                point.sweep_id,
                point.row_order,
                point.run_id,
            ),
        )
    )


def _group_by_key[T](values: Sequence[T], key_fn: Callable[[T], Any]) -> dict[Any, list[T]]:
    groups: dict[Any, list[T]] = {}
    for value in values:
        key = key_fn(value)
        groups.setdefault(key, []).append(value)
    return groups


def _ns_points(points: Sequence[ScalingStudyRunPoint]) -> tuple[ScalingStudyRunPoint, ...]:
    return tuple(point for point in points if point.family == "ns_core")


def _batch_points(points: Sequence[ScalingStudyRunPoint]) -> tuple[ScalingStudyRunPoint, ...]:
    return tuple(point for point in points if point.family == "batch_critical")


def _validation_backed_points(
    points: Sequence[ScalingStudyRunPoint],
) -> tuple[ScalingStudyRunPoint, ...]:
    return tuple(point for point in points if point.validation_loss is not None)


def _missing_validation_points(
    points: Sequence[ScalingStudyRunPoint],
) -> tuple[ScalingStudyRunPoint, ...]:
    return tuple(point for point in points if point.validation_loss is None)


def _validation_loss_value(point: ScalingStudyRunPoint, *, context: str) -> float:
    if point.validation_loss is None:
        raise RuntimeError(
            f"{context} requires validation_loss for run_id={point.run_id!r} "
            f"history_path={point.history_path!r}; missing_reason="
            f"{point.validation_loss_missing_reason or 'validation_loss_missing'}"
        )
    return float(point.validation_loss)


def _missing_validation_diagnostics(
    points: Sequence[ScalingStudyRunPoint],
) -> list[dict[str, Any]]:
    return [
        {
            "family": point.family,
            "sweep_id": point.sweep_id,
            "row_order": point.row_order,
            "row_label": point.row_label,
            "run_id": point.run_id,
            "history_path": point.history_path,
            "missing_reason": point.validation_loss_missing_reason or "validation_loss_missing",
        }
        for point in _missing_validation_points(points)
    ]


def _validation_coverage_payload(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    validation_points = _validation_backed_points(points)
    return {
        "total_points": len(points),
        "validation_backed_points": len(validation_points),
        "missing_validation_points": len(points) - len(validation_points),
        "ns_core_validation_backed_points": len(_validation_backed_points(_ns_points(points))),
        "batch_critical_validation_backed_points": len(
            _validation_backed_points(_batch_points(points))
        ),
        "missing": _missing_validation_diagnostics(points),
    }


def _strict_validation_missing_message(
    *,
    fit_name: str,
    points: Sequence[ScalingStudyRunPoint],
) -> str:
    missing = _missing_validation_diagnostics(points)
    missing_rows = ", ".join(
        f"{item['sweep_id']} order {item['row_order']} run_id={item['run_id']} "
        f"history_path={item['history_path']} reason={item['missing_reason']}"
        for item in missing
    )
    return (
        f"{fit_name} requires validation_loss for every selected point, but "
        f"{len(missing)} of {len(points)} points are missing validation records. "
        "These rows were likely run with runtime.val_batches=0; run posthoc validation "
        "backfill from completed checkpoints before fitting strict validation-backed "
        "alpha variables. "
        f"Missing points: {missing_rows}"
    )


def _require_validation_points(
    points: Sequence[ScalingStudyRunPoint],
    *,
    fit_name: str,
) -> tuple[ScalingStudyRunPoint, ...]:
    if _missing_validation_points(points):
        raise RuntimeError(_strict_validation_missing_message(fit_name=fit_name, points=points))
    return tuple(points)


def select_l_n_points(points: Sequence[ScalingStudyRunPoint]) -> tuple[ScalingStudyRunPoint, ...]:
    """Choose the highest completed S slice for ``L(N)``."""

    groups = _group_by_key(_ns_points(points), lambda point: point.steps)
    viable = [
        (int(steps), tuple(sorted(group, key=lambda point: point.n)))
        for steps, group in groups.items()
        if len({point.n for point in group}) >= _MIN_SLICE_POINTS
    ]
    if not viable:
        return ()
    return max(viable, key=lambda item: item[0])[1]


def select_l_d_points(points: Sequence[ScalingStudyRunPoint]) -> tuple[ScalingStudyRunPoint, ...]:
    """Choose the highest completed N slice for ``L(D)``."""

    groups = _group_by_key(_ns_points(points), lambda point: point.n)
    viable = [
        (float(n_value), tuple(sorted(group, key=lambda point: point.d)))
        for n_value, group in groups.items()
        if len({point.d for point in group}) >= _MIN_SLICE_POINTS
    ]
    if not viable:
        return ()
    return max(viable, key=lambda item: item[0])[1]


def _batch_envelope(points: Sequence[ScalingStudyRunPoint]) -> tuple[ScalingStudyRunPoint, ...]:
    validation_points = _validation_backed_points(points)
    ordered = sorted(
        validation_points,
        key=lambda point: (_validation_loss_value(point, context="Bcrit envelope"), point.b_eff),
        reverse=True,
    )
    envelope: list[ScalingStudyRunPoint] = []
    max_batch = -math.inf
    for point in ordered:
        if point.b_eff <= max_batch + _FLOAT_TOLERANCE:
            continue
        envelope.append(point)
        max_batch = point.b_eff
    return tuple(
        sorted(envelope, key=lambda point: _validation_loss_value(point, context="Bcrit envelope"))
    )


def _compute_lower_envelope(
    points: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> tuple[dict[str, Any], ...]:
    ordered = sorted(points, key=lambda point: float(point[x_key]))
    envelope: list[dict[str, Any]] = []
    best_y = math.inf
    for point in ordered:
        y_value = float(point[y_key])
        if y_value <= best_y + _FLOAT_TOLERANCE:
            envelope.append(point)
            best_y = y_value
    return tuple(envelope)


def inspect_scaling_study(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
) -> dict[str, Any]:
    """Inspect the configured study and summarize admissible completed points."""

    config = load_scaling_study_config(
        study_id=study_id,
        study_path=study_path,
        studies_root=studies_root,
    )
    points = collect_completed_scaling_points(
        config=config,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    validation_points = _validation_backed_points(points)
    navigation = build_scaling_navigation_payload(
        config=config,
        points=points,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    contract_issues = navigation.get("contract_issues")
    if isinstance(contract_issues, list) and contract_issues:
        raise RuntimeError(
            "scaling study contract validation failed:\n- "
            + "\n- ".join(str(issue) for issue in contract_issues)
        )
    return {
        "study": config.as_dict(),
        "available_points": [point.as_dict() for point in points],
        "counts": {
            "total_completed_points": len(points),
            "ns_core_completed_points": len(_ns_points(points)),
            "batch_critical_completed_points": len(_batch_points(points)),
            "validation_backed_points": len(validation_points),
            "missing_validation_points": len(points) - len(validation_points),
            "ns_core_validation_backed_points": len(_validation_backed_points(_ns_points(points))),
            "batch_critical_validation_backed_points": len(
                _validation_backed_points(_batch_points(points))
            ),
            "l_n_points": len(select_l_n_points(points)),
            "l_d_points": len(select_l_d_points(points)),
            "l_nd_points": len(_ns_points(points)),
            "l_ns_points": len(_validation_backed_points(_ns_points(points))),
            "bcrit_points": len(_batch_envelope(_batch_points(points))),
        },
        "validation_coverage": _validation_coverage_payload(points),
        "navigation": navigation,
    }


def render_scaling_study_text(payload: Mapping[str, Any]) -> str:
    """Render a short human-readable summary for `inspect` and `fit`."""

    study = payload.get("study", {})
    counts = payload.get("counts", {})
    navigation = payload.get("navigation", {})
    lines = [
        f"Study: {study.get('study_id', 'unknown')}",
        f"Phase: {study.get('phase', 'unknown')}",
        f"Completed points: {counts.get('total_completed_points', 0)}",
        f"NS core points: {counts.get('ns_core_completed_points', 0)}",
        f"Batch-critical points: {counts.get('batch_critical_completed_points', 0)}",
        f"Validation-backed points: {counts.get('validation_backed_points', 0)}",
        f"Missing validation points: {counts.get('missing_validation_points', 0)}",
    ]
    if isinstance(navigation, Mapping):
        contract = navigation.get("contract")
        if isinstance(contract, Mapping):
            lines.append(
                f"Benchmark manifest: {contract.get('benchmark_manifest_path', 'unknown')}"
            )
            lines.append(
                f"Control baseline: {contract.get('control_baseline_id', 'unknown')}"
            )
            if contract.get("corpus_ref") is not None:
                lines.append(f"Corpus ref: {contract.get('corpus_ref')}")
            if contract.get("carried_in_family_baseline_run_id") is not None:
                lines.append(
                    "Carried baseline run: "
                    f"{contract.get('carried_in_family_baseline_run_id')}"
                )
        winner = navigation.get("winner")
        if isinstance(winner, Mapping):
            winner_bits = [
                f"{winner.get('geometry_label', 'unknown')}",
                f"log_loss={float(winner['benchmark_log_loss']):.6f}",
            ]
            if winner.get("throughput_tokens_per_second") is not None:
                winner_bits.append(
                    f"throughput={float(winner['throughput_tokens_per_second']):.1f} tok/s"
                )
            if winner.get("end_to_end_wall_seconds") is not None:
                winner_bits.append(f"wall={float(winner['end_to_end_wall_seconds']):.1f}s")
            lines.append("Winner: " + ", ".join(winner_bits))
        fit_audit_state = navigation.get("fit_audit_state")
        if isinstance(fit_audit_state, Mapping):
            lines.append(
                "Fit/audit state: "
                f"overlay={bool(fit_audit_state.get('validation_overlay_exists'))} "
                f"fit={bool(fit_audit_state.get('fit_summary_exists'))} "
                f"audit={bool(fit_audit_state.get('audit_summary_exists'))}"
            )
        completeness = navigation.get("completeness")
        if isinstance(completeness, Mapping):
            lines.append(
                "Full-scope completeness: "
                f"{bool(completeness.get('all_expected_points_present'))}"
            )
    fit_summary = payload.get("fit_summary")
    if isinstance(fit_summary, Mapping):
        lines.append("Fits:")
        for fit_name in sorted(str(key) for key in fit_summary.keys()):
            fit_payload = fit_summary.get(fit_name)
            if not isinstance(fit_payload, Mapping):
                continue
            stats = fit_payload.get("stats")
            n_points = stats.get("n_points") if isinstance(stats, Mapping) else None
            lines.append(f"  {fit_name}: kind={fit_payload.get('fit_kind')} n_points={n_points}")
    return "\n".join(lines)


def _positive_vector(values: Sequence[float], *, context: str) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        raise RuntimeError(f"{context} requires at least one value")
    if not np.all(np.isfinite(array)) or not np.all(array > 0.0):
        raise RuntimeError(f"{context} requires strictly positive finite values")
    return array


def _bounded_log_parameter(value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        return 0.0
    return min(max(resolved, _MIN_LOG_POSITIVE), _MAX_LOG_PARAMETER_SEED)


def _log_space_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if not np.all(y_true > 0.0) or not np.all(y_pred > 0.0):
        return None
    log_true = np.log(y_true)
    log_pred = np.log(y_pred)
    total = np.sum((log_true - np.mean(log_true)) ** 2)
    if total <= _MIN_POSITIVE:
        return None
    return float(1.0 - np.sum((log_true - log_pred) ** 2) / total)


def _fit_stats(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_count: int,
) -> dict[str, Any]:
    residuals = y_true - y_pred
    rss = float(np.sum(residuals**2))
    rmse = float(math.sqrt(rss / float(max(1, y_true.size))))
    mae = float(np.mean(np.abs(residuals)))
    log_r2 = _log_space_r2(y_true, y_pred)
    effective_rss = max(rss, _MIN_POSITIVE)
    aic = float(
        y_true.size * math.log(effective_rss / float(max(1, y_true.size))) + 2.0 * param_count
    )
    bic = float(
        y_true.size * math.log(effective_rss / float(max(1, y_true.size)))
        + float(param_count) * math.log(float(max(1, y_true.size)))
    )
    return {
        "n_points": int(y_true.size),
        "rss": rss,
        "rmse": rmse,
        "mae": mae,
        "log_space_r2": log_r2,
        "aic": aic,
        "bic": bic,
        "residuals": residuals.tolist(),
    }


def _numeric_jacobian(
    theta: np.ndarray,
    *,
    model_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    baseline = model_fn(theta)
    jacobian = np.zeros((baseline.size, theta.size), dtype=float)
    for index in range(theta.size):
        step = 1.0e-5 * max(1.0, abs(float(theta[index])))
        candidate = theta.copy()
        candidate[index] += step
        forward = model_fn(candidate)
        jacobian[:, index] = (forward - baseline) / step
    return jacobian


def _fit_nonlinear_positive_model(
    *,
    y_true: np.ndarray,
    parameter_names: Sequence[str],
    theta_seed: Sequence[float],
    predict_fn: Callable[[dict[str, float]], np.ndarray],
) -> dict[str, Any]:
    theta = np.asarray([float(value) for value in theta_seed], dtype=float)
    if theta.size != len(tuple(parameter_names)):
        raise RuntimeError("theta seed size must match parameter_names")

    def _parameters_from_theta(raw_theta: np.ndarray) -> dict[str, float]:
        return {
            name: float(math.exp(_bounded_log_parameter(float(value))))
            for name, value in zip(parameter_names, raw_theta, strict=False)
        }

    def _model(raw_theta: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return predict_fn(_parameters_from_theta(raw_theta))

    damping = 1.0e-3
    last_rss: float | None = None
    for _ in range(_MAX_ITERATIONS):
        y_pred = _model(theta)
        residuals = y_true - y_pred
        rss = float(np.sum(residuals**2))
        jacobian = _numeric_jacobian(theta, model_fn=_model)
        lhs = jacobian.T @ jacobian + damping * np.eye(theta.size)
        rhs = jacobian.T @ residuals
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        if float(np.linalg.norm(delta)) <= _STEP_TOLERANCE:
            last_rss = rss
            break
        accepted = False
        step_scale = 1.0
        while step_scale >= _MIN_LINE_SEARCH_SCALE:
            candidate = theta + step_scale * delta
            candidate_rss = float(np.sum((y_true - _model(candidate)) ** 2))
            if candidate_rss + _FLOAT_TOLERANCE < rss:
                theta = candidate
                damping = max(damping * 0.5, 1.0e-6)
                last_rss = candidate_rss
                accepted = True
                break
            step_scale *= 0.5
            damping = min(damping * 2.0, 1.0e6)
        if not accepted:
            last_rss = rss
            break
    final_parameters = _parameters_from_theta(theta)
    final_pred = _model(theta)
    final_stats = _fit_stats(y_true=y_true, y_pred=final_pred, param_count=theta.size)
    final_jacobian = _numeric_jacobian(theta, model_fn=_model)
    covariance: np.ndarray | None = None
    if y_true.size > theta.size:
        try:
            sigma2 = max(
                float(last_rss if last_rss is not None else final_stats["rss"]), _MIN_POSITIVE
            ) / float(y_true.size - theta.size)
            covariance = sigma2 * np.linalg.inv(final_jacobian.T @ final_jacobian)
        except np.linalg.LinAlgError:
            covariance = None
    standard_errors: dict[str, float | None] = {}
    confidence_intervals: dict[str, dict[str, float] | None] = {}

    def _safe_positive_exp(value: float) -> float | None:
        try:
            resolved = float(math.exp(float(value)))
        except OverflowError:
            return None
        return resolved if math.isfinite(resolved) else None

    if covariance is None:
        for name in parameter_names:
            standard_errors[name] = None
            confidence_intervals[name] = None
    else:
        theta_standard_errors = np.sqrt(np.diag(covariance))
        for name, theta_value, theta_se in zip(
            parameter_names, theta, theta_standard_errors, strict=False
        ):
            parameter_value = float(math.exp(float(theta_value)))
            standard_errors[name] = float(parameter_value * theta_se)
            lower = _safe_positive_exp(float(theta_value - 1.96 * theta_se))
            upper = _safe_positive_exp(float(theta_value + 1.96 * theta_se))
            confidence_intervals[name] = {
                "lower": parameter_value if lower is None else lower,
                "upper": parameter_value if upper is None else upper,
            }
    return {
        "fit_kind": "nonlinear_positive_least_squares",
        "parameters": final_parameters,
        "stats": {
            **final_stats,
            "parameter_standard_errors": standard_errors,
            "parameter_confidence_intervals_95": confidence_intervals,
        },
        "predictions": final_pred.tolist(),
    }


def _power_law_seed(samples: Sequence[tuple[float, float]]) -> tuple[float, float]:
    power_fit = fit_power_law(samples)
    exponent_seed = max(0.02, -float(power_fit.exponent))
    scale_seed = float(
        math.exp(_bounded_log_parameter(float(power_fit.intercept) / exponent_seed))
    )
    return scale_seed, exponent_seed


def fit_loss_vs_scale(
    *,
    name: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    scale_name: str,
    alpha_name: str,
) -> dict[str, Any]:
    """Fit ``L(x) = E + (x_c / x)^alpha``."""

    x = _positive_vector(x_values, context=name)
    y = _positive_vector(y_values, context=name)
    floor_seed = max(_MIN_POSITIVE, 0.9 * float(np.min(y)))
    shifted = np.maximum(y - floor_seed, _MIN_POSITIVE)
    scale_seed, exponent_seed = _power_law_seed(
        tuple(zip(x.tolist(), shifted.tolist(), strict=False))
    )
    fit = _fit_nonlinear_positive_model(
        y_true=y,
        parameter_names=("irreducible_loss", scale_name, alpha_name),
        theta_seed=(
            math.log(floor_seed),
            math.log(max(scale_seed, _MIN_POSITIVE)),
            math.log(max(exponent_seed, 0.02)),
        ),
        predict_fn=lambda parameters: (
            parameters["irreducible_loss"] + (parameters[scale_name] / x) ** parameters[alpha_name]
        ),
    )
    return {
        "name": name,
        "x_axis": scale_name,
        **fit,
    }


def _target_values(
    points: Sequence[ScalingStudyRunPoint],
    *,
    target_key: str,
    context: str,
) -> list[float]:
    values: list[float] = []
    for point in points:
        value = getattr(point, target_key)
        if value is None:
            raise RuntimeError(
                f"{context} target {target_key!r} is missing for run_id={point.run_id!r} "
                f"history_path={point.history_path!r}; missing_reason="
                f"{point.validation_loss_missing_reason or 'target_missing'}"
            )
        values.append(float(value))
    return values


def fit_loss_vs_nd(
    *,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
) -> dict[str, Any]:
    """Fit the Kaplan-style joint law ``L(N, D)``."""

    n = _positive_vector([point.n for point in points], context="L(N,D) N")
    d = _positive_vector([point.d for point in points], context="L(N,D) D")
    target_values = _target_values(points, target_key=target_key, context="L(N,D)")
    y = _positive_vector(target_values, context="L(N,D) loss")
    floor_seed = max(_MIN_POSITIVE, 0.9 * float(np.min(y)))
    alpha_n_seed = max(
        0.02,
        -fit_power_law(
            tuple((point.n, value) for point, value in zip(points, target_values, strict=False))
        ).exponent,
    )
    alpha_d_seed = max(
        0.02,
        -fit_power_law(
            tuple((point.d, value) for point, value in zip(points, target_values, strict=False))
        ).exponent,
    )
    fit = _fit_nonlinear_positive_model(
        y_true=y,
        parameter_names=("irreducible_loss", "Nc", "Dc", "alpha_n", "alpha_d"),
        theta_seed=(
            math.log(floor_seed),
            math.log(float(np.median(n))),
            math.log(float(np.median(d))),
            math.log(alpha_n_seed),
            math.log(alpha_d_seed),
        ),
        predict_fn=lambda parameters: (
            parameters["irreducible_loss"]
            + (
                (parameters["Nc"] / n) ** (parameters["alpha_n"] / parameters["alpha_d"])
                + (parameters["Dc"] / d)
            )
            ** parameters["alpha_d"]
        ),
    )
    return {
        "name": "L(N,D)",
        "x_axis": "canonical_non_embedding_params,tokens_seen",
        **fit,
    }


def fit_loss_vs_ns(
    *,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
) -> dict[str, Any]:
    """Fit the paper-style joint law ``L(N, S)``."""

    n = _positive_vector([point.n for point in points], context="L(N,S) N")
    s = _positive_vector([point.s for point in points], context="L(N,S) S")
    target_values = _target_values(points, target_key=target_key, context="L(N,S)")
    y = _positive_vector(target_values, context="L(N,S) loss")
    floor_seed = max(_MIN_POSITIVE, 0.9 * float(np.min(y)))
    alpha_n_seed = max(
        0.02,
        -fit_power_law(
            tuple((point.n, value) for point, value in zip(points, target_values, strict=False))
        ).exponent,
    )
    alpha_s_seed = max(
        0.02,
        -fit_power_law(
            tuple((point.s, value) for point, value in zip(points, target_values, strict=False))
        ).exponent,
    )
    fit = _fit_nonlinear_positive_model(
        y_true=y,
        parameter_names=("irreducible_loss", "Nc", "Sc", "alpha_n", "alpha_s"),
        theta_seed=(
            math.log(floor_seed),
            math.log(float(np.median(n))),
            math.log(float(np.median(s))),
            math.log(alpha_n_seed),
            math.log(alpha_s_seed),
        ),
        predict_fn=lambda parameters: (
            parameters["irreducible_loss"]
            + (
                (parameters["Nc"] / n) ** (parameters["alpha_n"] / parameters["alpha_s"])
                + (parameters["Sc"] / s)
            )
            ** parameters["alpha_s"]
        ),
    )
    return {
        "name": "L(N,S)",
        "x_axis": "canonical_non_embedding_params,steps",
        **fit,
    }


def _validate_c_axis_points(
    points: Sequence[ScalingStudyRunPoint],
    *,
    context: str,
) -> None:
    missing_shape_summary = [
        point
        for point in points
        if (
            not point.uses_reuse_train_artifact
            and (
                not point.compute_training_shape_summary_present
                or point.compute_training_shape_signature_count <= 0
            )
        )
    ]
    if missing_shape_summary:
        examples = ", ".join(
            f"{point.run_id} row={point.row_order} label={point.row_label}"
            for point in missing_shape_summary[:_C_AXIS_ERROR_EXAMPLE_LIMIT]
        )
        suffix = (
            "..."
            if len(missing_shape_summary) > _C_AXIS_ERROR_EXAMPLE_LIMIT
            else ""
        )
        raise RuntimeError(
            f"{context} requires observed compute_accounting.training_shape_summary "
            f"for every C-axis point; missing for {len(missing_shape_summary)} point(s): "
            f"{examples}{suffix}"
        )

    ns_groups: dict[str, list[ScalingStudyRunPoint]] = {}
    for point in points:
        if point.family != "ns_core":
            continue
        ns_groups.setdefault(point.row_label, []).append(point)
    for row_label, group_points in sorted(ns_groups.items()):
        ordered = sorted(group_points, key=lambda point: (point.steps, point.c, point.run_id))
        for previous in ordered:
            for current in ordered:
                if current.steps <= previous.steps:
                    continue
                tolerated_floor = previous.c * (1.0 - _C_AXIS_MONOTONIC_RELATIVE_TOLERANCE)
                if current.c >= tolerated_floor:
                    continue
                raise RuntimeError(
                    f"{context} C-axis monotonicity check failed for ns_core row "
                    f"{row_label}: run_id={current.run_id!r} has steps={current.steps} "
                    f"C={current.c:g}, below earlier run_id={previous.run_id!r} "
                    f"steps={previous.steps} C={previous.c:g}"
                )


def fit_bcrit(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    """Fit ``Bcrit(L) = B* / L^(1/alpha_B)`` on the batch-envelope points."""

    validation_points = _require_validation_points(points, fit_name="Bcrit(L)")
    envelope = _batch_envelope(validation_points)
    loss = _positive_vector(
        [_validation_loss_value(point, context="Bcrit(L)") for point in envelope],
        context="Bcrit loss",
    )
    batch = _positive_vector([point.b_eff for point in envelope], context="Bcrit batch")
    power_fit = fit_power_law(
        tuple(
            (_validation_loss_value(point, context="Bcrit(L)"), point.b_eff) for point in envelope
        )
    )
    exponent_seed = float(power_fit.exponent)
    alpha_b_seed = max(0.02, -1.0 / exponent_seed) if exponent_seed < 0.0 else 0.2
    alpha_b_seed = min(alpha_b_seed, 10.0)
    log_b_star_seed = math.log(float(np.median(batch))) + (
        math.log(float(np.median(loss))) / alpha_b_seed
    )
    if math.isfinite(power_fit.intercept) and abs(power_fit.intercept) <= _MAX_LOG_PARAMETER_SEED:
        log_b_star_seed = float(power_fit.intercept)
    fit = _fit_nonlinear_positive_model(
        y_true=batch,
        parameter_names=("B_star", "alpha_b"),
        theta_seed=(
            _bounded_log_parameter(log_b_star_seed),
            math.log(alpha_b_seed),
        ),
        predict_fn=lambda parameters: (
            parameters["B_star"] / (loss ** (1.0 / parameters["alpha_b"]))
        ),
    )
    return {
        "name": "Bcrit(L)",
        "points": [point.as_dict() for point in envelope],
        **fit,
    }


def _cmin_points(
    *,
    points: Sequence[ScalingStudyRunPoint],
    bcrit_fit: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    validation_points = _require_validation_points(points, fit_name="L(Cmin)")
    parameters = _required_mapping(bcrit_fit, "parameters", context="Bcrit fit")
    b_star = _required_float(parameters, "B_star", context="Bcrit fit")
    alpha_b = _required_float(parameters, "alpha_b", context="Bcrit fit")
    adjusted: list[dict[str, Any]] = []
    for point in validation_points:
        bcrit = b_star / (_validation_loss_value(point, context="L(Cmin)") ** (1.0 / alpha_b))
        cmin = float(point.c) / (1.0 + float(point.b_eff) / max(bcrit, _MIN_POSITIVE))
        adjusted.append(
            {
                **point.as_dict(),
                "c": float(point.c),
                "cmin": cmin,
            }
        )
    return _compute_lower_envelope(adjusted, x_key="cmin", y_key="benchmark_log_loss")


def _derived_relations(
    *,
    l_nd_fit: Mapping[str, Any] | None,
    l_c_fit: Mapping[str, Any] | None,
    l_cmin_fit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if l_nd_fit is None:
        return {}
    parameters = _required_mapping(l_nd_fit, "parameters", context="L(N,D) fit")
    alpha_n = _required_float(parameters, "alpha_n", context="L(N,D) fit")
    alpha_d = _required_float(parameters, "alpha_d", context="L(N,D) fit")
    implied_alpha_c = (alpha_n * alpha_d) / max(alpha_n + alpha_d, _MIN_POSITIVE)
    payload: dict[str, Any] = {
        "alpha_c_implied_from_l_nd": implied_alpha_c,
        "n_opt_compute_exponent": alpha_d / max(alpha_n + alpha_d, _MIN_POSITIVE),
        "d_opt_compute_exponent": alpha_n / max(alpha_n + alpha_d, _MIN_POSITIVE),
    }
    if l_c_fit is not None:
        direct_alpha_c = _required_float(
            _required_mapping(l_c_fit, "parameters", context="L(C) fit"),
            "alpha_c",
            context="L(C) fit",
        )
        payload["direct_vs_implied_alpha_c"] = {
            "direct": direct_alpha_c,
            "implied": implied_alpha_c,
            "delta": direct_alpha_c - implied_alpha_c,
        }
    if l_cmin_fit is not None:
        direct_alpha_cmin = _required_float(
            _required_mapping(l_cmin_fit, "parameters", context="L(Cmin) fit"),
            "alpha_cmin",
            context="L(Cmin) fit",
        )
        payload["direct_vs_implied_alpha_cmin"] = {
            "direct": direct_alpha_cmin,
            "implied": implied_alpha_c,
            "delta": direct_alpha_cmin - implied_alpha_c,
        }
    return payload


def _plot_loglog_fit(
    *,
    out_path: Path,
    x_values: Sequence[float],
    y_values: Sequence[float],
    predicted_y: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        zip(x_values, y_values, predicted_y, strict=False),
        key=lambda item: float(item[0]),
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        color="#0f766e",
        label="observed",
    )
    ax.plot(
        [item[0] for item in ordered],
        [item[2] for item in ordered],
        color="#b91c1c",
        linewidth=2.0,
        label="fit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)


def _plot_residuals(
    *,
    out_path: Path,
    x_values: Sequence[float],
    residuals: Sequence[float],
    title: str,
    x_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.6)
    ax.scatter(x_values, residuals, color="#1d4ed8")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)


def _plot_joint_surface(
    *,
    out_path: Path,
    x_values: Sequence[float],
    y_values: Sequence[float],
    z_values: Sequence[float],
    predicted_z: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_values, y_values, z_values, color="#0f766e", label="observed")
    ax.scatter(x_values, y_values, predicted_z, color="#b91c1c", marker="^", label="fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_zscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)


def _plot_compute_frontier(
    *,
    out_path: Path,
    points: Sequence[dict[str, Any]],
    predicted_y: Sequence[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(
        zip(points, predicted_y, strict=False),
        key=lambda item: float(item[0]["cmin"]),
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(
        [float(point["c"]) for point, _ in ordered],
        [float(point["benchmark_log_loss"]) for point, _ in ordered],
        color="#94a3b8",
        label="all completed points",
    )
    ax.plot(
        [float(point["cmin"]) for point, _ in ordered],
        [float(point["benchmark_log_loss"]) for point, _ in ordered],
        color="#0f766e",
        linewidth=2.0,
        label="lower envelope",
    )
    ax.plot(
        [float(point["cmin"]) for point, _ in ordered],
        [float(prediction) for _, prediction in ordered],
        color="#b91c1c",
        linewidth=2.0,
        label="L(Cmin) fit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("L(Cmin) compute frontier")
    ax.set_xlabel("Cmin")
    ax.set_ylabel("benchmark log loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=144)
    plt.close(fig)


def _study_markdown(
    *,
    config: ScalingStudyConfig,
    fit_scope: FitScope,
    fit_summary: Mapping[str, Any],
    derived_relations: Mapping[str, Any],
    artifact_root: Path,
) -> str:
    lines = [
        f"# {config.study_id}",
        "",
        f"- Phase: `{config.phase}`",
        f"- Fit scope: `{fit_scope}`",
        f"- Artifact root: `{normalize_repo_relative_path(artifact_root)}`",
        "",
        "## Fits",
    ]
    for fit_name in sorted(fit_summary.keys()):
        fit_payload = fit_summary[fit_name]
        if not isinstance(fit_payload, Mapping):
            continue
        stats = fit_payload.get("stats")
        parameters = fit_payload.get("parameters")
        lines.append(f"- `{fit_name}`")
        if isinstance(parameters, Mapping):
            lines.append(f"  parameters: `{json.dumps(parameters, sort_keys=True)}`")
        if isinstance(stats, Mapping):
            lines.append(
                f"  rss=`{stats.get('rss')}` rmse=`{stats.get('rmse')}` "
                f"log_r2=`{stats.get('log_space_r2')}`"
            )
    if derived_relations:
        lines.extend(
            [
                "",
                "## Derived Relations",
                f"- `{json.dumps(dict(derived_relations), sort_keys=True)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def fit_scaling_study(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
    registry_path: Path,
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
    out_root: Path | None = None,
    fit_scope: str = FIT_SCOPE_ALL,
) -> dict[str, Any]:
    """Fit the configured scaling study and write JSON, PNG, Markdown, and W&B payloads."""

    normalized_fit_scope = _normalize_fit_scope(fit_scope)
    config = load_scaling_study_config(
        study_id=study_id,
        study_path=study_path,
        studies_root=studies_root,
    )
    all_points = collect_completed_scaling_points(
        config=config,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    navigation = build_scaling_navigation_payload(
        config=config,
        points=all_points,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    contract_issues = navigation.get("contract_issues")
    if isinstance(contract_issues, list) and contract_issues:
        raise RuntimeError(
            "scaling study contract validation failed:\n- "
            + "\n- ".join(str(issue) for issue in contract_issues)
        )
    points = _ns_points(all_points) if normalized_fit_scope == FIT_SCOPE_NS_ONLY else all_points
    artifact_root = (
        out_root.expanduser().resolve()
        if out_root is not None
        else config.output_root_path(root=repo_root())
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    plots_root = artifact_root / "plots"
    if normalized_fit_scope == FIT_SCOPE_NS_ONLY:
        for stale_plot in (
            "bcrit.png",
            "bcritl_residuals.png",
            "lcmin_residuals.png",
            "l_cmin_frontier.png",
        ):
            (plots_root / stale_plot).unlink(missing_ok=True)
    l_n_points = select_l_n_points(points)
    l_d_points = select_l_d_points(points)
    l_nd_points = _ns_points(points)
    l_ns_points = _require_validation_points(_ns_points(points), fit_name="L(N,S)")
    batch_points = _batch_points(points)
    completeness = cast(Mapping[str, Any], navigation.get("completeness", {}))
    if (
        normalized_fit_scope == FIT_SCOPE_ALL
        and batch_points
        and not bool(completeness.get("all_expected_points_present"))
    ):
        raise RuntimeError(
            "fit_scope='all' requires the full expected study surface before fitting; "
            f"expected={completeness.get('expected_counts_by_family')} "
            f"actual={completeness.get('actual_counts_by_family')}"
        )
    cmin_points: tuple[dict[str, Any], ...] = ()
    fit_summary: dict[str, Any] = {}
    fit_summary["L(N)"] = {
        **fit_loss_vs_scale(
            name="L(N)",
            x_values=[point.n for point in l_n_points],
            y_values=[point.benchmark_log_loss for point in l_n_points],
            scale_name="Nc",
            alpha_name="alpha_n",
        ),
        "points": [point.as_dict() for point in l_n_points],
        "target_key": "benchmark_log_loss",
    }
    fit_summary["L(D)"] = {
        **fit_loss_vs_scale(
            name="L(D)",
            x_values=[point.d for point in l_d_points],
            y_values=[point.benchmark_log_loss for point in l_d_points],
            scale_name="Dc",
            alpha_name="alpha_d",
        ),
        "points": [point.as_dict() for point in l_d_points],
        "target_key": "benchmark_log_loss",
    }
    _validate_c_axis_points(points, context="L(C)")
    fit_summary["L(C)"] = {
        **fit_loss_vs_scale(
            name="L(C)",
            x_values=[point.c for point in points],
            y_values=[point.benchmark_log_loss for point in points],
            scale_name="Cc",
            alpha_name="alpha_c",
        ),
        "points": [point.as_dict() for point in points],
        "target_key": "benchmark_log_loss",
    }
    fit_summary["L(N,D)"] = {
        **fit_loss_vs_nd(points=l_nd_points, target_key="benchmark_log_loss"),
        "points": [point.as_dict() for point in l_nd_points],
        "target_key": "benchmark_log_loss",
    }
    fit_summary["L(N,S)"] = {
        **fit_loss_vs_ns(points=l_ns_points, target_key="validation_loss"),
        "points": [point.as_dict() for point in l_ns_points],
        "target_key": "validation_loss",
    }
    if normalized_fit_scope == FIT_SCOPE_ALL:
        if not batch_points:
            raise RuntimeError(
                "fit_scope='all' requires completed batch_critical points for Bcrit(L) "
                "and L(Cmin); use fit_scope='ns-only' while batch-critical runs are pending"
            )
        bcrit_fit = fit_bcrit(batch_points)
        cmin_points = _cmin_points(points=points, bcrit_fit=bcrit_fit)
        fit_summary["Bcrit(L)"] = {
            **bcrit_fit,
            "target_key": "b_eff",
        }
        fit_summary["L(Cmin)"] = {
            **fit_loss_vs_scale(
                name="L(Cmin)",
                x_values=[float(point["cmin"]) for point in cmin_points],
                y_values=[float(point["benchmark_log_loss"]) for point in cmin_points],
                scale_name="Ccmin",
                alpha_name="alpha_cmin",
            ),
            "points": list(cmin_points),
            "target_key": "benchmark_log_loss",
        }
    derived_relations = _derived_relations(
        l_nd_fit=fit_summary.get("L(N,D)"),
        l_c_fit=fit_summary.get("L(C)"),
        l_cmin_fit=fit_summary.get("L(Cmin)"),
    )
    alphas = {
        "alpha_n": _required_float(
            _required_mapping(fit_summary["L(N)"], "parameters", context="L(N)"),
            "alpha_n",
            context="L(N)",
        ),
        "alpha_d": _required_float(
            _required_mapping(fit_summary["L(D)"], "parameters", context="L(D)"),
            "alpha_d",
            context="L(D)",
        ),
        "alpha_s": _required_float(
            _required_mapping(fit_summary["L(N,S)"], "parameters", context="L(N,S)"),
            "alpha_s",
            context="L(N,S)",
        ),
        "alpha_c": _required_float(
            _required_mapping(fit_summary["L(C)"], "parameters", context="L(C)"),
            "alpha_c",
            context="L(C)",
        ),
    }
    if "L(Cmin)" in fit_summary:
        alphas["alpha_cmin"] = _required_float(
            _required_mapping(fit_summary["L(Cmin)"], "parameters", context="L(Cmin)"),
            "alpha_cmin",
            context="L(Cmin)",
        )
    if "Bcrit(L)" in fit_summary:
        alphas["alpha_b"] = _required_float(
            _required_mapping(fit_summary["Bcrit(L)"], "parameters", context="Bcrit(L)"),
            "alpha_b",
            context="Bcrit(L)",
        )
    _plot_loglog_fit(
        out_path=plots_root / "l_n.png",
        x_values=[point.n for point in l_n_points],
        y_values=[point.benchmark_log_loss for point in l_n_points],
        predicted_y=fit_summary["L(N)"]["predictions"],
        title="L(N)",
        x_label="canonical non-embedding params",
        y_label="benchmark log loss",
    )
    _plot_loglog_fit(
        out_path=plots_root / "l_d.png",
        x_values=[point.d for point in l_d_points],
        y_values=[point.benchmark_log_loss for point in l_d_points],
        predicted_y=fit_summary["L(D)"]["predictions"],
        title="L(D)",
        x_label="tokens seen",
        y_label="benchmark log loss",
    )
    _plot_loglog_fit(
        out_path=plots_root / "l_c.png",
        x_values=[point.c for point in points],
        y_values=[point.benchmark_log_loss for point in points],
        predicted_y=fit_summary["L(C)"]["predictions"],
        title="L(C)",
        x_label="training FLOPs",
        y_label="benchmark log loss",
    )
    if "Bcrit(L)" in fit_summary:
        _plot_loglog_fit(
            out_path=plots_root / "bcrit.png",
            x_values=[
                float(point["validation_loss"]) for point in fit_summary["Bcrit(L)"]["points"]
            ],
            y_values=[point["tokens_per_step"] for point in fit_summary["Bcrit(L)"]["points"]],
            predicted_y=fit_summary["Bcrit(L)"]["predictions"],
            title="Bcrit(L)",
            x_label="validation loss",
            y_label="effective batch tokens",
        )
    _plot_joint_surface(
        out_path=plots_root / "l_nd_surface.png",
        x_values=[point.n for point in l_nd_points],
        y_values=[point.d for point in l_nd_points],
        z_values=[point.benchmark_log_loss for point in l_nd_points],
        predicted_z=fit_summary["L(N,D)"]["predictions"],
        title="L(N,D)",
        x_label="canonical non-embedding params",
        y_label="tokens seen",
        z_label="benchmark log loss",
    )
    _plot_joint_surface(
        out_path=plots_root / "l_ns_surface.png",
        x_values=[point.n for point in l_ns_points],
        y_values=[point.s for point in l_ns_points],
        z_values=[_validation_loss_value(point, context="L(N,S) plot") for point in l_ns_points],
        predicted_z=fit_summary["L(N,S)"]["predictions"],
        title="L(N,S)",
        x_label="canonical non-embedding params",
        y_label="steps",
        z_label="validation loss",
    )
    for fit_name, fit_payload in fit_summary.items():
        stats = fit_payload.get("stats")
        if not isinstance(stats, Mapping):
            continue
        residuals = stats.get("residuals")
        if not isinstance(residuals, list):
            continue
        if fit_name == "L(N)":
            x_values = [point.n for point in l_n_points]
            x_label = "canonical non-embedding params"
        elif fit_name == "L(D)":
            x_values = [point.d for point in l_d_points]
            x_label = "tokens seen"
        elif fit_name == "L(C)":
            x_values = [point.c for point in points]
            x_label = "training FLOPs"
        elif fit_name == "L(N,D)":
            x_values = [point.n * point.d for point in l_nd_points]
            x_label = "N * D"
        elif fit_name == "L(N,S)":
            x_values = [point.n * point.s for point in l_ns_points]
            x_label = "N * S"
        elif fit_name == "Bcrit(L)":
            x_values = [float(point["validation_loss"]) for point in fit_payload["points"]]
            x_label = "validation loss"
        elif fit_name == "L(Cmin)":
            x_values = [float(point["cmin"]) for point in cmin_points]
            x_label = "Cmin"
        else:  # pragma: no cover - fit summary names are fixed above.
            raise RuntimeError(f"unsupported fit residual plot: {fit_name}")
        _plot_residuals(
            out_path=plots_root
            / f"{fit_name.lower().replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')}_residuals.png",
            x_values=x_values,
            residuals=[float(value) for value in residuals],
            title=f"{fit_name} residuals",
            x_label=x_label,
        )
    if "L(Cmin)" in fit_summary:
        _plot_compute_frontier(
            out_path=plots_root / "l_cmin_frontier.png",
            points=cmin_points,
            predicted_y=fit_summary["L(Cmin)"]["predictions"],
        )
    wandb_summary_payload = {
        "research_scaling": {
            config.study_id: {
                "artifact_root": str(artifact_root),
                "alphas": alphas,
            }
        }
    }
    wandb_updates: list[dict[str, Any]] = []
    for telemetry_path in sorted({point.telemetry_path for point in points}):
        updated = posthoc_update_wandb_summary(
            telemetry_path=Path(telemetry_path),
            payload=wandb_summary_payload,
        )
        wandb_updates.append(
            {
                "telemetry_path": telemetry_path,
                "updated": bool(updated),
            }
        )
    fit_payload = {
        "study": config.as_dict(),
        "fit_scope": normalized_fit_scope,
        "navigation": navigation,
        "counts": {
            "total_completed_points": len(points),
            "all_completed_points": len(all_points),
            "ns_core_completed_points": len(l_nd_points),
            "batch_critical_completed_points": len(batch_points),
            "validation_backed_points": len(_validation_backed_points(points)),
            "missing_validation_points": len(_missing_validation_points(points)),
            "ns_core_validation_backed_points": len(l_ns_points),
            "batch_critical_validation_backed_points": len(_validation_backed_points(batch_points)),
        },
        "validation_coverage": _validation_coverage_payload(points),
        "available_points": [point.as_dict() for point in points],
        "fit_summary": fit_summary,
        "alphas": alphas,
        "derived_kaplan_relations": derived_relations,
        "artifact_paths": {
            "artifact_root": str(artifact_root),
            "json_path": str(artifact_root / "fit_summary.json"),
            "markdown_path": str(artifact_root / "summary.md"),
            "plots_root": str(plots_root),
            "wandb_summary_path": str(artifact_root / "wandb_summary.json"),
        },
        "wandb_updates": wandb_updates,
    }
    write_json(artifact_root / "fit_summary.json", fit_payload)
    (artifact_root / "summary.md").write_text(
        _study_markdown(
            config=config,
            fit_scope=normalized_fit_scope,
            fit_summary=fit_summary,
            derived_relations=derived_relations,
            artifact_root=artifact_root,
        ),
        encoding="utf-8",
    )
    write_json(artifact_root / "wandb_summary.json", wandb_summary_payload)
    return fit_payload
