"""Canonical programmatic surface for hardware architecture baseline freezing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import tab_foundry.benchmark_registry as read_benchmark_registry
import tab_foundry.hardware_architecture_registry as read_hardware_architecture_registry
from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.width_depth_scaling import (
    fit_affine_width_depth_parameter_bridge,
    WidthDepthParameterPoint,
)
from tab_foundry.bench.registry.summary_metrics import (
    ensure_mapping,
    ensure_non_empty_string,
    ensure_optional_finite_number,
    ensure_optional_positive_int,
)
from tab_foundry.hardware_profiles import build_hardware_profile_id
from tab_foundry.registry.common import copy_jsonable as _copy_jsonable
from tab_foundry.training.health import health_check


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


def _benchmark_timing_excerpt(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {
        "wall_elapsed_seconds": payload.get("wall_elapsed_seconds"),
        "mean_checkpoint_elapsed_seconds": payload.get("mean_checkpoint_elapsed_seconds"),
        "max_checkpoint_elapsed_seconds": payload.get("max_checkpoint_elapsed_seconds"),
        "attempted_checkpoint_count": payload.get("attempted_checkpoint_count"),
        "successful_checkpoint_count": payload.get("successful_checkpoint_count"),
        "failed_checkpoint_count": payload.get("failed_checkpoint_count"),
        "requested_device": payload.get("requested_device"),
        "resolved_device": payload.get("resolved_device"),
        "host_fingerprint": payload.get("host_fingerprint"),
    }


def _inference_timing_excerpt(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {
        "fixture_id": payload.get("fixture_id"),
        "requested_device": payload.get("requested_device"),
        "resolved_device": payload.get("resolved_device"),
        "device_type": payload.get("device_type"),
        "raw_device_name": payload.get("raw_device_name"),
        "gpu_class": payload.get("gpu_class"),
        "vram_class_gb": payload.get("vram_class_gb"),
        "hardware_profile_id": payload.get("hardware_profile_id"),
        "warmup_iterations": payload.get("warmup_iterations"),
        "measured_iterations": payload.get("measured_iterations"),
        "n_train": payload.get("n_train"),
        "n_test": payload.get("n_test"),
        "n_features": payload.get("n_features"),
        "num_classes": payload.get("num_classes"),
        "mean_ms": payload.get("mean_ms"),
        "p50_ms": payload.get("p50_ms"),
        "p95_ms": payload.get("p95_ms"),
        "max_ms": payload.get("max_ms"),
        "total_measured_seconds": payload.get("total_measured_seconds"),
    }


def _preferred_architecture_payload(run_entry: Mapping[str, Any]) -> dict[str, Any]:
    model = ensure_mapping(run_entry.get("model"), context="run_entry.model")
    build_spec = (
        ensure_mapping(model.get("build_spec"), context="run_entry.model.build_spec")
        if isinstance(model.get("build_spec"), Mapping)
        else None
    )
    payload = {
        "arch": ensure_non_empty_string(model.get("arch"), context="run_entry.model.arch"),
        "d_icl": int(model["d_icl"]),
        "head_hidden_dim": int(model["head_hidden_dim"]),
        "tficl_n_heads": int(model["tficl_n_heads"]),
        "tficl_n_layers": int(model["tficl_n_layers"]),
        "sandwich_heads": (
            None
            if build_spec is None or build_spec.get("sandwich_heads") is None
            else int(build_spec["sandwich_heads"])
        ),
        "sandwich_layers": (
            None
            if build_spec is None or build_spec.get("sandwich_layers") is None
            else int(build_spec["sandwich_layers"])
        ),
        "architecture": (
            dict(cast(Mapping[str, Any], model.get("architecture")))
            if isinstance(model.get("architecture"), Mapping)
            else None
        ),
        "build_spec": (
            dict(build_spec) if build_spec is not None else None
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


def _model_dimensions(run_entry: Mapping[str, Any]) -> tuple[int, int]:
    model = ensure_mapping(run_entry.get("model"), context="run_entry.model")
    build_spec = (
        ensure_mapping(model.get("build_spec"), context="run_entry.model.build_spec")
        if isinstance(model.get("build_spec"), Mapping)
        else None
    )
    if build_spec is not None and build_spec.get("sandwich_layers") is not None:
        return int(model["d_icl"]), int(build_spec["sandwich_layers"])
    return int(model["d_icl"]), int(model["tficl_n_layers"])


def _row_label(run_entry: Mapping[str, Any]) -> str:
    d_icl, layers = _model_dimensions(run_entry)
    return f"{d_icl}x{layers}"


def _effective_size(run_entry: Mapping[str, Any]) -> int:
    d_icl, layers = _model_dimensions(run_entry)
    return int(layers * (d_icl**2))


def _model_total_params(run_entry: Mapping[str, Any]) -> int | None:
    model_size = run_entry.get("model_size")
    if not isinstance(model_size, Mapping):
        return None
    raw_value = model_size.get("total_params")
    if raw_value is None:
        return None
    return int(raw_value)


def _reserved_vram_gb(run_entry: Mapping[str, Any]) -> float | None:
    runtime_summary = run_entry.get("runtime_summary")
    if not isinstance(runtime_summary, Mapping):
        return None
    raw_reserved = runtime_summary.get("peak_vram_reserved")
    if raw_reserved is None:
        return None
    return float(raw_reserved) / float(1024**3)


def _train_wall_seconds(run_entry: Mapping[str, Any]) -> float | None:
    diagnostics = run_entry.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    return ensure_optional_finite_number(
        diagnostics.get("wall_elapsed_seconds"),
        context="run_entry.training_diagnostics.wall_elapsed_seconds",
    )


def _benchmark_wall_seconds(run_entry: Mapping[str, Any]) -> float | None:
    timing = run_entry.get("benchmark_timing")
    if not isinstance(timing, Mapping):
        return None
    return ensure_optional_finite_number(
        timing.get("wall_elapsed_seconds"),
        context="run_entry.benchmark_timing.wall_elapsed_seconds",
    )


def _inference_timing_value(run_entry: Mapping[str, Any], key: str) -> float | None:
    payload = run_entry.get("inference_timing")
    if not isinstance(payload, Mapping):
        return None
    return ensure_optional_finite_number(
        payload.get(key),
        context=f"run_entry.inference_timing.{key}",
    )


def _health_verdict(run_entry: Mapping[str, Any]) -> str | None:
    artifacts = run_entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    raw_run_dir = artifacts.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        return None
    try:
        run_dir = read_benchmark_registry.resolve_registry_path_value(raw_run_dir)
    except RuntimeError:
        return None
    if not run_dir.exists():
        return None
    try:
        payload = health_check(run_dir)
    except RuntimeError:
        return None
    verdict = payload.get("verdict")
    return None if verdict is None else str(verdict)


def _fit_linear(
    samples: Sequence[tuple[float, float]],
) -> tuple[str, dict[str, float], Callable[[float], float | None]]:
    finite = [(float(x), float(y)) for x, y in samples if math.isfinite(x) and math.isfinite(y)]
    if not finite:
        return (
            "unfit",
            {"intercept": 0.0, "slope": 0.0},
            lambda _value: None,
        )
    if len(finite) == 1:
        intercept = float(finite[0][1])
        return (
            "constant",
            {"intercept": intercept, "slope": 0.0},
            lambda _value: float(intercept),
        )
    x_values = [sample[0] for sample in finite]
    y_values = [sample[1] for sample in finite]
    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(y_values) / float(len(y_values))
    denom = sum((value - x_mean) ** 2 for value in x_values)
    if denom == 0.0:
        intercept = float(y_mean)
        return (
            "constant",
            {"intercept": intercept, "slope": 0.0},
            lambda _value: float(intercept),
        )
    slope = sum((x - x_mean) * (y - y_mean) for x, y in finite) / float(denom)
    intercept = y_mean - slope * x_mean
    return (
        "linear",
        {"intercept": float(intercept), "slope": float(slope)},
        lambda value: float(intercept + slope * float(value)),
    )


def _fit_parameter_bridge(
    entries: Sequence[Mapping[str, Any]],
    *,
    evidence_run_ids: Sequence[str],
) -> tuple[dict[str, Any], Callable[[int, int, int], float | None]]:
    fixed_depth_coefficients: list[float] = []
    affine_points: list[WidthDepthParameterPoint] = []
    unique_layers: set[int] = set()
    for entry in entries:
        params = _model_total_params(entry)
        d_icl, layers = _model_dimensions(entry)
        effective_size = _effective_size(entry)
        if params is None or effective_size <= 0:
            continue
        unique_layers.add(int(layers))
        fixed_depth_coefficients.append(float(params) / float(effective_size))
        affine_points.append(
            WidthDepthParameterPoint(
                d_icl=int(d_icl),
                layers=int(layers),
                total_params=float(params),
                row_label=_row_label(entry),
            )
        )
    if not affine_points:
        return (
            {
                "expression": "P_local(d, L) ≈ 0.00 * L * d^2",
                "fit_kind": "unfit",
                "coefficients": {"coefficient": 0.0},
                "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
            },
            lambda _d_icl, _layers, _effective_size: None,
        )
    if len(unique_layers) <= 1:
        coefficient = sum(fixed_depth_coefficients) / float(len(fixed_depth_coefficients))
        return (
            {
                "expression": f"P_local(d, L) ≈ {coefficient:.2f} * L * d^2",
                "fit_kind": "mean_coefficient_fixed_depth",
                "coefficients": {"coefficient": float(coefficient)},
                "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
            },
            lambda _d_icl, _layers, effective_size: float(coefficient) * float(effective_size),
        )

    fit = fit_affine_width_depth_parameter_bridge(affine_points)
    return (
        {
            "expression": fit.expression(),
            "fit_kind": "affine_depth_aware_least_squares",
            "coefficients": {
                "intercept": float(fit.intercept),
                "d_squared_coefficient": float(fit.d_squared_coefficient),
                "layered_d_squared_coefficient": float(fit.layered_d_squared_coefficient),
            },
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        lambda d_icl, layers, _effective_size: fit.predict_total_params(
            d_icl=int(d_icl),
            layers=int(layers),
        ),
    )


def _constraint_model_payload(
    *,
    baseline_entry: Mapping[str, Any],
    evidence_entries: Sequence[Mapping[str, Any]],
    evidence_run_ids: Sequence[str],
    vram_class_gb: int,
) -> dict[str, Any]:
    parameter_formula, parameter_predictor = _fit_parameter_bridge(
        evidence_entries,
        evidence_run_ids=evidence_run_ids,
    )
    vram_fit_kind, vram_coefficients, vram_predictor = _fit_linear(
        [
            (float(params), float(reserved_gb))
            for entry in evidence_entries
            for params, reserved_gb in [(_model_total_params(entry), _reserved_vram_gb(entry))]
            if params is not None and reserved_gb is not None
        ]
    )
    train_fit_kind, train_coefficients, train_predictor = _fit_linear(
        [
            (float(params), float(train_seconds))
            for entry in evidence_entries
            for params, train_seconds in [(_model_total_params(entry), _train_wall_seconds(entry))]
            if params is not None and train_seconds is not None
        ]
    )
    benchmark_fit_kind, benchmark_coefficients, benchmark_predictor = _fit_linear(
        [
            (float(params), float(benchmark_seconds))
            for entry in evidence_entries
            for params, benchmark_seconds in [(_model_total_params(entry), _benchmark_wall_seconds(entry))]
            if params is not None and benchmark_seconds is not None
        ]
    )
    inference_mean_fit_kind, inference_mean_coefficients, inference_mean_predictor = _fit_linear(
        [
            (float(params), float(inference_ms))
            for entry in evidence_entries
            for params, inference_ms in [(_model_total_params(entry), _inference_timing_value(entry, "mean_ms"))]
            if params is not None and inference_ms is not None
        ]
    )
    inference_p50_fit_kind, inference_p50_coefficients, inference_p50_predictor = _fit_linear(
        [
            (float(params), float(inference_ms))
            for entry in evidence_entries
            for params, inference_ms in [(_model_total_params(entry), _inference_timing_value(entry, "p50_ms"))]
            if params is not None and inference_ms is not None
        ]
    )
    inference_p95_fit_kind, inference_p95_coefficients, inference_p95_predictor = _fit_linear(
        [
            (float(params), float(inference_ms))
            for entry in evidence_entries
            for params, inference_ms in [(_model_total_params(entry), _inference_timing_value(entry, "p95_ms"))]
            if params is not None and inference_ms is not None
        ]
    )

    formulas = {
        "parameter_count": parameter_formula,
        "reserved_vram_gb": {
            "expression": (
                f"reserved_vram_gb ≈ {vram_coefficients['intercept']:.2f} "
                f"+ {vram_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": vram_fit_kind,
            "coefficients": vram_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        "train_wall_seconds": {
            "expression": (
                f"train_wall_seconds ≈ {train_coefficients['intercept']:.2f} "
                f"+ {train_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": train_fit_kind,
            "coefficients": train_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        "benchmark_wall_seconds": {
            "expression": (
                f"benchmark_wall_seconds ≈ {benchmark_coefficients['intercept']:.2f} "
                f"+ {benchmark_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": benchmark_fit_kind,
            "coefficients": benchmark_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        "inference_mean_ms": {
            "expression": (
                f"inference_mean_ms ≈ {inference_mean_coefficients['intercept']:.2f} "
                f"+ {inference_mean_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": inference_mean_fit_kind,
            "coefficients": inference_mean_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        "inference_p50_ms": {
            "expression": (
                f"inference_p50_ms ≈ {inference_p50_coefficients['intercept']:.2f} "
                f"+ {inference_p50_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": inference_p50_fit_kind,
            "coefficients": inference_p50_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
        "inference_p95_ms": {
            "expression": (
                f"inference_p95_ms ≈ {inference_p95_coefficients['intercept']:.2f} "
                f"+ {inference_p95_coefficients['slope']:.3e} * params"
            ),
            "fit_kind": inference_p95_fit_kind,
            "coefficients": inference_p95_coefficients,
            "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        },
    }

    baseline_row = _row_label(baseline_entry)
    baseline_train = _train_wall_seconds(baseline_entry)
    baseline_benchmark = _benchmark_wall_seconds(baseline_entry)
    baseline_inference_mean = _inference_timing_value(baseline_entry, "mean_ms")
    baseline_inference_p50 = _inference_timing_value(baseline_entry, "p50_ms")
    baseline_inference_p95 = _inference_timing_value(baseline_entry, "p95_ms")

    rows: list[dict[str, Any]] = []
    sorted_entries = sorted(
        evidence_entries,
        key=lambda entry: (
            (
                _model_total_params(entry)
                if _model_total_params(entry) is not None
                else _effective_size(entry)
            ),
            _model_dimensions(entry)[1],
            _model_dimensions(entry)[0],
        ),
    )
    for entry in sorted_entries:
        d_icl, layers = _model_dimensions(entry)
        row_label = _row_label(entry)
        effective_size = _effective_size(entry)
        total_params = _model_total_params(entry)
        predicted_total_params = parameter_predictor(int(d_icl), int(layers), int(effective_size))
        predicted_reserved_vram = (
            None if predicted_total_params is None else vram_predictor(predicted_total_params)
        )
        predicted_train = (
            None if predicted_total_params is None else train_predictor(predicted_total_params)
        )
        predicted_benchmark = (
            None if predicted_total_params is None else benchmark_predictor(predicted_total_params)
        )
        predicted_inference_mean = (
            None if predicted_total_params is None else inference_mean_predictor(predicted_total_params)
        )
        predicted_inference_p50 = (
            None if predicted_total_params is None else inference_p50_predictor(predicted_total_params)
        )
        predicted_inference_p95 = (
            None if predicted_total_params is None else inference_p95_predictor(predicted_total_params)
        )
        observed_reserved_vram = _reserved_vram_gb(entry)
        observed_train = _train_wall_seconds(entry)
        observed_benchmark = _benchmark_wall_seconds(entry)
        observed_inference_mean = _inference_timing_value(entry, "mean_ms")
        observed_inference_p50 = _inference_timing_value(entry, "p50_ms")
        observed_inference_p95 = _inference_timing_value(entry, "p95_ms")
        best_reserved_for_headroom = (
            observed_reserved_vram
            if observed_reserved_vram is not None
            else predicted_reserved_vram
        )
        rows.append(
            {
                "row": row_label,
                "d_icl": int(d_icl),
                "sandwich_layers": int(layers),
                "effective_size": int(effective_size),
                "predicted": {
                    "total_params": (
                        None if predicted_total_params is None else int(round(predicted_total_params))
                    ),
                    "reserved_vram_gb": predicted_reserved_vram,
                    "train_wall_seconds": predicted_train,
                    "benchmark_wall_seconds": predicted_benchmark,
                    "inference_mean_ms": predicted_inference_mean,
                    "inference_p50_ms": predicted_inference_p50,
                    "inference_p95_ms": predicted_inference_p95,
                },
                "observed": {
                    "run_id": str(entry["run_id"]),
                    "delta_ref": (
                        None
                        if not isinstance(entry.get("sweep"), Mapping)
                        else cast(Mapping[str, Any], entry["sweep"]).get("delta_id")
                    ),
                    "health": _health_verdict(entry),
                    "total_params": total_params,
                    "reserved_vram_gb": observed_reserved_vram,
                    "train_wall_seconds": observed_train,
                    "benchmark_wall_seconds": observed_benchmark,
                    "inference_mean_ms": observed_inference_mean,
                    "inference_p50_ms": observed_inference_p50,
                    "inference_p95_ms": observed_inference_p95,
                },
                "headroom": {
                    "hardware_vram_ceiling_gb": float(vram_class_gb),
                    "reserved_vram_gb_to_ceiling": (
                        None
                        if best_reserved_for_headroom is None
                        else float(vram_class_gb) - float(best_reserved_for_headroom)
                    ),
                    "train_wall_seconds_delta_vs_baseline": (
                        None
                        if observed_train is None or baseline_train is None
                        else float(observed_train) - float(baseline_train)
                    ),
                    "benchmark_wall_seconds_delta_vs_baseline": (
                        None
                        if observed_benchmark is None or baseline_benchmark is None
                        else float(observed_benchmark) - float(baseline_benchmark)
                    ),
                    "inference_mean_ms_delta_vs_baseline": (
                        None
                        if observed_inference_mean is None or baseline_inference_mean is None
                        else float(observed_inference_mean) - float(baseline_inference_mean)
                    ),
                    "inference_p50_ms_delta_vs_baseline": (
                        None
                        if observed_inference_p50 is None or baseline_inference_p50 is None
                        else float(observed_inference_p50) - float(baseline_inference_p50)
                    ),
                    "inference_p95_ms_delta_vs_baseline": (
                        None
                        if observed_inference_p95 is None or baseline_inference_p95 is None
                        else float(observed_inference_p95) - float(baseline_inference_p95)
                    ),
                },
            }
        )

    return {
        "effective_size_expression": "S(d, L) = L * d^2",
        "formulas": formulas,
        "evidence_run_ids": [str(run_id) for run_id in evidence_run_ids],
        "baseline_row": baseline_row,
        "rows": rows,
    }


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
    constraint_model = _constraint_model_payload(
        baseline_entry=baseline_entry,
        evidence_entries=evidence_entries,
        evidence_run_ids=evidence_run_ids,
        vram_class_gb=vram_class_gb,
    )

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
        "preferred_benchmark_timing": _benchmark_timing_excerpt(
            cast(Mapping[str, Any] | None, preferred_entry.get("benchmark_timing"))
        ),
        "preferred_inference_timing": _inference_timing_excerpt(
            cast(Mapping[str, Any] | None, preferred_entry.get("inference_timing"))
        ),
        "constraint_model": constraint_model,
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
