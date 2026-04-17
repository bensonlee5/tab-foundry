"""Scaling-study fit audit diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import numpy as np

from tab_foundry.bench.artifacts import write_json
from tab_foundry.research.navigation import build_scaling_navigation_payload
from tab_foundry.repo_paths import normalize_repo_relative_path, repo_root
from tab_foundry.research.scaling.fit import (
    FIT_SCOPE_ALL,
    FIT_SCOPE_NS_ONLY,
    FIT_SCOPE_CHOICES,
    ScalingStudyRunPoint,
    _batch_envelope,
    _batch_points,
    _fit_stats,
    _missing_validation_points,
    _normalize_fit_scope,
    _ns_points,
    _target_values,
    _validate_c_axis_points,
    _validation_backed_points,
    _validation_coverage_payload,
    collect_completed_scaling_points,
    fit_bcrit,
    fit_loss_vs_nd,
    fit_loss_vs_ns,
    fit_loss_vs_scale,
    select_l_d_points,
    select_l_n_points,
)
from tab_foundry.research.scaling.study import ScalingStudyConfig, load_scaling_study_config


SCALING_AUDIT_SCHEMA = "tab-foundry-scaling-audit-v1"
AUDIT_SCOPE_CHOICES = FIT_SCOPE_CHOICES
AuditScope = Literal["all", "ns-only"]
_MIN_POSITIVE = 1.0e-12
_MIN_BOOTSTRAP_SAMPLES = 1
_DEFAULT_BOOTSTRAP_SAMPLES = 64
_MIN_UNIVARIATE_FIT_POINTS = 2
_MIN_JOINT_FIT_POINTS = 4
_MIN_BROKEN_POWER_POINTS = 5
_MIN_INTERPOLATED_BATCHES = 2
_PARAMETER_BOUND_TOLERANCE = 1.0e-10
_INTERPOLATION_TOLERANCE = 1.0e-12
_LARGE_ALPHA_WARNING = 25.0
_VERY_LARGE_ALPHA_WARNING = 100.0
_MIN_CMIN_ISO_LOSS_ESTIMATES = 4
_MIN_CMIN_GEOMETRIES = 3


def _target_points(
    points: Sequence[ScalingStudyRunPoint],
    *,
    target_key: str,
) -> tuple[ScalingStudyRunPoint, ...]:
    if target_key == "validation_loss":
        return _validation_backed_points(points)
    return tuple(points)


def _fit_parameter_flags(
    fit_payload: Mapping[str, Any],
    *,
    context: str,
) -> list[dict[str, Any]]:
    parameters = fit_payload.get("parameters")
    if not isinstance(parameters, Mapping):
        return []
    flags: list[dict[str, Any]] = []
    for raw_name, raw_value in parameters.items():
        name = str(raw_name)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            flags.append(
                {
                    "context": context,
                    "parameter": name,
                    "reason": "non_finite_parameter",
                    "value": raw_value,
                }
            )
            continue
        if value <= _PARAMETER_BOUND_TOLERANCE:
            flags.append(
                {
                    "context": context,
                    "parameter": name,
                    "reason": "near_lower_bound",
                    "value": value,
                }
            )
        if name.startswith("alpha") and value >= _VERY_LARGE_ALPHA_WARNING:
            flags.append(
                {
                    "context": context,
                    "parameter": name,
                    "reason": "very_large_alpha",
                    "value": value,
                }
            )
        elif name.startswith("alpha") and value >= _LARGE_ALPHA_WARNING:
            flags.append(
                {
                    "context": context,
                    "parameter": name,
                    "reason": "large_alpha",
                    "value": value,
                }
            )
    return flags


def _fit_univariate(
    *,
    name: str,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
    x_value: Callable[[ScalingStudyRunPoint], float],
    scale_name: str,
    alpha_name: str,
) -> dict[str, Any]:
    selected_points = _target_points(points, target_key=target_key)
    if len(selected_points) < _MIN_UNIVARIATE_FIT_POINTS:
        return {
            "status": "skipped",
            "reason": "insufficient_points",
            "n_points": len(selected_points),
        }
    fit_payload = fit_loss_vs_scale(
        name=name,
        x_values=[x_value(point) for point in selected_points],
        y_values=_target_values(selected_points, target_key=target_key, context=name),
        scale_name=scale_name,
        alpha_name=alpha_name,
    )
    return {
        "status": "fit",
        **fit_payload,
        "target_key": target_key,
        "points": [point.as_dict() for point in selected_points],
        "fit_quality_flags": _fit_parameter_flags(
            fit_payload,
            context=f"{name}:{target_key}",
        ),
    }


def _fit_joint(
    *,
    name: str,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
    family: Literal["nd", "ns"],
) -> dict[str, Any]:
    selected_points = _target_points(points, target_key=target_key)
    if len(selected_points) < _MIN_JOINT_FIT_POINTS:
        return {
            "status": "skipped",
            "reason": "insufficient_points",
            "n_points": len(selected_points),
        }
    fit_payload = (
        fit_loss_vs_nd(points=selected_points, target_key=target_key)
        if family == "nd"
        else fit_loss_vs_ns(points=selected_points, target_key=target_key)
    )
    return {
        "status": "fit",
        **fit_payload,
        "target_key": target_key,
        "points": [point.as_dict() for point in selected_points],
        "fit_quality_flags": _fit_parameter_flags(
            fit_payload,
            context=f"{name}:{target_key}",
        ),
    }


def _target_comparisons(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    ns_points = _ns_points(points)
    l_n_points = select_l_n_points(points)
    l_d_points = select_l_d_points(points)
    comparisons: dict[str, Any] = {
        "primary_target": "validation_loss",
        "external_transfer_target": "benchmark_log_loss",
        "fits": {},
    }
    c_axis_error: str | None = None
    try:
        _validate_c_axis_points(points, context="fit audit L(C)")
        c_axis_points: tuple[ScalingStudyRunPoint, ...] = tuple(points)
    except RuntimeError as exc:
        c_axis_error = str(exc)
        c_axis_points = ()

    specs: list[tuple[str, Callable[[str], dict[str, Any]]]] = [
        (
            "L(N)",
            lambda target_key: _fit_univariate(
                name="L(N)",
                points=l_n_points,
                target_key=target_key,
                x_value=lambda point: point.n,
                scale_name="Nc",
                alpha_name="alpha_n",
            ),
        ),
        (
            "L(D)",
            lambda target_key: _fit_univariate(
                name="L(D)",
                points=l_d_points,
                target_key=target_key,
                x_value=lambda point: point.d,
                scale_name="Dc",
                alpha_name="alpha_d",
            ),
        ),
        (
            "L(N,D)",
            lambda target_key: _fit_joint(
                name="L(N,D)",
                points=ns_points,
                target_key=target_key,
                family="nd",
            ),
        ),
        (
            "L(N,S)",
            lambda target_key: _fit_joint(
                name="L(N,S)",
                points=ns_points,
                target_key=target_key,
                family="ns",
            ),
        ),
    ]
    if c_axis_error is None:
        specs.insert(
            2,
            (
                "L(C)",
                lambda target_key: _fit_univariate(
                    name="L(C)",
                    points=c_axis_points,
                    target_key=target_key,
                    x_value=lambda point: point.c,
                    scale_name="Cc",
                    alpha_name="alpha_c",
                ),
            ),
        )
    else:
        comparisons["fits"]["L(C)"] = {
            "benchmark_log_loss": {"status": "skipped", "reason": c_axis_error},
            "validation_loss": {"status": "skipped", "reason": c_axis_error},
        }

    for fit_name, fit_fn in specs:
        comparisons["fits"].setdefault(fit_name, {})
        for target_key in ("benchmark_log_loss", "validation_loss"):
            try:
                comparisons["fits"][fit_name][target_key] = fit_fn(target_key)
            except RuntimeError as exc:
                comparisons["fits"][fit_name][target_key] = {
                    "status": "failed",
                    "reason": str(exc),
                }
    return comparisons


def _parameters(fit_payload: Mapping[str, Any], *, context: str) -> dict[str, float]:
    raw_parameters = fit_payload.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise RuntimeError(f"{context} missing fit parameters")
    resolved: dict[str, float] = {}
    for raw_name, raw_value in raw_parameters.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise RuntimeError(f"{context} parameter {raw_name!r} must be finite")
        resolved[str(raw_name)] = value
    return resolved


def _predict_joint(
    fit_payload: Mapping[str, Any],
    points: Sequence[ScalingStudyRunPoint],
    *,
    family: Literal["nd", "ns"],
) -> list[float]:
    parameters = _parameters(fit_payload, context=str(fit_payload.get("name") or "joint fit"))
    irreducible_loss = parameters["irreducible_loss"]
    nc = parameters["Nc"]
    alpha_n = parameters["alpha_n"]
    if family == "nd":
        dc = parameters["Dc"]
        alpha_d = parameters["alpha_d"]
        return [
            float(
                irreducible_loss
                + (
                    (nc / max(point.n, _MIN_POSITIVE)) ** (alpha_n / alpha_d)
                    + (dc / max(point.d, _MIN_POSITIVE))
                )
                ** alpha_d
            )
            for point in points
        ]
    sc = parameters["Sc"]
    alpha_s = parameters["alpha_s"]
    return [
        float(
            irreducible_loss
            + (
                (nc / max(point.n, _MIN_POSITIVE)) ** (alpha_n / alpha_s)
                + (sc / max(point.s, _MIN_POSITIVE))
            )
            ** alpha_s
        )
        for point in points
    ]


def _holdout_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped", "reason": "no_successful_holdouts", "n_holdout_points": 0}
    y_true = np.asarray([float(row["observed"]) for row in rows], dtype=float)
    y_pred = np.asarray([float(row["predicted"]) for row in rows], dtype=float)
    return {
        "status": "computed",
        "n_holdout_points": len(rows),
        "stats": _fit_stats(y_true=y_true, y_pred=y_pred, param_count=0),
        "rows": list(rows),
    }


def _holdout_joint_fit(
    *,
    name: str,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
    family: Literal["nd", "ns"],
    group_key: Callable[[ScalingStudyRunPoint], str],
    group_axis: str,
    min_train_points: int = 6,
) -> dict[str, Any]:
    selected_points = _target_points(points, target_key=target_key)
    groups: dict[str, list[ScalingStudyRunPoint]] = {}
    for point in selected_points:
        groups.setdefault(group_key(point), []).append(point)
    rows: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []
    fit_fn = fit_loss_vs_nd if family == "nd" else fit_loss_vs_ns
    for heldout_value, test_points in sorted(groups.items()):
        train_points = tuple(point for point in selected_points if group_key(point) != heldout_value)
        if len(train_points) < min_train_points:
            skipped_groups.append(
                {
                    "heldout": heldout_value,
                    "reason": "insufficient_train_points",
                    "train_points": len(train_points),
                }
            )
            continue
        try:
            fit_payload = fit_fn(points=train_points, target_key=target_key)
            predictions = _predict_joint(fit_payload, test_points, family=family)
            observed = _target_values(test_points, target_key=target_key, context=name)
        except RuntimeError as exc:
            skipped_groups.append(
                {
                    "heldout": heldout_value,
                    "reason": str(exc),
                    "train_points": len(train_points),
                }
            )
            continue
        for point, observed_value, predicted_value in zip(
            test_points,
            observed,
            predictions,
            strict=False,
        ):
            rows.append(
                {
                    "group_axis": group_axis,
                    "heldout": heldout_value,
                    "target_key": target_key,
                    "fit": name,
                    "run_id": point.run_id,
                    "row_label": point.row_label,
                    "steps": point.steps,
                    "observed": float(observed_value),
                    "predicted": float(predicted_value),
                    "residual": float(observed_value - predicted_value),
                }
            )
    payload = _holdout_summary(rows)
    payload["skipped_groups"] = skipped_groups
    return payload


def _holdout_residuals(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    ns_points = _ns_points(points)
    payload: dict[str, Any] = {}
    for fit_name, family in (("L(N,D)", "nd"), ("L(N,S)", "ns")):
        payload[fit_name] = {}
        for target_key in ("validation_loss", "benchmark_log_loss"):
            payload[fit_name][target_key] = {
                "leave_one_geometry": _holdout_joint_fit(
                    name=fit_name,
                    points=ns_points,
                    target_key=target_key,
                    family=family,  # type: ignore[arg-type]
                    group_key=lambda point: point.row_label,
                    group_axis="row_label",
                ),
                "leave_one_step": _holdout_joint_fit(
                    name=fit_name,
                    points=ns_points,
                    target_key=target_key,
                    family=family,  # type: ignore[arg-type]
                    group_key=lambda point: str(point.steps),
                    group_axis="steps",
                ),
            }
    return payload


def _bootstrap_parameter_summary(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for parameter_name, values in sorted(samples.items()):
        finite_values = np.asarray([float(value) for value in values if math.isfinite(float(value))])
        if finite_values.size == 0:
            summary[parameter_name] = None
            continue
        lower, median, upper = np.percentile(finite_values, [2.5, 50.0, 97.5])
        summary[parameter_name] = {
            "lower": float(lower),
            "median": float(median),
            "upper": float(upper),
            "n": int(finite_values.size),
        }
    return summary


def _bootstrap_fit(
    *,
    name: str,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
    samples: int,
    rng: np.random.Generator,
    fit_fn: Callable[[Sequence[ScalingStudyRunPoint]], dict[str, Any]],
    parameter_names: Sequence[str],
    min_points: int,
) -> dict[str, Any]:
    selected_points = _target_points(points, target_key=target_key)
    if len(selected_points) < min_points:
        return {
            "status": "skipped",
            "reason": "insufficient_points",
            "n_points": len(selected_points),
        }
    parameter_samples: dict[str, list[float]] = {name: [] for name in parameter_names}
    failures: list[str] = []
    for _ in range(max(_MIN_BOOTSTRAP_SAMPLES, int(samples))):
        indexes = rng.integers(0, len(selected_points), size=len(selected_points))
        sampled_points = tuple(selected_points[int(index)] for index in indexes)
        try:
            fit_payload = fit_fn(sampled_points)
            parameters = _parameters(fit_payload, context=name)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        for parameter_name in parameter_names:
            value = parameters.get(parameter_name)
            if value is not None and math.isfinite(value):
                parameter_samples[parameter_name].append(float(value))
    accepted = max((len(values) for values in parameter_samples.values()), default=0)
    if accepted == 0:
        return {
            "status": "failed",
            "reason": "no_successful_bootstrap_samples",
            "n_points": len(selected_points),
            "failures": failures[:10],
        }
    return {
        "status": "computed",
        "n_points": len(selected_points),
        "attempted_samples": max(_MIN_BOOTSTRAP_SAMPLES, int(samples)),
        "accepted_samples": accepted,
        "failed_samples": len(failures),
        "parameter_percentiles_95": _bootstrap_parameter_summary(parameter_samples),
        "failure_examples": failures[:10],
    }


def _bootstrap_univariate_fit_fn(
    *,
    name: str,
    target_key: str,
    x_value: Callable[[ScalingStudyRunPoint], float],
    scale_name: str,
    alpha_name: str,
) -> Callable[[Sequence[ScalingStudyRunPoint]], dict[str, Any]]:
    def _fit(sample_points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
        return fit_loss_vs_scale(
            name=name,
            x_values=[x_value(point) for point in sample_points],
            y_values=_target_values(sample_points, target_key=target_key, context=name),
            scale_name=scale_name,
            alpha_name=alpha_name,
        )

    return _fit


def _bootstrap_joint_fit_fn(
    *,
    target_key: str,
    family: Literal["nd", "ns"],
) -> Callable[[Sequence[ScalingStudyRunPoint]], dict[str, Any]]:
    def _fit(sample_points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
        if family == "nd":
            return fit_loss_vs_nd(points=sample_points, target_key=target_key)
        return fit_loss_vs_ns(points=sample_points, target_key=target_key)

    return _fit


def _bootstrap_confidence_intervals(
    points: Sequence[ScalingStudyRunPoint],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    ns_points = _ns_points(points)
    l_n_points = select_l_n_points(points)
    l_d_points = select_l_d_points(points)
    c_axis_points: tuple[ScalingStudyRunPoint, ...]
    try:
        _validate_c_axis_points(points, context="fit audit bootstrap L(C)")
        c_axis_points = tuple(points)
        c_axis_skip: str | None = None
    except RuntimeError as exc:
        c_axis_points = ()
        c_axis_skip = str(exc)

    payload: dict[str, Any] = {
        "samples": int(samples),
        "seed": int(seed),
        "fits": {},
    }
    for target_key in ("benchmark_log_loss", "validation_loss"):
        payload["fits"].setdefault("L(N)", {})[target_key] = _bootstrap_fit(
            name=f"L(N):{target_key}",
            points=l_n_points,
            target_key=target_key,
            samples=samples,
            rng=rng,
            min_points=_MIN_UNIVARIATE_FIT_POINTS,
            parameter_names=("irreducible_loss", "Nc", "alpha_n"),
            fit_fn=_bootstrap_univariate_fit_fn(
                name="L(N)",
                target_key=target_key,
                x_value=lambda point: point.n,
                scale_name="Nc",
                alpha_name="alpha_n",
            ),
        )
        payload["fits"].setdefault("L(D)", {})[target_key] = _bootstrap_fit(
            name=f"L(D):{target_key}",
            points=l_d_points,
            target_key=target_key,
            samples=samples,
            rng=rng,
            min_points=_MIN_UNIVARIATE_FIT_POINTS,
            parameter_names=("irreducible_loss", "Dc", "alpha_d"),
            fit_fn=_bootstrap_univariate_fit_fn(
                name="L(D)",
                target_key=target_key,
                x_value=lambda point: point.d,
                scale_name="Dc",
                alpha_name="alpha_d",
            ),
        )
        if c_axis_skip is None:
            payload["fits"].setdefault("L(C)", {})[target_key] = _bootstrap_fit(
                name=f"L(C):{target_key}",
                points=c_axis_points,
                target_key=target_key,
                samples=samples,
                rng=rng,
                min_points=_MIN_UNIVARIATE_FIT_POINTS,
                parameter_names=("irreducible_loss", "Cc", "alpha_c"),
                fit_fn=_bootstrap_univariate_fit_fn(
                    name="L(C)",
                    target_key=target_key,
                    x_value=lambda point: point.c,
                    scale_name="Cc",
                    alpha_name="alpha_c",
                ),
            )
        else:
            payload["fits"].setdefault("L(C)", {})[target_key] = {
                "status": "skipped",
                "reason": c_axis_skip,
            }
        payload["fits"].setdefault("L(N,D)", {})[target_key] = _bootstrap_fit(
            name=f"L(N,D):{target_key}",
            points=ns_points,
            target_key=target_key,
            samples=samples,
            rng=rng,
            min_points=_MIN_JOINT_FIT_POINTS,
            parameter_names=("irreducible_loss", "Nc", "Dc", "alpha_n", "alpha_d"),
            fit_fn=_bootstrap_joint_fit_fn(target_key=target_key, family="nd"),
        )
        payload["fits"].setdefault("L(N,S)", {})[target_key] = _bootstrap_fit(
            name=f"L(N,S):{target_key}",
            points=ns_points,
            target_key=target_key,
            samples=samples,
            rng=rng,
            min_points=_MIN_JOINT_FIT_POINTS,
            parameter_names=("irreducible_loss", "Nc", "Sc", "alpha_n", "alpha_s"),
            fit_fn=_bootstrap_joint_fit_fn(target_key=target_key, family="ns"),
        )
    batch_points = _batch_points(points)
    envelope_points = _batch_envelope(batch_points)
    if len(envelope_points) < _MIN_CMIN_ISO_LOSS_ESTIMATES:
        payload["fits"]["Bcrit(L)"] = {
            "validation_loss": {
                "status": "skipped",
                "reason": "insufficient_batch_envelope_points",
                "envelope_points": len(envelope_points),
                "required_envelope_points": _MIN_CMIN_ISO_LOSS_ESTIMATES,
            }
        }
    else:
        payload["fits"]["Bcrit(L)"] = {
            "validation_loss": _bootstrap_fit(
                name="Bcrit(L):validation_loss",
                points=batch_points,
                target_key="validation_loss",
                samples=samples,
                rng=rng,
                min_points=_MIN_CMIN_ISO_LOSS_ESTIMATES,
                parameter_names=("B_star", "alpha_b"),
                fit_fn=fit_bcrit,
            )
        }
    return payload


def _linear_log_fit(log_x: np.ndarray, log_y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    design = np.column_stack([np.ones(log_x.size), log_x])
    coefficients, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    predictions = design @ coefficients
    rss = float(np.sum((log_y - predictions) ** 2))
    return coefficients, rss, predictions


def _direction_change_count(values: Sequence[float]) -> int:
    signs: list[int] = []
    for left, right in zip(values, values[1:], strict=False):
        delta = float(right) - float(left)
        if abs(delta) <= _INTERPOLATION_TOLERANCE:
            continue
        signs.append(1 if delta > 0.0 else -1)
    return sum(1 for left, right in zip(signs, signs[1:], strict=False) if left != right)


def _broken_power_diagnostic(
    *,
    name: str,
    points: Sequence[ScalingStudyRunPoint],
    target_key: str,
    x_value: Callable[[ScalingStudyRunPoint], float],
) -> dict[str, Any]:
    selected_points = _target_points(points, target_key=target_key)
    if len(selected_points) < _MIN_BROKEN_POWER_POINTS:
        return {
            "status": "skipped",
            "reason": "insufficient_points",
            "n_points": len(selected_points),
        }
    ordered = sorted(selected_points, key=lambda point: x_value(point))
    x_values = np.asarray([float(x_value(point)) for point in ordered], dtype=float)
    y_values = np.asarray(
        _target_values(ordered, target_key=target_key, context=f"{name} broken-power diagnostic"),
        dtype=float,
    )
    unique_x = np.unique(x_values)
    if unique_x.size < _MIN_BROKEN_POWER_POINTS:
        return {
            "status": "skipped",
            "reason": "insufficient_unique_x_values",
            "n_points": len(selected_points),
            "unique_x_values": int(unique_x.size),
        }
    floor = max(_MIN_POSITIVE, 0.9 * float(np.min(y_values)))
    shifted = np.maximum(y_values - floor, _MIN_POSITIVE)
    log_x = np.log(x_values)
    log_y = np.log(shifted)
    single_coefficients, single_rss, single_predictions = _linear_log_fit(log_x, log_y)
    n_points = int(log_x.size)
    single_aic = float(n_points * math.log(max(single_rss, _MIN_POSITIVE) / n_points) + 4.0)
    best: dict[str, Any] | None = None
    for break_index in range(2, n_points - 1):
        if x_values[break_index - 1] == x_values[break_index]:
            continue
        left_coefficients, left_rss, left_predictions = _linear_log_fit(
            log_x[:break_index],
            log_y[:break_index],
        )
        right_coefficients, right_rss, right_predictions = _linear_log_fit(
            log_x[break_index:],
            log_y[break_index:],
        )
        rss = left_rss + right_rss
        aic = float(n_points * math.log(max(rss, _MIN_POSITIVE) / n_points) + 8.0)
        predictions = np.concatenate([left_predictions, right_predictions])
        candidate = {
            "break_index": int(break_index),
            "break_x": float(x_values[break_index]),
            "rss": float(rss),
            "aic": aic,
            "left_alpha": float(-left_coefficients[1]),
            "right_alpha": float(-right_coefficients[1]),
            "log_space_residuals": (log_y - predictions).tolist(),
        }
        if best is None or aic < float(best["aic"]):
            best = candidate
    if best is None:
        return {
            "status": "skipped",
            "reason": "no_valid_breakpoint",
            "n_points": len(selected_points),
        }
    return {
        "status": "computed",
        "target_key": target_key,
        "n_points": len(selected_points),
        "floor": floor,
        "single_power_law": {
            "rss": single_rss,
            "aic": single_aic,
            "alpha": float(-single_coefficients[1]),
            "log_space_residuals": (log_y - single_predictions).tolist(),
        },
        "best_broken_power_law": {
            **best,
            "delta_aic_vs_single": float(best["aic"] - single_aic),
        },
        "loss_direction_changes": _direction_change_count(y_values.tolist()),
        "diagnostic_only": True,
    }


def _broken_power_law_diagnostics(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    specs: list[tuple[str, Sequence[ScalingStudyRunPoint], Callable[[ScalingStudyRunPoint], float]]] = [
        ("L(N)", select_l_n_points(points), lambda point: point.n),
        ("L(D)", select_l_d_points(points), lambda point: point.d),
        ("L(C)", tuple(points), lambda point: point.c),
    ]
    for fit_name, selected_points, x_value in specs:
        diagnostics[fit_name] = {}
        for target_key in ("benchmark_log_loss", "validation_loss"):
            try:
                diagnostics[fit_name][target_key] = _broken_power_diagnostic(
                    name=fit_name,
                    points=selected_points,
                    target_key=target_key,
                    x_value=x_value,
                )
            except RuntimeError as exc:
                diagnostics[fit_name][target_key] = {"status": "failed", "reason": str(exc)}
    return diagnostics


def _interpolate_point_at_loss(
    points: Sequence[ScalingStudyRunPoint],
    *,
    target_loss: float,
) -> dict[str, float] | None:
    ordered = sorted(
        _validation_backed_points(points),
        key=lambda point: _target_values([point], target_key="validation_loss", context="iso-loss")[
            0
        ],
    )
    if not ordered:
        return None
    losses = [float(point.validation_loss) for point in ordered if point.validation_loss is not None]
    if (
        target_loss < min(losses) - _INTERPOLATION_TOLERANCE
        or target_loss > max(losses) + _INTERPOLATION_TOLERANCE
    ):
        return None
    for point in ordered:
        if (
            point.validation_loss is not None
            and abs(float(point.validation_loss) - target_loss) <= _INTERPOLATION_TOLERANCE
        ):
            return {"steps": point.s, "compute": point.c, "batch": point.b_eff}
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.validation_loss is None or right.validation_loss is None:
            continue
        left_loss = float(left.validation_loss)
        right_loss = float(right.validation_loss)
        if not (left_loss <= target_loss <= right_loss):
            continue
        denominator = max(right_loss - left_loss, _MIN_POSITIVE)
        weight = (target_loss - left_loss) / denominator
        return {
            "steps": float(left.s + weight * (right.s - left.s)),
            "compute": float(left.c + weight * (right.c - left.c)),
            "batch": float(left.b_eff + weight * (right.b_eff - left.b_eff)),
        }
    return None


def _iso_loss_bcrit_readiness(batch_points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    validation_points = _validation_backed_points(batch_points)
    by_geometry: dict[str, list[ScalingStudyRunPoint]] = {}
    for point in validation_points:
        by_geometry.setdefault(point.row_label, []).append(point)
    estimates: list[dict[str, Any]] = []
    for row_label, geometry_points in sorted(by_geometry.items()):
        by_batch: dict[float, list[ScalingStudyRunPoint]] = {}
        for point in geometry_points:
            by_batch.setdefault(float(point.b_eff), []).append(point)
        candidate_losses = sorted(
            {
                round(float(point.validation_loss), 12)
                for point in geometry_points
                if point.validation_loss is not None
            }
        )
        for target_loss in candidate_losses:
            interpolated = [
                interpolation
                for batch_group in by_batch.values()
                if (
                    interpolation := _interpolate_point_at_loss(
                        batch_group,
                        target_loss=float(target_loss),
                    )
                )
                is not None
            ]
            if len(interpolated) < _MIN_INTERPOLATED_BATCHES:
                continue
            emin = min(float(item["compute"]) for item in interpolated)
            smin = min(float(item["steps"]) for item in interpolated)
            estimates.append(
                {
                    "row_label": row_label,
                    "target_validation_loss": float(target_loss),
                    "candidate_batches": len(interpolated),
                    "emin": emin,
                    "smin": smin,
                    "bcrit_estimate": float(emin / max(smin, _MIN_POSITIVE)),
                }
            )
    distinct_geometries = sorted({estimate["row_label"] for estimate in estimates})
    ready = (
        len(estimates) >= _MIN_CMIN_ISO_LOSS_ESTIMATES
        and len(distinct_geometries) >= _MIN_CMIN_GEOMETRIES
    )
    return {
        "method": "diagnostic_iso_loss_interpolation",
        "ready_for_cmin": ready,
        "iso_loss_estimates": estimates,
        "iso_loss_estimate_count": len(estimates),
        "distinct_geometry_count": len(distinct_geometries),
        "distinct_geometries": distinct_geometries,
        "required_iso_loss_estimates": _MIN_CMIN_ISO_LOSS_ESTIMATES,
        "required_geometry_count": _MIN_CMIN_GEOMETRIES,
        "recommendation": (
            "Bcrit(L) may be used for Cmin once the iso-loss contour and geometry gates pass."
            if ready
            else "Do not use Bcrit(L) to derive Cmin; run the redesigned multi-geometry batch sweep first."
        ),
    }


def _fit_policy_diagnostics(points: Sequence[ScalingStudyRunPoint]) -> dict[str, Any]:
    batch_points = _batch_points(points)
    envelope = _batch_envelope(batch_points)
    return {
        "primary_target": "validation_loss",
        "benchmark_target_role": "external_transfer_validation_and_repo_ranking",
        "do_not_import_paper_exponents": True,
        "reject_or_quarantine_flags": [
            "near_lower_bound",
            "large_alpha",
            "very_large_alpha",
            "non_finite_parameter",
        ],
        "bcrit_envelope_points": len(envelope),
        "bcrit_iso_loss_readiness": _iso_loss_bcrit_readiness(batch_points),
        "missing_validation_points": [
            point.as_dict() for point in _missing_validation_points(points)
        ],
    }


def _audit_markdown(
    *,
    config: ScalingStudyConfig,
    artifact_root: Path,
    payload: Mapping[str, Any],
) -> str:
    counts = payload.get("counts", {})
    policy = payload.get("fit_policy", {})
    readiness = {}
    if isinstance(policy, Mapping):
        readiness = policy.get("bcrit_iso_loss_readiness", {})
    lines = [
        f"# {config.study_id} Scaling Fit Audit",
        "",
        f"- Artifact root: `{normalize_repo_relative_path(artifact_root)}`",
        f"- Completed points: `{counts.get('total_completed_points', 0)}`",
        f"- Validation-backed points: `{counts.get('validation_backed_points', 0)}`",
        "- Primary target: `validation_loss`",
        "- Benchmark target role: `external_transfer_validation_and_repo_ranking`",
        "",
        "## Bcrit Readiness",
        (
            f"- ready_for_cmin: `{readiness.get('ready_for_cmin')}` "
            f"iso_loss_estimates=`{readiness.get('iso_loss_estimate_count')}` "
            f"geometries=`{readiness.get('distinct_geometry_count')}`"
        ),
        f"- recommendation: {readiness.get('recommendation', 'unknown')}",
        "",
        "## Holdout Diagnostics",
    ]
    holdouts = payload.get("holdout_residuals", {})
    if isinstance(holdouts, Mapping):
        for fit_name in sorted(holdouts):
            fit_payload = holdouts.get(fit_name)
            if not isinstance(fit_payload, Mapping):
                continue
            lines.append(f"- `{fit_name}`")
            for target_key in sorted(fit_payload):
                target_payload = fit_payload.get(target_key)
                if not isinstance(target_payload, Mapping):
                    continue
                for group_axis, axis_payload in sorted(target_payload.items()):
                    if not isinstance(axis_payload, Mapping):
                        continue
                    stats = axis_payload.get("stats") if isinstance(axis_payload, Mapping) else None
                    rmse = stats.get("rmse") if isinstance(stats, Mapping) else None
                    lines.append(
                        f"  - `{target_key}` `{group_axis}`: "
                        f"status=`{axis_payload.get('status')}` rmse=`{rmse}`"
                    )
    return "\n".join(lines) + "\n"


def audit_scaling_study(
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
    bootstrap_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Audit a configured scaling study without executing new sweeps or W&B writes."""

    normalized_scope = _normalize_fit_scope(fit_scope)
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
    points = _ns_points(all_points) if normalized_scope == FIT_SCOPE_NS_ONLY else all_points
    artifact_root = (
        out_root.expanduser().resolve()
        if out_root is not None
        else config.output_root_path(root=repo_root()) / "audit"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    ns_points = _ns_points(points)
    batch_points = _batch_points(points)
    completeness = cast(Mapping[str, Any], navigation.get("completeness", {}))
    if normalized_scope == FIT_SCOPE_ALL and batch_points and not bool(completeness.get("all_expected_points_present")):
        raise RuntimeError(
            "fit_scope='all' requires the full expected study surface before auditing; "
            f"expected={completeness.get('expected_counts_by_family')} "
            f"actual={completeness.get('actual_counts_by_family')}"
        )
    payload: dict[str, Any] = {
        "schema": SCALING_AUDIT_SCHEMA,
        "study": config.as_dict(),
        "fit_scope": normalized_scope,
        "navigation": navigation,
        "counts": {
            "total_completed_points": len(points),
            "all_completed_points": len(all_points),
            "ns_core_completed_points": len(ns_points),
            "batch_critical_completed_points": len(batch_points),
            "validation_backed_points": len(_validation_backed_points(points)),
            "missing_validation_points": len(_missing_validation_points(points)),
            "ns_core_validation_backed_points": len(_validation_backed_points(ns_points)),
            "batch_critical_validation_backed_points": len(
                _validation_backed_points(batch_points)
            ),
        },
        "validation_coverage": _validation_coverage_payload(points),
        "target_comparisons": _target_comparisons(points),
        "holdout_residuals": _holdout_residuals(points),
        "bootstrap_confidence_intervals": _bootstrap_confidence_intervals(
            points,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "broken_power_law_diagnostics": _broken_power_law_diagnostics(points),
        "fit_policy": _fit_policy_diagnostics(points),
        "artifact_paths": {
            "artifact_root": str(artifact_root),
            "json_path": str(artifact_root / "audit_summary.json"),
            "markdown_path": str(artifact_root / "audit.md"),
        },
    }
    write_json(artifact_root / "audit_summary.json", payload)
    (artifact_root / "audit.md").write_text(
        _audit_markdown(config=config, artifact_root=artifact_root, payload=payload),
        encoding="utf-8",
    )
    return payload


def render_scaling_audit_text(payload: Mapping[str, Any]) -> str:
    """Render a short human-readable summary for `research scaling audit`."""

    study = payload.get("study", {})
    counts = payload.get("counts", {})
    policy = payload.get("fit_policy", {})
    readiness = policy.get("bcrit_iso_loss_readiness") if isinstance(policy, Mapping) else {}
    if not isinstance(readiness, Mapping):
        readiness = {}
    lines = [
        f"Study: {study.get('study_id', 'unknown')}",
        f"Completed points: {counts.get('total_completed_points', 0)}",
        f"Validation-backed points: {counts.get('validation_backed_points', 0)}",
        "Primary target: validation_loss",
        (
            "Bcrit Cmin ready: "
            f"{readiness.get('ready_for_cmin')} "
            f"(iso-loss estimates={readiness.get('iso_loss_estimate_count')}, "
            f"geometries={readiness.get('distinct_geometry_count')})"
        ),
        f"Audit JSON: {payload.get('artifact_paths', {}).get('json_path', 'unknown')}",
    ]
    return "\n".join(lines)


__all__ = [
    "AUDIT_SCOPE_CHOICES",
    "SCALING_AUDIT_SCHEMA",
    "audit_scaling_study",
    "render_scaling_audit_text",
]
