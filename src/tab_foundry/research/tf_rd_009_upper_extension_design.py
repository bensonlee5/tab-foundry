"""Deterministic TF-RD-009 upper-family reopening design helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tab_foundry.bench.width_depth_scaling import (
    AffineWidthDepthParameterFit,
    LinearFit,
    WidthDepthParameterPoint,
    fit_affine_width_depth_parameter_bridge,
    log_space_parameter_targets,
    round_to_width_rung,
)
from tab_foundry.benchmark_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_entry,
    resolve_registry_path_value,
)
from tab_foundry.repo_paths import repo_root
from tab_foundry.research.scaling.fit import (
    ScalingStudyRunPoint,
    _collect_run_point,
    _completed_benchmark_backed_row,
    _load_validation_overlay,
    fit_loss_vs_ns,
)
from tab_foundry.research.scaling.study import load_scaling_study_config
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tab_foundry.training.health import health_check


TF_RD_009_PHASE2_STUDY_ID = "tf_rd_009_phase2"
TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID = "tf_rd_009_width_depth_upper_extension_medium_v1"
TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID = "tf_rd_009_ns_upper_extension_medium_v1"
TF_RD_009_UPPER_EXTENSION_STUDY_ID = "tf_rd_009_phase2_upper_extension_v1"
TF_RD_009_UPPER_EXTENSION_SELECTION_ARTIFACT_ROOT = (
    "reference/system_delta_sweeps/tf_rd_009_width_depth_upper_extension_medium_v1/support"
)
TF_RD_009_UPPER_EXTENSION_SELECTION_JSON = (
    f"{TF_RD_009_UPPER_EXTENSION_SELECTION_ARTIFACT_ROOT}/selection_summary.json"
)
TF_RD_009_UPPER_EXTENSION_SELECTION_MD = (
    f"{TF_RD_009_UPPER_EXTENSION_SELECTION_ARTIFACT_ROOT}/selection_summary.md"
)
TF_RD_009_UPPER_EXTENSION_BASELINE_ROW_LABEL = "176x6"
TF_RD_009_UPPER_EXTENSION_BASELINE_WIDTH = 176
TF_RD_009_UPPER_EXTENSION_BASELINE_LAYERS = 6
TF_RD_009_UPPER_EXTENSION_WIDTH_RUNG = 8
TF_RD_009_UPPER_EXTENSION_GATE_STEPS = 2500
TF_RD_009_UPPER_EXTENSION_STEP_LADDER = (625, 1250, 2500, 5000)
TF_RD_009_UPPER_EXTENSION_CEILING_TARGET_GB = 40.0
TF_RD_009_UPPER_EXTENSION_CEILING_TOLERANCE_GB = 0.5
TF_RD_009_UPPER_EXTENSION_FINAL_LAYER_CHOICES = (8, 9, 10)
TF_RD_009_UPPER_EXTENSION_JACOBIAN_EPSILON = 1.0e-5
TF_RD_009_UPPER_EXTENSION_INFORMATION_RIDGE = 1.0e-8
_FIT_PARAMETER_ORDER = ("irreducible_loss", "Nc", "Sc", "alpha_n", "alpha_s")


@dataclass(frozen=True, slots=True)
class TfRd009UpperExtensionRow:
    row_label: str
    d_icl: int
    layers: int
    target_total_params: float
    predicted_total_params: float
    predicted_canonical_non_embedding_params: float
    predicted_reserved_vram_gb: float

    @property
    def delta_id(self) -> str:
        return f"delta_tf_rd_009_cls_sandwich_dicl{self.d_icl}_layers{self.layers}_upper_v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TfRd009UpperExtensionCandidate:
    candidate_id: str
    final_layers: int
    rows: tuple[TfRd009UpperExtensionRow, ...]
    d_optimal_gain: float
    alpha_uncertainty_width: float

    @property
    def row_labels(self) -> tuple[str, ...]:
        return tuple(row.row_label for row in self.rows)

    @property
    def max_predicted_reserved_vram_gb(self) -> float:
        return max(row.predicted_reserved_vram_gb for row in self.rows)

    @property
    def feasible_under_corrected_ceiling(self) -> bool:
        return (
            self.max_predicted_reserved_vram_gb
            <= TF_RD_009_UPPER_EXTENSION_CEILING_TARGET_GB
            + TF_RD_009_UPPER_EXTENSION_CEILING_TOLERANCE_GB
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row_labels"] = list(self.row_labels)
        payload["max_predicted_reserved_vram_gb"] = self.max_predicted_reserved_vram_gb
        payload["feasible_under_corrected_ceiling"] = self.feasible_under_corrected_ceiling
        return payload


@dataclass(frozen=True, slots=True)
class TfRd009UpperExtensionSelection:
    scoring_study_id: str
    selected_candidate_id: str
    selected_row_labels: tuple[str, ...]
    current_validation_fit: dict[str, Any]
    projected_canonical_parameter_bridge: dict[str, float]
    candidates: tuple[TfRd009UpperExtensionCandidate, ...]

    @property
    def selected_candidate(self) -> TfRd009UpperExtensionCandidate:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scoring_study_id": self.scoring_study_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_row_labels": list(self.selected_row_labels),
            "selection_rule": {
                "primary": "maximize D-optimal information gain on the current validation L(N,S) fit",
                "secondary": "minimize projected parameter-uncertainty width for alpha_n and alpha_s",
                "tie_breaks": [
                    "fewer new geometries",
                    "lower max predicted reserved VRAM",
                ],
            },
            "corrected_constraints": {
                "parameter_bridge_expression": corrected_tf_rd_009_parameter_bridge().expression(),
                "reserved_vram_expression": (
                    "reserved_vram_gb ≈ 8.69 + 9.271e-07 * params"
                ),
                "width_rung": TF_RD_009_UPPER_EXTENSION_WIDTH_RUNG,
                "baseline_row_label": TF_RD_009_UPPER_EXTENSION_BASELINE_ROW_LABEL,
                "ceiling_target_gb": TF_RD_009_UPPER_EXTENSION_CEILING_TARGET_GB,
                "ceiling_tolerance_gb": TF_RD_009_UPPER_EXTENSION_CEILING_TOLERANCE_GB,
                "step_ladder": list(TF_RD_009_UPPER_EXTENSION_STEP_LADDER),
                "gate_steps": TF_RD_009_UPPER_EXTENSION_GATE_STEPS,
            },
            "current_validation_fit": self.current_validation_fit,
            "projected_canonical_parameter_bridge": dict(self.projected_canonical_parameter_bridge),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def corrected_tf_rd_009_parameter_bridge() -> AffineWidthDepthParameterFit:
    return AffineWidthDepthParameterFit(
        intercept=18638.80,
        d_squared_coefficient=77.94,
        layered_d_squared_coefficient=47.93,
    )


def corrected_tf_rd_009_vram_fit() -> LinearFit:
    return LinearFit(
        intercept=8.69,
        slope=9.271e-07,
        fit_kind="linear",
    )


def load_tf_rd_009_phase2_ns_validation_points(
    *,
    studies_root: Path | None = None,
    registry_path: Path | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[ScalingStudyRunPoint, ...]:
    resolved_repo_root = repo_root()
    config = load_scaling_study_config(
        study_id=TF_RD_009_PHASE2_STUDY_ID,
        studies_root=studies_root,
    )
    resolved_registry_path = (
        registry_path or default_benchmark_run_registry_path()
    ).expanduser().resolve()
    resolved_index_path = (
        index_path or resolved_repo_root / "reference" / "system_delta_sweeps" / "index.yaml"
    ).expanduser().resolve()
    resolved_catalog_path = (
        catalog_path or resolved_repo_root / "reference" / "system_delta_catalog.yaml"
    ).expanduser().resolve()
    resolved_sweeps_root = (
        sweeps_root or resolved_repo_root / "reference" / "system_delta_sweeps"
    ).expanduser().resolve()
    validation_overlay = _load_validation_overlay(config)
    points: list[ScalingStudyRunPoint] = []
    for sweep_ref in config.sweeps:
        queue = load_system_delta_queue(
            path=resolved_sweeps_root / sweep_ref.sweep_id / "queue.yaml",
            index_path=resolved_index_path,
            catalog_path=resolved_catalog_path,
            sweeps_root=resolved_sweeps_root,
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
                    registry_path=resolved_registry_path,
                    validation_overlay=validation_overlay,
                )
            )
    points = sorted(
        points,
        key=lambda point: (
            point.family,
            point.sweep_id,
            point.row_order,
            point.run_id,
        ),
    )
    return tuple(
        point
        for point in points
        if point.family == "ns_core" and point.validation_loss is not None
    )


def fit_tf_rd_009_upper_extension_canonical_bridge(
    points: Sequence[ScalingStudyRunPoint],
) -> AffineWidthDepthParameterFit:
    representative_rows: dict[str, ScalingStudyRunPoint] = {}
    for point in points:
        representative_rows.setdefault(point.row_label, point)
    if not representative_rows:
        raise RuntimeError("at least one NS-core row is required to fit the canonical bridge")
    return fit_affine_width_depth_parameter_bridge(
        tuple(
            WidthDepthParameterPoint(
                d_icl=point.d_icl,
                layers=point.layers,
                total_params=float(point.canonical_non_embedding_params),
                row_label=point.row_label,
            )
            for point in sorted(
                representative_rows.values(),
                key=lambda row: (row.layers, row.d_icl, row.row_label),
            )
        )
    )


def enumerate_tf_rd_009_upper_extension_candidates(
    *,
    points: Sequence[ScalingStudyRunPoint] | None = None,
) -> tuple[TfRd009UpperExtensionCandidate, ...]:
    resolved_points = (
        tuple(points)
        if points is not None
        else load_tf_rd_009_phase2_ns_validation_points()
    )
    canonical_bridge = fit_tf_rd_009_upper_extension_canonical_bridge(resolved_points)
    parameter_bridge = corrected_tf_rd_009_parameter_bridge()
    vram_fit = corrected_tf_rd_009_vram_fit()
    baseline_params = parameter_bridge.predict_total_params(
        d_icl=TF_RD_009_UPPER_EXTENSION_BASELINE_WIDTH,
        layers=TF_RD_009_UPPER_EXTENSION_BASELINE_LAYERS,
    )
    ceiling_target_params = vram_fit.solve_x(TF_RD_009_UPPER_EXTENSION_CEILING_TARGET_GB)
    current_fit = fit_loss_vs_ns(points=resolved_points, target_key="validation_loss")
    current_information = _information_matrix_from_rows(
        _projected_ns_coordinates(resolved_points),
        parameters=current_fit["parameters"],
    )
    baseline_logdet = _stable_logdet(
        current_information + TF_RD_009_UPPER_EXTENSION_INFORMATION_RIDGE * np.eye(len(_FIT_PARAMETER_ORDER))
    )
    candidates: list[TfRd009UpperExtensionCandidate] = []
    for final_layers in TF_RD_009_UPPER_EXTENSION_FINAL_LAYER_CHOICES:
        final_width = round_to_width_rung(
            parameter_bridge.solve_width_for_target_params(
                layers=final_layers,
                target_params=ceiling_target_params,
            ),
            rung=TF_RD_009_UPPER_EXTENSION_WIDTH_RUNG,
        )
        final_predicted_total = parameter_bridge.predict_total_params(
            d_icl=final_width,
            layers=final_layers,
        )
        interior_targets = log_space_parameter_targets(
            start_value=baseline_params,
            end_value=final_predicted_total,
            count=max(0, final_layers - 7),
        )
        rows: list[TfRd009UpperExtensionRow] = []
        for layers, target_total_params in zip(
            range(TF_RD_009_UPPER_EXTENSION_BASELINE_LAYERS + 1, final_layers),
            interior_targets,
            strict=True,
        ):
            width = round_to_width_rung(
                parameter_bridge.solve_width_for_target_params(
                    layers=layers,
                    target_params=target_total_params,
                ),
                rung=TF_RD_009_UPPER_EXTENSION_WIDTH_RUNG,
            )
            rows.append(
                _projected_row(
                    d_icl=width,
                    layers=layers,
                    target_total_params=float(target_total_params),
                    parameter_bridge=parameter_bridge,
                    canonical_bridge=canonical_bridge,
                    vram_fit=vram_fit,
                )
            )
        rows.append(
            _projected_row(
                d_icl=final_width,
                layers=final_layers,
                target_total_params=float(ceiling_target_params),
                parameter_bridge=parameter_bridge,
                canonical_bridge=canonical_bridge,
                vram_fit=vram_fit,
            )
        )
        projected_information = _information_matrix_from_rows(
            _projected_ns_coordinates(rows),
            parameters=current_fit["parameters"],
        )
        total_information = (
            current_information
            + projected_information
            + TF_RD_009_UPPER_EXTENSION_INFORMATION_RIDGE * np.eye(len(_FIT_PARAMETER_ORDER))
        )
        alpha_uncertainty_width = _alpha_uncertainty_width(total_information)
        candidates.append(
            TfRd009UpperExtensionCandidate(
                candidate_id="->".join(row.row_label for row in rows),
                final_layers=final_layers,
                rows=tuple(rows),
                d_optimal_gain=_stable_logdet(total_information) - baseline_logdet,
                alpha_uncertainty_width=alpha_uncertainty_width,
            )
        )
    feasible_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.feasible_under_corrected_ceiling
            ),
            key=lambda candidate: (
                -candidate.d_optimal_gain,
                candidate.alpha_uncertainty_width,
                len(candidate.rows),
                candidate.max_predicted_reserved_vram_gb,
            ),
        )
    )
    if not feasible_candidates:
        raise RuntimeError("no feasible TF-RD-009 upper-extension candidates were found")
    return feasible_candidates


def select_tf_rd_009_upper_extension(
    *,
    points: Sequence[ScalingStudyRunPoint] | None = None,
) -> TfRd009UpperExtensionSelection:
    resolved_points = (
        tuple(points)
        if points is not None
        else load_tf_rd_009_phase2_ns_validation_points()
    )
    current_fit = fit_loss_vs_ns(points=resolved_points, target_key="validation_loss")
    canonical_bridge = fit_tf_rd_009_upper_extension_canonical_bridge(resolved_points)
    candidates = enumerate_tf_rd_009_upper_extension_candidates(points=resolved_points)
    selected_candidate = candidates[0]
    return TfRd009UpperExtensionSelection(
        scoring_study_id=TF_RD_009_PHASE2_STUDY_ID,
        selected_candidate_id=selected_candidate.candidate_id,
        selected_row_labels=selected_candidate.row_labels,
        current_validation_fit=current_fit,
        projected_canonical_parameter_bridge={
            "intercept": canonical_bridge.intercept,
            "d_squared_coefficient": canonical_bridge.d_squared_coefficient,
            "layered_d_squared_coefficient": canonical_bridge.layered_d_squared_coefficient,
        },
        candidates=candidates,
    )


def build_tf_rd_009_upper_extension_gate_queue_rows(
    *,
    selection: TfRd009UpperExtensionSelection | None = None,
) -> tuple[dict[str, Any], ...]:
    resolved_selection = selection or select_tf_rd_009_upper_extension()
    shared_hypothesis = (
        "The reopened upper-family gate should widen the validation L(N,S) design space "
        "without reopening any non-scaling knobs, even if a row is weaker than 152x5 "
        "on the matched-budget benchmark objective."
    )
    notes = [
        "This gate is law-information-first: do not reinterpret it as a best-model chase.",
        (
            "Selection artifact: "
            f"`{TF_RD_009_UPPER_EXTENSION_SELECTION_JSON}` and "
            f"`{TF_RD_009_UPPER_EXTENSION_SELECTION_MD}`."
        ),
        (
            "Do not replace the frozen preferred RTX 8000 baseline from medium evidence alone; "
            "any new preferred row still requires a fresh one-row large-rung validation gate."
        ),
    ]
    return tuple(
        {
            "order": order,
            "delta_ref": row.delta_id,
            "status": "ready",
            "rationale": (
                f"Gate `{row.row_label}` at the carried fixed-budget row before spending "
                "NS-ladder budget on the reopened TF-RD-009 upper family."
            ),
            "hypothesis": shared_hypothesis,
            "anchor_delta": (
                "Keep the closed TF-RD-010 medium benchmark contract, TF-RD-022 runtime "
                "surface, TF-RD-024 non-scaling sandwich freeze, and matched regime budget "
                f"fixed while changing only geometry to `{row.row_label}`."
            ),
            "training": {
                "overrides": {
                    "runtime": {
                        "max_steps": TF_RD_009_UPPER_EXTENSION_GATE_STEPS,
                    },
                    "schedule": {
                        "stages": [
                            {
                                "name": "prior_dump",
                                "steps": TF_RD_009_UPPER_EXTENSION_GATE_STEPS,
                            }
                        ]
                    },
                }
            },
            "parameter_adequacy_plan": [
                (
                    "Run only at the carried fixed-budget gate row "
                    f"`max_steps={TF_RD_009_UPPER_EXTENSION_GATE_STEPS}` before any "
                    "upper-family NS expansion."
                ),
                (
                    "Promote to `tf_rd_009_ns_upper_extension_medium_v1` only if the row is "
                    "benchmark-backed and returns health=`ok`; keep `warn` rows as "
                    "upper-family evidence only."
                ),
                (
                    "Use this row for validation L(N,S) information gain even when it does not "
                    "beat `152x5` on `final_log_loss_at_matched_regime_budget`."
                ),
            ],
            "run_id": None,
            "followup_run_ids": [],
            "decision": None,
            "interpretation_status": "pending",
            "confounders": [],
            "next_action": (
                f"Execute gate order {order} in `{TF_RD_009_UPPER_EXTENSION_GATE_SWEEP_ID}` "
                "and expand only after health=`ok`."
            ),
            "notes": list(notes),
        }
        for order, row in enumerate(resolved_selection.selected_candidate.rows, start=1)
    )


def promoted_tf_rd_009_upper_extension_row_labels(
    *,
    gate_rows: Sequence[Mapping[str, Any]],
    registry_path: Path | None = None,
) -> tuple[str, ...]:
    resolved_registry_path = (
        registry_path or default_benchmark_run_registry_path()
    ).expanduser().resolve()
    promoted: list[str] = []
    for row in gate_rows:
        status = str(row.get("status") or "").strip().lower()
        if status != "completed":
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        run_entry = load_benchmark_run_entry(run_id, path=resolved_registry_path)
        verdict = _run_health_verdict(run_entry)
        if verdict != "ok":
            continue
        promoted.append(_row_label_from_row_payload(row))
    return tuple(promoted)


def build_tf_rd_009_upper_extension_ns_queue_rows(
    *,
    promoted_row_labels: Sequence[str],
    selection: TfRd009UpperExtensionSelection | None = None,
) -> tuple[dict[str, Any], ...]:
    resolved_selection = selection or select_tf_rd_009_upper_extension()
    promoted = set(promoted_row_labels)
    rows: list[dict[str, Any]] = []
    order = 1
    for row in resolved_selection.selected_candidate.rows:
        if row.row_label not in promoted:
            continue
        for step in TF_RD_009_UPPER_EXTENSION_STEP_LADDER:
            rows.append(
                {
                    "order": order,
                    "delta_ref": row.delta_id,
                    "status": "ready",
                    "rationale": (
                        f"Execute `{row.row_label}` at `max_steps={step}` for the reopened "
                        "TF-RD-009 upper-family NS extension."
                    ),
                    "hypothesis": (
                        "Healthy upper-family survivors should expand into the validation "
                        "L(N,S) ladder even if they are not the current matched-budget winner."
                    ),
                    "anchor_delta": (
                        "Reuse the locked medium benchmark surface and change only geometry or "
                        "step budget for the upper-family extension NS ladder."
                    ),
                    "training": {
                        "overrides": {
                            "runtime": {"max_steps": step},
                            "schedule": {"stages": [{"name": "prior_dump", "steps": step}]},
                        }
                    },
                    "parameter_adequacy_plan": [
                        (
                            "Only materialize this NS ladder after the corresponding fixed-budget "
                            "gate row returns health=`ok`."
                        ),
                        (
                            "Keep validation `L(N,S)` as the primary fit target and treat the "
                            "benchmark objective as transfer evidence."
                        ),
                        (
                            "Do not freeze a new preferred RTX 8000 baseline from this medium-only "
                            "evidence; require a fresh large-rung validation gate first."
                        ),
                    ],
                    "run_id": None,
                    "followup_run_ids": [],
                    "decision": None,
                    "interpretation_status": "pending",
                    "confounders": [],
                    "next_action": (
                        f"Execute order {order} in `{TF_RD_009_UPPER_EXTENSION_NS_SWEEP_ID}` "
                        "after the gate row is benchmark-backed and healthy."
                    ),
                    "notes": [
                        (
                            "This row was materialized by the deterministic upper-extension "
                            "promotion rule from a health=`ok` fixed-budget gate outcome."
                        )
                    ],
                }
            )
            order += 1
    return tuple(rows)


def render_tf_rd_009_upper_extension_selection_markdown(
    selection: TfRd009UpperExtensionSelection,
) -> str:
    fit_parameters = selection.current_validation_fit["parameters"]
    lines = [
        "# TF-RD-009 Upper-Family Selection",
        "",
        "## Decision",
        "",
        f"- Selected continuation: `{' -> '.join(selection.selected_row_labels)}`",
        "- Primary score: maximize D-optimal information gain on the current validation `L(N,S)` fit.",
        "- Secondary score: minimize projected parameter-uncertainty width for `alpha_n` and `alpha_s`.",
        "- Tie-breaks: fewer new geometries, then lower max predicted reserved VRAM.",
        "",
        "## Current Validation L(N,S)",
        "",
        f"- `alpha_n = {fit_parameters['alpha_n']:.9f}`",
        f"- `alpha_s = {fit_parameters['alpha_s']:.9f}`",
        f"- `Nc = {fit_parameters['Nc']:.3f}`",
        f"- `Sc = {fit_parameters['Sc']:.3f}`",
        f"- `irreducible_loss = {fit_parameters['irreducible_loss']:.12f}`",
        "",
        "## Candidate Scores",
        "",
        "| Continuation | Rows | D-opt gain | alpha width | Max predicted reserved GB |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for candidate in selection.candidates:
        lines.append(
            "| "
            f"`{candidate.candidate_id}` | "
            f"`{' -> '.join(candidate.row_labels)}` | "
            f"{candidate.d_optimal_gain:.6f} | "
            f"{candidate.alpha_uncertainty_width:.6f} | "
            f"{candidate.max_predicted_reserved_vram_gb:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Run the selected rows first at the carried fixed-budget gate row.",
            "- Promote only health=`ok` survivors into the full `{625,1250,2500,5000}` NS ladder.",
            "- Keep health=`warn` rows as upper-family evidence only.",
            "- Require a fresh one-row large-rung validation before replacing the frozen preferred RTX 8000 baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def _projected_row(
    *,
    d_icl: int,
    layers: int,
    target_total_params: float,
    parameter_bridge: AffineWidthDepthParameterFit,
    canonical_bridge: AffineWidthDepthParameterFit,
    vram_fit: LinearFit,
) -> TfRd009UpperExtensionRow:
    predicted_total_params = parameter_bridge.predict_total_params(
        d_icl=d_icl,
        layers=layers,
    )
    predicted_canonical_non_embedding_params = canonical_bridge.predict_total_params(
        d_icl=d_icl,
        layers=layers,
    )
    predicted_reserved_vram_gb = vram_fit.predict(predicted_total_params)
    if predicted_reserved_vram_gb is None:
        raise RuntimeError("corrected TF-RD-009 VRAM fit should always be available")
    return TfRd009UpperExtensionRow(
        row_label=f"{d_icl}x{layers}",
        d_icl=d_icl,
        layers=layers,
        target_total_params=float(target_total_params),
        predicted_total_params=float(predicted_total_params),
        predicted_canonical_non_embedding_params=float(predicted_canonical_non_embedding_params),
        predicted_reserved_vram_gb=float(predicted_reserved_vram_gb),
    )


def _projected_ns_coordinates(
    rows: Sequence[ScalingStudyRunPoint] | Sequence[TfRd009UpperExtensionRow],
) -> tuple[tuple[float, float], ...]:
    coordinates: list[tuple[float, float]] = []
    for row in rows:
        if isinstance(row, ScalingStudyRunPoint):
            coordinates.append((float(row.n), float(row.s)))
            continue
        for step in TF_RD_009_UPPER_EXTENSION_STEP_LADDER:
            coordinates.append((float(row.predicted_canonical_non_embedding_params), float(step)))
    return tuple(coordinates)


def _prediction_from_log_parameters(
    *,
    theta: np.ndarray,
    n_value: float,
    s_value: float,
) -> float:
    parameters = {
        name: float(math.exp(value))
        for name, value in zip(_FIT_PARAMETER_ORDER, theta, strict=True)
    }
    return float(
        parameters["irreducible_loss"]
        + (
            (parameters["Nc"] / float(n_value))
            ** (parameters["alpha_n"] / parameters["alpha_s"])
            + (parameters["Sc"] / float(s_value))
        )
        ** parameters["alpha_s"]
    )


def _jacobian_for_point(
    *,
    theta: np.ndarray,
    n_value: float,
    s_value: float,
) -> np.ndarray:
    gradient = np.zeros(len(_FIT_PARAMETER_ORDER), dtype=float)
    for index in range(len(_FIT_PARAMETER_ORDER)):
        theta_plus = np.array(theta, dtype=float)
        theta_minus = np.array(theta, dtype=float)
        theta_plus[index] += TF_RD_009_UPPER_EXTENSION_JACOBIAN_EPSILON
        theta_minus[index] -= TF_RD_009_UPPER_EXTENSION_JACOBIAN_EPSILON
        gradient[index] = (
            _prediction_from_log_parameters(theta=theta_plus, n_value=n_value, s_value=s_value)
            - _prediction_from_log_parameters(theta=theta_minus, n_value=n_value, s_value=s_value)
        ) / (2.0 * TF_RD_009_UPPER_EXTENSION_JACOBIAN_EPSILON)
    return gradient


def _information_matrix_from_rows(
    rows: Sequence[tuple[float, float]],
    *,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    theta = np.array(
        [math.log(float(parameters[name])) for name in _FIT_PARAMETER_ORDER],
        dtype=float,
    )
    information = np.zeros((len(_FIT_PARAMETER_ORDER), len(_FIT_PARAMETER_ORDER)), dtype=float)
    for n_value, s_value in rows:
        gradient = _jacobian_for_point(theta=theta, n_value=float(n_value), s_value=float(s_value))
        information += np.outer(gradient, gradient)
    return information


def _alpha_uncertainty_width(information: np.ndarray) -> float:
    covariance = np.linalg.pinv(information)
    alpha_n_index = _FIT_PARAMETER_ORDER.index("alpha_n")
    alpha_s_index = _FIT_PARAMETER_ORDER.index("alpha_s")
    return float(
        math.sqrt(max(0.0, float(covariance[alpha_n_index, alpha_n_index])))
        + math.sqrt(max(0.0, float(covariance[alpha_s_index, alpha_s_index])))
    )


def _stable_logdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        raise RuntimeError("TF-RD-009 upper-extension information matrix was not positive definite")
    return float(logdet)


def _row_label_from_row_payload(row: Mapping[str, Any]) -> str:
    model = row.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("gate row payload missing model mapping")
    d_icl = model.get("d_icl")
    layers = model.get("sandwich_layers")
    if d_icl is None or layers is None:
        raise RuntimeError("gate row payload missing d_icl or sandwich_layers")
    return f"{int(d_icl)}x{int(layers)}"


def _run_health_verdict(run_entry: Mapping[str, Any]) -> str | None:
    artifacts = run_entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    run_dir = artifacts.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        return None
    try:
        payload = health_check(resolve_registry_path_value(run_dir))
    except RuntimeError:
        return None
    verdict = payload.get("verdict")
    return None if verdict is None else str(verdict)
