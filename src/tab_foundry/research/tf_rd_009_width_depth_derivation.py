"""Executable derivation helpers for the TF-RD-009 width-depth family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tab_foundry.bench.width_depth_scaling import (
    AffineWidthDepthParameterFit,
    LinearFit,
    PowerLawFit,
    WidthDepthParameterPoint,
    fit_affine_width_depth_parameter_bridge,
    fit_linear,
    fit_power_law,
    log_space_parameter_targets,
    round_to_width_rung,
)
from tab_foundry.benchmark_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_registry,
    load_benchmark_run_entry,
)
from tab_foundry.hardware_architecture_registry import load_hardware_architecture_registry
from tab_foundry.research.sweep.materialize import load_system_delta_queue


TF_RD_009_WIDTH_DEPTH_SWEEP_ID = "tf_rd_009_width_depth_medium_v1"
TF_RD_009_REPORTED_FIT_ROW_LABELS = (
    "72x1",
    "96x2",
    "112x3",
    "128x4",
    "152x5",
    "176x6",
)
TF_RD_009_REPORTED_FIT_OBJECTIVE_METRIC = "final_log_loss_at_matched_regime_budget"
FORMAL_ANCHOR_RUN_ID = (
    "sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_"
    "delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2"
)
CARRIED_BASELINE_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1"
UPPER_WIDTH_EVIDENCE_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
PRACTICAL_WIDTH_RUNG = 8
CEILING_RESERVED_VRAM_TARGET_GB = 32.5
TF_RD_009_MUON_WIDTH_SCREEN_SWEEP_ID = "tf_rd_009_muon_width_screen_medium_v1"
TF_RD_009_MUON_WIDTH_DEPTH_SWEEP_ID = "tf_rd_009_muon_width_depth_medium_v1"
TF_RD_009_MUON_FORMAL_ANCHOR_ROW_LABEL = "60x2"
TF_RD_009_MUON_CARRIED_BASELINE_ROW_LABEL = "128x2"
TF_RD_009_MUON_FORMAL_ANCHOR_RUN_ID = (
    "sd_tf_rd_009_muon_width_screen_medium_v1_01_"
    "delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1"
)
TF_RD_009_MUON_CARRIED_BASELINE_RUN_ID = (
    "sd_tf_rd_009_muon_width_screen_medium_v1_04_"
    "delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
)
TF_RD_009_MUON_REPORTED_FIT_ROW_LABELS = (
    "72x1",
    "112x3",
    "128x2",
    "144x4",
    "192x5",
    "264x6",
)
TF_RD_009_FROZEN_HARDWARE_BASELINE_ID = "tf_rd_009_rtx8000_44gb_classification_medium_v1"


@dataclass(frozen=True, slots=True)
class TfRd009ObservedPoint:
    row_label: str
    d_icl: int
    layers: int
    total_params: int
    reserved_vram_gb: float | None = None
    train_wall_seconds: float | None = None
    run_id: str | None = None

    @property
    def effective_size(self) -> int:
        return int(self.layers * (self.d_icl**2))

    def as_parameter_point(self) -> WidthDepthParameterPoint:
        return WidthDepthParameterPoint(
            d_icl=self.d_icl,
            layers=self.layers,
            total_params=float(self.total_params),
            row_label=self.row_label,
        )


@dataclass(frozen=True, slots=True)
class TfRd009DerivedRow:
    row_label: str
    d_icl: int
    layers: int
    raw_d_icl: float
    target_params: float
    predicted_total_params: float
    predicted_reserved_vram_gb: float | None
    predicted_train_wall_seconds: float | None

    @property
    def effective_size(self) -> int:
        return int(self.layers * (self.d_icl**2))

    @property
    def delta_id(self) -> str:
        return f"delta_tf_rd_009_cls_sandwich_dicl{self.d_icl}_layers{self.layers}_v1"


@dataclass(frozen=True, slots=True)
class TfRd009WidthDepthDerivation:
    parameter_bridge: AffineWidthDepthParameterFit
    vram_fit: LinearFit
    train_fit: LinearFit
    formal_anchor: TfRd009ObservedPoint
    carried_baseline: TfRd009ObservedPoint
    upper_width_evidence: TfRd009ObservedPoint
    historical_joint_draft_points: tuple[TfRd009ObservedPoint, ...]
    lower_seed: TfRd009DerivedRow
    upper_seed: TfRd009DerivedRow
    interpolated_rows: tuple[TfRd009DerivedRow, ...]
    ceiling_probe: TfRd009DerivedRow

    @property
    def queue_rows(self) -> tuple[TfRd009DerivedRow, ...]:
        return (
            self.lower_seed,
            self.upper_seed,
            *self.interpolated_rows,
            self.ceiling_probe,
        )

    @property
    def in_family_row_labels(self) -> tuple[str, ...]:
        ordered_rows = sorted(
            (
                (self.lower_seed.row_label, float(self.lower_seed.predicted_total_params)),
                (self.carried_baseline.row_label, float(self.carried_baseline.total_params)),
                (self.upper_seed.row_label, float(self.upper_seed.predicted_total_params)),
                *(
                    (row.row_label, float(row.predicted_total_params))
                    for row in self.interpolated_rows
                ),
                (self.ceiling_probe.row_label, float(self.ceiling_probe.predicted_total_params)),
            ),
            key=lambda item: item[1],
        )
        return tuple(row_label for row_label, _ in ordered_rows)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TfRd009MeasuredLawFitPoint:
    row_label: str
    d_icl: int
    layers: int
    run_id: str
    total_params: int
    final_log_loss: float

    @property
    def effective_size(self) -> int:
        return int(self.layers * (self.d_icl**2))


@dataclass(frozen=True, slots=True)
class TfRd009MeasuredPowerLawFit:
    fit: PowerLawFit
    points: tuple[TfRd009MeasuredLawFitPoint, ...]
    x_axis: str = "model_size.total_params"
    y_axis: str = TF_RD_009_REPORTED_FIT_OBJECTIVE_METRIC
    fit_family: str = "power_law_log_log"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _registry_point_from_run(run_id: str, *, registry_path: Path | None = None) -> TfRd009ObservedPoint:
    entry = load_benchmark_run_entry(run_id, path=registry_path)
    model = entry["model"]
    build_spec = model["build_spec"]
    return TfRd009ObservedPoint(
        row_label=f"{int(model['d_icl'])}x{int(build_spec['sandwich_layers'])}",
        d_icl=int(model["d_icl"]),
        layers=int(build_spec["sandwich_layers"]),
        total_params=int(entry["model_size"]["total_params"]),
        reserved_vram_gb=(
            None
            if entry.get("runtime_summary", {}).get("peak_vram_reserved") is None
            else float(entry["runtime_summary"]["peak_vram_reserved"]) / float(1024**3)
        ),
        train_wall_seconds=(
            None
            if entry.get("training_diagnostics", {}).get("wall_elapsed_seconds") is None
            else float(entry["training_diagnostics"]["wall_elapsed_seconds"])
        ),
        run_id=run_id,
    )


def load_tf_rd_009_registry_evidence(
    *,
    registry_path: Path | None = None,
) -> tuple[TfRd009ObservedPoint, TfRd009ObservedPoint, TfRd009ObservedPoint]:
    resolved_registry_path = (registry_path or default_benchmark_run_registry_path()).expanduser().resolve()
    return (
        _registry_point_from_run(FORMAL_ANCHOR_RUN_ID, registry_path=resolved_registry_path),
        _registry_point_from_run(CARRIED_BASELINE_RUN_ID, registry_path=resolved_registry_path),
        _registry_point_from_run(UPPER_WIDTH_EVIDENCE_RUN_ID, registry_path=resolved_registry_path),
    )


def historical_tf_rd_009_joint_draft_points() -> tuple[TfRd009ObservedPoint, ...]:
    """Legacy mixed-depth bridge evidence used for queue construction only."""

    return (
        TfRd009ObservedPoint(row_label="88x1", d_icl=88, layers=1, total_params=986886),
        TfRd009ObservedPoint(row_label="104x3", d_icl=104, layers=3, total_params=2419862),
        TfRd009ObservedPoint(row_label="112x4", d_icl=112, layers=4, total_params=3410046),
        TfRd009ObservedPoint(row_label="128x5", d_icl=128, layers=5, total_params=5234830),
        TfRd009ObservedPoint(row_label="144x6", d_icl=144, layers=6, total_params=7615262),
    )


def _model_payload_value(model_payload: Mapping[str, Any], key: str) -> Any:
    if key in model_payload:
        return model_payload.get(key)
    build_spec = model_payload.get("build_spec")
    if isinstance(build_spec, Mapping):
        return build_spec.get(key)
    return None


def _canonical_total_params_for_model_payload(
    *,
    model_payload: Mapping[str, Any],
    registry_path: Path | None = None,
) -> int:
    registry = load_benchmark_run_registry(registry_path)
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        raise RuntimeError("benchmark run registry is missing runs payload")
    observed_total_params: set[int] = set()
    key_fields = tuple(str(key) for key in model_payload.keys())
    for entry in runs.values():
        if not isinstance(entry, Mapping):
            continue
        model = entry.get("model")
        if not isinstance(model, Mapping):
            continue
        if any(
            _model_payload_value(model, key) != model_payload.get(key)
            for key in key_fields
        ):
            continue
        model_size = entry.get("model_size")
        if not isinstance(model_size, Mapping) or model_size.get("total_params") is None:
            continue
        observed_total_params.add(int(model_size["total_params"]))
    if not observed_total_params:
        raise RuntimeError(
            "benchmark run registry is missing canonical parameter-count evidence for "
            f"model payload {dict(model_payload)!r}"
        )
    if len(observed_total_params) != 1:
        raise RuntimeError(
            "benchmark run registry recorded inconsistent parameter counts for "
            f"model payload {dict(model_payload)!r}: {sorted(observed_total_params)}"
        )
    return next(iter(observed_total_params))


def load_tf_rd_009_muon_width_screen_evidence(
    *,
    registry_path: Path | None = None,
    sweep_id: str = TF_RD_009_MUON_WIDTH_SCREEN_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[TfRd009ObservedPoint, TfRd009ObservedPoint]:
    resolved_registry_path = (registry_path or default_benchmark_run_registry_path()).expanduser().resolve()
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    required_labels = {
        TF_RD_009_MUON_FORMAL_ANCHOR_ROW_LABEL,
        TF_RD_009_MUON_CARRIED_BASELINE_ROW_LABEL,
    }
    observed_points: dict[str, TfRd009ObservedPoint] = {}
    for row in _ordered_queue_rows(queue):
        model_payload = row.get("model")
        if not isinstance(model_payload, Mapping):
            continue
        row_label = _row_label_from_model_payload(model_payload)
        if row_label not in required_labels:
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError(
                "Muon width-screen evidence rows must record benchmark-backed run ids: "
                f"row_label={row_label!r}"
            )
        observed_points[row_label] = TfRd009ObservedPoint(
            row_label=row_label,
            d_icl=int(model_payload["d_icl"]),
            layers=int(model_payload["sandwich_layers"]),
            total_params=_canonical_total_params_for_model_payload(
                model_payload=model_payload,
                registry_path=resolved_registry_path,
            ),
            run_id=str(run_id),
        )
    missing_labels = required_labels.difference(observed_points)
    if missing_labels:
        raise RuntimeError(
            "Muon width-screen sweep is missing canonical evidence rows required for derivation: "
            f"missing={sorted(missing_labels)}"
        )
    return (
        observed_points[TF_RD_009_MUON_FORMAL_ANCHOR_ROW_LABEL],
        observed_points[TF_RD_009_MUON_CARRIED_BASELINE_ROW_LABEL],
    )


def load_tf_rd_009_frozen_hardware_constraint_fits(
    *,
    registry_path: Path | None = None,
    baseline_id: str = TF_RD_009_FROZEN_HARDWARE_BASELINE_ID,
) -> tuple[AffineWidthDepthParameterFit, LinearFit, LinearFit]:
    registry = load_hardware_architecture_registry(registry_path)
    baselines = registry.get("baselines")
    if not isinstance(baselines, Mapping):
        raise RuntimeError("hardware architecture registry is missing baselines payload")
    baseline = baselines.get(baseline_id)
    if not isinstance(baseline, Mapping):
        raise RuntimeError(f"hardware architecture baseline {baseline_id!r} is missing")
    constraint_model = baseline.get("constraint_model")
    if not isinstance(constraint_model, Mapping):
        raise RuntimeError(
            f"hardware architecture baseline {baseline_id!r} is missing constraint_model"
        )
    formulas = constraint_model.get("formulas")
    if not isinstance(formulas, Mapping):
        raise RuntimeError(
            f"hardware architecture baseline {baseline_id!r} is missing constraint formulas"
        )

    def _require_formula(formula_name: str) -> Mapping[str, Any]:
        formula = formulas.get(formula_name)
        if not isinstance(formula, Mapping):
            raise RuntimeError(
                f"hardware architecture baseline {baseline_id!r} is missing formula {formula_name!r}"
            )
        coefficients = formula.get("coefficients")
        if not isinstance(coefficients, Mapping):
            raise RuntimeError(
                f"hardware architecture baseline {baseline_id!r} formula {formula_name!r} "
                "is missing coefficients"
            )
        return formula

    parameter_formula = _require_formula("parameter_count")
    parameter_coefficients = parameter_formula["coefficients"]
    vram_formula = _require_formula("reserved_vram_gb")
    vram_coefficients = vram_formula["coefficients"]
    train_formula = _require_formula("train_wall_seconds")
    train_coefficients = train_formula["coefficients"]

    return (
        AffineWidthDepthParameterFit(
            intercept=float(parameter_coefficients["intercept"]),
            d_squared_coefficient=float(parameter_coefficients["d_squared_coefficient"]),
            layered_d_squared_coefficient=float(parameter_coefficients["layered_d_squared_coefficient"]),
        ),
        LinearFit(
            intercept=float(vram_coefficients["intercept"]),
            slope=float(vram_coefficients["slope"]),
            fit_kind=str(vram_formula["fit_kind"]),
        ),
        LinearFit(
            intercept=float(train_coefficients["intercept"]),
            slope=float(train_coefficients["slope"]),
            fit_kind=str(train_formula["fit_kind"]),
        ),
    )


def _derive_row_from_target(
    *,
    parameter_bridge: AffineWidthDepthParameterFit,
    vram_fit: LinearFit,
    train_fit: LinearFit,
    layers: int,
    target_params: float,
    width_rung: int,
) -> TfRd009DerivedRow:
    raw_d_icl = parameter_bridge.solve_width_for_target_params(
        layers=layers,
        target_params=target_params,
    )
    d_icl = round_to_width_rung(raw_d_icl, rung=width_rung)
    predicted_total_params = parameter_bridge.predict_total_params(d_icl=d_icl, layers=layers)
    return TfRd009DerivedRow(
        row_label=f"{d_icl}x{layers}",
        d_icl=d_icl,
        layers=layers,
        raw_d_icl=raw_d_icl,
        target_params=float(target_params),
        predicted_total_params=predicted_total_params,
        predicted_reserved_vram_gb=vram_fit.predict(predicted_total_params),
        predicted_train_wall_seconds=train_fit.predict(predicted_total_params),
    )


def _build_tf_rd_009_width_depth_derivation(
    *,
    parameter_bridge: AffineWidthDepthParameterFit,
    vram_fit: LinearFit,
    train_fit: LinearFit,
    formal_anchor: TfRd009ObservedPoint,
    carried_baseline: TfRd009ObservedPoint,
    upper_width_evidence: TfRd009ObservedPoint,
    historical_joint_draft_points: tuple[TfRd009ObservedPoint, ...],
    width_rung: int,
    ceiling_reserved_vram_target_gb: float,
) -> TfRd009WidthDepthDerivation:
    lower_seed = _derive_row_from_target(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        layers=1,
        target_params=float(formal_anchor.total_params),
        width_rung=width_rung,
    )
    upper_seed = _derive_row_from_target(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        layers=3,
        target_params=float(upper_width_evidence.total_params),
        width_rung=width_rung,
    )
    ceiling_target_params = vram_fit.solve_x(ceiling_reserved_vram_target_gb)
    ceiling_probe = _derive_row_from_target(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        layers=6,
        target_params=ceiling_target_params,
        width_rung=width_rung,
    )
    interpolated_targets = log_space_parameter_targets(
        start_value=upper_seed.predicted_total_params,
        end_value=ceiling_probe.predicted_total_params,
        count=2,
    )
    interpolated_rows = (
        _derive_row_from_target(
            parameter_bridge=parameter_bridge,
            vram_fit=vram_fit,
            train_fit=train_fit,
            layers=4,
            target_params=interpolated_targets[0],
            width_rung=width_rung,
        ),
        _derive_row_from_target(
            parameter_bridge=parameter_bridge,
            vram_fit=vram_fit,
            train_fit=train_fit,
            layers=5,
            target_params=interpolated_targets[1],
            width_rung=width_rung,
        ),
    )
    return TfRd009WidthDepthDerivation(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        formal_anchor=formal_anchor,
        carried_baseline=carried_baseline,
        upper_width_evidence=upper_width_evidence,
        historical_joint_draft_points=historical_joint_draft_points,
        lower_seed=lower_seed,
        upper_seed=upper_seed,
        interpolated_rows=interpolated_rows,
        ceiling_probe=ceiling_probe,
    )


def derive_tf_rd_009_width_depth_family(
    *,
    registry_path: Path | None = None,
    width_rung: int = PRACTICAL_WIDTH_RUNG,
    ceiling_reserved_vram_target_gb: float = CEILING_RESERVED_VRAM_TARGET_GB,
) -> TfRd009WidthDepthDerivation:
    formal_anchor, carried_baseline, upper_width_evidence = load_tf_rd_009_registry_evidence(
        registry_path=registry_path,
    )
    historical_joint_draft_points = historical_tf_rd_009_joint_draft_points()
    parameter_bridge = fit_affine_width_depth_parameter_bridge(
        tuple(
            point.as_parameter_point()
            for point in (
                formal_anchor,
                carried_baseline,
                upper_width_evidence,
                *historical_joint_draft_points,
            )
        )
    )
    vram_fit = fit_linear(
        tuple(
            (float(point.total_params), float(point.reserved_vram_gb))
            for point in (formal_anchor, carried_baseline, upper_width_evidence)
            if point.reserved_vram_gb is not None
        )
    )
    train_fit = fit_linear(
        tuple(
            (float(point.total_params), float(point.train_wall_seconds))
            for point in (formal_anchor, carried_baseline, upper_width_evidence)
            if point.train_wall_seconds is not None
        )
    )
    return _build_tf_rd_009_width_depth_derivation(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        formal_anchor=formal_anchor,
        carried_baseline=carried_baseline,
        upper_width_evidence=upper_width_evidence,
        historical_joint_draft_points=historical_joint_draft_points,
        width_rung=width_rung,
        ceiling_reserved_vram_target_gb=ceiling_reserved_vram_target_gb,
    )


def derive_tf_rd_009_muon_width_depth_family(
    *,
    registry_path: Path | None = None,
    hardware_registry_path: Path | None = None,
    width_rung: int = PRACTICAL_WIDTH_RUNG,
    ceiling_reserved_vram_target_gb: float = CEILING_RESERVED_VRAM_TARGET_GB,
    width_screen_sweep_id: str = TF_RD_009_MUON_WIDTH_SCREEN_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> TfRd009WidthDepthDerivation:
    formal_anchor, carried_baseline = load_tf_rd_009_muon_width_screen_evidence(
        registry_path=registry_path,
        sweep_id=width_screen_sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    parameter_bridge, vram_fit, train_fit = load_tf_rd_009_frozen_hardware_constraint_fits(
        registry_path=hardware_registry_path,
    )
    return _build_tf_rd_009_width_depth_derivation(
        parameter_bridge=parameter_bridge,
        vram_fit=vram_fit,
        train_fit=train_fit,
        formal_anchor=formal_anchor,
        carried_baseline=carried_baseline,
        upper_width_evidence=carried_baseline,
        historical_joint_draft_points=(),
        width_rung=width_rung,
        ceiling_reserved_vram_target_gb=ceiling_reserved_vram_target_gb,
    )


def _row_label_from_model_payload(model_payload: Mapping[str, Any]) -> str:
    raw_d_icl = model_payload.get("d_icl")
    if raw_d_icl is None:
        raise RuntimeError("queue row model payload must include d_icl")
    raw_layers = model_payload.get("sandwich_layers")
    if raw_layers is None:
        build_spec = model_payload.get("build_spec")
        if isinstance(build_spec, Mapping):
            raw_layers = build_spec.get("sandwich_layers")
    if raw_layers is None:
        raise RuntimeError("queue row model payload must include sandwich_layers")
    return f"{int(raw_d_icl)}x{int(raw_layers)}"


def _ordered_queue_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = queue.get("rows")
    if not isinstance(raw_rows, list):
        raise RuntimeError("TF-RD-009 queue payload must include a rows list")
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("TF-RD-009 queue rows must be mappings")
        rows.append({str(key): value for key, value in row.items()})
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("order", 0)),
            str(row.get("delta_id") or row.get("delta_ref") or ""),
        ),
    )


def _measured_law_fit_point_from_registry_run(
    run_id: str,
    *,
    registry_path: Path | None = None,
) -> TfRd009MeasuredLawFitPoint:
    entry = load_benchmark_run_entry(run_id, path=registry_path)
    model = entry.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError(f"benchmark registry run {run_id!r} is missing model payload")
    row_label = _row_label_from_model_payload(model)
    regime_budget = entry.get("regime_budget")
    if not isinstance(regime_budget, Mapping):
        raise RuntimeError(f"benchmark registry run {run_id!r} is missing regime_budget payload")
    objective_metric = regime_budget.get("objective_metric")
    if objective_metric != TF_RD_009_REPORTED_FIT_OBJECTIVE_METRIC:
        raise RuntimeError(
            "TF-RD-009 reported fit requires matched-budget objective metric; "
            f"run {run_id!r} recorded {objective_metric!r}"
        )
    model_size = entry.get("model_size")
    if not isinstance(model_size, Mapping) or model_size.get("total_params") is None:
        raise RuntimeError(f"benchmark registry run {run_id!r} is missing model_size.total_params")
    metrics = entry.get("tab_foundry_metrics")
    if not isinstance(metrics, Mapping) or metrics.get("final_log_loss") is None:
        raise RuntimeError(f"benchmark registry run {run_id!r} is missing tab_foundry_metrics.final_log_loss")
    build_spec = model.get("build_spec")
    layers = model.get("sandwich_layers")
    if layers is None and isinstance(build_spec, Mapping):
        layers = build_spec.get("sandwich_layers")
    if layers is None:
        raise RuntimeError(f"benchmark registry run {run_id!r} is missing sandwich_layers")
    return TfRd009MeasuredLawFitPoint(
        row_label=row_label,
        d_icl=int(model["d_icl"]),
        layers=int(layers),
        run_id=str(run_id),
        total_params=int(model_size["total_params"]),
        final_log_loss=float(metrics["final_log_loss"]),
    )


def _completed_queue_row_for_measured_fit(row: Mapping[str, Any]) -> bool:
    if str(row.get("interpretation_status", "")).strip().lower() != "completed":
        return False
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return False
    benchmark_metrics = row.get("benchmark_metrics")
    if not isinstance(benchmark_metrics, Mapping):
        return False
    if benchmark_metrics.get("final_log_loss") is None:
        return False
    return benchmark_metrics.get("objective_metric") == TF_RD_009_REPORTED_FIT_OBJECTIVE_METRIC


def _collect_tf_rd_009_completed_measured_fit_points(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    reported_fit_row_labels: tuple[str, ...],
    default_anchor_run_id: str,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    points_by_label: dict[str, TfRd009MeasuredLawFitPoint] = {}
    carried_baseline_run_id = str(queue.get("anchor_run_id") or default_anchor_run_id)
    baseline_point = _measured_law_fit_point_from_registry_run(
        carried_baseline_run_id,
        registry_path=registry_path,
    )
    if baseline_point.row_label in reported_fit_row_labels:
        points_by_label[baseline_point.row_label] = baseline_point
    for row in _ordered_queue_rows(queue):
        model_payload = row.get("model")
        if not isinstance(model_payload, Mapping):
            continue
        row_label = _row_label_from_model_payload(model_payload)
        if row_label not in reported_fit_row_labels:
            continue
        if not _completed_queue_row_for_measured_fit(row):
            continue
        run_id = row.get("run_id")
        assert isinstance(run_id, str)
        point = _measured_law_fit_point_from_registry_run(run_id, registry_path=registry_path)
        if point.row_label != row_label:
            raise RuntimeError(
                "TF-RD-009 completed row run_id does not match the queue row geometry: "
                f"queue={row_label!r}, registry={point.row_label!r}, run_id={run_id!r}"
            )
        points_by_label[row_label] = point
    return tuple(
        points_by_label[row_label]
        for row_label in reported_fit_row_labels
        if row_label in points_by_label
    )


def collect_tf_rd_009_completed_measured_fit_points(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    return _collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=CARRIED_BASELINE_RUN_ID,
    )


def collect_tf_rd_009_muon_completed_measured_fit_points(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    return _collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_MUON_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=TF_RD_009_MUON_CARRIED_BASELINE_RUN_ID,
    )


def _load_tf_rd_009_completed_measured_fit_points(
    *,
    queue_path: Path | None = None,
    registry_path: Path | None = None,
    sweep_id: str,
    reported_fit_row_labels: tuple[str, ...],
    default_anchor_run_id: str,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    queue = load_system_delta_queue(
        queue_path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    return _collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=reported_fit_row_labels,
        default_anchor_run_id=default_anchor_run_id,
    )


def load_tf_rd_009_completed_measured_fit_points(
    *,
    queue_path: Path | None = None,
    registry_path: Path | None = None,
    sweep_id: str = TF_RD_009_WIDTH_DEPTH_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    return _load_tf_rd_009_completed_measured_fit_points(
        queue_path=queue_path,
        registry_path=registry_path,
        sweep_id=sweep_id,
        reported_fit_row_labels=TF_RD_009_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=CARRIED_BASELINE_RUN_ID,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )


def load_tf_rd_009_muon_completed_measured_fit_points(
    *,
    queue_path: Path | None = None,
    registry_path: Path | None = None,
    sweep_id: str = TF_RD_009_MUON_WIDTH_DEPTH_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> tuple[TfRd009MeasuredLawFitPoint, ...]:
    return _load_tf_rd_009_completed_measured_fit_points(
        queue_path=queue_path,
        registry_path=registry_path,
        sweep_id=sweep_id,
        reported_fit_row_labels=TF_RD_009_MUON_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=TF_RD_009_MUON_CARRIED_BASELINE_RUN_ID,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )


def _fit_tf_rd_009_completed_measured_power_law(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    reported_fit_row_labels: tuple[str, ...],
    default_anchor_run_id: str,
) -> TfRd009MeasuredPowerLawFit:
    points = _collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=reported_fit_row_labels,
        default_anchor_run_id=default_anchor_run_id,
    )
    fit = fit_power_law(
        tuple((float(point.total_params), float(point.final_log_loss)) for point in points)
    )
    return TfRd009MeasuredPowerLawFit(fit=fit, points=points)


def load_tf_rd_009_completed_measured_power_law(
    *,
    queue_path: Path | None = None,
    registry_path: Path | None = None,
    sweep_id: str = TF_RD_009_WIDTH_DEPTH_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> TfRd009MeasuredPowerLawFit:
    queue = load_system_delta_queue(
        queue_path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    return _fit_tf_rd_009_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=CARRIED_BASELINE_RUN_ID,
    )


def fit_tf_rd_009_completed_measured_power_law(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> TfRd009MeasuredPowerLawFit:
    return _fit_tf_rd_009_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=CARRIED_BASELINE_RUN_ID,
    )


def fit_tf_rd_009_muon_completed_measured_power_law(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> TfRd009MeasuredPowerLawFit:
    return _fit_tf_rd_009_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_MUON_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=TF_RD_009_MUON_CARRIED_BASELINE_RUN_ID,
    )


def load_tf_rd_009_muon_completed_measured_power_law(
    *,
    queue_path: Path | None = None,
    registry_path: Path | None = None,
    sweep_id: str = TF_RD_009_MUON_WIDTH_DEPTH_SWEEP_ID,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> TfRd009MeasuredPowerLawFit:
    queue = load_system_delta_queue(
        queue_path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    return _fit_tf_rd_009_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
        reported_fit_row_labels=TF_RD_009_MUON_REPORTED_FIT_ROW_LABELS,
        default_anchor_run_id=TF_RD_009_MUON_CARRIED_BASELINE_RUN_ID,
    )
