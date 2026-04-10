"""Executable derivation helpers for the TF-RD-009 width-depth family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tab_foundry.bench.width_depth_scaling import (
    AffineWidthDepthParameterFit,
    LinearFit,
    WidthDepthParameterPoint,
    fit_affine_width_depth_parameter_bridge,
    fit_linear,
    log_space_parameter_targets,
    round_to_width_rung,
)
from tab_foundry.benchmark_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_entry,
)


FORMAL_ANCHOR_RUN_ID = (
    "sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_"
    "delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2"
)
CARRIED_BASELINE_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1"
UPPER_WIDTH_EVIDENCE_RUN_ID = "sd_tf_rd_009_width_transfer_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
PRACTICAL_WIDTH_RUNG = 8
CEILING_RESERVED_VRAM_TARGET_GB = 32.5


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
    parameter_fit: AffineWidthDepthParameterFit
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
        return (
            self.lower_seed.row_label,
            self.carried_baseline.row_label,
            self.upper_seed.row_label,
            *(row.row_label for row in self.interpolated_rows),
            self.ceiling_probe.row_label,
        )

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
    return (
        TfRd009ObservedPoint(row_label="88x1", d_icl=88, layers=1, total_params=986886),
        TfRd009ObservedPoint(row_label="104x3", d_icl=104, layers=3, total_params=2419862),
        TfRd009ObservedPoint(row_label="112x4", d_icl=112, layers=4, total_params=3410046),
        TfRd009ObservedPoint(row_label="128x5", d_icl=128, layers=5, total_params=5234830),
        TfRd009ObservedPoint(row_label="144x6", d_icl=144, layers=6, total_params=7615262),
    )


def _derive_row_from_target(
    *,
    parameter_fit: AffineWidthDepthParameterFit,
    vram_fit: LinearFit,
    train_fit: LinearFit,
    layers: int,
    target_params: float,
    width_rung: int,
) -> TfRd009DerivedRow:
    raw_d_icl = parameter_fit.solve_width_for_target_params(
        layers=layers,
        target_params=target_params,
    )
    d_icl = round_to_width_rung(raw_d_icl, rung=width_rung)
    predicted_total_params = parameter_fit.predict_total_params(d_icl=d_icl, layers=layers)
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
    parameter_fit = fit_affine_width_depth_parameter_bridge(
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
    lower_seed = _derive_row_from_target(
        parameter_fit=parameter_fit,
        vram_fit=vram_fit,
        train_fit=train_fit,
        layers=1,
        target_params=float(formal_anchor.total_params),
        width_rung=width_rung,
    )
    upper_seed = _derive_row_from_target(
        parameter_fit=parameter_fit,
        vram_fit=vram_fit,
        train_fit=train_fit,
        layers=3,
        target_params=float(upper_width_evidence.total_params),
        width_rung=width_rung,
    )
    ceiling_target_params = vram_fit.solve_x(ceiling_reserved_vram_target_gb)
    ceiling_probe = _derive_row_from_target(
        parameter_fit=parameter_fit,
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
            parameter_fit=parameter_fit,
            vram_fit=vram_fit,
            train_fit=train_fit,
            layers=4,
            target_params=interpolated_targets[0],
            width_rung=width_rung,
        ),
        _derive_row_from_target(
            parameter_fit=parameter_fit,
            vram_fit=vram_fit,
            train_fit=train_fit,
            layers=5,
            target_params=interpolated_targets[1],
            width_rung=width_rung,
        ),
    )
    return TfRd009WidthDepthDerivation(
        parameter_fit=parameter_fit,
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
