"""Shared helpers for width-depth scaling fits and target derivations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


LINEAR_SOLVER_PIVOT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class WidthDepthParameterPoint:
    """Observed parameter count for one width-depth row."""

    d_icl: int
    layers: int
    total_params: float
    row_label: str | None = None

    @property
    def effective_size(self) -> int:
        return int(self.layers * (self.d_icl**2))

    @property
    def label(self) -> str:
        if self.row_label is not None:
            return str(self.row_label)
        return f"{self.d_icl}x{self.layers}"


@dataclass(frozen=True, slots=True)
class LinearFit:
    """Simple least-squares fit of ``y ≈ intercept + slope * x``."""

    intercept: float
    slope: float
    fit_kind: str

    def predict(self, x_value: float) -> float | None:
        if self.fit_kind == "unfit":
            return None
        return float(self.intercept + self.slope * float(x_value))

    def solve_x(self, target_y: float) -> float:
        if abs(self.slope) <= LINEAR_SOLVER_PIVOT_TOLERANCE:
            raise RuntimeError("cannot solve linear fit with zero slope")
        return float((float(target_y) - self.intercept) / self.slope)


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """Least-squares fit of ``log(y) ≈ intercept + exponent * log(x)``."""

    intercept: float
    exponent: float
    fit_kind: str

    @property
    def coefficient(self) -> float:
        if self.fit_kind == "unfit":
            return 0.0
        return float(math.exp(self.intercept))

    def expression(self, *, x_label: str = "x", y_label: str = "y") -> str:
        if self.fit_kind == "unfit":
            return f"{y_label} power-law fit unavailable"
        if self.fit_kind == "constant":
            return f"{y_label} ≈ {self.coefficient:.6g}"
        return f"log({y_label}) ≈ {self.intercept:.6f} + {self.exponent:.6f} * log({x_label})"

    def predict(self, x_value: float) -> float | None:
        if self.fit_kind == "unfit":
            return None
        if x_value <= 0.0:
            raise ValueError("power-law prediction requires a strictly positive x_value")
        return float(self.coefficient * (float(x_value) ** self.exponent))


@dataclass(frozen=True, slots=True)
class AffineWidthDepthParameterFit:
    """Mixed-depth parameter fit ``P(d, L) ≈ a + b*d^2 + c*L*d^2``."""

    intercept: float
    d_squared_coefficient: float
    layered_d_squared_coefficient: float

    def expression(self) -> str:
        return (
            f"P_local(d, L) ≈ {self.intercept:.2f} + {self.d_squared_coefficient:.2f} * d^2 "
            f"+ {self.layered_d_squared_coefficient:.2f} * L * d^2"
        )

    def predict_total_params(self, *, d_icl: int | float, layers: int | float) -> float:
        d_squared = float(d_icl) * float(d_icl)
        return float(
            self.intercept
            + self.d_squared_coefficient * d_squared
            + self.layered_d_squared_coefficient * float(layers) * d_squared
        )

    def solve_width_for_target_params(self, *, layers: int, target_params: float) -> float:
        width_coefficient = self.d_squared_coefficient + (
            self.layered_d_squared_coefficient * float(layers)
        )
        if width_coefficient <= 0.0:
            raise RuntimeError("width-depth fit is not invertible for non-positive width coefficient")
        numerator = float(target_params) - self.intercept
        if numerator <= 0.0:
            raise RuntimeError("target_params must exceed fit intercept to solve for width")
        return float(math.sqrt(numerator / width_coefficient))


def _solve_square_linear_system(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
) -> list[float] | None:
    size = len(matrix)
    if size == 0 or len(rhs) != size:
        return None
    augmented = [
        [float(value) for value in row] + [float(rhs_value)]
        for row, rhs_value in zip(matrix, rhs, strict=False)
    ]
    for row in augmented:
        if len(row) != size + 1:
            return None
    for pivot_column in range(size):
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row_index: abs(augmented[row_index][pivot_column]),
        )
        pivot_value = augmented[pivot_row][pivot_column]
        if abs(pivot_value) <= LINEAR_SOLVER_PIVOT_TOLERANCE:
            return None
        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_column],
            )
        pivot_value = augmented[pivot_column][pivot_column]
        for column in range(pivot_column, size + 1):
            augmented[pivot_column][column] /= pivot_value
        for row_index in range(size):
            if row_index == pivot_column:
                continue
            factor = augmented[row_index][pivot_column]
            if factor == 0.0:
                continue
            for column in range(pivot_column, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot_column][column]
    return [float(augmented[row_index][size]) for row_index in range(size)]


def fit_linear(samples: Sequence[tuple[float, float]]) -> LinearFit:
    finite = [(float(x), float(y)) for x, y in samples if math.isfinite(x) and math.isfinite(y)]
    if not finite:
        return LinearFit(intercept=0.0, slope=0.0, fit_kind="unfit")
    if len(finite) == 1:
        return LinearFit(intercept=float(finite[0][1]), slope=0.0, fit_kind="constant")
    x_values = [sample[0] for sample in finite]
    y_values = [sample[1] for sample in finite]
    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(y_values) / float(len(y_values))
    denom = sum((value - x_mean) ** 2 for value in x_values)
    if denom <= LINEAR_SOLVER_PIVOT_TOLERANCE:
        return LinearFit(intercept=float(y_mean), slope=0.0, fit_kind="constant")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in finite) / float(denom)
    intercept = y_mean - slope * x_mean
    return LinearFit(intercept=float(intercept), slope=float(slope), fit_kind="linear")


def fit_power_law(samples: Sequence[tuple[float, float]]) -> PowerLawFit:
    positive_finite = [
        (float(x), float(y))
        for x, y in samples
        if math.isfinite(x) and math.isfinite(y) and x > 0.0 and y > 0.0
    ]
    if not positive_finite:
        return PowerLawFit(intercept=0.0, exponent=0.0, fit_kind="unfit")
    if len(positive_finite) == 1:
        _, y_value = positive_finite[0]
        return PowerLawFit(intercept=float(math.log(y_value)), exponent=0.0, fit_kind="constant")
    log_x_values = [math.log(sample[0]) for sample in positive_finite]
    log_y_values = [math.log(sample[1]) for sample in positive_finite]
    x_mean = sum(log_x_values) / float(len(log_x_values))
    y_mean = sum(log_y_values) / float(len(log_y_values))
    denom = sum((value - x_mean) ** 2 for value in log_x_values)
    if denom <= LINEAR_SOLVER_PIVOT_TOLERANCE:
        return PowerLawFit(intercept=float(y_mean), exponent=0.0, fit_kind="constant")
    exponent = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(log_x_values, log_y_values, strict=False)
    ) / float(denom)
    intercept = y_mean - exponent * x_mean
    return PowerLawFit(
        intercept=float(intercept),
        exponent=float(exponent),
        fit_kind="power_law",
    )


def fit_affine_width_depth_parameter_bridge(
    points: Sequence[WidthDepthParameterPoint],
) -> AffineWidthDepthParameterFit:
    if not points:
        raise RuntimeError("at least one width-depth point is required to fit the parameter bridge")
    feature_rows = [
        (1.0, float(point.d_icl**2), float(point.effective_size))
        for point in points
    ]
    normal_matrix = [
        [
            sum(feature_row[row] * feature_row[column] for feature_row in feature_rows)
            for column in range(3)
        ]
        for row in range(3)
    ]
    rhs = [
        sum(
            feature_row[column] * float(point.total_params)
            for feature_row, point in zip(feature_rows, points, strict=False)
        )
        for column in range(3)
    ]
    solution = _solve_square_linear_system(normal_matrix, rhs)
    if solution is None:
        raise RuntimeError(
            "unable to fit depth-aware parameter bridge from the supplied width-depth points"
        )
    intercept, d_squared_coefficient, layered_d_squared_coefficient = solution
    return AffineWidthDepthParameterFit(
        intercept=float(intercept),
        d_squared_coefficient=float(d_squared_coefficient),
        layered_d_squared_coefficient=float(layered_d_squared_coefficient),
    )


def log_space_parameter_targets(
    *,
    start_value: float,
    end_value: float,
    count: int,
) -> tuple[float, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return ()
    if start_value <= 0.0 or end_value <= 0.0:
        raise ValueError("log-space interpolation requires strictly positive endpoints")
    log_start = math.log(float(start_value))
    log_end = math.log(float(end_value))
    span = log_end - log_start
    return tuple(
        float(math.exp(log_start + (float(index + 1) / float(count + 1)) * span))
        for index in range(count)
    )


def round_to_width_rung(raw_width: float, *, rung: int = 8) -> int:
    if rung <= 0:
        raise ValueError("rung must be positive")
    if raw_width <= 0.0:
        raise ValueError("raw_width must be positive")
    rounded = int(rung * round(float(raw_width) / float(rung)))
    return max(int(rung), rounded)
