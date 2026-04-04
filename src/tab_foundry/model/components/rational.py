"""Local rational activation implementation for sandwich experiments."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn


RATIONAL_VERSION_A_5_4_GELU_NUMERATOR: Final[tuple[float, ...]] = (
    -0.0012423594497499122,
    0.5080497063245629,
    0.41586363182937475,
    0.13022718688035761,
    0.024355900098993424,
    0.00290283948155535,
)
RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR: Final[tuple[float, ...]] = (
    -0.06675015696494944,
    0.17927646217001553,
    0.03746682605496631,
    1.6561610853276082e-10,
)
SUPPORTED_RATIONAL_VERSION: Final = "A"
SUPPORTED_RATIONAL_DEGREES: Final[tuple[int, int]] = (5, 4)
SUPPORTED_RATIONAL_APPROX_FUNC: Final = "gelu"


def _evaluate_polynomial(coefficients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate one polynomial with coefficients ordered by increasing degree."""

    value = torch.zeros_like(inputs)
    for coefficient in reversed(coefficients):
        value = (value * inputs) + coefficient
    return value


class RationalActivation(nn.Module):
    """Trainable version-A rational activation with GELU 5/4 initialization."""

    def __init__(
        self,
        *,
        version: str = SUPPORTED_RATIONAL_VERSION,
        degrees: tuple[int, int] = SUPPORTED_RATIONAL_DEGREES,
        approx_func: str = SUPPORTED_RATIONAL_APPROX_FUNC,
    ) -> None:
        super().__init__()
        normalized_version = str(version).strip().upper()
        normalized_degrees = (int(degrees[0]), int(degrees[1]))
        normalized_approx_func = str(approx_func).strip().lower()
        if normalized_version != SUPPORTED_RATIONAL_VERSION:
            raise ValueError(
                "Only rational version "
                f"{SUPPORTED_RATIONAL_VERSION!r} is supported locally, got {version!r}"
            )
        if normalized_degrees != SUPPORTED_RATIONAL_DEGREES:
            raise ValueError(
                "Only rational degrees "
                f"{SUPPORTED_RATIONAL_DEGREES!r} are supported locally, got {degrees!r}"
            )
        if normalized_approx_func != SUPPORTED_RATIONAL_APPROX_FUNC:
            raise ValueError(
                "Only rational GELU initialization is supported locally, "
                f"got {approx_func!r}"
            )
        self.version = normalized_version
        self.degrees = normalized_degrees
        self.approx_func = normalized_approx_func
        self.numerator = nn.Parameter(
            torch.tensor(RATIONAL_VERSION_A_5_4_GELU_NUMERATOR, dtype=torch.float32)
        )
        self.denominator = nn.Parameter(
            torch.tensor(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        numerator = _evaluate_polynomial(self.numerator, inputs)
        denominator = torch.ones_like(inputs)
        powered_inputs = inputs
        for coefficient in self.denominator:
            denominator = denominator + torch.abs(coefficient * powered_inputs)
            powered_inputs = powered_inputs * inputs
        return numerator / denominator

    def extra_repr(self) -> str:
        return (
            f"version={self.version!r}, degrees={self.degrees!r}, "
            f"approx_func={self.approx_func!r}"
        )


def rational_parameter_ids(model: nn.Module) -> set[int]:
    """Return the ids for all trainable local rational coefficients in one model."""

    ids: set[int] = set()
    for module in model.modules():
        if not isinstance(module, RationalActivation):
            continue
        ids.add(id(module.numerator))
        ids.add(id(module.denominator))
    return ids


__all__ = [
    "RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR",
    "RATIONAL_VERSION_A_5_4_GELU_NUMERATOR",
    "RationalActivation",
    "SUPPORTED_RATIONAL_APPROX_FUNC",
    "SUPPORTED_RATIONAL_DEGREES",
    "SUPPORTED_RATIONAL_VERSION",
    "rational_parameter_ids",
]
