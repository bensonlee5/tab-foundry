from __future__ import annotations

import torch

from tab_foundry.model.components.rational import (
    RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR,
    RATIONAL_VERSION_A_5_4_GELU_NUMERATOR,
    RationalActivation,
)


def test_rational_activation_matches_version_a_formula() -> None:
    activation = RationalActivation()
    inputs = torch.tensor([-2.0, -0.5, 0.0, 0.75, 1.5], dtype=torch.float32)

    expected_numerator = (
        RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[0]
        + (RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[1] * inputs)
        + (RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[2] * inputs.square())
        + (RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[3] * inputs.pow(3))
        + (RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[4] * inputs.pow(4))
        + (RATIONAL_VERSION_A_5_4_GELU_NUMERATOR[5] * inputs.pow(5))
    )
    expected_denominator = (
        1.0
        + torch.abs(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR[0] * inputs)
        + torch.abs(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR[1] * inputs.square())
        + torch.abs(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR[2] * inputs.pow(3))
        + torch.abs(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR[3] * inputs.pow(4))
    )

    torch.testing.assert_close(activation(inputs), expected_numerator / expected_denominator)


def test_rational_activation_uses_upstream_gelu_5_4_coefficients() -> None:
    activation = RationalActivation()

    torch.testing.assert_close(
        activation.numerator.detach(),
        torch.tensor(RATIONAL_VERSION_A_5_4_GELU_NUMERATOR, dtype=torch.float32),
    )
    torch.testing.assert_close(
        activation.denominator.detach(),
        torch.tensor(RATIONAL_VERSION_A_5_4_GELU_DENOMINATOR, dtype=torch.float32),
    )
