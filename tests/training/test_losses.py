from __future__ import annotations

import torch

from tab_foundry.training.losses import (
    classification_loss,
    classification_z_loss,
    cross_entropy_bits,
    gaussian_nll_bits,
    mixture_bits,
    quantile_pinball_loss,
)


def test_classification_z_loss_matches_logsumexp_square_mean() -> None:
    logits = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float32)

    z_loss = classification_z_loss(logits)

    expected = torch.logsumexp(logits, dim=-1).square().mean()
    torch.testing.assert_close(z_loss, expected)


def test_classification_loss_adds_weighted_z_loss() -> None:
    logits = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.int64)

    loss = classification_loss(logits, targets, z_loss_coeff=1.0e-4)

    expected = classification_loss(logits, targets) + (1.0e-4 * classification_z_loss(logits))
    torch.testing.assert_close(loss, expected)


def test_pinball_loss_finite() -> None:
    pred = torch.randn(7, 999)
    target = torch.randn(7)
    levels = torch.arange(1, 1000, dtype=torch.float32) / 1000.0
    loss = quantile_pinball_loss(pred, target, levels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_cross_entropy_bits_matches_base2_information_units() -> None:
    logits = torch.log(torch.tensor([[3.0, 1.0]], dtype=torch.float32))
    target = torch.tensor([0], dtype=torch.int64)

    bits = cross_entropy_bits(logits, target)

    torch.testing.assert_close(bits, torch.tensor([0.4150375], dtype=torch.float32))


def test_gaussian_nll_bits_matches_manual_conversion() -> None:
    mean = torch.tensor([0.0], dtype=torch.float32)
    log_variance = torch.tensor([0.0], dtype=torch.float32)
    target = torch.tensor([0.0], dtype=torch.float32)

    bits = gaussian_nll_bits(mean, log_variance, target)
    expected = 0.5 * torch.log2(torch.tensor(2.0 * torch.pi))

    torch.testing.assert_close(bits, expected.reshape(1))


def test_mixture_bits_matches_manual_logsumexp_mix() -> None:
    gate_logit = torch.tensor(0.0, dtype=torch.float32)
    discrete_bits = torch.tensor([1.0], dtype=torch.float32)
    continuous_bits = torch.tensor([3.0], dtype=torch.float32)

    bits = mixture_bits(
        gate_logit=gate_logit,
        discrete_bits=discrete_bits,
        continuous_bits=continuous_bits,
    )

    expected_prob = 0.5 * (2.0 ** -1.0) + 0.5 * (2.0 ** -3.0)
    expected = torch.tensor([-torch.log2(torch.tensor(expected_prob))], dtype=torch.float32)
    torch.testing.assert_close(bits, expected)
