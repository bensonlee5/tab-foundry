"""Shared likelihood helpers used by models and training code."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_LOG2 = math.log(2.0)


def cross_entropy_bits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Categorical cross-entropy converted from nats to bits."""

    return F.cross_entropy(logits, targets, reduction="none") / _LOG2


def gaussian_nll_bits(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Single-Gaussian negative log-likelihood converted to bits."""

    variance = torch.exp(log_variance).clamp_min(1.0e-8)
    squared_error = (targets - mean).square()
    nll = 0.5 * ((squared_error / variance) + log_variance + math.log(2.0 * math.pi))
    return nll / _LOG2


def mixture_bits(
    *,
    gate_logit: torch.Tensor,
    discrete_bits: torch.Tensor,
    continuous_bits: torch.Tensor,
) -> torch.Tensor:
    """Two-branch mixture NLL in bits with one Bernoulli gate logit."""

    gate_log_prob = F.logsigmoid(gate_logit)
    continuous_gate_log_prob = F.logsigmoid(-gate_logit)
    mixture_nll = -torch.logsumexp(
        torch.stack(
            [
                gate_log_prob - (discrete_bits * _LOG2),
                continuous_gate_log_prob - (continuous_bits * _LOG2),
            ],
            dim=0,
        ),
        dim=0,
    )
    return mixture_nll / _LOG2
