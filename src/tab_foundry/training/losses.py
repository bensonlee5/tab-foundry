"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tab_foundry.likelihoods import cross_entropy_bits, gaussian_nll_bits, mixture_bits


MIN_CLASS_PROB = 1.0e-12


def classification_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Canonical logit z-loss: mean(logsumexp(logits)^2)."""

    return torch.logsumexp(logits, dim=-1).square().mean()


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    z_loss_coeff: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy classification loss."""

    loss = F.cross_entropy(logits, targets)
    coeff = float(z_loss_coeff)
    if coeff <= 0.0:
        return loss
    return loss + (coeff * classification_z_loss(logits))


def hierarchical_nll_loss(class_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """NLL on class probabilities for many-class hierarchical outputs."""

    probs = class_probs.clamp_min(MIN_CLASS_PROB)
    selected = probs[torch.arange(targets.shape[0], device=targets.device), targets]
    return -torch.log(selected).mean()


def quantile_pinball_loss(
    pred_quantiles: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: torch.Tensor,
) -> torch.Tensor:
    """Pinball loss summed over quantiles and averaged over rows."""

    if targets.ndim == 1:
        targets = targets[:, None]
    error = targets - pred_quantiles
    tau = quantile_levels[None, :].to(pred_quantiles.device, pred_quantiles.dtype)
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


__all__ = [
    "classification_loss",
    "classification_z_loss",
    "cross_entropy_bits",
    "gaussian_nll_bits",
    "hierarchical_nll_loss",
    "mixture_bits",
    "quantile_pinball_loss",
]
