"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy classification loss."""

    return F.cross_entropy(logits, targets)


def hierarchical_nll_loss(class_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """NLL on class probabilities for many-class hierarchical outputs."""

    probs = class_probs.clamp_min(1e-12)
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
