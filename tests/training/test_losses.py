from __future__ import annotations

import torch

from tab_foundry.training.losses import quantile_pinball_loss


def test_pinball_loss_finite() -> None:
    pred = torch.randn(7, 999)
    target = torch.randn(7)
    levels = torch.arange(1, 1000, dtype=torch.float32) / 1000.0
    loss = quantile_pinball_loss(pred, target, levels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
