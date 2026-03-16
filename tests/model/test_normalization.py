from __future__ import annotations

import pytest
import torch

from tab_foundry.model.components.normalization import RMSNorm, build_norm


def test_rmsnorm_normalizes_by_root_mean_square() -> None:
    norm = RMSNorm(3, eps=1.0e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    x = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
    observed = norm(x)

    rms = torch.sqrt(torch.tensor((3.0**2 + 4.0**2 + 0.0**2) / 3.0, dtype=torch.float32) + 1.0e-6)
    expected = x / rms * torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_build_norm_supports_layernorm_and_rmsnorm() -> None:
    assert isinstance(build_norm("layernorm", 4), torch.nn.LayerNorm)
    assert isinstance(build_norm("rmsnorm", 4), RMSNorm)

    with pytest.raises(ValueError, match="Unsupported norm_type"):
        _ = build_norm("bogus", 4)


def test_rmsnorm_exposes_layernorm_compatible_bias_attribute() -> None:
    norm = RMSNorm(4)

    assert hasattr(norm, "bias")
    assert norm.bias is None
