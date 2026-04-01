"""Shared tabular-model primitives used across architecture families."""

from __future__ import annotations

import torch
from torch import nn

from .non_finite import encode_non_finite_token_features


class ScalarPerFeatureTokenizer(nn.Module):
    """One scalar token per feature."""

    token_dim = 1

    def forward(self, x_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x_all.unsqueeze(-1), None


class ScalarPerFeatureMissingnessTokenizer(nn.Module):
    """One per-feature token with explicit non-finite-type flags."""

    token_dim = 4

    def forward(self, x_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return encode_non_finite_token_features(x_all), None


class SharedLinearFeatureEncoder(nn.Module):
    """Linear feature encoder for pre-tokenized feature vectors."""

    def __init__(self, token_dim: int, embedding_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(token_dim, embedding_size, bias=False)

    def forward(self, tokenized_x: torch.Tensor) -> torch.Tensor:
        return self.linear(tokenized_x)


class FeatureTypeFiLM(nn.Module):
    """Feature-type FiLM modulation applied after the shared feature encoder."""

    def __init__(self, num_feature_types: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.params = nn.Embedding(int(num_feature_types), int(embedding_size) * 2)
        nn.init.zeros_(self.params.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        feature_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        film_params = self.params(feature_type_ids)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(1).to(dtype=hidden.dtype)
        shift = shift.unsqueeze(1).to(dtype=hidden.dtype)
        return hidden * (1.0 + scale) + shift


class LabelTokenTargetConditioner(nn.Module):
    """Train-label embeddings plus a learned test token."""

    def __init__(self, num_embeddings: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.test_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

    def forward(self, y_train: torch.Tensor, *, num_rows: int) -> torch.Tensor:
        y_train_i64 = y_train.to(torch.int64).clamp(max=self.embedding.num_embeddings - 1)
        train_embed = self.embedding(y_train_i64)
        n_train = int(train_embed.shape[1])
        n_test = num_rows - n_train
        test_embed = self.test_token.expand(int(train_embed.shape[0]), n_test, -1)
        return torch.cat([train_embed, test_embed], dim=1).unsqueeze(2)


class DirectMulticlassHead(nn.Module):
    """Direct multiclass MLP head with fixed output width."""

    def __init__(self, embedding_size: int, hidden_size: int, output_width: int) -> None:
        super().__init__()
        self.output_width = output_width
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_width),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows)


__all__ = [
    "DirectMulticlassHead",
    "FeatureTypeFiLM",
    "LabelTokenTargetConditioner",
    "ScalarPerFeatureMissingnessTokenizer",
    "ScalarPerFeatureTokenizer",
    "SharedLinearFeatureEncoder",
]
