"""Transformer-based proposal distribution over the scored robust-prior table."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .search_space import RobustPriorProposal, RobustPriorSearchSpace


_STAT_KEYS = (
    "normalized_gap",
    "raw_gap",
    "class_prior_headroom",
    "depth_ratio",
    "feature_count_center",
    "categorical_ratio_center",
    "class_count_center",
    "mechanism_nonlinearity_mass",
    "shift_graph_scale",
    "shift_variance_scale",
)


@dataclass(frozen=True, slots=True)
class ProposerFitResult:
    """One fitted proposer distribution plus compact training diagnostics."""

    probabilities: dict[str, list[float]]
    fit_summary: dict[str, Any]


class _TrialTableTransformer(nn.Module):
    def __init__(self, *, search_space: RobustPriorSearchSpace, d_model: int = 64) -> None:
        super().__init__()
        self.search_space = search_space
        self.d_model = int(d_model)
        self.embeddings = nn.ModuleDict(
            {
                dimension.name: nn.Embedding(len(dimension.values), self.d_model)
                for dimension in search_space.dimensions
            }
        )
        self.stats_projection = nn.Linear(len(_STAT_KEYS), self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=4 * self.d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.heads = nn.ModuleDict(
            {
                dimension.name: nn.Linear(self.d_model, len(dimension.values))
                for dimension in search_space.dimensions
            }
        )

    def forward(
        self,
        proposal_indices: torch.Tensor,
        stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = self.stats_projection(stats)
        for column, dimension in enumerate(self.search_space.dimensions):
            token = token + self.embeddings[dimension.name](proposal_indices[:, column])
        encoded = self.encoder(token.unsqueeze(0)).mean(dim=1)
        return {
            dimension.name: self.heads[dimension.name](encoded)
            for dimension in self.search_space.dimensions
        }


def _trial_stats_tensor(trials: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    rows: list[list[float]] = []
    for trial in trials:
        summary = trial.get("aggregate", {})
        proposal_vector = trial.get("proposal_vector", {})
        row: list[float] = []
        for key in _STAT_KEYS:
            if key in summary:
                row.append(float(summary[key]))
            else:
                row.append(float(proposal_vector.get(key, 0.0)))
        rows.append(row)
    if not rows:
        return torch.zeros((0, len(_STAT_KEYS)), dtype=torch.float32)
    stats = np.asarray(rows, dtype=np.float32)
    if stats.shape[0] > 1:
        mean = stats.mean(axis=0, keepdims=True)
        std = stats.std(axis=0, keepdims=True)
        stats = (stats - mean) / np.clip(std, 1.0e-6, None)
    return torch.as_tensor(stats, dtype=torch.float32)


def _top_quartile_feasible_trials(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    feasible = [
        dict(trial)
        for trial in trials
        if bool(trial.get("feasible", False))
        and trial.get("aggregate") is not None
        and trial["aggregate"].get("normalized_gap") is not None
    ]
    if not feasible:
        return []
    feasible.sort(
        key=lambda trial: float(trial["aggregate"]["normalized_gap"]),
        reverse=True,
    )
    keep_count = max(1, int(math.ceil(float(len(feasible)) / 4.0)))
    return feasible[:keep_count]


def _entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _apply_entropy_floor(probabilities: np.ndarray, *, entropy_floor_ratio: float) -> np.ndarray:
    if probabilities.size <= 1:
        return probabilities
    uniform = np.full_like(probabilities, 1.0 / float(probabilities.size))
    target_entropy = float(entropy_floor_ratio) * math.log(float(probabilities.size))
    if _entropy(probabilities) >= target_entropy:
        return probabilities
    low = 0.0
    high = 1.0
    candidate = uniform
    for _ in range(24):
        alpha = 0.5 * (low + high)
        mixed = (1.0 - alpha) * probabilities + alpha * uniform
        if _entropy(mixed) >= target_entropy:
            high = alpha
            candidate = mixed
        else:
            low = alpha
    candidate /= float(candidate.sum())
    return candidate


def fit_proposer_distribution(
    *,
    search_space: RobustPriorSearchSpace,
    trial_history: Sequence[Mapping[str, Any]],
    seed: int,
) -> ProposerFitResult | None:
    """Fit one transformer proposer and return per-dimension probabilities."""

    selected = _top_quartile_feasible_trials(trial_history)
    if not selected:
        return None
    model = _TrialTableTransformer(search_space=search_space)
    torch.manual_seed(int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    history_indices = torch.as_tensor(
        [search_space.encode(RobustPriorProposal(**dict(trial["proposal"]))) for trial in trial_history],
        dtype=torch.long,
    )
    history_stats = _trial_stats_tensor(trial_history)
    target_indices = torch.as_tensor(
        [search_space.encode(RobustPriorProposal(**dict(trial["proposal"]))) for trial in selected],
        dtype=torch.long,
    )
    weights = torch.as_tensor(
        [
            max(1.0e-6, float(cast_mapping(trial["aggregate"]).get("normalized_gap", 0.0)))
            for trial in selected
        ],
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    loss_history: list[float] = []
    for _epoch in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits = model(history_indices, history_stats)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for dim_index, dimension in enumerate(search_space.dimensions):
            target = target_indices[:, dim_index]
            per_target = F.cross_entropy(
                logits[dimension.name].expand(target.shape[0], -1),
                target,
                reduction="none",
            )
            loss = loss + torch.sum(weights * per_target)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))
    with torch.no_grad():
        logits = model(history_indices, history_stats)
        probabilities = {
            dimension.name: (
                F.softmax(logits[dimension.name], dim=-1)
                .reshape(-1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            for dimension in search_space.dimensions
        }
    return ProposerFitResult(
        probabilities={
            key: [float(value) for value in values]
            for key, values in probabilities.items()
        },
        fit_summary={
            "history_trial_count": int(len(trial_history)),
            "selected_target_count": int(len(selected)),
            "loss_history": [float(value) for value in loss_history],
        },
    )


def cast_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return value


def sample_proposal(
    *,
    search_space: RobustPriorSearchSpace,
    rng: np.random.Generator,
    probabilities: Mapping[str, Sequence[float]] | None,
    exploration_rate: float,
    entropy_floor_ratio: float,
) -> tuple[RobustPriorProposal, dict[str, Any]]:
    """Sample one proposal with uniform exploration and an entropy floor."""

    if probabilities is None:
        proposal = search_space.sample_uniform(rng)
        return proposal, {
            "source": "uniform",
            "exploration_rate": float(exploration_rate),
            "entropy_floor_ratio": float(entropy_floor_ratio),
        }
    adjusted: dict[str, list[float]] = {}
    entropies: dict[str, float] = {}
    for dimension in search_space.dimensions:
        raw = np.asarray(probabilities[dimension.name], dtype=np.float64)
        raw = np.clip(raw, 0.0, None)
        if float(raw.sum()) <= 0.0:
            raw = np.full((len(dimension.values),), 1.0 / float(len(dimension.values)))
        else:
            raw /= float(raw.sum())
        raw = (1.0 - float(exploration_rate)) * raw + float(exploration_rate) * (
            np.full_like(raw, 1.0 / float(len(dimension.values)))
        )
        floored = _apply_entropy_floor(raw, entropy_floor_ratio=float(entropy_floor_ratio))
        floored /= float(floored.sum())
        adjusted_probs = [float(value) for value in floored]
        adjusted[dimension.name] = adjusted_probs
        entropies[dimension.name] = _entropy(np.asarray(adjusted_probs, dtype=np.float64))
    proposal = search_space.sample_from_distribution(adjusted, rng=rng)
    return proposal, {
        "source": "transformer",
        "exploration_rate": float(exploration_rate),
        "entropy_floor_ratio": float(entropy_floor_ratio),
        "entropies": {key: float(value) for key, value in entropies.items()},
        "probabilities": dict(adjusted),
    }
