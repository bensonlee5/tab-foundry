"""Synthetic adequacy specification and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml

from tab_foundry.repo_paths import repo_root as shared_repo_root


SYNTHETIC_ADEQUACY_SCHEMA = "tab-foundry-synthetic-adequacy-v2"
LABEL_TARGET_LOG_LOSS_PER_TEST_CELL = "label-target log loss per test cell"
_PROBABILITY_MATRIX_NDIM = 2
_REPLICATE_TENSOR_NDIM = 3
_MIN_CLASS_COUNT = 2
_MIN_REPLICATE_COUNT = 2


def _repo_root() -> Path:
    return shared_repo_root()


def synthetic_adequacy_root(*, repo_root: Path | None = None) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return resolved_repo_root / "reference" / "synthetic_adequacy"


def synthetic_adequacy_spec_path(adequacy_id: str, *, repo_root: Path | None = None) -> Path:
    normalized = adequacy_id.strip()
    if not normalized:
        raise RuntimeError("adequacy_id must be a non-empty string")
    return synthetic_adequacy_root(repo_root=repo_root) / f"{normalized}.yaml"


def _require_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _require_string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_require_non_empty_string(item, context=f"{context}[{index}]"))
    return normalized


def _require_int_list(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    normalized: list[int] = []
    for index, item in enumerate(value):
        if item is None or isinstance(item, bool):
            raise RuntimeError(f"{context}[{index}] must be an integer")
        try:
            normalized.append(int(item))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{context}[{index}] must be an integer") from exc
    return normalized


def _load_yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a mapping: {path}")
    return {str(key): item for key, item in cast(Mapping[str, Any], payload).items()}


@dataclass(frozen=True, slots=True)
class SyntheticAdequacyBlock:
    block_id: str
    description: str
    corpus_ref: str
    n_ladder: tuple[int, ...]
    predictors: tuple[str, ...]
    repeats_per_n: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "description": self.description,
            "corpus_ref": self.corpus_ref,
            "n_ladder": list(self.n_ladder),
            "predictors": list(self.predictors),
            "repeats_per_n": self.repeats_per_n,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SyntheticAdequacySpec:
    adequacy_id: str
    status: str
    metric_definition: str
    blocked_sweeps: tuple[str, ...]
    blocks: tuple[SyntheticAdequacyBlock, ...]
    decision_buckets: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYNTHETIC_ADEQUACY_SCHEMA,
            "adequacy_id": self.adequacy_id,
            "status": self.status,
            "metric_definition": self.metric_definition,
            "blocked_sweeps": list(self.blocked_sweeps),
            "blocks": [block.to_dict() for block in self.blocks],
            "decision_buckets": dict(self.decision_buckets),
            "notes": list(self.notes),
        }


def load_synthetic_adequacy_spec(
    adequacy_id: str,
    *,
    repo_root: Path | None = None,
) -> SyntheticAdequacySpec:
    path = synthetic_adequacy_spec_path(adequacy_id, repo_root=repo_root)
    payload = _load_yaml_mapping(path, context=f"synthetic adequacy spec {adequacy_id!r}")
    if payload.get("schema") != SYNTHETIC_ADEQUACY_SCHEMA:
        raise RuntimeError(
            f"synthetic adequacy spec schema must be {SYNTHETIC_ADEQUACY_SCHEMA!r}, "
            f"got {payload.get('schema')!r}: {path}"
        )
    blocks_payload = payload.get("blocks")
    if not isinstance(blocks_payload, list) or not blocks_payload:
        raise RuntimeError("synthetic adequacy spec blocks must be a non-empty list")
    blocks: list[SyntheticAdequacyBlock] = []
    for index, raw_block in enumerate(blocks_payload):
        block = _require_mapping(raw_block, context=f"blocks[{index}]")
        repeats_per_n = block.get("repeats_per_n")
        if repeats_per_n is None or isinstance(repeats_per_n, bool):
            raise RuntimeError(f"blocks[{index}].repeats_per_n must be an integer")
        blocks.append(
            SyntheticAdequacyBlock(
                block_id=_require_non_empty_string(
                    block.get("block_id"),
                    context=f"blocks[{index}].block_id",
                ),
                description=_require_non_empty_string(
                    block.get("description"),
                    context=f"blocks[{index}].description",
                ),
                corpus_ref=_require_non_empty_string(
                    block.get("corpus_ref"),
                    context=f"blocks[{index}].corpus_ref",
                ),
                n_ladder=tuple(
                    _require_int_list(block.get("n_ladder"), context=f"blocks[{index}].n_ladder")
                ),
                predictors=tuple(
                    _require_string_list(
                        block.get("predictors"),
                        context=f"blocks[{index}].predictors",
                    )
                ),
                repeats_per_n=int(repeats_per_n),
                notes=tuple(
                    _require_string_list(block.get("notes", []), context=f"blocks[{index}].notes")
                    if block.get("notes")
                    else []
                ),
            )
        )
    decision_buckets = _require_mapping(
        payload.get("decision_buckets"),
        context="synthetic adequacy spec decision_buckets",
    )
    normalized_decision_buckets = {
        key: _require_non_empty_string(value, context=f"decision_buckets.{key}")
        for key, value in decision_buckets.items()
    }
    return SyntheticAdequacySpec(
        adequacy_id=_require_non_empty_string(payload.get("adequacy_id"), context="adequacy_id"),
        status=_require_non_empty_string(payload.get("status"), context="status"),
        metric_definition=_require_non_empty_string(
            payload.get("metric_definition"),
            context="metric_definition",
        ),
        blocked_sweeps=tuple(
            _require_string_list(payload.get("blocked_sweeps"), context="blocked_sweeps")
        ),
        blocks=tuple(blocks),
        decision_buckets=normalized_decision_buckets,
        notes=tuple(_require_string_list(payload.get("notes", []), context="notes") if payload.get("notes") else []),
    )


def _probability_matrix(probabilities: Any, *, context: str) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != _PROBABILITY_MATRIX_NDIM:
        raise ValueError(f"{context} must have shape [items, classes]")
    if array.shape[1] < _MIN_CLASS_COUNT:
        raise ValueError(f"{context} must include at least two classes")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must be finite")
    row_sums = array.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{context} rows must have positive mass")
    normalized = np.clip(array / row_sums, 1.0e-12, 1.0)
    normalized /= normalized.sum(axis=1, keepdims=True)
    return normalized


def _replicate_probability_tensor(probabilities: Any, *, context: str) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != _REPLICATE_TENSOR_NDIM:
        raise ValueError(f"{context} must have shape [replicates, items, classes]")
    if array.shape[0] < _MIN_REPLICATE_COUNT:
        raise ValueError(f"{context} must include at least two replicates")
    normalized = np.stack(
        [_probability_matrix(array[index], context=f"{context}[{index}]") for index in range(array.shape[0])],
        axis=0,
    )
    return normalized


def _target_indices(targets: Any, *, n_items: int, n_classes: int) -> np.ndarray:
    array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if array.shape[0] != n_items:
        raise ValueError("targets length must match the number of test cells")
    if np.any(array < 0) or np.any(array >= n_classes):
        raise ValueError("targets must be valid class indices")
    return array


def label_target_log_loss_per_test_cell(probabilities: Any, targets: Any) -> float:
    normalized = _probability_matrix(probabilities, context="probabilities")
    target_indices = _target_indices(targets, n_items=normalized.shape[0], n_classes=normalized.shape[1])
    losses = -np.log(normalized[np.arange(normalized.shape[0]), target_indices])
    return float(losses.mean())


def teacher_excess_log_loss_per_test_cell(probabilities: Any, teacher_probabilities: Any) -> float:
    model = _probability_matrix(probabilities, context="probabilities")
    teacher = _probability_matrix(teacher_probabilities, context="teacher_probabilities")
    if teacher.shape != model.shape:
        raise ValueError("teacher_probabilities must match probabilities shape")
    teacher_cross_entropy = -np.sum(teacher * np.log(model), axis=1)
    teacher_entropy = -np.sum(teacher * np.log(teacher), axis=1)
    return float(np.mean(teacher_cross_entropy - teacher_entropy))


def prediction_variance_per_test_cell(probabilities_by_replicate: Any) -> float:
    normalized = _replicate_probability_tensor(
        probabilities_by_replicate,
        context="probabilities_by_replicate",
    )
    return float(np.var(normalized, axis=0).sum(axis=1).mean())


def summarize_replicate_predictions(
    probabilities_by_replicate: Any,
    targets: Any,
    *,
    teacher_probabilities: Any | None = None,
) -> dict[str, float]:
    normalized = _replicate_probability_tensor(
        probabilities_by_replicate,
        context="probabilities_by_replicate",
    )
    target_indices = _target_indices(
        targets,
        n_items=normalized.shape[1],
        n_classes=normalized.shape[2],
    )
    replicate_log_losses = np.asarray(
        [
            label_target_log_loss_per_test_cell(probabilities=replicate, targets=target_indices)
            for replicate in normalized
        ],
        dtype=np.float64,
    )
    summary = {
        "mean_log_loss_per_test_cell": float(replicate_log_losses.mean()),
        "std_log_loss_per_test_cell": float(replicate_log_losses.std(ddof=0)),
        "prediction_variance_per_test_cell": prediction_variance_per_test_cell(normalized),
    }
    if teacher_probabilities is not None:
        teacher = _probability_matrix(teacher_probabilities, context="teacher_probabilities")
        if teacher.shape != normalized.shape[1:]:
            raise ValueError(
                "teacher_probabilities must match a single replicate shape [items, classes]"
            )
        summary["mean_teacher_excess_log_loss_per_test_cell"] = float(
            np.mean(
                [
                    teacher_excess_log_loss_per_test_cell(
                        probabilities=replicate,
                        teacher_probabilities=teacher,
                    )
                    for replicate in normalized
                ]
            )
        )
        summary["teacher_optimal_log_loss_per_test_cell"] = float(
            -np.sum(teacher * np.log(teacher), axis=1).mean()
        )
    return summary


__all__ = [
    "LABEL_TARGET_LOG_LOSS_PER_TEST_CELL",
    "SYNTHETIC_ADEQUACY_SCHEMA",
    "SyntheticAdequacyBlock",
    "SyntheticAdequacySpec",
    "label_target_log_loss_per_test_cell",
    "load_synthetic_adequacy_spec",
    "prediction_variance_per_test_cell",
    "summarize_replicate_predictions",
    "synthetic_adequacy_root",
    "synthetic_adequacy_spec_path",
    "teacher_excess_log_loss_per_test_cell",
]
