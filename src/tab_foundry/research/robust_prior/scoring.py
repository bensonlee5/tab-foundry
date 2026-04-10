"""Probe scoring against TabFoundry and classical baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pyarrow.parquet as pq

from tab_foundry.bench.checkpoint import TabFoundryClassifier
from tab_foundry.data.dataset import _load_manifest_task_record


def _sklearn_imports() -> tuple[Any, Any, Any, Any, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.neural_network import MLPClassifier

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - exercised in runtime only
        raise RuntimeError(
            "robust-prior scoring requires the optional research dependency "
            "`catboost>=1.2.10`; install the `research` extra before running "
            "`tab-foundry research robust-prior ...`"
        ) from exc
    return RandomForestClassifier, LogisticRegression, MLPClassifier, log_loss, CatBoostClassifier


@dataclass(frozen=True, slots=True)
class ProbeDatasetScore:
    """One scored synthetic dataset within a probe trial."""

    dataset_id: str
    tfm_log_loss: float
    class_prior_log_loss: float
    baseline_log_losses: dict[str, float]
    raw_gap: float
    normalized_gap: float
    class_prior_headroom: float
    class_entropy: float
    graph_target_depth_ratio: float | None
    feature_count_center: float
    class_count_center: float
    categorical_ratio_center: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeTrialScore:
    """Aggregate probe score for one proposal."""

    dataset_scores: tuple[ProbeDatasetScore, ...]
    aggregate: dict[str, float | None]
    feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_scores": [score.as_dict() for score in self.dataset_scores],
            "aggregate": dict(self.aggregate),
            "feasible": bool(self.feasible),
        }


def _manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(manifest_path)
    return [
        {str(key): value for key, value in row.items()}
        for row in table.to_pylist()
    ]


def _class_entropy(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=np.int64)
    if values.size <= 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(counts.sum())
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def _empirical_class_prior_log_loss(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
) -> float:
    train = np.asarray(y_train, dtype=np.int64)
    test = np.asarray(y_test, dtype=np.int64)
    counts = np.bincount(train, minlength=int(n_classes)).astype(np.float64)
    probabilities = np.clip(counts / max(float(counts.sum()), 1.0), 1.0e-12, 1.0)
    return float(-np.log(probabilities[test]).mean())


def _aligned_proba(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    aligned = np.full((int(probabilities.shape[0]), int(n_classes)), 1.0e-12, dtype=np.float64)
    aligned[:, classes.astype(np.int64)] = probabilities
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _baseline_log_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    RandomForestClassifier, LogisticRegression, MLPClassifier, log_loss, CatBoostClassifier = (
        _sklearn_imports()
    )
    baselines = {
        "catboost": CatBoostClassifier(
            iterations=128,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            verbose=False,
            random_seed=int(seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=int(seed),
            n_jobs=1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=400,
            random_state=int(seed),
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=int(seed),
        ),
    }
    losses: dict[str, float] = {}
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, estimator in baselines.items():
            try:
                estimator.fit(x_train, y_train)
                probabilities = estimator.predict_proba(x_test)
                aligned = _aligned_proba(
                    np.asarray(probabilities, dtype=np.float64),
                    np.asarray(estimator.classes_, dtype=np.int64),
                    n_classes=n_classes,
                )
                losses[name] = float(
                    log_loss(
                        y_test,
                        aligned,
                        labels=list(range(int(n_classes))),
                    )
                )
            except Exception:
                continue
    if not losses:
        raise RuntimeError("all classical baseline fits failed for one probe dataset")
    return losses


def compute_gap_metrics(
    *,
    tfm_log_loss: float,
    baseline_log_losses: Mapping[str, float],
    class_prior_log_loss: float,
) -> dict[str, float]:
    """Compute the adversarial objective metrics for one probe dataset."""

    min_baseline_log_loss = min(float(value) for value in baseline_log_losses.values())
    raw_gap = float(tfm_log_loss) - float(min_baseline_log_loss)
    denominator = max(1.0e-6, float(class_prior_log_loss) - float(min_baseline_log_loss))
    return {
        "raw_gap": float(raw_gap),
        "normalized_gap": float(raw_gap / denominator),
        "class_prior_headroom": float(class_prior_log_loss - min_baseline_log_loss),
        "min_baseline_log_loss": float(min_baseline_log_loss),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_depth_ratio(metadata: Mapping[str, Any]) -> float | None:
    direct = metadata.get("graph_target_depth_ratio")
    if direct is not None:
        try:
            value = float(direct)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return value
    graph_nodes = metadata.get("graph_nodes")
    target_depth_nodes = metadata.get("graph_target_depth_nodes")
    graph_nodes_i = _int_or_none(graph_nodes)
    target_depth_nodes_i = _int_or_none(target_depth_nodes)
    if graph_nodes_i is None or target_depth_nodes_i is None:
        return None
    if graph_nodes_i <= 0:
        return None
    return float(target_depth_nodes_i) / float(graph_nodes_i)


def _extract_categorical_ratio(metadata: Mapping[str, Any], *, feature_types: Sequence[str]) -> float:
    raw_n_categorical = metadata.get("n_categorical_features")
    try:
        if raw_n_categorical is not None:
            n_categorical = int(raw_n_categorical)
            if len(feature_types) > 0:
                return float(n_categorical) / float(len(feature_types))
    except (TypeError, ValueError):
        pass
    if not feature_types:
        return 0.0
    n_categorical = sum(1 for feature_type in feature_types if str(feature_type).strip() == "cat")
    return float(n_categorical) / float(len(feature_types))


def score_probe_manifest(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    device: str,
    seed: int,
    class_entropy_floor: float,
    min_class_prior_headroom: float,
    authored_depth_ratio_band: tuple[float, float],
) -> ProbeTrialScore:
    """Score one probe manifest against the checkpoint and classical baselines."""

    records = _manifest_records(manifest_path)
    classifier = TabFoundryClassifier(checkpoint_path, device=device)
    dataset_scores: list[ProbeDatasetScore] = []
    for record_index, record in enumerate(records):
        loaded = _load_manifest_task_record(
            manifest_path,
            split=str(record.get("split", "train")),
            task=str(record.get("task", "classification")),
            record=record,
            include_metadata=True,
        )
        feature_types = list(loaded.feature_types)
        classifier.set_benchmark_feature_types(feature_types if feature_types else None)
        classifier.fit(loaded.x_train, loaded.y_train)
        tfm_proba = classifier.predict_proba(loaded.x_test)
        n_classes = int(max(np.max(loaded.y_train), np.max(loaded.y_test)) + 1)
        _, _, _, log_loss, _ = _sklearn_imports()
        tfm_log_loss = float(
            log_loss(
                loaded.y_test,
                tfm_proba,
                labels=list(range(int(n_classes))),
            )
        )
        baseline_log_losses = _baseline_log_losses(
            loaded.x_train,
            loaded.y_train,
            loaded.x_test,
            loaded.y_test,
            seed=int(seed) + int(record_index),
        )
        class_prior_log_loss = _empirical_class_prior_log_loss(
            loaded.y_train,
            loaded.y_test,
            n_classes=n_classes,
        )
        gap_metrics = compute_gap_metrics(
            tfm_log_loss=tfm_log_loss,
            baseline_log_losses=baseline_log_losses,
            class_prior_log_loss=class_prior_log_loss,
        )
        labels_all = np.concatenate(
            [
                np.asarray(loaded.y_train, dtype=np.int64),
                np.asarray(loaded.y_test, dtype=np.int64),
            ]
        )
        dataset_scores.append(
            ProbeDatasetScore(
                dataset_id=str(record.get("dataset_id", f"dataset_{record_index:05d}")),
                tfm_log_loss=float(tfm_log_loss),
                class_prior_log_loss=float(class_prior_log_loss),
                baseline_log_losses=dict(baseline_log_losses),
                raw_gap=float(gap_metrics["raw_gap"]),
                normalized_gap=float(gap_metrics["normalized_gap"]),
                class_prior_headroom=float(gap_metrics["class_prior_headroom"]),
                class_entropy=float(_class_entropy(labels_all)),
                graph_target_depth_ratio=_extract_depth_ratio(loaded.metadata),
                feature_count_center=float(loaded.x_train.shape[1]),
                class_count_center=float(n_classes),
                categorical_ratio_center=_extract_categorical_ratio(
                    loaded.metadata,
                    feature_types=feature_types,
                ),
            )
        )
    if not dataset_scores:
        raise RuntimeError(f"probe manifest produced no scoreable records: {manifest_path}")
    raw_gap = float(np.mean([score.raw_gap for score in dataset_scores]))
    normalized_gap = float(np.mean([score.normalized_gap for score in dataset_scores]))
    class_prior_headroom = float(np.mean([score.class_prior_headroom for score in dataset_scores]))
    class_entropy = float(np.mean([score.class_entropy for score in dataset_scores]))
    depth_values = [
        float(score.graph_target_depth_ratio)
        for score in dataset_scores
        if score.graph_target_depth_ratio is not None
    ]
    depth_ratio = None if not depth_values else float(np.mean(depth_values))
    depth_band_ok = (
        depth_ratio is not None
        and authored_depth_ratio_band[0] - 1.0e-9 <= depth_ratio <= authored_depth_ratio_band[1] + 1.0e-9
    )
    feasible = bool(
        raw_gap > 0.0
        and class_prior_headroom >= float(min_class_prior_headroom)
        and class_entropy >= float(class_entropy_floor)
        and depth_band_ok
    )
    aggregate = {
        "raw_gap": float(raw_gap),
        "normalized_gap": float(normalized_gap),
        "class_prior_headroom": float(class_prior_headroom),
        "class_entropy": float(class_entropy),
        "depth_ratio": None if depth_ratio is None else float(depth_ratio),
        "feature_count_center": float(np.mean([score.feature_count_center for score in dataset_scores])),
        "class_count_center": float(np.mean([score.class_count_center for score in dataset_scores])),
        "categorical_ratio_center": float(
            np.mean([score.categorical_ratio_center for score in dataset_scores])
        ),
    }
    return ProbeTrialScore(
        dataset_scores=tuple(dataset_scores),
        aggregate=aggregate,
        feasible=feasible,
    )
