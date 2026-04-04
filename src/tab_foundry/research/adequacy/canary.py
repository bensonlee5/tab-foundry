"""Local canary baseline scoring for adequacy pilot corpora."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.research.synthetic_adequacy import (
    SyntheticAdequacyBlock,
    label_target_log_loss_per_test_cell,
)
from tab_foundry.types import TaskBatch

from .contract import (
    _classification_manifest_records,
    _manifest_path_from_corpus_record,
    _row_total_from_record,
)
from .shared import _CANARY_PREDICTORS, _MAX_REPORTED_TASK_ERRORS


def score_task_local_predictors(
    batch: TaskBatch,
    *,
    predictors: tuple[str, ...] | list[str],
) -> tuple[dict[str, dict[str, float | int]], dict[str, str]]:
    n_test = int(batch.y_test.shape[0])
    if batch.num_classes is None:
        raise RuntimeError("classification baselines require batch.num_classes")
    num_classes = int(batch.num_classes)
    y_train = batch.y_train.detach().cpu().numpy().astype(np.int64, copy=False)
    y_test = batch.y_test.detach().cpu().numpy().astype(np.int64, copy=False)
    x_train = batch.x_train.detach().cpu().numpy().astype(np.float64, copy=False)
    x_test = batch.x_test.detach().cpu().numpy().astype(np.float64, copy=False)

    results: dict[str, dict[str, float | int]] = {}
    errors: dict[str, str] = {}
    for predictor in predictors:
        try:
            if predictor == "chance":
                probabilities = np.full(
                    (n_test, num_classes),
                    1.0 / float(num_classes),
                    dtype=np.float64,
                )
            elif predictor == "logistic_regression":
                estimator = LogisticRegression(max_iter=1000, random_state=0)
                estimator.fit(x_train, y_train)
                raw_probabilities = estimator.predict_proba(x_test)
                probabilities = np.zeros((n_test, num_classes), dtype=np.float64)
                for column_index, class_index in enumerate(estimator.classes_):
                    probabilities[:, int(class_index)] = raw_probabilities[:, column_index]
            else:
                raise RuntimeError(f"unsupported canary predictor {predictor!r}")
            results[predictor] = {
                "n_test": n_test,
                "num_classes": num_classes,
                "label_target_log_loss_per_test_cell": label_target_log_loss_per_test_cell(
                    probabilities,
                    y_test,
                ),
            }
        except Exception as exc:
            errors[predictor] = f"{type(exc).__name__}: {exc}"
    return results, errors


def _new_metric_accumulator() -> dict[str, float | int]:
    return {
        "task_count": 0,
        "test_cell_count": 0,
        "log_loss_weighted_sum": 0.0,
    }


def _update_metric_accumulator(
    accumulator: dict[str, float | int],
    metrics: Mapping[str, float | int],
) -> None:
    test_cell_count = int(metrics["n_test"])
    accumulator["task_count"] = int(accumulator["task_count"]) + 1
    accumulator["test_cell_count"] = int(accumulator["test_cell_count"]) + test_cell_count
    accumulator["log_loss_weighted_sum"] = float(accumulator["log_loss_weighted_sum"]) + (
        float(metrics["label_target_log_loss_per_test_cell"]) * float(test_cell_count)
    )


def _finalize_metric_accumulator(accumulator: Mapping[str, float | int]) -> dict[str, Any] | None:
    task_count = int(accumulator["task_count"])
    test_cell_count = int(accumulator["test_cell_count"])
    if task_count <= 0 or test_cell_count <= 0:
        return None
    return {
        "task_count": task_count,
        "test_cell_count": test_cell_count,
        "label_target_log_loss_per_test_cell": (
            float(accumulator["log_loss_weighted_sum"]) / float(test_cell_count)
        ),
    }


def score_canary_block(
    block: SyntheticAdequacyBlock,
    *,
    corpus_record: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _manifest_path_from_corpus_record(corpus_record)
    records = _classification_manifest_records(manifest_path)
    splits = sorted({str(record["split"]) for record in records})
    expected_counts = Counter(_row_total_from_record(record) for record in records)
    predictors_evaluated = tuple(
        predictor for predictor in block.predictors if predictor in _CANARY_PREDICTORS
    )
    predictors_omitted = [
        predictor for predictor in block.predictors if predictor not in predictors_evaluated
    ]

    accumulators = {
        predictor: {int(row_total): _new_metric_accumulator() for row_total in block.n_ladder}
        for predictor in predictors_evaluated
    }
    predictor_errors: list[dict[str, Any]] = []

    for split in splits:
        dataset = PackedParquetTaskDataset(
            manifest_path,
            split=split,
            task="classification",
            load_metadata=False,
        )
        for index in range(len(dataset)):
            batch = dataset[index]
            record = dataset.records[index]
            row_total = _row_total_from_record(record)
            if row_total not in {int(value) for value in block.n_ladder}:
                continue
            results, errors = score_task_local_predictors(batch, predictors=predictors_evaluated)
            for predictor, metrics in results.items():
                _update_metric_accumulator(accumulators[predictor][row_total], metrics)
            for predictor, error_text in errors.items():
                predictor_errors.append(
                    {
                        "predictor": predictor,
                        "dataset_id": str(record.get("dataset_id", "unknown")),
                        "split": str(record.get("split", "unknown")),
                        "row_total": row_total,
                        "error": error_text,
                    }
                )

    scores_by_predictor: dict[str, dict[str, Any] | None] = {}
    for predictor, predictor_accumulators in accumulators.items():
        scores_by_predictor[predictor] = {
            str(row_total): _finalize_metric_accumulator(predictor_accumulators[int(row_total)])
            for row_total in block.n_ladder
        }

    comparisons: dict[str, Any] = {}
    chance_scores = cast(dict[str, Any], scores_by_predictor.get("chance", {}))
    logistic_scores = cast(dict[str, Any], scores_by_predictor.get("logistic_regression", {}))
    for row_total in block.n_ladder:
        chance_summary = chance_scores.get(str(row_total))
        logistic_summary = logistic_scores.get(str(row_total))
        if not isinstance(chance_summary, Mapping) or not isinstance(logistic_summary, Mapping):
            continue
        chance_log_loss = float(chance_summary["label_target_log_loss_per_test_cell"])
        logistic_log_loss = float(logistic_summary["label_target_log_loss_per_test_cell"])
        improvement = chance_log_loss - logistic_log_loss
        comparisons[str(row_total)] = {
            "chance_minus_logistic_log_loss": improvement,
            "logistic_beats_chance": improvement > 0.0,
        }

    return {
        "block_id": block.block_id,
        "manifest_path": str(manifest_path),
        "predictors_evaluated": list(predictors_evaluated),
        "predictors_omitted": predictors_omitted,
        "expected_task_count_by_row_total": {
            str(row_total): int(expected_counts.get(int(row_total), 0))
            for row_total in block.n_ladder
        },
        "scores_by_predictor": scores_by_predictor,
        "comparisons": comparisons,
        "predictor_error_count": len(predictor_errors),
        "predictor_errors": predictor_errors[:_MAX_REPORTED_TASK_ERRORS],
    }


__all__ = [
    "score_canary_block",
    "score_task_local_predictors",
]
