"""Lean synthetic adequacy pilot for TF-RD-010."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, cast

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression

from tab_foundry.config import compose_config
from tab_foundry.data.corpus_materialization import materialize_corpus_refs_batch
from tab_foundry.data.dataset import (
    PackedParquetTaskDataset,
    load_manifest_record_catalog,
)
from tab_foundry.repo_paths import repo_root as shared_repo_root
from tab_foundry.research.synthetic_adequacy import (
    SyntheticAdequacyBlock,
    label_target_log_loss_per_test_cell,
    load_synthetic_adequacy_spec,
)
from tab_foundry.training.health import run_inspect
from tab_foundry.training.trainer import train
from tab_foundry.types import TaskBatch


_SUPPORTED_ADEQUACY_ID = "tf_rd_010_synthetic_adequacy_v3"
_SUPPORTED_DEVICE = "cpu"
_LATENT_TARGET_DERIVATION = "tabiclv2_latent_node"
_CANARY_BLOCK_ID = "latent_target_canary_curated_v3"
_PRODUCTION_BLOCK_ID = "production_control_curated_v5"
_TRAINING_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_v1"
_CANARY_PREDICTORS = frozenset({"chance", "logistic_regression"})
_SUMMARY_JSON_NAME = "summary.json"
_SUMMARY_MARKDOWN_NAME = "summary.md"
_MAX_REPORTED_TASK_ERRORS = 12
_ABSOLUTE_CANARY_IMPROVEMENT_THRESHOLD = 0.05

_MEDIUM_V4_TRAINING_SURFACE = {
    "experiment": _TRAINING_EXPERIMENT,
    "task_batch_size": 16,
    "grad_accum_steps": 4,
    "grad_clip": 0.0,
    "max_steps": 2500,
    "optimizer_min_lr": 1.0e-5,
    "runtime": {
        "device": "cpu",
        "mixed_precision": "no",
        "num_workers": 0,
        "eval_every": 25,
        "checkpoint_every": 25,
        "val_batches": 0,
        "seed": 1,
    },
    "schedule_stage": {
        "name": "prior_dump",
        "steps": 2500,
        "lr_max": 1.0e-3,
        "lr_schedule": "linear",
        "warmup_ratio": 0.10,
    },
}


def _repo_root() -> Path:
    return shared_repo_root()


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_pilot_output_root(
    adequacy_id: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    return (
        (repo_root or _repo_root()).expanduser().resolve()
        / "outputs"
        / "research"
        / "adequacy"
        / adequacy_id
        / "pilot"
    )


def _ensure_supported_configuration(*, adequacy_id: str, device: str) -> None:
    if adequacy_id != _SUPPORTED_ADEQUACY_ID:
        raise RuntimeError(
            "the lean adequacy pilot currently supports only "
            f"{_SUPPORTED_ADEQUACY_ID!r}, got {adequacy_id!r}"
        )
    if str(device).strip().lower() != _SUPPORTED_DEVICE:
        raise RuntimeError(
            f"the lean adequacy pilot supports device={_SUPPORTED_DEVICE!r} only, got {device!r}"
        )


def _manifest_path_from_corpus_record(corpus_record: Mapping[str, Any]) -> Path:
    manifest = _ensure_mapping(corpus_record.get("manifest"), context="corpus_record.manifest")
    raw_manifest_path = manifest.get("manifest_path")
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
        raise RuntimeError("corpus_record.manifest.manifest_path must be a non-empty string")
    return Path(raw_manifest_path).expanduser().resolve()


def _row_total_from_record(record: Mapping[str, Any]) -> int:
    return int(record.get("n_train", 0)) + int(record.get("n_test", 0))


def _classification_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(manifest_path)
    records = []
    for raw_record in table.to_pylist():
        record = {str(key): value for key, value in cast(Mapping[str, Any], raw_record).items()}
        if str(record.get("task", "")).strip().lower() != "classification":
            continue
        records.append(record)
    return records


def validate_latent_target_metadata(
    catalog_record: Mapping[str, Any],
    *,
    n_features: int | None = None,
) -> dict[str, Any]:
    missing_reasons: list[str] = []
    target_derivation = catalog_record.get("target_derivation")
    if target_derivation != _LATENT_TARGET_DERIVATION:
        missing_reasons.append(
            f"catalog.target_derivation must equal {_LATENT_TARGET_DERIVATION!r}"
        )

    feature_count = int(n_features) if n_features is not None else None
    target_relevant_feature_count: int | None = None
    target_relevant_feature_fraction: float | None = None
    target_relevance = _optional_mapping(catalog_record.get("target_relevance"))
    if target_relevance is None:
        missing_reasons.append("catalog.target_relevance is missing")
    else:
        target_relevant_feature_count = _int_or_none(target_relevance.get("feature_count"))
        if target_relevant_feature_count is None:
            missing_reasons.append("catalog.target_relevance.feature_count must be an integer")
        elif target_relevant_feature_count < 0:
            missing_reasons.append(
                "catalog.target_relevance.feature_count must be non-negative"
            )

        target_relevant_feature_fraction = _finite_float_or_none(
            target_relevance.get("feature_fraction")
        )
        if target_relevant_feature_fraction is None or not (
            0.0 <= target_relevant_feature_fraction <= 1.0
        ):
            missing_reasons.append(
                "catalog.target_relevance.feature_fraction must be finite in [0, 1]"
            )

        if (
            feature_count is not None
            and target_relevant_feature_count is not None
            and target_relevant_feature_count > feature_count
        ):
            missing_reasons.append(
                "catalog.target_relevance.feature_count must be within [0, n_features]"
            )
        if (
            feature_count is not None
            and target_relevant_feature_count is not None
            and target_relevant_feature_fraction is not None
            and feature_count > 0
        ):
            expected_fraction = float(target_relevant_feature_count) / float(feature_count)
            if not math.isclose(
                target_relevant_feature_fraction,
                expected_fraction,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                missing_reasons.append(
                    "catalog.target_relevance.feature_fraction does not match feature_count / n_features"
                )

    return {
        "present": not missing_reasons,
        "target_derivation": (
            None if target_derivation is None else str(target_derivation)
        ),
        "feature_count": feature_count,
        "target_relevant_feature_count": target_relevant_feature_count,
        "target_relevant_feature_fraction": target_relevant_feature_fraction,
        "missing_reasons": missing_reasons,
    }


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


def inspect_corpus_latent_target_contract(
    *,
    block: SyntheticAdequacyBlock,
    corpus_record: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _manifest_path_from_corpus_record(corpus_record)
    records = _classification_manifest_records(manifest_path)
    row_total_counts = Counter(_row_total_from_record(record) for record in records)
    split_counts = Counter(str(record.get("split", "")).strip() for record in records)

    sample_records: list[dict[str, Any]] = []
    missing_reasons: list[str] = []
    for row_total in block.n_ladder:
        sample_record = next(
            (record for record in records if _row_total_from_record(record) == int(row_total)),
            None,
        )
        if sample_record is None:
            missing_reasons.append(
                f"manifest is missing a classification record for row_total={int(row_total)}"
            )
            continue
        catalog_record = load_manifest_record_catalog(
            manifest_path,
            record=sample_record,
        )
        validation = validate_latent_target_metadata(
            catalog_record,
            n_features=int(sample_record["n_features"]),
        )
        sample_entry = {
            "row_total": int(row_total),
            "dataset_id": str(sample_record.get("dataset_id", "unknown")),
            "split": str(sample_record.get("split", "unknown")),
            "dataset_index": int(sample_record["dataset_index"]),
            "n_train": int(sample_record["n_train"]),
            "n_test": int(sample_record["n_test"]),
            "n_features": int(sample_record["n_features"]),
            "n_classes": int(sample_record["n_classes"]),
            **validation,
        }
        sample_records.append(sample_entry)
        if not bool(validation["present"]):
            for reason in cast(list[str], validation["missing_reasons"]):
                missing_reasons.append(f"row_total={int(row_total)}: {reason}")

    provenance_summary = _optional_mapping(corpus_record.get("dagzoo_provenance_summary")) or {}
    dagzoo_provenance = _optional_mapping(corpus_record.get("dagzoo_provenance")) or {}
    if provenance_summary.get("target_derivation") != _LATENT_TARGET_DERIVATION:
        missing_reasons.append(
            f"corpus_record.dagzoo_provenance_summary.target_derivation must equal {_LATENT_TARGET_DERIVATION!r}"
        )

    invocation_payloads = cast(list[Any], dagzoo_provenance.get("invocations", []))
    target_accepted_datasets = 0
    if not invocation_payloads:
        missing_reasons.append("corpus_record.dagzoo_provenance.invocations is missing")
    for invocation in invocation_payloads:
        if not isinstance(invocation, Mapping):
            missing_reasons.append("corpus_record.dagzoo_provenance.invocations must contain mappings")
            continue
        normalized_invocation = {
            str(key): value for key, value in cast(Mapping[str, Any], invocation).items()
        }
        invocation_id = str(normalized_invocation.get("invocation_id", "unknown"))
        requested_count = _int_or_none(normalized_invocation.get("num_datasets"))
        if requested_count is None or requested_count <= 0:
            missing_reasons.append(
                f"invocation {invocation_id!r} is missing a positive num_datasets target"
            )
            continue
        target_accepted_datasets += requested_count
        filter_payload = _optional_mapping(normalized_invocation.get("filter"))
        if filter_payload is None:
            missing_reasons.append(
                f"invocation {invocation_id!r} is missing accepted_only filter provenance"
            )
            continue
        if str(filter_payload.get("filter_policy", "")).strip() != "accepted_only":
            missing_reasons.append(
                f"invocation {invocation_id!r} filter_policy must equal 'accepted_only'"
            )
        curated_accepted = _int_or_none(filter_payload.get("curated_accepted_datasets"))
        if curated_accepted != requested_count:
            missing_reasons.append(
                f"invocation {invocation_id!r} curated_accepted_datasets "
                f"must equal authored target {requested_count}, got {curated_accepted!r}"
            )
        accepted_count = _int_or_none(filter_payload.get("accepted_datasets"))
        if accepted_count is None or accepted_count < requested_count:
            missing_reasons.append(
                f"invocation {invocation_id!r} accepted_datasets must be at least {requested_count}"
            )
        for required_path_key in ("filter_manifest_path", "filter_summary_path", "curated_dir"):
            raw_path = filter_payload.get(required_path_key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                missing_reasons.append(
                    f"invocation {invocation_id!r} filter provenance is missing {required_path_key}"
                )
                continue
            if not Path(raw_path).expanduser().resolve().exists():
                missing_reasons.append(
                    f"invocation {invocation_id!r} {required_path_key} does not exist"
                )

    accepted_datasets = _int_or_none(provenance_summary.get("accepted_datasets"))
    curated_accepted_datasets = _int_or_none(provenance_summary.get("curated_accepted_datasets"))
    if str(provenance_summary.get("filter_policy", "")).strip() != "accepted_only":
        missing_reasons.append(
            "corpus_record.dagzoo_provenance_summary.filter_policy must equal 'accepted_only'"
        )
    if target_accepted_datasets > 0:
        if accepted_datasets is None or accepted_datasets < target_accepted_datasets:
            missing_reasons.append(
                "corpus_record.dagzoo_provenance_summary.accepted_datasets must meet the authored target"
            )
        if curated_accepted_datasets != target_accepted_datasets:
            missing_reasons.append(
                "corpus_record.dagzoo_provenance_summary.curated_accepted_datasets must equal the authored target"
            )

    return {
        "required": True,
        "present": not missing_reasons,
        "provenance": {
            "target_derivation": provenance_summary.get("target_derivation"),
            "target_relevant_feature_count_range": provenance_summary.get(
                "target_relevant_feature_count_range"
            ),
            "target_relevant_feature_fraction_range": provenance_summary.get(
                "target_relevant_feature_fraction_range"
            ),
        },
        "filter_provenance": {
            "filter_policy": provenance_summary.get("filter_policy"),
            "target_accepted_datasets": target_accepted_datasets,
            "accepted_datasets": accepted_datasets,
            "rejected_datasets": _int_or_none(provenance_summary.get("rejected_datasets")),
            "curated_accepted_datasets": curated_accepted_datasets,
            "acceptance_rate": _finite_float_or_none(provenance_summary.get("acceptance_rate")),
        },
        "manifest_path": str(manifest_path),
        "classification_task_count": len(records),
        "row_total_counts": {
            str(row_total): int(count)
            for row_total, count in sorted(row_total_counts.items())
        },
        "split_counts": {
            split: int(count)
            for split, count in sorted(split_counts.items())
        },
        "sample_records": sample_records,
        "missing_reasons": missing_reasons,
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


def build_production_control_config(
    *,
    corpus_ref: str,
    run_dir: Path,
    device: str,
) -> Any:
    cfg = compose_config(
        [
            f"experiment={_TRAINING_EXPERIMENT}",
            "logging.use_wandb=false",
        ]
    )
    cfg.data.corpus_ref = str(corpus_ref)
    cfg.runtime.device = str(device)
    cfg.runtime.mixed_precision = "no"
    cfg.runtime.num_workers = 0
    cfg.runtime.grad_accum_steps = 4
    cfg.runtime.grad_clip = 0.0
    cfg.runtime.max_steps = 2500
    cfg.runtime.eval_every = 25
    cfg.runtime.checkpoint_every = 25
    cfg.runtime.val_batches = 0
    cfg.runtime.seed = 1
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.training.task_batch_size = 16
    cfg.optimizer.min_lr = 1.0e-5
    cfg.schedule.stages = [
        dict(
            cast(
                Mapping[str, Any],
                _MEDIUM_V4_TRAINING_SURFACE["schedule_stage"],
            )
        )
    ]
    cfg.logging.run_name = f"{_SUPPORTED_ADEQUACY_ID}-production-control-v4"
    cfg.logging.history_jsonl_path = str((run_dir / "train_history.jsonl").resolve())
    return cfg


def _run_inspect_excerpt(payload: Mapping[str, Any]) -> dict[str, Any]:
    health = payload.get("health")
    health_excerpt = None
    if isinstance(health, Mapping):
        health_excerpt = {
            "verdict": health.get("verdict"),
            "summary": health.get("summary"),
            "metrics": _json_safe(_optional_mapping(health.get("metrics")) or {}),
        }
    artifacts = _optional_mapping(payload.get("artifacts")) or {}
    selected_artifacts = {
        key: artifacts[key]
        for key in (
            "training_surface_record_json",
            "telemetry_json",
            "gradient_history_jsonl",
            "train_history_jsonl",
            "latest_checkpoint_pt",
            "best_checkpoint_pt",
            "checkpoints_dir",
        )
        if key in artifacts
    }
    return {
        "surface_labels": _json_safe(_optional_mapping(payload.get("surface_labels")) or {}),
        "health": health_excerpt,
        "artifacts": _json_safe(selected_artifacts),
    }


def run_production_control_pilot(
    *,
    corpus_ref: str,
    pilot_root: Path,
    device: str,
    force: bool,
) -> dict[str, Any]:
    run_root = pilot_root / _PRODUCTION_BLOCK_ID
    run_dir = run_root / "train"
    if force and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = build_production_control_config(
        corpus_ref=corpus_ref,
        run_dir=run_dir,
        device=device,
    )
    config_excerpt = {
        "experiment": _TRAINING_EXPERIMENT,
        "corpus_ref": corpus_ref,
        "task_batch_size": int(cfg.training.task_batch_size),
        "runtime": {
            "device": str(cfg.runtime.device),
            "mixed_precision": str(cfg.runtime.mixed_precision),
            "num_workers": int(cfg.runtime.num_workers),
            "grad_accum_steps": int(cfg.runtime.grad_accum_steps),
            "grad_clip": float(cfg.runtime.grad_clip),
            "max_steps": int(cfg.runtime.max_steps),
            "eval_every": int(cfg.runtime.eval_every),
            "checkpoint_every": int(cfg.runtime.checkpoint_every),
            "val_batches": int(cfg.runtime.val_batches),
            "seed": int(cfg.runtime.seed),
        },
        "optimizer": {
            "name": str(cfg.optimizer.name),
            "min_lr": float(cfg.optimizer.min_lr),
        },
        "schedule_stages": cast(
            list[dict[str, Any]],
            OmegaConf.to_container(cfg.schedule.stages, resolve=True),
        ),
        "logging": {
            "run_name": str(cfg.logging.run_name),
            "use_wandb": bool(cfg.logging.use_wandb),
            "history_jsonl_path": str(cfg.logging.history_jsonl_path),
        },
        "output_dir": str(run_dir.resolve()),
    }

    try:
        result = train(cfg)
    except Exception as exc:
        inspect_payload = None
        if run_dir.exists():
            try:
                inspect_payload = run_inspect(run_dir)
            except Exception:
                inspect_payload = None
        if inspect_payload is not None:
            _write_json(run_root / "run_inspect.json", cast(Mapping[str, Any], inspect_payload))
        return {
            "block_id": _PRODUCTION_BLOCK_ID,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "run_dir": str(run_dir.resolve()),
            "config_excerpt": config_excerpt,
            "run_inspect": (
                None
                if inspect_payload is None
                else _run_inspect_excerpt(cast(Mapping[str, Any], inspect_payload))
            ),
        }

    inspect_payload = run_inspect(result.output_dir)
    _write_json(run_root / "run_inspect.json", cast(Mapping[str, Any], inspect_payload))
    return {
        "block_id": _PRODUCTION_BLOCK_ID,
        "status": "completed",
        "run_dir": str(result.output_dir.resolve()),
        "config_excerpt": config_excerpt,
        "metrics": {
            "best_val_loss": _finite_float_or_none(result.metrics.get("best_val_loss")),
            "best_val_step": _finite_float_or_none(result.metrics.get("best_val_step")),
            "final_val_loss": _finite_float_or_none(result.metrics.get("final_val_loss")),
            "train_elapsed_seconds": _finite_float_or_none(result.metrics.get("train_elapsed_seconds")),
            "wall_elapsed_seconds": _finite_float_or_none(result.metrics.get("wall_elapsed_seconds")),
        },
        "checkpoints": {
            "best_checkpoint": (
                None
                if result.best_checkpoint is None
                else str(result.best_checkpoint.resolve())
            ),
            "latest_checkpoint": (
                None
                if result.latest_checkpoint is None
                else str(result.latest_checkpoint.resolve())
            ),
        },
        "run_inspect": _run_inspect_excerpt(inspect_payload),
    }


def _canary_failure_reasons(canary_summary: Mapping[str, Any] | None) -> list[str]:
    if canary_summary is None:
        return ["canary baseline summary is missing"]
    if int(canary_summary.get("predictor_error_count", 0)) > 0:
        return ["one or more canary baseline tasks failed to score cleanly"]
    comparisons = _optional_mapping(canary_summary.get("comparisons")) or {}
    if not comparisons:
        return ["canary baseline comparisons are missing"]
    healthy_buckets = 0
    checked_buckets = 0
    for comparison in comparisons.values():
        comparison_payload = _optional_mapping(comparison)
        if comparison_payload is None:
            continue
        improvement = _finite_float_or_none(
            comparison_payload.get("chance_minus_logistic_log_loss")
        )
        if improvement is None:
            continue
        checked_buckets += 1
        if improvement >= _ABSOLUTE_CANARY_IMPROVEMENT_THRESHOLD:
            healthy_buckets += 1
    if checked_buckets == 0:
        return ["canary baseline comparisons did not produce any scored row-total buckets"]
    if healthy_buckets * 2 < checked_buckets:
        return [
            "logistic regression does not beat chance by a convincing margin on most canary row totals"
        ]
    return []


def _production_training_problem_reasons(production_summary: Mapping[str, Any] | None) -> list[str]:
    if production_summary is None:
        return []
    if production_summary.get("status") == "error":
        error_payload = _optional_mapping(production_summary.get("error")) or {}
        return [
            "production-control sandwich pilot errored: "
            f"{error_payload.get('type', 'RuntimeError')}: {error_payload.get('message', 'unknown error')}"
        ]
    run_inspect_payload = _optional_mapping(production_summary.get("run_inspect")) or {}
    health = _optional_mapping(run_inspect_payload.get("health"))
    if health is None:
        return []
    verdict = str(health.get("verdict", "")).strip().lower()
    metrics = _optional_mapping(health.get("metrics")) or {}
    initial_train_loss = _finite_float_or_none(metrics.get("initial_train_loss"))
    final_train_loss = _finite_float_or_none(metrics.get("final_train_loss"))
    if verdict == "fail":
        return ["production-control sandwich pilot tripped the run-health fail thresholds"]
    if (
        initial_train_loss is not None
        and final_train_loss is not None
        and final_train_loss >= initial_train_loss * 0.98
    ):
        return ["production-control sandwich pilot does not show a meaningful train-loss reduction"]
    return []


def select_provisional_interpretation(
    *,
    decision_buckets: Mapping[str, str],
    latent_target_contract: Mapping[str, Mapping[str, Any]],
    canary_summary: Mapping[str, Any] | None,
    production_control_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    missing_contract_blocks = [
        block_id
        for block_id, payload in latent_target_contract.items()
        if bool(payload.get("required")) and not bool(payload.get("present"))
    ]
    if missing_contract_blocks:
        reasoning = [
            "latent-target contract validation failed for "
            + ", ".join(sorted(missing_contract_blocks))
        ]
        bucket = "generator_problem"
    else:
        canary_reasons = _canary_failure_reasons(canary_summary)
        if canary_reasons:
            reasoning = canary_reasons
            bucket = "generator_problem"
        else:
            production_reasons = _production_training_problem_reasons(production_control_summary)
            if production_reasons:
                reasoning = production_reasons
                bucket = "training_regime_problem"
            else:
                reasoning = [
                    "latent-target lineage metadata validates, the canary baselines beat chance, and the "
                    "single production-control CPU pilot is not decisively broken"
                ]
                bucket = "inconclusive"
    return {
        "bucket": bucket,
        "definition": decision_buckets.get(bucket),
        "reasoning": reasoning,
    }


def _markdown_float(value: Any) -> str:
    numeric = _finite_float_or_none(value)
    return "n/a" if numeric is None else f"{numeric:.4f}"


def render_adequacy_pilot_markdown(summary: Mapping[str, Any]) -> str:
    materialized_corpora = _optional_mapping(summary.get("materialized_corpora")) or {}
    latent_target_contract = _optional_mapping(summary.get("latent_target_contract")) or {}
    canary_summary = _optional_mapping(summary.get("canary_baselines"))
    production_control_summary = _optional_mapping(summary.get("production_control_pilot"))
    interpretation = _ensure_mapping(
        summary.get("provisional_interpretation"),
        context="summary.provisional_interpretation",
    )

    lines = [
        f"# {summary['adequacy_id']} adequacy pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Provisional interpretation: `{interpretation['bucket']}`",
        f"- Blocked sweeps remain: {', '.join(cast(list[str], summary['blocked_sweeps']))}",
        "",
        "## Corpora",
        "",
        "| Block | Requested | Materialized | Latent target contract | Curated accepted | Acceptance rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for block_id, payload in materialized_corpora.items():
        corpus_payload = _ensure_mapping(payload, context=f"materialized_corpora.{block_id}")
        contract_payload = _optional_mapping(latent_target_contract.get(block_id)) or {}
        filter_payload = _optional_mapping(contract_payload.get("filter_provenance")) or {}
        target_accepted = _int_or_none(filter_payload.get("target_accepted_datasets"))
        curated_accepted = _int_or_none(filter_payload.get("curated_accepted_datasets"))
        curated_display = "n/a"
        if target_accepted is not None:
            curated_display = (
                f"`{curated_accepted}`/`{target_accepted}`"
                if curated_accepted is not None
                else f"`?`/`{target_accepted}`"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    block_id,
                    f"`{corpus_payload['requested_corpus_ref']}`",
                    f"`{corpus_payload['materialized_corpus_ref']}`",
                    "`present`" if contract_payload.get("present") else "`missing`",
                    curated_display,
                    _markdown_float(filter_payload.get("acceptance_rate")),
                ]
            )
            + " |"
        )
    if canary_summary is not None:
        lines.extend(
            [
                "",
                "## Canary Baselines",
                "",
                "| n | chance log loss | logistic log loss | chance - logistic |",
                "| --- | --- | --- | --- |",
            ]
        )
        scores_by_predictor = _optional_mapping(canary_summary.get("scores_by_predictor")) or {}
        chance_scores = _optional_mapping(scores_by_predictor.get("chance")) or {}
        logistic_scores = _optional_mapping(scores_by_predictor.get("logistic_regression")) or {}
        comparisons = _optional_mapping(canary_summary.get("comparisons")) or {}
        for row_total in ("128", "256", "512", "1024"):
            chance_payload = _optional_mapping(chance_scores.get(row_total)) or {}
            logistic_payload = _optional_mapping(logistic_scores.get(row_total)) or {}
            comparison_payload = _optional_mapping(comparisons.get(row_total)) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        row_total,
                        _markdown_float(chance_payload.get("label_target_log_loss_per_test_cell")),
                        _markdown_float(logistic_payload.get("label_target_log_loss_per_test_cell")),
                        _markdown_float(comparison_payload.get("chance_minus_logistic_log_loss")),
                    ]
                )
                + " |"
            )
    if production_control_summary is not None:
        lines.extend(
            [
                "",
                "## Production Control Pilot",
                "",
                f"- Status: `{production_control_summary.get('status', 'unknown')}`",
                f"- Run dir: `{production_control_summary.get('run_dir', 'unknown')}`",
            ]
        )
        metrics = _optional_mapping(production_control_summary.get("metrics")) or {}
        if metrics:
            lines.append(
                "- Validation losses: "
                f"best={_markdown_float(metrics.get('best_val_loss'))}, "
                f"final={_markdown_float(metrics.get('final_val_loss'))}"
            )
        run_inspect_payload = _optional_mapping(production_control_summary.get("run_inspect")) or {}
        health = _optional_mapping(run_inspect_payload.get("health")) or {}
        if health:
            lines.append(f"- Health verdict: `{health.get('verdict', 'unknown')}`")
            if health.get("summary") is not None:
                lines.append(f"- Health summary: {health['summary']}")

    reasoning = cast(list[str], interpretation.get("reasoning", []))
    if reasoning:
        lines.extend(
            [
                "",
                "## Interpretation Notes",
                "",
            ]
        )
        for reason in reasoning:
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _write_blocking_summary(
    *,
    adequacy_id: str,
    blocked_sweeps: tuple[str, ...] | list[str],
    pilot_root: Path,
    materialized_corpora: Mapping[str, Any],
    latent_target_contract: Mapping[str, Any],
    canary_summary: Mapping[str, Any] | None,
    definition: str | None,
    reasoning: list[str],
) -> None:
    summary = {
        "adequacy_id": adequacy_id,
        "status": "blocked",
        "blocked_sweeps": list(blocked_sweeps),
        "materialized_corpora": _json_safe(materialized_corpora),
        "latent_target_contract": _json_safe(latent_target_contract),
        "canary_baselines": _json_safe(canary_summary),
        "production_control_pilot": None,
        "provisional_interpretation": {
            "bucket": "generator_problem",
            "definition": definition,
            "reasoning": list(reasoning),
        },
        "summary_paths": {
            "summary_json": str((pilot_root / _SUMMARY_JSON_NAME).resolve()),
            "summary_md": str((pilot_root / _SUMMARY_MARKDOWN_NAME).resolve()),
        },
    }
    _write_json(pilot_root / _SUMMARY_JSON_NAME, summary)
    (pilot_root / _SUMMARY_MARKDOWN_NAME).write_text(
        render_adequacy_pilot_markdown(summary),
        encoding="utf-8",
    )


def _materialized_corpus_payload(
    *,
    block: SyntheticAdequacyBlock,
    corpus_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "requested_corpus_ref": block.corpus_ref,
        "materialized_corpus_ref": str(corpus_record["corpus_ref"]),
        "recipe_id": str(corpus_record["recipe_id"]),
        "corpus_id": str(corpus_record["corpus_id"]),
        "surface_label": str(corpus_record["surface_label"]),
        "manifest_path": str(_manifest_path_from_corpus_record(corpus_record)),
        "corpus_record_path": str(corpus_record["corpus_record_path"]),
    }


def run_adequacy_pilot(
    *,
    adequacy_id: str,
    dagzoo_root: Path,
    device: str = _SUPPORTED_DEVICE,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    out_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    _ensure_supported_configuration(adequacy_id=adequacy_id, device=device)
    spec = load_synthetic_adequacy_spec(adequacy_id, repo_root=repo_root)
    pilot_root = (
        out_root.expanduser().resolve()
        if out_root is not None
        else default_pilot_output_root(adequacy_id, repo_root=repo_root)
    )
    pilot_root.mkdir(parents=True, exist_ok=True)

    materialized_corpora: dict[str, dict[str, Any]] = {}
    latent_target_contract: dict[str, dict[str, Any]] = {}
    canary_summary: dict[str, Any] | None = None
    blocking_summary_written = False

    blocks_by_recipe_id: dict[str, list[SyntheticAdequacyBlock]] = {}
    for block in spec.blocks:
        recipe_id, _separator, _corpus_id = str(block.corpus_ref).partition("/")
        blocks_by_recipe_id.setdefault(recipe_id, []).append(block)

    summary_md_path = pilot_root / _SUMMARY_MARKDOWN_NAME
    canary_recipe_id = next(
        (
            str(block.corpus_ref).partition("/")[0]
            for block in spec.blocks
            if block.block_id == _CANARY_BLOCK_ID
        ),
        None,
    )

    def _on_corpus_materialized(corpus_record: dict[str, Any]) -> None:
        nonlocal blocking_summary_written, canary_summary
        recipe_id = str(corpus_record["recipe_id"])
        for block in blocks_by_recipe_id.get(recipe_id, []):
            materialized_corpora[block.block_id] = _materialized_corpus_payload(
                block=block,
                corpus_record=corpus_record,
            )
            latent_target_contract[block.block_id] = inspect_corpus_latent_target_contract(
                block=block,
                corpus_record=corpus_record,
            )
            filter_provenance = _optional_mapping(
                latent_target_contract[block.block_id].get("filter_provenance")
            )
            if filter_provenance is not None:
                materialized_corpora[block.block_id]["filter_provenance"] = filter_provenance
            if block.block_id != _CANARY_BLOCK_ID:
                continue
            canary_summary = score_canary_block(
                block,
                corpus_record=corpus_record,
            )
            contract_payload = latent_target_contract[block.block_id]
            if bool(contract_payload.get("required")) and not bool(contract_payload.get("present")):
                _write_blocking_summary(
                    adequacy_id=adequacy_id,
                    blocked_sweeps=spec.blocked_sweeps,
                    pilot_root=pilot_root,
                    materialized_corpora=materialized_corpora,
                    latent_target_contract=latent_target_contract,
                    canary_summary=canary_summary,
                    definition=spec.decision_buckets.get("generator_problem"),
                    reasoning=[
                        "latent-target contract validation failed for "
                        + str(block.block_id)
                    ],
                )
                blocking_summary_written = True
                raise RuntimeError(
                    "latent-target contract validation failed for the canary corpus; "
                    f"wrote blocking summary to {summary_md_path.resolve()}"
                )
            canary_failure_reasons = _canary_failure_reasons(canary_summary)
            if canary_failure_reasons:
                _write_blocking_summary(
                    adequacy_id=adequacy_id,
                    blocked_sweeps=spec.blocked_sweeps,
                    pilot_root=pilot_root,
                    materialized_corpora=materialized_corpora,
                    latent_target_contract=latent_target_contract,
                    canary_summary=canary_summary,
                    definition=spec.decision_buckets.get("generator_problem"),
                    reasoning=canary_failure_reasons,
                )
                blocking_summary_written = True
                raise RuntimeError(
                    "canary baseline validation failed for the adequacy pilot; "
                    f"wrote blocking summary to {summary_md_path.resolve()}"
                )

    try:
        _ = materialize_corpus_refs_batch(
            corpus_refs=[block.corpus_ref for block in spec.blocks],
            dagzoo_root=dagzoo_root,
            force=force,
            materialize_processes=materialize_processes,
            materialize_worker_threads=materialize_worker_threads,
            prioritized_recipe_ids=(
                [] if canary_recipe_id is None else [canary_recipe_id]
            ),
            on_corpus_materialized=_on_corpus_materialized,
            repo_root=repo_root,
        )
    except Exception as exc:
        if blocking_summary_written:
            raise
        _write_blocking_summary(
            adequacy_id=adequacy_id,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "corpus materialization or validation failed: "
                f"{type(exc).__name__}: {exc}"
            ],
        )
        blocking_summary_written = True
        raise RuntimeError(
            "adequacy pilot blocked during corpus materialization or validation; "
            f"wrote blocking summary to {summary_md_path.resolve()}"
        ) from exc

    missing_block_ids = [
        block.block_id for block in spec.blocks if block.block_id not in materialized_corpora
    ]
    if missing_block_ids:
        _write_blocking_summary(
            adequacy_id=adequacy_id,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "missing materialized adequacy blocks: "
                + ", ".join(sorted(missing_block_ids))
            ],
        )
        raise RuntimeError(
            "adequacy pilot did not materialize every required corpus; "
            f"wrote blocking summary to {summary_md_path.resolve()}"
        )

    missing_contract_blocks = [
        block_id
        for block_id, payload in latent_target_contract.items()
        if bool(payload.get("required")) and not bool(payload.get("present"))
    ]
    if missing_contract_blocks:
        _write_blocking_summary(
            adequacy_id=adequacy_id,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "latent-target contract validation failed for "
                + ", ".join(sorted(missing_contract_blocks))
            ],
        )
        raise RuntimeError(
            "latent-target contract validation failed for one or more adequacy blocks; "
            f"wrote blocking summary to {(pilot_root / _SUMMARY_MARKDOWN_NAME).resolve()}"
        )

    canary_failure_reasons = _canary_failure_reasons(canary_summary)
    if canary_failure_reasons:
        _write_blocking_summary(
            adequacy_id=adequacy_id,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=canary_failure_reasons,
        )
        raise RuntimeError(
            "canary baseline validation failed for the adequacy pilot; "
            f"wrote blocking summary to {(pilot_root / _SUMMARY_MARKDOWN_NAME).resolve()}"
        )

    production_control_corpus_ref = materialized_corpora[_PRODUCTION_BLOCK_ID]["materialized_corpus_ref"]
    production_control_summary = run_production_control_pilot(
        corpus_ref=cast(str, production_control_corpus_ref),
        pilot_root=pilot_root,
        device=device,
        force=force,
    )

    interpretation = select_provisional_interpretation(
        decision_buckets=spec.decision_buckets,
        latent_target_contract=latent_target_contract,
        canary_summary=canary_summary,
        production_control_summary=production_control_summary,
    )
    summary = {
        "adequacy_id": adequacy_id,
        "status": "completed",
        "blocked_sweeps": list(spec.blocked_sweeps),
        "materialized_corpora": materialized_corpora,
        "latent_target_contract": latent_target_contract,
        "canary_baselines": canary_summary,
        "production_control_pilot": production_control_summary,
        "provisional_interpretation": interpretation,
        "summary_paths": {
            "summary_json": str((pilot_root / _SUMMARY_JSON_NAME).resolve()),
            "summary_md": str((pilot_root / _SUMMARY_MARKDOWN_NAME).resolve()),
        },
    }
    _write_json(pilot_root / _SUMMARY_JSON_NAME, summary)
    (pilot_root / _SUMMARY_MARKDOWN_NAME).write_text(
        render_adequacy_pilot_markdown(summary),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "build_production_control_config",
    "default_pilot_output_root",
    "inspect_corpus_latent_target_contract",
    "render_adequacy_pilot_markdown",
    "run_adequacy_pilot",
    "run_production_control_pilot",
    "score_canary_block",
    "score_task_local_predictors",
    "select_provisional_interpretation",
    "validate_latent_target_metadata",
]
