"""Missingness and telemetry-summary helpers for prior-dump training."""

from __future__ import annotations

from typing import Any, cast

import torch

from tab_foundry.bench.prior_dump import PriorDumpBatchMissingness


def _initial_missingness_summary(
    prior_dump_path,
    *,
    prior_missingness_config: dict[str, Any] | None,
    prior_dump_non_finite_policy: str,
) -> dict[str, Any]:
    synthetic_enabled = prior_missingness_config is not None
    return {
        "prior_dump": {
            "path": str(prior_dump_path.expanduser().resolve()),
            "allow_missing_values": False,
            "non_finite_policy": str(prior_dump_non_finite_policy),
            "batches_seen": 0,
            "affected_batch_count": 0,
            "affected_dataset_indices": [],
            "non_finite_feature_count": 0,
            "non_finite_label_count": 0,
            "last_batch": None,
            "skipped_batch_count": 0,
            "skipped_dataset_indices": [],
            "skipped_non_finite_feature_count": 0,
            "skipped_non_finite_label_count": 0,
            "last_skipped_batch": None,
        },
        "synthetic_prior": {
            "enabled": synthetic_enabled,
            "min_rate": None if prior_missingness_config is None else float(prior_missingness_config["min_rate"]),
            "max_rate": None if prior_missingness_config is None else float(prior_missingness_config["max_rate"]),
            "batches_seen": 0,
            "affected_batch_count": 0,
            "affected_dataset_indices": [],
            "masked_feature_count": 0,
            "last_batch": None,
        },
    }


def _accumulate_missingness(
    summary: dict[str, Any],
    *,
    batch_missingness: PriorDumpBatchMissingness,
    skipped: bool = False,
) -> None:
    prior_dump = cast(dict[str, Any], summary["prior_dump"])
    prior_dump["batches_seen"] = int(prior_dump["batches_seen"]) + 1
    prior_dump["non_finite_feature_count"] = int(prior_dump["non_finite_feature_count"]) + int(
        batch_missingness.non_finite_feature_count
    )
    prior_dump["non_finite_label_count"] = int(prior_dump["non_finite_label_count"]) + int(
        batch_missingness.non_finite_label_count
    )
    if batch_missingness.affected_batch_count > 0:
        prior_dump["affected_batch_count"] = int(prior_dump["affected_batch_count"]) + 1
        known_dataset_indices = {
            int(dataset_index) for dataset_index in prior_dump["affected_dataset_indices"]
        }
        known_dataset_indices.update(int(index) for index in batch_missingness.affected_dataset_indices)
        prior_dump["affected_dataset_indices"] = sorted(known_dataset_indices)
        prior_dump["last_batch"] = batch_missingness.to_dict()
    if skipped:
        prior_dump["skipped_batch_count"] = int(prior_dump["skipped_batch_count"]) + 1
        prior_dump["skipped_non_finite_feature_count"] = int(
            prior_dump["skipped_non_finite_feature_count"]
        ) + int(batch_missingness.non_finite_feature_count)
        prior_dump["skipped_non_finite_label_count"] = int(
            prior_dump["skipped_non_finite_label_count"]
        ) + int(batch_missingness.non_finite_label_count)
        skipped_dataset_indices = {
            int(dataset_index) for dataset_index in prior_dump["skipped_dataset_indices"]
        }
        skipped_dataset_indices.update(int(index) for index in batch_missingness.affected_dataset_indices)
        prior_dump["skipped_dataset_indices"] = sorted(skipped_dataset_indices)
        prior_dump["last_skipped_batch"] = batch_missingness.to_dict()


def _accumulate_synthetic_missingness(
    summary: dict[str, Any],
    *,
    batch_missingness: dict[str, Any] | None,
) -> None:
    synthetic_prior = cast(dict[str, Any], summary["synthetic_prior"])
    synthetic_prior["batches_seen"] = int(synthetic_prior["batches_seen"]) + 1
    if batch_missingness is None:
        return
    masked_feature_count = int(batch_missingness["masked_feature_count"])
    synthetic_prior["masked_feature_count"] = int(synthetic_prior["masked_feature_count"]) + masked_feature_count
    if masked_feature_count <= 0:
        return
    synthetic_prior["affected_batch_count"] = int(synthetic_prior["affected_batch_count"]) + 1
    known_dataset_indices = {
        int(dataset_index) for dataset_index in synthetic_prior["affected_dataset_indices"]
    }
    known_dataset_indices.update(int(index) for index in batch_missingness["affected_dataset_indices"])
    synthetic_prior["affected_dataset_indices"] = sorted(known_dataset_indices)
    synthetic_prior["last_batch"] = batch_missingness


def _apply_prior_missingness(
    x_batch: torch.Tensor,
    *,
    prior_step: Any,
    generator: torch.Generator | None,
    prior_missingness_config: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if prior_missingness_config is None:
        return x_batch, None

    masked = x_batch.clone()
    min_rate = float(prior_missingness_config["min_rate"])
    max_rate = float(prior_missingness_config["max_rate"])
    affected_dataset_indices: list[int] = []
    affected_datasets: list[dict[str, int | float]] = []
    masked_feature_count = 0
    for local_index, task in enumerate(prior_step.tasks):
        num_rows = int(task.x_train.shape[0] + task.x_test.shape[0])
        num_features = int(task.x_train.shape[1])
        if num_rows <= 0 or num_features <= 0:
            continue
        if max_rate > min_rate:
            assert generator is not None
            rate = min_rate + ((max_rate - min_rate) * float(torch.rand((), generator=generator).item()))
        else:
            rate = min_rate
        if rate <= 0.0:
            continue
        mask = torch.rand((num_rows, num_features), generator=generator, device="cpu") < rate
        dataset_masked_feature_count = int(mask.sum().item())
        if dataset_masked_feature_count <= 0:
            continue
        masked_feature_count += dataset_masked_feature_count
        dataset_index = int(prior_step.dataset_indices[local_index])
        affected_dataset_indices.append(dataset_index)
        affected_datasets.append(
            {
                "dataset_index": dataset_index,
                "masked_feature_count": dataset_masked_feature_count,
                "applied_rate": float(rate),
            }
        )
        masked[local_index, :num_rows, :num_features][mask.to(device=masked.device)] = float("nan")
    return masked, {
        "step_index": int(prior_step.step_index),
        "dataset_indices": [int(index) for index in prior_step.dataset_indices],
        "masked_feature_count": int(masked_feature_count),
        "affected_batch_count": 1 if masked_feature_count > 0 else 0,
        "affected_dataset_count": int(len(affected_dataset_indices)),
        "affected_dataset_indices": sorted(affected_dataset_indices),
        "affected_datasets": affected_datasets,
    }


def _prior_wandb_summary_payload(
    *,
    output_dir,
    global_step: int,
    telemetry_payload: dict[str, Any],
) -> dict[str, Any]:
    raw_gradient_summary = telemetry_payload.get("gradient_summary")
    gradient_global = (
        raw_gradient_summary.get("global")
        if isinstance(raw_gradient_summary, dict)
        else None
    )
    raw_missingness = telemetry_payload.get("missingness")
    prior_dump_missingness = (
        raw_missingness.get("prior_dump")
        if isinstance(raw_missingness, dict)
        else None
    )
    synthetic_prior_missingness = (
        raw_missingness.get("synthetic_prior")
        if isinstance(raw_missingness, dict)
        else None
    )
    affected_indices = (
        prior_dump_missingness.get("affected_dataset_indices")
        if isinstance(prior_dump_missingness, dict)
        else None
    )
    raw_artifacts = telemetry_payload.get("artifacts")
    latest_checkpoint = (
        raw_artifacts.get("latest_checkpoint")
        if isinstance(raw_artifacts, dict)
        else None
    )
    summary: dict[str, Any] = {
        "run": {
            "output_dir": str(output_dir),
            "global_step": int(global_step),
        },
        "telemetry": {
            "success": telemetry_payload.get("success"),
            "checkpoint_snapshot_count": len(telemetry_payload.get("checkpoint_snapshots", [])),
        },
        "artifacts": {
            "latest_checkpoint": latest_checkpoint,
        },
        "loss_summary": telemetry_payload.get("loss_summary"),
        "gradient_summary": {
            "global": gradient_global,
        },
        "diagnostics": telemetry_payload.get("diagnostics"),
    }
    if isinstance(prior_dump_missingness, dict) or isinstance(synthetic_prior_missingness, dict):
        summary["missingness"] = {}
        if isinstance(prior_dump_missingness, dict):
            summary["missingness"]["prior_dump"] = {
                "batches_seen": prior_dump_missingness.get("batches_seen"),
                "non_finite_policy": prior_dump_missingness.get("non_finite_policy"),
                "affected_batch_count": prior_dump_missingness.get("affected_batch_count"),
                "affected_dataset_count": len(affected_indices)
                if isinstance(affected_indices, list)
                else None,
                "non_finite_feature_count": prior_dump_missingness.get("non_finite_feature_count"),
                "non_finite_label_count": prior_dump_missingness.get("non_finite_label_count"),
                "skipped_batch_count": prior_dump_missingness.get("skipped_batch_count"),
                "skipped_dataset_count": len(prior_dump_missingness.get("skipped_dataset_indices", []))
                if isinstance(prior_dump_missingness.get("skipped_dataset_indices"), list)
                else None,
            }
        if isinstance(synthetic_prior_missingness, dict):
            summary["missingness"]["synthetic_prior"] = {
                "enabled": synthetic_prior_missingness.get("enabled"),
                "batches_seen": synthetic_prior_missingness.get("batches_seen"),
                "affected_batch_count": synthetic_prior_missingness.get("affected_batch_count"),
                "masked_feature_count": synthetic_prior_missingness.get("masked_feature_count"),
            }
    raw_error = telemetry_payload.get("error")
    if isinstance(raw_error, dict):
        summary["error"] = {
            "type": raw_error.get("type"),
            "message": raw_error.get("message"),
        }
    return summary
