"""Corpus comparison and benchmark-report payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_realdata_hub.manifest import compare_jsonlike_payloads

import tab_foundry.benchmark_registry as benchmark_registry

from .corpus_loading import _ensure_mapping, _ensure_non_empty_string, _read_json_mapping
from .corpus_lookup import load_corpus_record


def corpus_compare_payload(
    *,
    left: str,
    right: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    left_record = load_corpus_record(left, repo_root=repo_root, hydrate_characteristics=True)
    right_record = load_corpus_record(right, repo_root=repo_root, hydrate_characteristics=True)
    left_manifest = cast(Mapping[str, Any], left_record["manifest"])
    right_manifest = cast(Mapping[str, Any], right_record["manifest"])
    left_payload = {
        "recipe_id": left_record.get("recipe_id"),
        "corpus_id": left_record.get("corpus_id"),
        "surface_label": left_record.get("surface_label"),
        "inspection": left_manifest.get("inspection"),
        "characteristics": left_manifest.get("characteristics"),
    }
    right_payload = {
        "recipe_id": right_record.get("recipe_id"),
        "corpus_id": right_record.get("corpus_id"),
        "surface_label": right_record.get("surface_label"),
        "inspection": right_manifest.get("inspection"),
        "characteristics": right_manifest.get("characteristics"),
    }
    differences = compare_jsonlike_payloads(left_payload, right_payload)
    return {
        "left": left_record,
        "right": right_record,
        "difference_count": len(differences),
        "differences": differences,
    }


def corpus_results_payload(
    *,
    corpus_ref: str,
    registry_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    record = load_corpus_record(corpus_ref, repo_root=repo_root)
    normalized_corpus_ref = _ensure_non_empty_string(record.get("corpus_ref"), context="corpus record corpus_ref")
    registry = benchmark_registry.load_benchmark_run_registry(
        registry_path or benchmark_registry.default_benchmark_run_registry_path()
    )
    runs = _ensure_mapping(registry.get("runs"), context="benchmark run registry runs")
    matched_runs: list[dict[str, Any]] = []
    for run_id in sorted(runs):
        entry = runs.get(run_id)
        if not isinstance(entry, Mapping):
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        training_surface_record_path = artifacts.get("training_surface_record_path")
        if not isinstance(training_surface_record_path, str) or not training_surface_record_path.strip():
            continue
        resolved_surface_path = benchmark_registry.resolve_registry_path_value(training_surface_record_path)
        if not resolved_surface_path.exists():
            continue
        training_surface_record = _read_json_mapping(
            resolved_surface_path,
            context=f"training surface record for run {run_id!r}",
        )
        data_payload = training_surface_record.get("data")
        if not isinstance(data_payload, Mapping):
            continue
        if data_payload.get("corpus_ref") != normalized_corpus_ref:
            continue
        sweep_payload = entry.get("sweep")
        metrics = entry.get("tab_foundry_metrics")
        matched_runs.append(
            {
                "run_id": str(run_id),
                "experiment": entry.get("experiment"),
                "config_profile": entry.get("config_profile"),
                "decision": entry.get("decision"),
                "surface_labels": entry.get("surface_labels"),
                "sweep": {
                    "sweep_id": None if not isinstance(sweep_payload, Mapping) else sweep_payload.get("sweep_id"),
                    "delta_id": None if not isinstance(sweep_payload, Mapping) else sweep_payload.get("delta_id"),
                    "queue_order": None
                    if not isinstance(sweep_payload, Mapping)
                    else sweep_payload.get("queue_order"),
                },
                "headline_metrics": (
                    None
                    if not isinstance(metrics, Mapping)
                    else {
                        key: metrics.get(key)
                        for key in (
                            "best_bpc",
                            "final_bpc",
                            "best_bpf",
                            "final_bpf",
                            "best_roc_auc",
                            "final_roc_auc",
                            "best_log_loss",
                            "final_log_loss",
                            "best_brier_score",
                            "final_brier_score",
                            "best_step",
                        )
                        if key in metrics
                    }
                ),
                "training_surface_record_path": str(resolved_surface_path),
            }
        )
    return {
        "corpus_ref": normalized_corpus_ref,
        "recipe_id": record.get("recipe_id"),
        "corpus_id": record.get("corpus_id"),
        "run_count": len(matched_runs),
        "runs": matched_runs,
    }
