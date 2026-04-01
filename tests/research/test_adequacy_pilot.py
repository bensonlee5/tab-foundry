from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tab_foundry.research.adequacy.pilot as pilot_module
from tab_foundry.research.synthetic_adequacy import load_synthetic_adequacy_spec
from tab_foundry.types import TaskBatch


def _latent_target_metadata(*, n_features: int = 1) -> dict[str, object]:
    return {
        "prior": {
            "target_derivation": "tabiclv2_latent_node",
        },
        "lineage": {
            "assignments": {
                "feature_to_node": [0] * n_features,
                "target_to_node": 0,
                "target_relevant_features": list(range(n_features)),
                "target_relevant_feature_count": n_features,
                "target_relevant_feature_fraction": 1.0,
            }
        },
    }


def _latent_target_catalog_record(
    *,
    n_features: int,
    target_derivation: str = "tabiclv2_latent_node",
    relevant_feature_count: int | None = None,
    relevant_feature_fraction: float | None = None,
) -> dict[str, object]:
    resolved_relevant_feature_count = (
        n_features if relevant_feature_count is None else int(relevant_feature_count)
    )
    resolved_relevant_feature_fraction = (
        float(resolved_relevant_feature_count) / float(n_features)
        if relevant_feature_fraction is None
        else float(relevant_feature_fraction)
    )
    return {
        "target_derivation": target_derivation,
        "target_relevance": {
            "feature_count": resolved_relevant_feature_count,
            "feature_fraction": resolved_relevant_feature_fraction,
        },
    }


def _task_batch() -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [[-2.0], [-1.0], [1.0], [2.0]],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        x_test=torch.tensor([[-1.5], [1.5]], dtype=torch.float32),
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        metadata=_latent_target_metadata(),
        num_classes=2,
    )


def _fake_corpus_record(tmp_path: Path, *, corpus_ref: str) -> dict[str, object]:
    manifest_path = tmp_path / f"{corpus_ref.replace('/', '__')}.parquet"
    manifest_path.write_bytes(b"manifest")
    corpus_record_path = tmp_path / f"{corpus_ref.replace('/', '__')}_record.json"
    corpus_record_path.write_text("{}", encoding="utf-8")
    recipe_id, _, corpus_id = corpus_ref.partition("/")
    resolved_corpus_id = corpus_id or "materialized"
    return {
        "corpus_ref": f"{recipe_id}/{resolved_corpus_id}",
        "recipe_id": recipe_id,
        "corpus_id": resolved_corpus_id,
        "surface_label": recipe_id,
        "corpus_record_path": str(corpus_record_path.resolve()),
        "manifest": {
            "manifest_path": str(manifest_path.resolve()),
        },
    }


def _healthy_canary_summary() -> dict[str, object]:
    return {
        "scores_by_predictor": {
            "chance": {
                "128": {
                    "task_count": 32,
                    "test_cell_count": 1024,
                    "label_target_log_loss_per_test_cell": 0.6931,
                },
                "256": {
                    "task_count": 32,
                    "test_cell_count": 2048,
                    "label_target_log_loss_per_test_cell": 0.6931,
                },
            },
            "logistic_regression": {
                "128": {
                    "task_count": 32,
                    "test_cell_count": 1024,
                    "label_target_log_loss_per_test_cell": 0.2100,
                },
                "256": {
                    "task_count": 32,
                    "test_cell_count": 2048,
                    "label_target_log_loss_per_test_cell": 0.1800,
                },
            },
        },
        "comparisons": {
            "128": {"chance_minus_logistic_log_loss": 0.4831},
            "256": {"chance_minus_logistic_log_loss": 0.5131},
        },
        "predictor_error_count": 0,
    }


def test_validate_latent_target_metadata_accepts_fixture_payload() -> None:
    payload = pilot_module.validate_latent_target_metadata(
        _latent_target_catalog_record(n_features=2),
        n_features=2,
    )

    assert payload["present"] is True
    assert payload["target_derivation"] == "tabiclv2_latent_node"
    assert payload["feature_count"] == 2
    assert payload["target_relevant_feature_count"] == 2
    assert payload["target_relevant_feature_fraction"] == pytest.approx(1.0)
    assert payload["missing_reasons"] == []


def test_validate_latent_target_metadata_reports_missing_payload() -> None:
    payload = pilot_module.validate_latent_target_metadata(
        _latent_target_catalog_record(
            n_features=2,
            target_derivation="wrong",
            relevant_feature_fraction=0.25,
        ),
        n_features=2,
    )

    assert payload["present"] is False
    assert any(
        "catalog.target_relevance.feature_fraction does not match feature_count / n_features"
        in reason
        for reason in payload["missing_reasons"]
    )
    assert any("catalog.target_derivation must equal" in reason for reason in payload["missing_reasons"])


def test_inspect_corpus_latent_target_contract_accepts_public_catalog_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = load_synthetic_adequacy_spec("tf_rd_010_synthetic_adequacy_v3")
    block = next(
        candidate
        for candidate in spec.blocks
        if candidate.block_id == "latent_target_canary_curated_v3"
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.write_bytes(b"manifest")
    filter_manifest_path = tmp_path / "filter_manifest.ndjson"
    filter_manifest_path.write_text("{}\n", encoding="utf-8")
    filter_summary_path = tmp_path / "filter_summary.json"
    filter_summary_path.write_text("{}\n", encoding="utf-8")
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    sample_records = [
        {
            "dataset_id": f"dataset-{row_total}",
            "dataset_index": 0,
            "split": "train",
            "n_train": row_total - (row_total // 4),
            "n_test": row_total // 4,
            "n_features": 6,
            "n_classes": 2,
            "task": "classification",
            "catalog_path": str(manifest_path),
            "catalog_offset_bytes": 0,
            "catalog_size_bytes": 1,
            "catalog_sha256": "0" * 64,
        }
        for row_total in block.n_ladder
    ]
    corpus_record = {
        "manifest": {"manifest_path": str(manifest_path.resolve())},
        "dagzoo_provenance_summary": {
            "target_derivation": "tabiclv2_latent_node",
            "filter_policy": "accepted_only",
            "accepted_datasets": 128,
            "curated_accepted_datasets": 128,
            "rejected_datasets": 53,
            "acceptance_rate": 128.0 / 181.0,
            "target_relevant_feature_count_range": {"min": 2, "max": 6},
            "target_relevant_feature_fraction_range": {
                "min": 2.0 / 6.0,
                "max": 1.0,
            },
        },
        "dagzoo_provenance": {
            "invocations": [
                {
                    "invocation_id": f"r{row_total:04d}_canary",
                    "num_datasets": 32,
                    "filter": {
                        "filter_policy": "accepted_only",
                        "accepted_datasets": 40,
                        "curated_accepted_datasets": 32,
                        "filter_manifest_path": str(filter_manifest_path.resolve()),
                        "filter_summary_path": str(filter_summary_path.resolve()),
                        "curated_dir": str(curated_dir.resolve()),
                    },
                }
                for row_total in block.n_ladder
            ]
        },
    }

    monkeypatch.setattr(
        pilot_module,
        "_classification_manifest_records",
        lambda path: sample_records,
    )
    monkeypatch.setattr(
        pilot_module,
        "load_manifest_record_catalog",
        lambda path, *, record: _latent_target_catalog_record(
            n_features=int(record["n_features"]),
            relevant_feature_count=int(record["n_features"]),
        ),
    )

    payload = pilot_module.inspect_corpus_latent_target_contract(
        block=block,
        corpus_record=corpus_record,
    )

    assert payload["present"] is True
    assert payload["missing_reasons"] == []
    assert payload["sample_records"][0]["target_derivation"] == "tabiclv2_latent_node"
    assert payload["sample_records"][0]["target_relevant_feature_count"] == 6
    assert payload["sample_records"][0]["target_relevant_feature_fraction"] == pytest.approx(1.0)
    assert "target_to_node" not in payload["sample_records"][0]


def test_score_task_local_predictors_beats_chance_on_easy_canary() -> None:
    results, errors = pilot_module.score_task_local_predictors(
        _task_batch(),
        predictors=("chance", "logistic_regression"),
    )

    assert errors == {}
    assert results["chance"]["label_target_log_loss_per_test_cell"] == pytest.approx(math.log(2.0), rel=1.0e-4)
    assert (
        float(results["logistic_regression"]["label_target_log_loss_per_test_cell"])
        < float(results["chance"]["label_target_log_loss_per_test_cell"])
    )


def test_select_provisional_interpretation_marks_training_regime_problem() -> None:
    interpretation = pilot_module.select_provisional_interpretation(
        decision_buckets={
            "generator_problem": "generator",
            "training_regime_problem": "training",
            "inconclusive": "inconclusive",
        },
        latent_target_contract={
            "latent_target_canary_curated_v3": {"required": True, "present": True},
            "production_control_curated_v5": {"required": True, "present": True},
        },
        canary_summary=_healthy_canary_summary(),
        production_control_summary={
            "status": "completed",
            "run_inspect": {
                "health": {
                    "verdict": "fail",
                    "metrics": {
                        "initial_train_loss": 1.0,
                        "final_train_loss": 1.1,
                    },
                }
            },
        },
    )

    assert interpretation["bucket"] == "training_regime_problem"
    assert interpretation["definition"] == "training"


def test_run_adequacy_pilot_writes_summary_with_monkeypatched_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_batch: dict[str, object] = {}

    def _fake_materialize_corpus_refs_batch(
        *,
        corpus_refs: list[str],
        dagzoo_root: Path,
        force: bool = False,
        materialize_processes: int | None = None,
        materialize_worker_threads: int | None = None,
        prioritized_recipe_ids: list[str] | tuple[str, ...] = (),
        on_corpus_materialized=None,
        repo_root: Path | None = None,
        sweep_id: str | None = None,
        sweeps_root: Path | None = None,
    ) -> list[dict[str, object]]:
        del dagzoo_root, force, repo_root, sweep_id, sweeps_root
        captured_batch.update(
            {
                "corpus_refs": list(corpus_refs),
                "materialize_processes": materialize_processes,
                "materialize_worker_threads": materialize_worker_threads,
                "prioritized_recipe_ids": list(prioritized_recipe_ids),
            }
        )
        records = [
            _fake_corpus_record(tmp_path, corpus_ref=corpus_ref) for corpus_ref in corpus_refs
        ]
        for record in records:
            assert on_corpus_materialized is not None
            on_corpus_materialized(record)
        return records

    def _fake_contract_inspection(
        *,
        block: object,
        corpus_record: dict[str, object],
    ) -> dict[str, object]:
        del block, corpus_record
        return {
            "required": True,
            "present": True,
            "missing_reasons": [],
            "sample_records": [],
        }

    def _fake_run_production_control_pilot(
        *,
        corpus_ref: str,
        pilot_root: Path,
        device: str,
        force: bool,
    ) -> dict[str, object]:
        del corpus_ref, pilot_root, device, force
        return {
            "status": "completed",
            "run_dir": str((tmp_path / "pilot" / "train").resolve()),
            "metrics": {
                "best_val_loss": None,
                "final_val_loss": None,
            },
            "run_inspect": {
                "health": {
                    "verdict": "ok",
                    "summary": "stable",
                    "metrics": {
                        "initial_train_loss": 1.0,
                        "final_train_loss": 0.4,
                    },
                }
            },
        }

    monkeypatch.setattr(
        pilot_module,
        "materialize_corpus_refs_batch",
        _fake_materialize_corpus_refs_batch,
    )
    monkeypatch.setattr(
        pilot_module,
        "inspect_corpus_latent_target_contract",
        _fake_contract_inspection,
    )
    monkeypatch.setattr(
        pilot_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )
    monkeypatch.setattr(
        pilot_module,
        "run_production_control_pilot",
        _fake_run_production_control_pilot,
    )

    out_root = tmp_path / "adequacy"
    summary = pilot_module.run_adequacy_pilot(
        adequacy_id="tf_rd_010_synthetic_adequacy_v3",
        dagzoo_root=tmp_path / "dagzoo",
        out_root=out_root,
        materialize_processes=3,
        materialize_worker_threads=2,
    )

    summary_json_path = out_root / "summary.json"
    summary_md_path = out_root / "summary.md"
    assert captured_batch == {
        "corpus_refs": [
            "tf_rd_010_latent_target_canary_curated_v3",
            "tf_rd_010_dagzoo_medium_control_curated_v5",
        ],
        "materialize_processes": 3,
        "materialize_worker_threads": 2,
        "prioritized_recipe_ids": ["tf_rd_010_latent_target_canary_curated_v3"],
    }
    assert summary["provisional_interpretation"]["bucket"] == "inconclusive"
    assert summary_json_path.exists()
    assert summary_md_path.exists()
    persisted = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert persisted["adequacy_id"] == "tf_rd_010_synthetic_adequacy_v3"
    assert "Production Control Pilot" in summary_md_path.read_text(encoding="utf-8")


def test_run_adequacy_pilot_fails_fast_and_writes_blocking_summary_when_contract_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_materialize_corpus_refs_batch(
        *,
        corpus_refs: list[str],
        dagzoo_root: Path,
        force: bool = False,
        materialize_processes: int | None = None,
        materialize_worker_threads: int | None = None,
        prioritized_recipe_ids: list[str] | tuple[str, ...] = (),
        on_corpus_materialized=None,
        repo_root: Path | None = None,
        sweep_id: str | None = None,
        sweeps_root: Path | None = None,
    ) -> list[dict[str, object]]:
        del (
            dagzoo_root,
            force,
            materialize_processes,
            materialize_worker_threads,
            prioritized_recipe_ids,
            repo_root,
            sweep_id,
            sweeps_root,
        )
        records = [
            _fake_corpus_record(tmp_path, corpus_ref=corpus_ref) for corpus_ref in corpus_refs
        ]
        for record in records:
            assert on_corpus_materialized is not None
            on_corpus_materialized(record)
        return records

    monkeypatch.setattr(
        pilot_module,
        "materialize_corpus_refs_batch",
        _fake_materialize_corpus_refs_batch,
    )
    monkeypatch.setattr(
        pilot_module,
        "inspect_corpus_latent_target_contract",
        lambda *, block, corpus_record: {
            "required": True,
            "present": False,
            "missing_reasons": [f"{block.block_id} missing latent-target lineage"],
            "sample_records": [],
        },
    )
    monkeypatch.setattr(
        pilot_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )

    out_root = tmp_path / "adequacy"
    with pytest.raises(RuntimeError, match="latent-target contract validation failed"):
        _ = pilot_module.run_adequacy_pilot(
            adequacy_id="tf_rd_010_synthetic_adequacy_v3",
            dagzoo_root=tmp_path / "dagzoo",
            out_root=out_root,
        )

    summary_json_path = out_root / "summary.json"
    assert summary_json_path.exists()
    persisted = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "blocked"
    assert persisted["provisional_interpretation"]["bucket"] == "generator_problem"
