from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tab_foundry.research.adequacy.canary as canary_module
import tab_foundry.research.adequacy.contract as contract_module
import tab_foundry.research.adequacy.pilot as pilot_module
import tab_foundry.research.adequacy.production_control as production_control_module
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


def _write_existing_production_control_run(pilot_root: Path) -> Path:
    run_dir = pilot_root / "production_control_curated_v5" / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    telemetry_payload = {
        "artifacts": {
            "best_checkpoint": str((checkpoints_dir / "best.pt").resolve()),
            "latest_checkpoint": str((checkpoints_dir / "latest_prior_dump.pt").resolve()),
            "train_history_jsonl": str((run_dir / "train_history.jsonl").resolve()),
            "telemetry_json": str((run_dir / "telemetry.json").resolve()),
            "training_surface_record_json": str(
                (run_dir / "training_surface_record.json").resolve()
            ),
        },
        "wandb": {
            "entity": "test-entity",
            "mode": "online",
            "project": "tab-foundry",
            "run_id": "test-run-id",
            "run_name": "tf_rd_010_synthetic_adequacy_v3-production-control-curated-v5-direct-manifest",
        },
    }
    training_surface_payload = {
        "data": {
            "surface_label": "tf_rd_010_dagzoo_medium_control_curated_v5",
            "source": "manifest",
            "corpus_ref": None,
            "manifest": {
                "manifest_path": str((pilot_root.parent / "direct_training" / "manifest.parquet").resolve())
            },
        },
        "runtime": {
            "mixed_precision": "no",
            "num_workers": 0,
            "grad_accum_steps": 4,
            "grad_clip": 0.0,
            "max_steps": 2500,
            "eval_every": 25,
            "checkpoint_every": 25,
            "val_batches": 0,
            "seed": 1,
        },
        "training": {
            "task_batch_size": 16,
            "optimizer_name": "schedulefree_adamw",
            "optimizer_min_lr": 1.0e-5,
            "schedule_stages": [
                {
                    "name": "prior_dump",
                    "steps": 2500,
                    "lr_max": 1.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.10,
                }
            ],
        },
    }
    (pilot_root.parent / "direct_training").mkdir(parents=True, exist_ok=True)
    (pilot_root.parent / "direct_training" / "manifest.parquet").write_bytes(b"manifest")
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "training_surface_record.json").write_text(
        json.dumps(training_surface_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train_history.jsonl").write_text(
        json.dumps(
            {
                "elapsed_seconds": 13.0,
                "train_elapsed_seconds": 12.5,
                "step": 2500,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


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
        contract_module,
        "_classification_manifest_records",
        lambda path: sample_records,
    )
    monkeypatch.setattr(
        contract_module,
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


def test_inspect_corpus_latent_target_contract_fast_skips_catalog_reads_for_production_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = load_synthetic_adequacy_spec("tf_rd_010_synthetic_adequacy_v3")
    block = next(
        candidate
        for candidate in spec.blocks
        if candidate.block_id == "production_control_curated_v5"
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
        },
        "dagzoo_provenance": {
            "invocations": [
                {
                    "invocation_id": f"r{row_total:04d}_control",
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
        contract_module,
        "_classification_manifest_records",
        lambda path: sample_records,
    )
    monkeypatch.setattr(
        contract_module,
        "load_manifest_record_catalog",
        lambda path, *, record: pytest.fail("fast production contract check should skip catalog loads"),
    )
    monkeypatch.setattr(
        contract_module,
        "_load_manifest_contract_stats",
        lambda *, manifest_path, corpus_record: contract_module._manifest_contract_stats_from_records(
            sample_records
        ),
    )

    payload = pilot_module.inspect_corpus_latent_target_contract(
        block=block,
        corpus_record=corpus_record,
        mode="fast",
    )

    assert payload["present"] is True
    assert payload["contract_check_mode"] == "fast"
    assert payload["catalog_validation_mode"] == "skipped"
    assert payload["sample_records"] == []


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


def test_build_production_control_config_keeps_wandb_enabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot" / "train"

    cfg = pilot_module.build_production_control_config(
        requested_corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5",
        corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5/materialized",
        manifest_path=None,
        materialization_state="finalized",
        run_dir=run_dir,
        device="cpu",
    )

    assert bool(cfg.logging.use_wandb) is True
    assert str(cfg.logging.history_jsonl_path) == str((run_dir / "train_history.jsonl").resolve())
    assert str(cfg.data.requested_corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert str(cfg.data.materialization_state) == "finalized"


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
        mode: str,
    ) -> dict[str, object]:
        del block, corpus_record
        assert mode == "fast"
        return {
            "required": True,
            "present": True,
            "missing_reasons": [],
            "sample_records": [],
        }

    def _fake_run_production_control_pilot(
        *,
        requested_corpus_ref: str,
        corpus_ref: str | None,
        manifest_path: Path | None,
        materialization_state: str,
        pilot_root: Path,
        device: str,
        force: bool,
    ) -> dict[str, object]:
        assert requested_corpus_ref == "tf_rd_010_dagzoo_medium_control_curated_v5"
        assert corpus_ref == "tf_rd_010_dagzoo_medium_control_curated_v5/materialized"
        assert manifest_path is not None
        assert materialization_state == "finalized"
        del pilot_root, device, force
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
        contract_module,
        "inspect_corpus_latent_target_contract",
        _fake_contract_inspection,
    )
    monkeypatch.setattr(
        canary_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )
    monkeypatch.setattr(
        production_control_module,
        "load_corpus_record",
        lambda corpus_ref, *, repo_root=None: _fake_corpus_record(tmp_path, corpus_ref=corpus_ref),
    )
    monkeypatch.setattr(
        production_control_module,
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
        contract_check="fast",
    )

    summary_json_path = out_root / "summary.json"
    summary_md_path = out_root / "summary.md"
    assert captured_batch == {
        "corpus_refs": [
            "tf_rd_010_latent_target_canary_curated_v3",
        ],
        "materialize_processes": 3,
        "materialize_worker_threads": 2,
        "prioritized_recipe_ids": ["tf_rd_010_latent_target_canary_curated_v3"],
    }
    assert summary["contract_check"]["mode"] == "fast"
    assert summary["provisional_interpretation"]["bucket"] == "inconclusive"
    assert summary_json_path.exists()
    assert summary_md_path.exists()
    persisted = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert persisted["adequacy_id"] == "tf_rd_010_synthetic_adequacy_v3"
    assert persisted["contract_check"]["mode"] == "fast"
    assert "Contract check: `fast`" in summary_md_path.read_text(encoding="utf-8")
    assert "Production Control Pilot" in summary_md_path.read_text(encoding="utf-8")
    assert (
        persisted["materialized_corpora"]["production_control_curated_v5"]["materialization_state"]
        == "finalized"
    )


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
        contract_module,
        "inspect_corpus_latent_target_contract",
        lambda *, block, corpus_record, mode: {
            "required": True,
            "present": False,
            "missing_reasons": [f"{block.block_id} missing latent-target lineage"],
            "sample_records": [],
        },
    )
    monkeypatch.setattr(
        canary_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )
    monkeypatch.setattr(
        production_control_module,
        "load_corpus_record",
        lambda corpus_ref, *, repo_root=None: _fake_corpus_record(tmp_path, corpus_ref=corpus_ref),
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
    assert persisted["contract_check"]["mode"] == "fast"


def test_finalize_adequacy_pilot_writes_summary_from_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "adequacy"
    run_dir = _write_existing_production_control_run(out_root)
    canary_record = _fake_corpus_record(
        tmp_path,
        corpus_ref="tf_rd_010_latent_target_canary_curated_v3/materialized",
    )
    production_record = _fake_corpus_record(
        tmp_path,
        corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5/materialized",
    )
    stale_summary_path = out_root / "summary.json"
    stale_summary_path.parent.mkdir(parents=True, exist_ok=True)
    stale_summary_path.write_text('{"status":"blocked"}\n', encoding="utf-8")
    (out_root / "summary.md").write_text("stale summary\n", encoding="utf-8")

    monkeypatch.setattr(
        pilot_module,
        "materialize_corpus_refs_batch",
        lambda **kwargs: pytest.fail("finalize should not materialize corpora"),
    )
    monkeypatch.setattr(
        production_control_module,
        "train",
        lambda cfg: pytest.fail("finalize should not retrain"),
    )
    monkeypatch.setattr(
        production_control_module,
        "load_corpus_record",
        lambda corpus_ref, *, repo_root=None: canary_record,
    )
    monkeypatch.setattr(
        production_control_module,
        "_resolve_production_control_corpus",
        lambda **kwargs: {
            "corpus_record": production_record,
            "materialization_state": "finalized",
        },
    )
    monkeypatch.setattr(
        contract_module,
        "inspect_corpus_latent_target_contract",
        lambda *, block, corpus_record, mode: {
            "required": True,
            "present": True,
            "missing_reasons": [],
            "sample_records": [],
            "filter_provenance": {
                "filter_policy": "accepted_only",
                "target_accepted_datasets": 128,
                "accepted_datasets": 128,
                "curated_accepted_datasets": 128,
                "acceptance_rate": 1.0,
            },
        },
    )
    monkeypatch.setattr(
        canary_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )
    monkeypatch.setattr(
        production_control_module,
        "run_inspect",
        lambda run_dir_arg: {
            "surface_labels": {
                "data": "tf_rd_010_dagzoo_medium_control_curated_v5",
                "model": "tabfoundry_sandwich",
                "preprocessing": "runtime_default",
                "training": "prior_cosine_warmup",
            },
            "health": {
                "verdict": "ok",
                "summary": "stable",
                "metrics": {
                    "initial_train_loss": 2.0,
                    "final_train_loss": 1.4,
                },
            },
            "artifacts": {
                "train_history_jsonl": str((run_dir / "train_history.jsonl").resolve()),
                "telemetry_json": str((run_dir / "telemetry.json").resolve()),
                "training_surface_record_json": str(
                    (run_dir / "training_surface_record.json").resolve()
                ),
                "best_checkpoint_pt": str((run_dir / "checkpoints" / "best.pt").resolve()),
                "latest_checkpoint_pt": str(
                    (run_dir / "checkpoints" / "latest_prior_dump.pt").resolve()
                ),
            },
        },
    )

    summary = pilot_module.finalize_adequacy_pilot(
        adequacy_id="tf_rd_010_synthetic_adequacy_v3",
        dagzoo_root=tmp_path / "dagzoo",
        out_root=out_root,
        contract_check="fast",
    )

    assert summary["status"] == "completed"
    assert summary["contract_check"]["mode"] == "fast"
    assert summary["production_control_pilot"]["status"] == "completed"
    assert summary["production_control_pilot"]["config_excerpt"]["materialization_state"] == "staged"
    assert summary["production_control_pilot"]["metrics"]["train_elapsed_seconds"] == pytest.approx(12.5)
    assert summary["production_control_pilot"]["metrics"]["wall_elapsed_seconds"] == pytest.approx(13.0)
    persisted = json.loads(stale_summary_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    assert persisted["production_control_pilot"] is not None
    assert "Production Control Pilot" in (out_root / "summary.md").read_text(encoding="utf-8")


def test_finalize_adequacy_pilot_leaves_existing_summary_untouched_when_run_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "adequacy"
    stale_summary_json = out_root / "summary.json"
    stale_summary_md = out_root / "summary.md"
    stale_summary_json.parent.mkdir(parents=True, exist_ok=True)
    stale_summary_json.write_text('{"status":"blocked","sentinel":true}\n', encoding="utf-8")
    stale_summary_md.write_text("sentinel summary\n", encoding="utf-8")
    canary_record = _fake_corpus_record(
        tmp_path,
        corpus_ref="tf_rd_010_latent_target_canary_curated_v3/materialized",
    )
    production_record = _fake_corpus_record(
        tmp_path,
        corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5/materialized",
    )
    (out_root / "production_control_curated_v5" / "train").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        production_control_module,
        "load_corpus_record",
        lambda corpus_ref, *, repo_root=None: canary_record,
    )
    monkeypatch.setattr(
        production_control_module,
        "_resolve_production_control_corpus",
        lambda **kwargs: {
            "corpus_record": production_record,
            "materialization_state": "finalized",
        },
    )
    monkeypatch.setattr(
        contract_module,
        "inspect_corpus_latent_target_contract",
        lambda *, block, corpus_record, mode: {
            "required": True,
            "present": True,
            "missing_reasons": [],
            "sample_records": [],
        },
    )
    monkeypatch.setattr(
        canary_module,
        "score_canary_block",
        lambda block, *, corpus_record: _healthy_canary_summary(),
    )

    with pytest.raises(RuntimeError, match="production control telemetry is missing"):
        _ = pilot_module.finalize_adequacy_pilot(
            adequacy_id="tf_rd_010_synthetic_adequacy_v3",
            dagzoo_root=tmp_path / "dagzoo",
            out_root=out_root,
        )

    assert json.loads(stale_summary_json.read_text(encoding="utf-8")) == {
        "status": "blocked",
        "sentinel": True,
    }
    assert stale_summary_md.read_text(encoding="utf-8") == "sentinel summary\n"


def test_resolve_production_control_corpus_prefers_existing_finalized_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    finalized_record = _fake_corpus_record(
        tmp_path,
        corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5/materialized",
    )

    monkeypatch.setattr(
        production_control_module,
        "load_corpus_record",
        lambda corpus_ref, *, repo_root=None: finalized_record,
    )
    monkeypatch.setattr(
        production_control_module,
        "_staged_direct_manifest_record",
        lambda **kwargs: pytest.fail("existing finalized corpora should be preferred"),
    )

    resolved = production_control_module._resolve_production_control_corpus(
        requested_corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5",
        pilot_root=tmp_path / "adequacy",
        dagzoo_root=tmp_path / "dagzoo",
        force=False,
        repo_root=tmp_path / "repo",
    )

    assert resolved["materialization_state"] == "finalized"
    assert resolved["corpus_record"] == finalized_record


def test_resolve_production_control_corpus_falls_back_to_staged_direct_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged_record = {
        "recipe_id": "tf_rd_010_dagzoo_medium_control_curated_v5",
        "corpus_id": None,
        "corpus_ref": None,
        "surface_label": "production_control_curated_v5",
        "corpus_record_path": None,
        "manifest": {
            "manifest_path": str((tmp_path / "adequacy" / "direct_training" / "manifest.parquet").resolve())
        },
        "dagzoo_provenance": {"invocations": []},
        "dagzoo_provenance_summary": {"target_derivation": "tabiclv2_latent_node"},
    }

    def _missing_record(corpus_ref: str, *, repo_root: Path | None = None) -> dict[str, object]:
        raise RuntimeError(f"missing corpus {corpus_ref}")

    monkeypatch.setattr(production_control_module, "load_corpus_record", _missing_record)
    monkeypatch.setattr(
        production_control_module,
        "_staged_direct_manifest_record",
        lambda **kwargs: staged_record,
    )

    resolved = production_control_module._resolve_production_control_corpus(
        requested_corpus_ref="tf_rd_010_dagzoo_medium_control_curated_v5",
        pilot_root=tmp_path / "adequacy",
        dagzoo_root=tmp_path / "dagzoo",
        force=False,
        repo_root=tmp_path / "repo",
    )

    assert resolved["materialization_state"] == "staged"
    assert resolved["corpus_record"] == staged_record
