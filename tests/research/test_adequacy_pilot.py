from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tab_foundry.research.adequacy.pilot as pilot_module
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
        _latent_target_metadata(n_features=2),
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
        {"prior": {"target_derivation": "wrong"}},
        n_features=2,
    )

    assert payload["present"] is False
    assert any(
        "metadata.lineage.assignments is missing" in reason for reason in payload["missing_reasons"]
    )


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
            "latent_target_canary_easy_v2": {"required": True, "present": True},
            "production_control_v4": {"required": True, "present": True},
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
    def _fake_materialize_corpus_ref(
        *,
        corpus_ref: str,
        dagzoo_root: Path,
        force: bool = False,
        repo_root: Path | None = None,
        sweep_id: str | None = None,
        sweeps_root: Path | None = None,
    ) -> dict[str, object]:
        del dagzoo_root, force, repo_root, sweep_id, sweeps_root
        return _fake_corpus_record(tmp_path, corpus_ref=corpus_ref)

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

    monkeypatch.setattr(pilot_module, "materialize_corpus_ref", _fake_materialize_corpus_ref)
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
        adequacy_id="tf_rd_010_synthetic_adequacy_v2",
        dagzoo_root=tmp_path / "dagzoo",
        out_root=out_root,
    )

    summary_json_path = out_root / "summary.json"
    summary_md_path = out_root / "summary.md"
    assert summary["provisional_interpretation"]["bucket"] == "inconclusive"
    assert summary_json_path.exists()
    assert summary_md_path.exists()
    persisted = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert persisted["adequacy_id"] == "tf_rd_010_synthetic_adequacy_v2"
    assert "Production Control Pilot" in summary_md_path.read_text(encoding="utf-8")


def test_run_adequacy_pilot_fails_fast_and_writes_blocking_summary_when_contract_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_materialize_corpus_ref(
        *,
        corpus_ref: str,
        dagzoo_root: Path,
        force: bool = False,
        repo_root: Path | None = None,
        sweep_id: str | None = None,
        sweeps_root: Path | None = None,
    ) -> dict[str, object]:
        del dagzoo_root, force, repo_root, sweep_id, sweeps_root
        return _fake_corpus_record(tmp_path, corpus_ref=corpus_ref)

    monkeypatch.setattr(pilot_module, "materialize_corpus_ref", _fake_materialize_corpus_ref)
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
            adequacy_id="tf_rd_010_synthetic_adequacy_v2",
            dagzoo_root=tmp_path / "dagzoo",
            out_root=out_root,
        )

    summary_json_path = out_root / "summary.json"
    assert summary_json_path.exists()
    persisted = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "blocked"
    assert persisted["provisional_interpretation"]["bucket"] == "generator_problem"
