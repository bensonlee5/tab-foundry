from __future__ import annotations

import json
from pathlib import Path

from tab_foundry.training.health import run_inspect
from tab_foundry.training.instability import build_training_telemetry


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _training_surface_record() -> dict[str, object]:
    return {
        "labels": {
            "model": "row_cls_pool_test",
            "data": "anchor_manifest_default",
            "preprocessing": "runtime_default",
            "training": "training_default",
        },
        "model": {
            "arch": "tabfoundry_staged",
            "stage": "row_cls_pool",
            "stage_label": "row_cls_pool_test",
        },
        "data": {
            "surface_label": "anchor_manifest_default",
        },
        "preprocessing": {
            "surface_label": "runtime_default",
        },
        "training": {
            "surface_label": "training_default",
        },
    }


def _history_records() -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "train_loss": 1.0 - (0.01 * float(step)),
            "train_loss_delta": None if step == 1 else -0.01,
        }
        for step in range(1, 21)
    ]


def _gradient_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(1, 21):
        block_value = 4.0 + (0.001 * float(step))
        records.append(
            {
                "step": step,
                "global_grad_norm": 0.4 + (0.01 * float(step)),
                "grad_clip_triggered": False,
                "activation_norms": {
                    "post_transformer_block_8": block_value,
                    "post_transformer_block_9": block_value + 0.1,
                    "post_transformer_block_10": block_value + 0.2,
                    "post_transformer_block_11": block_value + 0.3,
                },
            }
        )
    return records


def test_run_inspect_reports_health_surface_labels_and_benchmark_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "row_one_run" / "train"
    benchmark_dir = run_dir.parent / "benchmark"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_records = _history_records()
    gradient_records = _gradient_records()
    _write_jsonl(run_dir / "train_history.jsonl", history_records)
    _write_jsonl(run_dir / "gradient_history.jsonl", gradient_records)
    training_surface_record = _training_surface_record()
    (run_dir / "training_surface_record.json").write_text(
        json.dumps(training_surface_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    telemetry = build_training_telemetry(
        run_dir=run_dir,
        success=True,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=history_records,
        gradient_records=gradient_records,
        training_surface_record=training_surface_record,
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "tab_foundry": {
                    "benchmark_profile": "row_cls_pool_test",
                    "model_arch": "tabfoundry_staged",
                    "model_stage": "row_cls_pool",
                    "run_dir": str(run_dir),
                    "best_roc_auc": 0.71,
                    "final_roc_auc": 0.70,
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (benchmark_dir / "benchmark_run_record.json").write_text(
        json.dumps(
            {
                "run_id": "row_one_run",
                "track": "test_track",
                "experiment": "cls_smoke",
                "config_profile": "cls_smoke",
                "surface_labels": {
                    "model": "row_cls_pool_test",
                    "data": "anchor_manifest_default",
                    "preprocessing": "runtime_default",
                },
                "tab_foundry_metrics": {"best_roc_auc": 0.71},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_inspect(run_dir)

    assert payload["surface_labels"]["model"] == "row_cls_pool_test"
    assert payload["health"]["verdict"] == "ok"
    assert payload["comparison_summary"]["best_roc_auc"] == 0.71
    assert payload["benchmark_run_record"]["run_id"] == "row_one_run"
    assert payload["artifacts"]["comparison_summary_json"]["exists"] is True


def test_run_inspect_keeps_partial_runs_inspectable_when_health_is_unavailable(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial_run" / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_surface_record.json").write_text(
        json.dumps(_training_surface_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = run_inspect(run_dir)

    assert payload["surface_labels"]["model"] == "row_cls_pool_test"
    assert payload["health"] is None
    assert "health-check requires telemetry.json" in payload["health_error"]


def test_run_inspect_falls_back_to_benchmark_training_surface_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "benchmarked_run" / "train"
    benchmark_dir = run_dir.parent / "benchmark"
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_surface_record = _training_surface_record()
    benchmark_surface_record["labels"] = {
        "model": "benchmark_row_cls_pool_test",
        "data": "benchmark_anchor_manifest_default",
        "preprocessing": "benchmark_runtime_default",
        "training": "benchmark_training_default",
    }
    benchmark_surface_record_path = benchmark_dir / "training_surface_record.json"
    benchmark_surface_record_path.write_text(
        json.dumps(benchmark_surface_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (benchmark_dir / "benchmark_run_record.json").write_text(
        json.dumps(
            {
                "run_id": "benchmarked_run",
                "surface_labels": {
                    "model": "benchmark_row_cls_pool_test",
                },
                "artifacts": {
                    "training_surface_record_path": str(benchmark_surface_record_path),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_inspect(run_dir)

    assert payload["surface_labels"]["model"] == "benchmark_row_cls_pool_test"
    assert payload["training_surface_record"]["labels"]["training"] == "benchmark_training_default"
    assert payload["artifacts"]["training_surface_record_json"]["exists"] is True
    assert payload["artifacts"]["training_surface_record_json"]["path"] == str(
        benchmark_surface_record_path.resolve()
    )
