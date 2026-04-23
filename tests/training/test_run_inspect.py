from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import pytest

import tab_foundry.cli as cli_module
import tab_foundry.training.posthoc_accounting as posthoc_accounting
from tab_foundry.cli.dev import render_run_inspect_text
from tab_foundry.training.health import health_check, run_inspect
from tab_foundry.training.instability import build_training_telemetry


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _training_surface_record(
    *,
    arch: str = "tabfoundry_staged",
    model_label: str = "row_cls_pool_test",
    stage: str | None = "row_cls_pool",
    loss_surface: str = "classification",
) -> dict[str, object]:
    model: dict[str, object] = {
        "arch": arch,
        "stage_label": model_label,
    }
    if stage is not None:
        model["stage"] = stage
    return {
        "labels": {
            "model": model_label,
            "data": "anchor_manifest_default",
            "preprocessing": "runtime_default",
            "training": "training_default",
        },
        "model": model,
        "data": {
            "surface_label": "anchor_manifest_default",
        },
        "preprocessing": {
            "surface_label": "runtime_default",
        },
        "training": {
            "loss_surface": loss_surface,
            "surface_label": "training_default",
        },
        "runtime": {
            "mixed_precision": "bf16",
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


def _runtime_summary() -> dict[str, object]:
    return {
        "peak_vram_allocated": 1024,
        "peak_vram_reserved": 2048,
        "peak_vram_allocated_fraction": 1024 / float(80 * 1024**3),
        "peak_vram_reserved_fraction": 2048 / float(80 * 1024**3),
        "throughput_examples_per_second": 12.5,
        "throughput_tokens_per_second": 6400.0,
        "non_train_overhead_seconds": 0.8,
        "non_train_overhead_fraction": 0.2,
    }


def _hardware_summary() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "raw_device_name": "NVIDIA A100-SXM4-80GB",
        "gpu_class": "a100",
        "total_device_vram_bytes": 80 * 1024**3,
        "vram_class_gb": 80,
        "hardware_profile_id": "a100_80gb",
    }


def _regime_budget() -> dict[str, object]:
    return {
        "tokens_per_step": 512.0,
        "tokens_seen": 38400,
        "token_budget": 38400,
        "unique_task_budget": 96,
        "objective_metric": "final_log_loss_at_matched_regime_budget",
        "curriculum_id": "dagzoo_shape_aware_multi_invocation",
    }


def _utilization_summary() -> dict[str, object]:
    return {
        "peak_vram_allocated_fraction": 1024 / float(80 * 1024**3),
        "peak_vram_reserved_fraction": 2048 / float(80 * 1024**3),
        "non_train_overhead_fraction": 0.2,
        "achieved_train_tflops_per_second": 4.0,
        "theoretical_peak_tflops_per_second": 312.0,
        "compute_utilization_fraction": 4.0 / 312.0,
        "theoretical_hbm_bandwidth_gbps": 2039.0,
        "roofline_knee_flops_per_byte": 153.0161844031388,
        "peak_compute_basis": "tensorcore_bf16_dense",
    }


def _gradient_records(*, activation_shape: str = "staged") -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(1, 21):
        block_value = 4.0 + (0.001 * float(step))
        activation_norms: dict[str, float] = {}
        if activation_shape == "sandwich":
            activation_norms["post_stage_0_self"] = block_value
            activation_norms["post_stage_1_self"] = block_value + 0.1
        else:
            activation_norms["post_transformer_block_8"] = block_value
            activation_norms["post_transformer_block_9"] = block_value + 0.1
            activation_norms["post_transformer_block_10"] = block_value + 0.2
            activation_norms["post_transformer_block_11"] = block_value + 0.3
        records.append(
            {
                "step": step,
                "global_grad_norm": 0.4 + (0.01 * float(step)),
                "grad_clip_triggered": False,
                "activation_norms": activation_norms,
            }
        )
    return records


def _timed_gradient_records() -> list[dict[str, object]]:
    records = _gradient_records()
    for record in records:
        record["timing_seconds"] = {
            "data_wait": 0.2,
            "batch_diagnostics": 0.1,
            "h2d_transfer": 0.01,
            "forward_backward": 0.5,
            "activation_trace": 0.0,
            "grad_diagnostics": 0.02,
            "optimizer": 0.05,
            "checkpoint": 0.0,
        }
    return records


def _warmup_sensitive_gradient_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(1, 41):
        if step <= 10:
            block_value = 4.0 + (0.5 * float(step))
        else:
            block_value = 9.0
        records.append(
            {
                "step": step,
                "global_grad_norm": 0.5 + (0.01 * float(step)),
                "grad_clip_triggered": False,
                "module_grad_norms": {
                    "feature_encoder": 1.0,
                    "direct_head": 4.0,
                },
                "activation_norms": {
                    "post_transformer_block_8": block_value,
                    "post_transformer_block_9": block_value + 0.2,
                    "post_transformer_block_10": block_value + 0.4,
                    "post_transformer_block_11": block_value + 0.6,
                },
            }
        )
    return records


def _warmup_sensitive_training_surface_record() -> dict[str, object]:
    payload = _training_surface_record()
    payload["training"] = {
        "loss_surface": "classification",
        "surface_label": "benchmark_training_default",
        "schedule_stages": [
            {
                "name": "stage1",
                "steps": 40,
                "lr_max": 1.0e-3,
                "warmup_ratio": 0.25,
            }
        ],
    }
    return payload


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
        runtime_summary=_runtime_summary(),
        hardware_summary=_hardware_summary(),
        regime_budget=_regime_budget(),
        training_surface_record=training_surface_record,
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest_stage1.pt").write_bytes(b"latest")
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
                "runtime_summary": _runtime_summary(),
                "utilization_summary": _utilization_summary(),
                "hardware_summary": _hardware_summary(),
                "regime_budget": _regime_budget(),
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
    assert payload["runtime_summary"]["peak_vram_reserved"] == 2048
    assert payload["utilization_summary"]["compute_utilization_fraction"] == pytest.approx(4.0 / 312.0)
    assert payload["hardware_summary"]["hardware_profile_id"] == "a100_80gb"
    assert payload["regime_budget"]["token_budget"] == 38400
    assert payload["benchmark_run_record"]["runtime_summary"]["throughput_tokens_per_second"] == 6400.0
    assert payload["benchmark_run_record"]["utilization_summary"]["peak_compute_basis"] == "tensorcore_bf16_dense"
    assert payload["benchmark_run_record"]["hardware_summary"]["gpu_class"] == "a100"
    assert payload["artifacts"]["comparison_summary_json"]["exists"] is True
    assert payload["artifacts"]["latest_checkpoint_pt"]["exists"] is True
    assert payload["artifacts"]["latest_checkpoint_pt"]["path"].endswith("latest_stage1.pt")
    rendered = render_run_inspect_text(payload)
    assert "runtime_summary=" in rendered
    assert "\"throughput_tokens_per_second\": 6400.0" in rendered
    assert "utilization_summary=" in rendered
    assert "\"compute_utilization_fraction\": 0.01282051282051282" in rendered
    assert "hardware_summary=" in rendered
    assert "\"hardware_profile_id\": \"a100_80gb\"" in rendered
    assert "regime_budget=" in rendered
    assert "\"token_budget\": 38400" in rendered


def test_run_inspect_derives_posthoc_compute_and_bottleneck_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "profiled_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_records = _history_records()
    gradient_records = _timed_gradient_records()
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
        runtime_summary=_runtime_summary(),
        hardware_summary=_hardware_summary(),
        regime_budget=_regime_budget(),
        training_surface_record=training_surface_record,
    )
    telemetry_path = run_dir / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    original_telemetry = telemetry_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        posthoc_accounting,
        "derive_compute_accounting_for_run",
        lambda *_args, **_kwargs: {"train_flops_per_token": 625_000_000.0},
    )

    plain_payload = run_inspect(run_dir)
    payload = run_inspect(run_dir, derive_compute_accounting=True)

    assert plain_payload["compute_accounting"] is None
    assert plain_payload["utilization_summary"]["achieved_train_tflops_per_second"] is None
    assert payload["compute_accounting"] == {"train_flops_per_token": 625_000_000.0}
    assert payload["utilization_summary"]["achieved_train_tflops_per_second"] == pytest.approx(4.0)
    assert payload["utilization_summary"]["compute_utilization_fraction"] == pytest.approx(4.0 / 312.0)
    assert payload["bottleneck_summary"]["dominant_bucket"] == "forward_backward"
    assert payload["bottleneck_summary"]["host_pipeline_fraction"] == pytest.approx(0.3 / 0.88)
    assert payload["bottleneck_summary"]["h2d_transfer_fraction"] == pytest.approx(0.01 / 0.88)
    assert payload["bottleneck_summary"]["compute_utilization_fraction"] == pytest.approx(4.0 / 312.0)
    assert telemetry_path.read_text(encoding="utf-8") == original_telemetry

    rendered = render_run_inspect_text(payload)
    assert "bottleneck_summary=" in rendered
    assert "compute_accounting=" in rendered


def test_run_inspect_cli_derives_compute_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cli_profiled_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_records = _history_records()
    gradient_records = _timed_gradient_records()
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
        runtime_summary=_runtime_summary(),
        hardware_summary=_hardware_summary(),
        regime_budget=_regime_budget(),
        training_surface_record=training_surface_record,
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        posthoc_accounting,
        "derive_compute_accounting_for_run",
        lambda *_args, **_kwargs: {"train_flops_per_token": 625_000_000.0},
    )

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "dev",
            "run-inspect",
            "--run-dir",
            str(run_dir),
            "--derive-compute-accounting",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["compute_accounting"] == {"train_flops_per_token": 625_000_000.0}
    assert payload["utilization_summary"]["achieved_train_tflops_per_second"] == pytest.approx(4.0)
    assert payload["bottleneck_summary"]["forward_backward_fraction"] == pytest.approx(0.5 / 0.88)


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


def test_run_inspect_reports_summary_markdown_for_smoke_style_train_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "iris_smoke_run" / "train_outputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_surface_record.json").write_text(
        json.dumps(_training_surface_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Iris Smoke Report\n", encoding="utf-8")

    payload = run_inspect(run_dir)

    assert payload["artifacts"]["summary_md"]["exists"] is True
    assert payload["artifacts"]["summary_md"]["path"] == str((run_dir / "summary.md").resolve())


def test_run_inspect_reports_non_null_upper_block_metrics_for_sandwich_runs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sandwich_run" / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_records = _history_records()
    gradient_records = _gradient_records(activation_shape="sandwich")
    _write_jsonl(run_dir / "train_history.jsonl", history_records)
    _write_jsonl(run_dir / "gradient_history.jsonl", gradient_records)
    training_surface_record = _training_surface_record(
        arch="tabfoundry_sandwich",
        model_label="tabfoundry_sandwich",
        stage=None,
    )
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
        runtime_summary=_runtime_summary(),
        regime_budget=_regime_budget(),
        training_surface_record=training_surface_record,
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = run_inspect(run_dir)

    assert payload["surface_labels"]["model"] == "tabfoundry_sandwich"
    assert payload["health"]["verdict"] == "ok"
    assert payload["health"]["metrics"]["upper_block_post_warmup_mean_slope"] is not None
    assert payload["health"]["metrics"]["upper_block_final_to_early_ratio"] is not None


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


def test_run_inspect_uses_effective_training_surface_record_for_health_reconstruction(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "benchmarked_health_run" / "train"
    benchmark_dir = run_dir.parent / "benchmark"
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    history_records = _history_records()
    gradient_records = _warmup_sensitive_gradient_records()
    _write_jsonl(run_dir / "train_history.jsonl", history_records)
    _write_jsonl(run_dir / "gradient_history.jsonl", gradient_records)

    benchmark_surface_record = _warmup_sensitive_training_surface_record()
    benchmark_surface_record_path = benchmark_dir / "training_surface_record.json"
    benchmark_surface_record_path.write_text(
        json.dumps(benchmark_surface_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (benchmark_dir / "benchmark_run_record.json").write_text(
        json.dumps(
            {
                "run_id": "benchmarked_health_run",
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
    expected_health = health_check(
        run_dir,
        training_surface_record_path=benchmark_surface_record_path,
    )
    local_health = health_check(run_dir)

    assert payload["health"] == expected_health
    assert payload["health"]["verdict"] == "ok"
    assert local_health["verdict"] == "warn"
    assert local_health["metrics"]["upper_block_post_warmup_mean_slope"] > (
        payload["health"]["metrics"]["upper_block_post_warmup_mean_slope"]
    )
