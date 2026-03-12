from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tab_foundry.bench.iris_smoke as iris_smoke_module
from tab_foundry.data.manifest import ManifestSummary
from tab_foundry.types import EvalResult, TrainResult


def test_build_parser_defaults_match_ci_profile() -> None:
    args = iris_smoke_module.build_parser().parse_args([])
    assert args.device == "cpu"
    assert args.initial_num_tasks == 64
    assert args.max_num_tasks == 512
    assert args.iris_benchmark_seeds == 5
    assert args.checkpoint_every == 2


def test_write_summary_markdown_includes_timings_and_benchmark(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.md"
    telemetry = {
        "generated_at_utc": "2026-03-12T00:00:00+00:00",
        "config": {
            "device": "cpu",
            "final_num_tasks": 64,
            "task_count_attempts": [64],
        },
        "manifest": {"train_records": 56, "val_records": 3, "test_records": 5},
        "train_metrics": {"global_step": 6, "best_val_loss": 0.5},
        "eval_metrics": {"loss": 0.4, "acc": 0.8},
        "iris_benchmark": {
            "means": {"tab_foundry": 0.9, "LogReg": 0.95},
            "stddevs": {"tab_foundry": 0.1, "LogReg": 0.05},
        },
        "checkpoint_snapshots": [{"step": 2, "train_elapsed_seconds": 0.2}],
        "timings_seconds": {
            "generate_iris_tasks": 0.1,
            "build_manifest": 0.2,
            "train": 0.3,
            "eval": 0.4,
            "iris_benchmark": 0.5,
            "total": 1.5,
        },
        "artifacts": {
            "best_checkpoint": "/tmp/best.pt",
            "generated_dir": "/tmp/generated",
            "manifest_path": "/tmp/manifest.parquet",
            "train_output_dir": "/tmp/train_outputs",
            "train_history_jsonl": "/tmp/train_history.jsonl",
            "loss_curve_png": "/tmp/loss_curve.png",
            "telemetry_json": "/tmp/telemetry.json",
            "summary_md": str(summary_path),
        },
    }

    iris_smoke_module._write_summary_markdown(summary_path, telemetry)

    content = summary_path.read_text(encoding="utf-8")
    assert "# Iris Smoke Report" in content
    assert "| generate_iris_tasks | 0.100 |" in content
    assert "| tab_foundry | 0.900000 | 0.100000 |" in content
    assert "/tmp/best.pt" in content


def test_run_iris_smoke_expands_task_count_until_test_split_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_task_counts: list[int] = []
    build_calls = {"count": 0}

    def _fake_write_iris_tasks(
        generated_dir: Path,
        *,
        num_tasks: int,
        seed: int,
        test_size: float,
    ) -> Path:
        captured_task_counts.append(num_tasks)
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir

    def _fake_build_manifest(*_args: Any, **kwargs: Any) -> ManifestSummary:
        build_calls["count"] += 1
        out_path = Path(str(kwargs["out_path"]))
        if build_calls["count"] == 1:
            return ManifestSummary(
                out_path=out_path,
                filter_policy="accepted_only",
                discovered_records=64,
                excluded_records=0,
                total_records=64,
                train_records=60,
                val_records=4,
                test_records=0,
                warnings=[],
            )
        return ManifestSummary(
            out_path=out_path,
            filter_policy="accepted_only",
            discovered_records=128,
            excluded_records=0,
            total_records=128,
            train_records=112,
            val_records=8,
            test_records=8,
            warnings=[],
        )

    def _fake_train(cfg: Any) -> TrainResult:
        history_path = Path(str(cfg.logging.history_jsonl_path))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "step": 2,
                    "stage": "stage1",
                    "train_loss": 0.9,
                    "train_acc": 0.6,
                    "lr": 8.0e-4,
                    "elapsed_seconds": 0.2,
                    "train_elapsed_seconds": 0.1,
                    "val_loss": 0.8,
                    "val_acc": 0.7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_dir = Path(str(cfg.runtime.output_dir)) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_dir / "best.pt"
        best_checkpoint.write_bytes(b"best")
        (checkpoint_dir / "step_000002.pt").write_bytes(b"step")
        latest_checkpoint = checkpoint_dir / "latest_stage2.pt"
        latest_checkpoint.write_bytes(b"latest")
        return TrainResult(
            output_dir=Path(str(cfg.runtime.output_dir)),
            best_checkpoint=best_checkpoint,
            latest_checkpoint=latest_checkpoint,
            global_step=6,
            metrics={"best_val_loss": 0.6, "train_elapsed_seconds": 0.1},
        )

    monkeypatch.setattr(iris_smoke_module, "_write_iris_tasks", _fake_write_iris_tasks)
    monkeypatch.setattr(iris_smoke_module, "build_manifest", _fake_build_manifest)
    monkeypatch.setattr(iris_smoke_module, "train", _fake_train)
    monkeypatch.setattr(
        iris_smoke_module,
        "evaluate_checkpoint",
        lambda *_args, **_kwargs: EvalResult(
            checkpoint=tmp_path / "best.pt",
            metrics={"loss": 0.5, "acc": 0.8},
        ),
    )
    monkeypatch.setattr(
        iris_smoke_module,
        "evaluate_iris_checkpoint",
        lambda *_args, **_kwargs: iris_smoke_module.IrisEvalSummary(
            checkpoint=tmp_path / "best.pt",
            results={"tab_foundry": [0.9, 0.8], "LogReg": [0.95, 0.96]},
        ),
    )

    telemetry = iris_smoke_module.run_iris_smoke(
        iris_smoke_module.IrisSmokeConfig(out_root=tmp_path / "run")
    )

    assert telemetry["success"] is True
    assert captured_task_counts == [64, 128]
    assert telemetry["config"]["final_num_tasks"] == 128
    assert telemetry["manifest"]["test_records"] == 8
    assert (tmp_path / "run" / "summary.md").exists()
    assert (tmp_path / "run" / "telemetry.json").exists()


def test_run_iris_smoke_rejects_non_finite_train_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        iris_smoke_module,
        "_write_iris_tasks",
        lambda generated_dir, **_kwargs: generated_dir.mkdir(parents=True, exist_ok=True) or generated_dir,
    )
    monkeypatch.setattr(
        iris_smoke_module,
        "build_manifest",
        lambda *_args, **kwargs: ManifestSummary(
            out_path=Path(str(kwargs["out_path"])),
            filter_policy="accepted_only",
            discovered_records=64,
            excluded_records=0,
            total_records=64,
            train_records=56,
            val_records=3,
            test_records=5,
            warnings=[],
        ),
    )

    def _fake_train(cfg: Any) -> TrainResult:
        history_path = Path(str(cfg.logging.history_jsonl_path))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "step": 2,
                    "stage": "stage1",
                    "train_loss": 0.9,
                    "train_acc": 0.6,
                    "lr": 8.0e-4,
                    "elapsed_seconds": 0.2,
                    "train_elapsed_seconds": 0.1,
                    "val_loss": 0.8,
                    "val_acc": 0.7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_dir = Path(str(cfg.runtime.output_dir)) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_dir / "best.pt"
        best_checkpoint.write_bytes(b"best")
        (checkpoint_dir / "step_000002.pt").write_bytes(b"step")
        return TrainResult(
            output_dir=Path(str(cfg.runtime.output_dir)),
            best_checkpoint=best_checkpoint,
            latest_checkpoint=None,
            global_step=6,
            metrics={"best_val_loss": float("inf"), "train_elapsed_seconds": 0.1},
        )

    monkeypatch.setattr(iris_smoke_module, "train", _fake_train)
    monkeypatch.setattr(
        iris_smoke_module,
        "evaluate_checkpoint",
        lambda *_args, **_kwargs: EvalResult(
            checkpoint=tmp_path / "best.pt",
            metrics={"loss": 0.5, "acc": 0.8},
        ),
    )
    monkeypatch.setattr(
        iris_smoke_module,
        "evaluate_iris_checkpoint",
        lambda *_args, **_kwargs: iris_smoke_module.IrisEvalSummary(
            checkpoint=tmp_path / "best.pt",
            results={"tab_foundry": [0.9], "LogReg": [0.95]},
        ),
    )

    with pytest.raises(RuntimeError, match="train metric must be finite"):
        iris_smoke_module.run_iris_smoke(
            iris_smoke_module.IrisSmokeConfig(out_root=tmp_path / "run")
        )

    telemetry = json.loads((tmp_path / "run" / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["success"] is False
    assert "RuntimeError" in telemetry["error"]
