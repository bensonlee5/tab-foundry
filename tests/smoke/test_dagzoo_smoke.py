from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tab_foundry.bench.dagzoo_smoke as smoke_module
from tab_foundry.data.dagzoo_handoff import DagzooHandoffInfo
from tab_foundry.data.dagzoo_workflow import DagzooGenerateManifestResult
from tab_foundry.data.manifest import ManifestSummary
from tab_foundry.types import EvalResult, TrainResult


def test_build_parser_defaults_match_indicative_profile() -> None:
    args = smoke_module.build_parser().parse_args([])
    assert args.num_datasets == 128
    assert args.rows == 1024
    assert args.train_steps == 250
    assert args.checkpoint_every == 25
    assert args.device == "cpu"


def test_plot_loss_curve_writes_png(tmp_path: Path) -> None:
    history_path = tmp_path / "train_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "stage": "stage1",
                        "train_loss": 1.2,
                        "train_acc": 0.4,
                        "lr": 8.0e-4,
                        "elapsed_seconds": 0.1,
                        "val_loss": 1.0,
                        "val_acc": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "step": 2,
                        "stage": "stage1",
                        "train_loss": 0.9,
                        "train_acc": 0.6,
                        "lr": 7.9e-4,
                        "elapsed_seconds": 0.2,
                        "val_loss": 0.8,
                        "val_acc": 0.7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = smoke_module.plot_loss_curve(history_path, tmp_path / "loss_curve.png")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_run_dagzoo_smoke_writes_expected_telemetry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dagzoo_root = tmp_path / "dagzoo"
    (dagzoo_root / "configs").mkdir(parents=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    out_root = tmp_path / "run"
    generated_dir = out_root / "nested" / "generated"
    handoff_manifest_path = out_root / "handoff_manifest.json"

    captured: dict[str, Any] = {}

    def _fake_run_dagzoo_generate_manifest(
        cfg: Any,
    ) -> DagzooGenerateManifestResult:
        captured["workflow_cfg"] = cfg
        return DagzooGenerateManifestResult(
            handoff=DagzooHandoffInfo(
                handoff_manifest_path=handoff_manifest_path,
                handoff_manifest_sha256="a" * 64,
                source_family="dagzoo.fixed_layout_scm",
                generate_run_id="1" * 32,
                generated_corpus_id="2" * 32,
                generated_dir=generated_dir,
                recommended_training_corpus="generated",
                recommended_training_artifact_key="generated_dir",
                curation_policy="none",
            ),
            summary=ManifestSummary(
                out_path=out_root / "manifest.parquet",
                filter_policy="include_all",
                discovered_records=128,
                excluded_records=0,
                total_records=128,
                train_records=80,
                val_records=24,
                test_records=24,
                warnings=[],
            ),
        )

    def _fake_train(cfg: Any) -> TrainResult:
        captured["train_cfg"] = cfg
        history_path = Path(str(cfg.logging.history_jsonl_path))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "step": 25,
                            "stage": "stage1",
                            "train_loss": 1.0,
                            "train_acc": 0.5,
                            "lr": 8.0e-4,
                            "elapsed_seconds": 0.1,
                            "val_loss": 0.9,
                            "val_acc": 0.6,
                        }
                    ),
                    json.dumps(
                        {
                            "step": 250,
                            "stage": "stage1",
                            "train_loss": 0.8,
                            "train_acc": 0.7,
                            "lr": 7.5e-4,
                            "elapsed_seconds": 1.5,
                            "val_loss": 0.7,
                            "val_acc": 0.8,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = Path(str(cfg.runtime.output_dir))
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
        (checkpoint_dir / "step_000250.pt").write_bytes(b"step250")
        best = checkpoint_dir / "best.pt"
        latest = checkpoint_dir / "latest_stage1.pt"
        best.write_bytes(b"best")
        latest.write_bytes(b"latest")
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=best,
            latest_checkpoint=latest,
            global_step=250,
            metrics={"best_val_loss": 0.75},
        )

    def _fake_eval(cfg: Any) -> EvalResult:
        captured["eval_cfg"] = cfg
        return EvalResult(checkpoint=Path(str(cfg.eval.checkpoint)), metrics={"loss": 0.7, "acc": 0.8})

    monkeypatch.setattr(
        smoke_module,
        "run_dagzoo_generate_manifest",
        _fake_run_dagzoo_generate_manifest,
    )
    monkeypatch.setattr(smoke_module, "train", _fake_train)
    monkeypatch.setattr(smoke_module, "evaluate_checkpoint", _fake_eval)

    telemetry = smoke_module.run_dagzoo_smoke(
        smoke_module.SmokeConfig(
            dagzoo_root=dagzoo_root,
            out_root=out_root,
        )
    )

    workflow_cfg = captured["workflow_cfg"]
    assert workflow_cfg.dagzoo_root == dagzoo_root.resolve()
    assert workflow_cfg.dagzoo_config == Path("configs/default.yaml")
    assert workflow_cfg.handoff_root == out_root.resolve()
    assert workflow_cfg.out_manifest == (out_root / "manifest.parquet").resolve()
    assert workflow_cfg.num_datasets == 128
    assert workflow_cfg.rows == "1024"
    assert workflow_cfg.seed == 1
    assert workflow_cfg.device == "cpu"
    assert workflow_cfg.train_ratio == smoke_module.DEFAULT_TRAIN_RATIO
    assert workflow_cfg.val_ratio == smoke_module.DEFAULT_VAL_RATIO
    assert workflow_cfg.filter_policy == smoke_module.DEFAULT_FILTER_POLICY

    assert telemetry["success"] is True
    assert telemetry["dagzoo_handoff"]["generate_run_id"] == "1" * 32
    assert telemetry["dagzoo_handoff"]["generated_corpus_id"] == "2" * 32
    assert telemetry["dagzoo_handoff"]["handoff_manifest_path"] == str(handoff_manifest_path)
    assert telemetry["artifacts"]["generated_dir"] == str(generated_dir)
    assert "dagzoo_generate_manifest" in telemetry["timings_seconds"]
    assert "dagzoo_generate" not in telemetry["timings_seconds"]
    assert "build_manifest" not in telemetry["timings_seconds"]

    train_stage = captured["train_cfg"].schedule.stages[0]
    assert int(train_stage["steps"]) == 250
    assert int(captured["train_cfg"].data.train_row_cap) == 96
    assert int(captured["train_cfg"].data.test_row_cap) == 48
    assert str(captured["train_cfg"].runtime.device) == "cpu"
    assert int(captured["train_cfg"].runtime.checkpoint_every) == 25
    assert int(captured["eval_cfg"].data.train_row_cap) == 96
    assert int(captured["eval_cfg"].data.test_row_cap) == 48
    assert str(captured["eval_cfg"].eval.split) == "test"

    telemetry_path = out_root / "telemetry.json"
    assert telemetry_path.exists()
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["generated_dir"] == str(generated_dir)
    assert payload["artifacts"]["train_history_jsonl"].endswith("train_history.jsonl")
    assert payload["artifacts"]["loss_curve_png"].endswith("loss_curve.png")
    assert payload["dagzoo_handoff"]["generate_run_id"] == "1" * 32
    assert payload["dagzoo_handoff"]["generated_corpus_id"] == "2" * 32
    assert payload["dagzoo_handoff"]["handoff_manifest_path"] == str(handoff_manifest_path)
    assert "dagzoo_generate_manifest" in payload["timings_seconds"]
    assert "dagzoo_generate" not in payload["timings_seconds"]
    assert "build_manifest" not in payload["timings_seconds"]
    assert payload["manifest"]["train_records"] == 80
    assert payload["checkpoint_snapshots"][0]["step"] == 25
    assert payload["checkpoint_snapshots"][-1]["step"] == 250
    assert payload["train_metrics"]["global_step"] == 250
    assert payload["eval_metrics"]["acc"] == 0.8
