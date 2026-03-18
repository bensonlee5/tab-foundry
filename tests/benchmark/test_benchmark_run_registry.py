from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

import tab_foundry.bench.benchmark_run_registry as registry_module
from tab_foundry.model.factory import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_checkpoint(
    path: Path,
    *,
    manifest_path: str,
    data_cfg: dict[str, object] | None = None,
    seed: int = 1,
    arch: str | None = "tabfoundry_staged",
    stage: str | None = "nano_exact",
    training_cfg: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cfg: dict[str, object] = {
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
    }
    if arch is not None:
        model_cfg["arch"] = arch
    if stage is not None:
        model_cfg["stage"] = stage
    checkpoint_data_cfg: dict[str, object] = {"manifest_path": manifest_path}
    if data_cfg is not None:
        checkpoint_data_cfg.update(data_cfg)
    torch.save(
        {
            "model": {},
            "config": {
                "task": "classification",
                "data": checkpoint_data_cfg,
                "runtime": {"seed": int(seed)},
                "model": model_cfg,
                **({} if training_cfg is None else {"training": training_cfg}),
                "schedule": {
                    "stages": [
                        {
                            "name": "stage1",
                            "steps": 400,
                            "lr_max": 8.0e-4,
                            "lr_schedule": "linear",
                            "warmup_ratio": 0.05,
                        }
                    ]
                },
            },
        },
        path,
    )
    return path


def _write_history(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "step": 25,
            "stage": "stage1",
            "train_loss": 0.62,
            "train_acc": 0.55,
            "lr": 8.0e-4,
            "grad_norm": 0.9,
            "elapsed_seconds": 1.3,
            "train_elapsed_seconds": 1.0,
            "val_loss": 0.41,
            "val_acc": 0.58,
        },
        {
            "step": 50,
            "stage": "stage1",
            "train_loss": 0.44,
            "train_acc": 0.66,
            "lr": 8.0e-4,
            "grad_norm": 0.7,
            "elapsed_seconds": 2.5,
            "train_elapsed_seconds": 2.0,
            "val_loss": 0.31,
            "val_acc": 0.64,
        },
        {
            "step": 75,
            "stage": "stage1",
            "train_loss": 0.48,
            "train_acc": 0.63,
            "lr": 8.0e-4,
            "grad_norm": 0.8,
            "elapsed_seconds": 3.8,
            "train_elapsed_seconds": 3.0,
            "val_loss": 0.35,
            "val_acc": 0.61,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
    return path


def _write_comparison_summary(
    path: Path,
    *,
    run_dir: Path,
    source_bundle_path: str,
    final_roc_auc: float = 0.83,
    final_log_loss: float = 0.42,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_count": 1,
        "benchmark_bundle": {
            "name": "test_bundle",
            "version": 1,
            "source_path": source_bundle_path,
            "task_count": 1,
            "task_ids": [1],
        },
        "tab_foundry": {
            "best_step": 50.0,
            "best_training_time": 2.0,
            "best_roc_auc": 0.84,
            "final_step": 75.0,
            "final_training_time": 3.0,
            "final_roc_auc": float(final_roc_auc),
            "final_log_loss": float(final_log_loss),
            "run_dir": str(run_dir.resolve()),
            "model_arch": "tabfoundry_staged",
            "model_stage": "nano_exact",
            "benchmark_profile": "nano_exact",
        },
        "nanotabpfn": {
            "best_step": 25.0,
            "best_training_time": 2.2,
            "best_roc_auc": 0.8,
            "final_step": 25.0,
            "final_training_time": 2.2,
            "final_roc_auc": 0.8,
            "root": "/tmp/nano",
            "python": "/tmp/nano/.venv/bin/python",
            "num_seeds": 2,
        },
        "artifacts": {
            "comparison_curve_png": str((path.parent / "comparison_curve.png").resolve()),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path.parent / "comparison_curve.png").write_bytes(b"png")
    return path


def _prepare_run(
    repo_root: Path,
    *,
    run_name: str,
    checkpoint_data_cfg: dict[str, object] | None = None,
    seed: int = 1,
    final_roc_auc: float = 0.83,
    final_log_loss: float = 0.42,
    training_cfg: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    manifest_path = repo_root / "data" / "manifests" / "default.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"manifest")
    bundle_path = repo_root / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_benchmark_v1.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text("{}\n", encoding="utf-8")

    run_dir = repo_root / "outputs" / run_name / "train"
    summary_path = repo_root / "outputs" / run_name / "benchmark" / "comparison_summary.json"
    _write_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        manifest_path="data/manifests/default.parquet",
        data_cfg=checkpoint_data_cfg,
        seed=seed,
        training_cfg=training_cfg,
    )
    _write_history(run_dir / "train_history.jsonl")
    _write_comparison_summary(
        summary_path,
        run_dir=run_dir,
        source_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json",
        final_roc_auc=final_roc_auc,
        final_log_loss=final_log_loss,
    )
    return run_dir, summary_path


def test_derive_benchmark_run_record_extracts_diagnostics_and_model_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(repo_root, run_name="stage_01")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    record = registry_module.derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
    )

    assert record["manifest_path"] == "data/manifests/default.parquet"
    assert record["seed_set"] == [1]
    assert record["model"]["arch"] == "tabfoundry_staged"
    assert record["model"]["stage"] == "nano_exact"
    assert record["training_diagnostics"]["best_val_loss"] == pytest.approx(0.31)
    assert record["training_diagnostics"]["best_val_step"] == pytest.approx(50.0)
    assert record["training_diagnostics"]["final_grad_norm"] == pytest.approx(0.8)
    assert record["model_size"]["total_params"] > 0
    assert record["model_size"]["trainable_params"] > 0
    assert record["artifacts"]["run_dir"] == "outputs/stage_01/train"
    assert record["artifacts"]["training_surface_record_path"] == (
        "outputs/stage_01/benchmark/training_surface_record.json"
    )
    assert record["surface_labels"]["model"] == "nano_exact"
    assert "training" not in record["surface_labels"]
    assert record["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json"
    )
    assert record["tab_foundry_metrics"]["final_log_loss"] == pytest.approx(0.42)


def test_derive_benchmark_run_record_captures_optional_training_surface_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(
        repo_root,
        run_name="stage_training",
        training_cfg={
            "surface_label": "prior_linear_decay",
            "apply_schedule": True,
            "overrides": {"optimizer": {"min_lr": 4.0e-4}},
        },
    )
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    record = registry_module.derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
    )

    assert record["surface_labels"]["training"] == "prior_linear_decay"


def test_derive_benchmark_run_record_includes_optional_sweep_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(repo_root, run_name="stage_01")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    record = registry_module.derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
        sweep_id="binary_md_v1",
        delta_id="delta_label_token",
        parent_sweep_id=None,
        queue_order=1,
        run_kind="primary",
    )

    assert record["sweep"] == {
        "sweep_id": "binary_md_v1",
        "delta_id": "delta_label_token",
        "parent_sweep_id": None,
        "queue_order": 1,
        "run_kind": "primary",
    }


def test_derive_benchmark_run_record_uses_manifest_path_from_resolved_data_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    override_manifest = (
        repo_root
        / "outputs"
        / "staged_ladder_support"
        / "binary_iris_manifest"
        / "manifest.parquet"
    )
    override_manifest.parent.mkdir(parents=True, exist_ok=True)
    override_manifest.write_bytes(b"manifest")
    run_dir, summary_path = _prepare_run(
        repo_root,
        run_name="surface_override",
        checkpoint_data_cfg={
            "surface_label": "binary_iris_manifest",
            "surface_overrides": {
                "manifest_path": str(override_manifest),
            },
        },
    )
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    record = registry_module.derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
    )

    surface_record = json.loads(
        (summary_path.parent / "training_surface_record.json").read_text(encoding="utf-8")
    )
    assert record["manifest_path"] == "outputs/staged_ladder_support/binary_iris_manifest/manifest.parquet"
    assert registry_module.resolve_registry_path_value(record["manifest_path"]) == override_manifest.resolve()
    assert (
        Path(surface_record["data"]["manifest"]["manifest_path"]).resolve()
        == override_manifest.resolve()
    )


def test_derive_benchmark_run_record_falls_back_to_best_benchmark_step_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(repo_root, run_name="prior_like")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    best_checkpoint.unlink()
    _write_checkpoint(
        run_dir / "checkpoints" / "step_000050.pt",
        manifest_path="data/manifests/default.parquet",
        seed=1,
    )
    _write_checkpoint(
        run_dir / "checkpoints" / "step_000075.pt",
        manifest_path="data/manifests/default.parquet",
        seed=1,
    )
    _write_checkpoint(
        run_dir / "checkpoints" / "latest.pt",
        manifest_path="data/manifests/default.parquet",
        seed=1,
    )

    record = registry_module.derive_benchmark_run_record(
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
    )

    assert record["artifacts"]["best_checkpoint_path"] == (
        "outputs/prior_like/train/checkpoints/step_000050.pt"
    )
    assert record["model_size"]["total_params"] > 0


def test_register_benchmark_run_writes_repo_relative_entry_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    anchor_run_dir, anchor_summary = _prepare_run(
        repo_root,
        run_name="anchor",
        seed=1,
        final_roc_auc=0.83,
        final_log_loss=0.42,
    )
    child_run_dir, child_summary = _prepare_run(
        repo_root,
        run_name="label_token",
        seed=2,
        final_roc_auc=0.87,
        final_log_loss=0.39,
    )
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    _ = registry_module.register_benchmark_run(
        run_id="00_simple_anchor",
        track="binary_ladder",
        experiment="cls_benchmark_linear_simple",
        config_profile="cls_benchmark_linear_simple",
        budget_class="short-run",
        run_dir=anchor_run_dir,
        comparison_summary_path=anchor_summary,
        decision="keep",
        conclusion="Frozen anchor for staged comparisons.",
        registry_path=registry_path,
    )
    result = registry_module.register_benchmark_run(
        run_id="01_nano_exact",
        track="binary_ladder",
        experiment="cls_benchmark_staged",
        config_profile="cls_benchmark_staged",
        budget_class="short-run",
        run_dir=child_run_dir,
        comparison_summary_path=child_summary,
        decision="keep",
        conclusion="Exact staged repro matches the frozen anchor contract.",
        parent_run_id="00_simple_anchor",
        anchor_run_id="00_simple_anchor",
        sweep_id="binary_md_v1",
        delta_id="delta_label_token",
        parent_sweep_id=None,
        queue_order=1,
        run_kind="primary",
        registry_path=registry_path,
    )

    assert result["registry_path"] == str(registry_path.resolve())
    run_entry = result["run"]
    assert run_entry["artifacts"]["run_dir"] == "outputs/label_token/train"
    assert run_entry["sweep"] == {
        "sweep_id": "binary_md_v1",
        "delta_id": "delta_label_token",
        "parent_sweep_id": None,
        "queue_order": 1,
        "run_kind": "primary",
    }
    assert run_entry["comparisons"]["vs_parent"]["reference_run_id"] == "00_simple_anchor"
    assert run_entry["comparisons"]["vs_parent"]["final_roc_auc_delta"] == pytest.approx(0.04)
    assert run_entry["comparisons"]["vs_parent"]["final_log_loss_delta"] == pytest.approx(-0.03)
    assert run_entry["decision"] == "keep"
    child_record_path = repo_root / str(run_entry["artifacts"]["benchmark_run_record_path"])
    assert child_record_path.exists()
    child_record = json.loads(child_record_path.read_text(encoding="utf-8"))
    assert child_record["manifest_path"] == "data/manifests/default.parquet"
    assert child_record["artifacts"]["training_surface_record_path"] == (
        "outputs/label_token/benchmark/training_surface_record.json"
    )

    registry = registry_module.load_benchmark_run_registry(registry_path)
    assert set(registry["runs"]) == {"00_simple_anchor", "01_nano_exact"}
    assert registry["runs"]["01_nano_exact"]["comparisons"]["vs_anchor"]["final_roc_auc_delta"] == pytest.approx(0.04)
    assert registry["runs"]["01_nano_exact"]["comparisons"]["vs_anchor"]["final_log_loss_delta"] == pytest.approx(-0.03)


def test_register_benchmark_run_rejects_unknown_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    run_dir, summary_path = _prepare_run(repo_root, run_name="stage_01")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    with pytest.raises(RuntimeError, match="unknown parent_run_id"):
        registry_module.register_benchmark_run(
            run_id="01_nano_exact",
            track="binary_ladder",
            experiment="cls_benchmark_staged",
            config_profile="cls_benchmark_staged",
            budget_class="short-run",
            run_dir=run_dir,
            comparison_summary_path=summary_path,
            decision="defer",
            conclusion="Waiting on the frozen anchor comparison.",
            parent_run_id="missing",
            registry_path=registry_path,
        )


def test_register_benchmark_run_main_parses_cli_and_defaults_config_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_register_benchmark_run(**kwargs):
        captured.update(kwargs)
        return {
            "registry_path": str((tmp_path / "benchmark_run_registry.json").resolve()),
            "run": {"run_id": kwargs["run_id"], "decision": kwargs["decision"]},
        }

    monkeypatch.setattr(registry_module, "register_benchmark_run", _fake_register_benchmark_run)

    exit_code = registry_module.main(
        [
            "--run-id",
            "02_label_token_md_prior",
            "--track",
            "prior_compact_buildout",
            "--run-dir",
            str(tmp_path / "run"),
            "--comparison-summary",
            str(tmp_path / "comparison_summary.json"),
            "--experiment",
            "cls_benchmark_staged_prior",
            "--decision",
            "keep",
            "--conclusion",
            "CLI smoke coverage",
            "--parent-run-id",
            "01_nano_exact_md_prior_parity_fix",
            "--anchor-run-id",
            "01_nano_exact_md_prior_parity_fix",
            "--prior-dir",
            str(tmp_path / "prior"),
            "--control-baseline-id",
            "cls_benchmark_linear_v1",
            "--sweep-id",
            "binary_md_v1",
            "--delta-id",
            "delta_label_token",
            "--parent-sweep-id",
            "binary_xs_v0",
            "--queue-order",
            "3",
            "--run-kind",
            "followup",
            "--registry-path",
            str(tmp_path / "benchmark_run_registry.json"),
        ]
    )

    assert exit_code == 0
    assert captured["run_id"] == "02_label_token_md_prior"
    assert captured["track"] == "prior_compact_buildout"
    assert captured["experiment"] == "cls_benchmark_staged_prior"
    assert captured["config_profile"] == "cls_benchmark_staged_prior"
    assert captured["budget_class"] == registry_module.DEFAULT_BUDGET_CLASS
    assert captured["run_dir"] == tmp_path / "run"
    assert captured["comparison_summary_path"] == tmp_path / "comparison_summary.json"
    assert captured["decision"] == "keep"
    assert captured["conclusion"] == "CLI smoke coverage"
    assert captured["parent_run_id"] == "01_nano_exact_md_prior_parity_fix"
    assert captured["anchor_run_id"] == "01_nano_exact_md_prior_parity_fix"
    assert captured["prior_dir"] == tmp_path / "prior"
    assert captured["control_baseline_id"] == "cls_benchmark_linear_v1"
    assert captured["sweep_id"] == "binary_md_v1"
    assert captured["delta_id"] == "delta_label_token"
    assert captured["parent_sweep_id"] == "binary_xs_v0"
    assert captured["queue_order"] == 3
    assert captured["run_kind"] == "followup"
    assert captured["registry_path"] == tmp_path / "benchmark_run_registry.json"
    assert "Benchmark run registered:" in capsys.readouterr().out


def test_register_benchmark_run_script_delegates_to_registry_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(registry_module, "main", _fake_main)
    monkeypatch.setattr(sys, "argv", ["register_benchmark_run.py"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(REPO_ROOT / "scripts" / "register_benchmark_run.py"), run_name="__main__")

    assert exc_info.value.code == 0
    assert captured["argv"] is None


def test_derive_benchmark_run_record_rejects_legacy_checkpoint_without_arch_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(repo_root, run_name="legacy_simple_anchor")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    simple_model = build_model(
        task="classification",
        arch="tabfoundry_simple",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=4,
        tficl_n_layers=3,
        head_hidden_dim=192,
    )
    torch.save(
        {
            "model": simple_model.state_dict(),
            "config": {
                "task": "classification",
                "data": {"manifest_path": "data/manifests/default.parquet"},
                "runtime": {"seed": 1},
                "model": {
                    "d_icl": 96,
                    "input_normalization": "train_zscore_clip",
                    "many_class_base": 2,
                    "tficl_n_heads": 4,
                    "tficl_n_layers": 3,
                    "head_hidden_dim": 192,
                },
                "schedule": {
                    "stages": [
                        {
                            "name": "stage1",
                            "steps": 400,
                            "lr_max": 8.0e-4,
                            "lr_schedule": "linear",
                            "warmup_ratio": 0.05,
                        }
                    ]
                },
            },
        },
        run_dir / "checkpoints" / "best.pt",
    )

    with pytest.raises(RuntimeError, match="persisted model.arch"):
        registry_module.derive_benchmark_run_record(
            run_dir=run_dir,
            comparison_summary_path=summary_path,
            benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
        )


def test_derive_benchmark_run_record_rejects_legacy_grouped_checkpoint_without_feature_group_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir, summary_path = _prepare_run(repo_root, run_name="legacy_grouped")
    monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

    grouped_model = build_model(task="classification", feature_group_size=32)
    torch.save(
        {
            "model": grouped_model.state_dict(),
            "config": {
                "task": "classification",
                "data": {"manifest_path": "data/manifests/default.parquet"},
                "runtime": {"seed": 1},
                "model": {"arch": "tabfoundry"},
                "schedule": {
                    "stages": [
                        {
                            "name": "stage1",
                            "steps": 400,
                            "lr_max": 8.0e-4,
                            "lr_schedule": "linear",
                            "warmup_ratio": 0.05,
                        }
                    ]
                },
            },
        },
        run_dir / "checkpoints" / "best.pt",
    )

    with pytest.raises(ValueError, match="omitted feature_group_size"):
        registry_module.derive_benchmark_run_record(
            run_dir=run_dir,
            comparison_summary_path=summary_path,
            benchmark_run_record_path=summary_path.parent / "benchmark_run_record.json",
        )


def test_checked_in_benchmark_run_registry_contains_medium_binary_anchor() -> None:
    registry_path = REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"

    registry = registry_module.load_benchmark_run_registry(registry_path)
    run = registry["runs"]["01_nano_exact_md_prior_parity_fix_binary_medium_v1"]

    assert run["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json"
    )
    assert run["lineage"]["control_baseline_id"] == "cls_benchmark_linear_v2"
    assert run["artifacts"]["run_dir"] == "outputs/staged_ladder/01_nano_exact_md/prior_parity_fix"
    assert run.get("sweep") is None
