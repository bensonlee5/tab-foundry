from __future__ import annotations

import json
from pathlib import Path

import pytest

import tab_foundry.bench.instability_audit as audit_module


def _write_history(path: Path, *, grad_norms: list[float], train_losses: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (grad_norm, train_loss) in enumerate(zip(grad_norms, train_losses, strict=True), start=1):
        records.append(
            {
                "step": index,
                "stage": "prior_dump",
                "train_loss": float(train_loss),
                "train_acc": 0.5,
                "lr": 4.0e-3,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": float(index),
                "train_elapsed_seconds": float(index),
                "val_loss": None,
                "val_acc": None,
            }
        )
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_comparison_summary(
    path: Path,
    *,
    run_dir: Path,
    best_roc_auc: float,
    final_roc_auc: float,
    final_log_loss: float = 0.44,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_bundle": {
            "name": "toy_bundle",
            "version": 1,
            "source_path": "src/tab_foundry/bench/toy_bundle.json",
            "task_count": 1,
            "task_ids": [1],
        },
        "tab_foundry": {
            "run_dir": str(run_dir.resolve()),
            "best_step": 2.0,
            "best_training_time": 2.0,
            "best_roc_auc": float(best_roc_auc),
            "final_step": 3.0,
            "final_training_time": 3.0,
            "final_roc_auc": float(final_roc_auc),
            "final_log_loss": float(final_log_loss),
            "best_to_final_roc_auc_delta": float(final_roc_auc) - float(best_roc_auc),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_build_instability_audit_ranks_runs_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged_ladder_root = tmp_path / "outputs" / "staged_ladder"
    anchor_run_dir = staged_ladder_root / "anchor_reference" / "train"
    _write_history(anchor_run_dir / "train_history.jsonl", grad_norms=[0.5, 0.4, 0.3], train_losses=[0.8, 0.7, 0.6])
    anchor_summary = _write_comparison_summary(
        staged_ladder_root / "anchor_reference" / "benchmark" / "comparison_summary.json",
        run_dir=anchor_run_dir,
        best_roc_auc=0.81,
        final_roc_auc=0.8,
    )

    stable_run_dir = staged_ladder_root / "sd_binary_md_v1_01_delta_stable_v1" / "train"
    _write_history(stable_run_dir / "train_history.jsonl", grad_norms=[1.0, 0.8, 0.7], train_losses=[0.9, 0.7, 0.65])
    _write_comparison_summary(
        staged_ladder_root / "sd_binary_md_v1_01_delta_stable_v1" / "benchmark" / "comparison_summary.json",
        run_dir=stable_run_dir,
        best_roc_auc=0.79,
        final_roc_auc=0.78,
    )

    unstable_run_dir = staged_ladder_root / "sd_binary_md_v1_02_delta_unstable_v1" / "train"
    _write_history(
        unstable_run_dir / "train_history.jsonl",
        grad_norms=[2.0, 125.0, 4.0],
        train_losses=[0.9, 2.4, 0.8],
    )
    _write_comparison_summary(
        staged_ladder_root / "sd_binary_md_v1_02_delta_unstable_v1" / "benchmark" / "comparison_summary.json",
        run_dir=unstable_run_dir,
        best_roc_auc=0.7,
        final_roc_auc=0.66,
    )
    result_card_path = (
        staged_ladder_root / "research" / "binary_md_v1" / "delta_unstable" / "result_card.md"
    )
    result_card_path.parent.mkdir(parents=True, exist_ok=True)
    result_card_path.write_text(
        "\n".join(
            [
                "# Result Card",
                "",
                "- run id: `sd_binary_md_v1_02_delta_unstable_v1`",
                "- decision recommendation: `defer`",
                "- recommended next action: `needs_followup`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        audit_module,
        "load_benchmark_run_registry",
        lambda _path=None: {
            "runs": {
                audit_module.DEFAULT_ANCHOR_RUN_ID: {
                    "artifacts": {
                        "run_dir": str(anchor_run_dir.resolve()),
                        "comparison_summary_path": str(anchor_summary.resolve()),
                    }
                }
            }
        },
    )
    monkeypatch.setattr(audit_module, "resolve_registry_path_value", lambda value: Path(str(value)).resolve())

    payload = audit_module.build_instability_audit(
        staged_ladder_root=staged_ladder_root,
        registry_path=tmp_path / "benchmark_run_registry.json",
    )

    assert payload["run_count"] == 3
    assert payload["recommendations"][0]["run_id"] == audit_module.DEFAULT_ANCHOR_RUN_ID
    assert payload["runs"][1]["run_id"] == "sd_binary_md_v1_02_delta_unstable_v1"
    assert payload["runs"][1]["decision_recommendation"] == "defer"
    assert payload["runs"][1]["next_action"] == "needs_followup"

    report_paths = audit_module.write_instability_audit(
        payload,
        out_root=staged_ladder_root / "reports",
        sweep_id=audit_module.DEFAULT_SWEEP_ID,
    )

    assert Path(report_paths["json"]).exists()
    markdown_path = Path(report_paths["markdown"])
    assert markdown_path.exists()
    report_text = markdown_path.read_text(encoding="utf-8")
    assert "sd_binary_md_v1_02_delta_unstable_v1" in report_text
    assert "Anchor reference rerun with module telemetry" in report_text
    assert "final log loss" in report_text


def test_build_instability_audit_indexes_primary_run_id_result_cards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged_ladder_root = tmp_path / "outputs" / "staged_ladder"
    candidate_run_dir = staged_ladder_root / "sd_binary_md_v1_06_delta_row_cls_pool_v1" / "train"
    _write_history(
        candidate_run_dir / "train_history.jsonl",
        grad_norms=[5.0, 25.0, 4.0],
        train_losses=[0.9, 1.6, 0.7],
    )
    _write_comparison_summary(
        staged_ladder_root / "sd_binary_md_v1_06_delta_row_cls_pool_v1" / "benchmark" / "comparison_summary.json",
        run_dir=candidate_run_dir,
        best_roc_auc=0.58,
        final_roc_auc=0.5,
    )
    result_card_path = (
        staged_ladder_root / "research" / "binary_md_v1" / "delta_row_cls_pool" / "result_card.md"
    )
    result_card_path.parent.mkdir(parents=True, exist_ok=True)
    result_card_path.write_text(
        "\n".join(
            [
                "# Result Card",
                "",
                "- primary run id: `sd_binary_md_v1_06_delta_row_cls_pool_v1`",
                "- decision recommendation: `defer`",
                "- recommended next action: `delta_row_cls_pool_rmsnorm`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(audit_module, "load_benchmark_run_registry", lambda _path=None: {"runs": {}})

    payload = audit_module.build_instability_audit(
        staged_ladder_root=staged_ladder_root,
        registry_path=tmp_path / "benchmark_run_registry.json",
    )

    assert payload["run_count"] == 1
    assert payload["runs"][0]["run_id"] == "sd_binary_md_v1_06_delta_row_cls_pool_v1"
    assert payload["runs"][0]["result_card_path"] == str(result_card_path.resolve())
    assert payload["runs"][0]["decision_recommendation"] == "defer"
    assert payload["runs"][0]["next_action"] == "delta_row_cls_pool_rmsnorm"
