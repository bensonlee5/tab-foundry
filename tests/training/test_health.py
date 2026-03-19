from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.training.health import health_check
from tab_foundry.training.instability import build_training_telemetry


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _training_surface_record() -> dict[str, object]:
    return {
        "training": {
            "schedule_stages": [
                {
                    "name": "stage1",
                    "steps": 100,
                    "lr_max": 1.0e-3,
                    "warmup_ratio": 0.1,
                }
            ]
        }
    }


def _history_records(*, initial_loss: float, delta: float) -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "train_loss": initial_loss + (delta * float(step - 1)),
            "train_loss_delta": None if step == 1 else delta,
        }
        for step in range(1, 41)
    ]


def _gradient_records(
    *,
    clipped_steps: set[int],
    block_base: float,
    block_slope: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(1, 41):
        block_value = block_base + (block_slope * float(step))
        records.append(
            {
                "step": step,
                "global_grad_norm": 0.5 + (0.01 * float(step)),
                "grad_clip_triggered": step in clipped_steps,
                "module_grad_norms": {
                    "feature_encoder": 1.0,
                    "direct_head": 4.0,
                },
                "activation_norms": {
                    "post_feature_encoder": 2.0 + (0.001 * float(step)),
                    "pre_transformer": 3.0 + (0.001 * float(step)),
                    "post_transformer_block_8": block_value,
                    "post_transformer_block_9": block_value + 0.2,
                    "post_transformer_block_10": block_value + 0.4,
                    "post_transformer_block_11": block_value + 0.6,
                },
            }
        )
    return records


def _write_run_artifacts(
    run_dir: Path,
    *,
    history_records: list[dict[str, object]],
    gradient_records: list[dict[str, object]],
    include_telemetry: bool,
    success: bool = True,
    error: BaseException | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "train_history.jsonl", history_records)
    _write_jsonl(run_dir / "gradient_history.jsonl", gradient_records)
    training_surface_record = _training_surface_record()
    (run_dir / "training_surface_record.json").write_text(
        json.dumps(training_surface_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not include_telemetry:
        return
    telemetry = build_training_telemetry(
        run_dir=run_dir,
        success=success,
        artifacts={},
        checkpoint_snapshots=[],
        history_records=history_records,
        gradient_records=gradient_records,
        training_surface_record=training_surface_record,
        error=error,
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_health_check_reports_ok_from_existing_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "ok_run"
    _write_run_artifacts(
        run_dir,
        history_records=_history_records(initial_loss=1.0, delta=-0.005),
        gradient_records=_gradient_records(clipped_steps={1}, block_base=8.0, block_slope=0.001),
        include_telemetry=True,
    )

    payload = health_check(run_dir)

    assert payload["verdict"] == "ok"
    assert payload["source"] == "telemetry"
    assert payload["metrics"]["clipped_step_fraction"] == pytest.approx(0.025)


def test_health_check_reconstructs_telemetry_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "reconstructed_run"
    _write_run_artifacts(
        run_dir,
        history_records=_history_records(initial_loss=1.0, delta=-0.002),
        gradient_records=_gradient_records(clipped_steps=set(), block_base=6.0, block_slope=0.001),
        include_telemetry=False,
    )

    payload = health_check(run_dir)

    assert payload["verdict"] == "ok"
    assert payload["source"] == "reconstructed"
    assert payload["metrics"]["upper_block_final_to_early_ratio"] is not None


def test_health_check_reports_warn_for_loss_regression(tmp_path: Path) -> None:
    run_dir = tmp_path / "warn_run"
    _write_run_artifacts(
        run_dir,
        history_records=_history_records(initial_loss=1.0, delta=0.004),
        gradient_records=_gradient_records(clipped_steps={1}, block_base=7.0, block_slope=0.001),
        include_telemetry=True,
    )

    payload = health_check(run_dir)

    assert payload["verdict"] == "warn"
    assert "instability heuristic" in payload["summary"]


def test_health_check_reports_fail_for_recorded_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "fail_run"
    _write_run_artifacts(
        run_dir,
        history_records=_history_records(initial_loss=1.0, delta=-0.001),
        gradient_records=_gradient_records(clipped_steps=set(), block_base=9.0, block_slope=0.001),
        include_telemetry=True,
        success=False,
        error=RuntimeError("boom"),
    )

    payload = health_check(run_dir)

    assert payload["verdict"] == "fail"
    assert payload["telemetry_error"] == {"message": "boom", "type": "RuntimeError"}


def test_health_check_requires_reconstructable_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="health-check requires telemetry.json"):
        _ = health_check(run_dir)
