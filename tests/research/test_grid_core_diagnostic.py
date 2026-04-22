from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from tab_foundry.research import grid_core_diagnostic as diagnostic


def test_contiguous_layer_chunks_enumerates_all_four_layer_chunks() -> None:
    chunks = diagnostic.contiguous_layer_chunks(4)

    assert len(chunks) == 10
    assert set(chunks) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 2),
        (1, 3),
        (0, 3),
    }


def test_contiguous_layer_chunks_middle_scope_excludes_boundaries() -> None:
    assert diagnostic.contiguous_layer_chunks(4, scope="middle") == [
        (1, 1),
        (1, 2),
        (2, 2),
    ]


def test_grid_core_perturbation_diagnostic_writes_json_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    manifest = tmp_path / "manifest.parquet"
    manifest.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(
        diagnostic,
        "_load_benchmark_surface",
        lambda _manifest_path: ({}, "supervised_classification", False, manifest),
    )

    def _fake_evaluate_checkpoint_metrics(
        *,
        checkpoint_path: Path,
        datasets: Mapping[str, Any],
        device: str,
        allow_missing_values: bool,
        intervention: Mapping[str, int | str] | None,
    ) -> dict[str, Any]:
        del checkpoint_path, datasets, device, allow_missing_values
        if intervention is None:
            log_loss = 1.0
            brier = 0.25
            roc_auc = 0.7
            elapsed = 10.0
        else:
            start = int(intervention["start_layer"])
            end = int(intervention["end_layer"])
            mode = str(intervention["mode"])
            chunk_width = end - start + 1
            if mode == "ablate_chunk":
                log_loss = 1.0 + 0.01 * chunk_width
            else:
                log_loss = 1.0 - 0.01 * chunk_width - 0.001 * int(intervention["repeat_count"])
            brier = 0.25 + (log_loss - 1.0)
            roc_auc = 0.7 - (log_loss - 1.0)
            elapsed = 11.0 + float(chunk_width)
        return {
            "metrics": {
                "log_loss": log_loss,
                "brier_score": brier,
                "roc_auc": roc_auc,
            },
            "parameter_count": 1234,
            "elapsed_seconds": elapsed,
        }

    monkeypatch.setattr(
        diagnostic,
        "_evaluate_checkpoint_metrics",
        _fake_evaluate_checkpoint_metrics,
    )

    payload = diagnostic.run_grid_core_perturbation_diagnostic(
        checkpoint_path=checkpoint,
        benchmark_manifest_path=manifest,
        out_dir=tmp_path / "diagnostic",
        device="cpu",
        repeat_counts=(2, 4),
        chunk_scope="all",
        layer_count=2,
    )

    assert payload["layer_count"] == 2
    assert len(payload["chunks"]) == 3
    assert payload["repeat_counts"] == [2, 4]
    assert len(payload["candidates"]) == 9
    assert payload["baseline"]["parameter_count"] == 1234
    assert payload["rankings"]["repeat_by_log_loss_delta"][0] == "repeat_chunk_r4_0_1"
    assert payload["chunk_decisions"][0]["decision_label"] == "recurrence_promising"
    assert payload["chunk_decisions"][0]["best_repeat_count"] == 4

    json_path = Path(str(payload["artifacts"]["json"]))
    markdown_path = Path(str(payload["artifacts"]["markdown"]))
    assert json_path.exists()
    assert markdown_path.exists()
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["baseline"]["parameter_count"] == 1234
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Grid-Core Perturbation Diagnostic" in markdown
    assert "| `repeat_chunk` |" in markdown
