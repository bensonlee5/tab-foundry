"""Reporting helpers for the Iris smoke harness."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from tab_foundry.bench.artifacts import ensure_finite_metrics
from tab_foundry.bench.iris import IrisEvalSummary


def iris_benchmark_payload(summary: IrisEvalSummary) -> dict[str, Any]:
    means = {name: float(np.mean(values)) for name, values in summary.results.items()}
    stddevs = {name: float(np.std(values)) for name, values in summary.results.items()}
    ensure_finite_metrics(means, context="iris benchmark mean")
    ensure_finite_metrics(stddevs, context="iris benchmark std")
    return {
        "checkpoint": str(summary.checkpoint),
        "means": means,
        "stddevs": stddevs,
        "raw": {name: [float(value) for value in values] for name, values in summary.results.items()},
    }


def format_float(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def write_summary_markdown(path: Path, telemetry: dict[str, Any]) -> Path:
    config = telemetry["config"]
    manifest = telemetry["manifest"]
    train_metrics = telemetry["train_metrics"]
    eval_metrics = telemetry["eval_metrics"]
    benchmark = telemetry["iris_benchmark"]
    timings = telemetry["timings_seconds"]
    artifacts = telemetry["artifacts"]
    checkpoint_snapshots = telemetry["checkpoint_snapshots"]

    lines: list[str] = [
        "# Iris Smoke Report",
        "",
        f"- Generated at: `{telemetry.get('generated_at_utc', '-')}`",
        f"- Device: `{config['device']}`",
        f"- Task count: `{config['final_num_tasks']}`",
        f"- Task count attempts: `{', '.join(str(value) for value in config['task_count_attempts'])}`",
        f"- Manifest splits: `train={manifest['train_records']}, val={manifest['val_records']}, test={manifest['test_records']}`",
        f"- Best checkpoint: `{artifacts['best_checkpoint']}`",
        "",
        "## Timings",
        "",
        "| Stage | Seconds |",
        "|---|---:|",
    ]
    for key in (
        "generate_iris_tasks",
        "build_manifest",
        "train",
        "eval",
        "iris_benchmark",
        "total",
    ):
        lines.append(f"| {key} | {format_float(timings.get(key), 3)} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Category | Key | Value |",
            "|---|---|---:|",
        ]
    )
    for key, value in sorted(train_metrics.items()):
        lines.append(f"| train | {key} | {format_float(value, 6)} |")
    for key, value in sorted(eval_metrics.items()):
        lines.append(f"| eval | {key} | {format_float(value, 6)} |")

    lines.extend(
        [
            "",
            "## Iris Benchmark",
            "",
            "| Method | Mean ROC AUC | Std |",
            "|---|---:|---:|",
        ]
    )
    for name in sorted(benchmark["means"]):
        lines.append(
            "| "
            f"{name} | "
            f"{format_float(benchmark['means'][name], 6)} | "
            f"{format_float(benchmark['stddevs'][name], 6)} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| Step | Train elapsed (s) |",
            "|---|---:|",
        ]
    )
    for snapshot in checkpoint_snapshots:
        lines.append(
            f"| {int(snapshot['step'])} | {format_float(snapshot['train_elapsed_seconds'], 3)} |"
        )

    lines.extend(["", "## Artifacts", ""])
    for key in (
        "generated_dir",
        "manifest_path",
        "train_output_dir",
        "train_history_jsonl",
        "loss_curve_png",
        "telemetry_json",
        "summary_md",
    ):
        if key in artifacts:
            lines.append(f"- `{artifacts[key]}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
