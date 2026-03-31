"""Shared benchmark and smoke artifact helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def write_json(path: Path, payload: Any) -> Path:
    """Write one JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write newline-delimited JSON records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise RuntimeError(f"JSONL record must be an object: path={path}")
            records.append(payload)
    return records


def ensure_finite_metrics(
    metrics: Mapping[str, float | None],
    *,
    context: str,
) -> dict[str, float]:
    """Reject missing or non-finite metric payloads and return normalized floats."""

    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            raise RuntimeError(f"{context} metric must be finite: key={key}, value=None")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise RuntimeError(f"{context} metric must be finite: key={key}, value={value_f!r}")
        normalized[str(key)] = value_f
    return normalized


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty training-history JSONL file."""

    records = load_jsonl(path)
    if not records:
        raise RuntimeError(f"history file contains no records: path={path}")
    return records


def resolve_train_elapsed_seconds(record: Mapping[str, Any], *, context: str) -> float:
    """Resolve a training-time field from history or telemetry payloads."""

    if "train_elapsed_seconds" in record:
        elapsed_raw = record["train_elapsed_seconds"]
    elif "elapsed_seconds" in record:
        elapsed_raw = record["elapsed_seconds"]
    else:
        keys = ", ".join(sorted(str(key) for key in record.keys()))
        raise RuntimeError(
            f"{context} record is missing elapsed time; expected train_elapsed_seconds or "
            f"elapsed_seconds, keys=[{keys}]"
        )
    elapsed_seconds = float(elapsed_raw)
    if not math.isfinite(elapsed_seconds):
        raise RuntimeError(f"{context} record has non-finite elapsed time: {elapsed_seconds!r}")
    return max(0.0, elapsed_seconds)


def plot_loss_curve(
    history_path: Path,
    out_path: Path,
    *,
    title: str = "tab-foundry loss curve",
) -> Path:
    """Render a train/validation loss curve from JSONL history."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_history(history_path)
    steps = [int(record["step"]) for record in records]
    train_losses = [float(record["train_loss"]) for record in records]
    val_points = [
        (int(record["step"]), float(record["val_loss"]))
        for record in records
        if record.get("val_loss") is not None
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, label="train_loss", color="#1f77b4", linewidth=2.0)
    if val_points:
        ax.plot(
            [step for step, _ in val_points],
            [value for _, value in val_points],
            label="val_loss",
            color="#d62728",
            linewidth=2.0,
        )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    return out_path


def checkpoint_snapshots_from_history(history_path: Path, checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Resolve step checkpoints and their elapsed training times."""

    history_records = load_history(history_path)
    step_times = {
        int(record["step"]): resolve_train_elapsed_seconds(
            record,
            context=f"history step={record['step']}",
        )
        for record in history_records
    }
    snapshots: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt")):
        try:
            step = int(checkpoint.stem.removeprefix("step_"))
        except ValueError as exc:
            raise RuntimeError(f"invalid step checkpoint name: {checkpoint.name}") from exc
        elapsed_seconds = step_times.get(step)
        if elapsed_seconds is None:
            raise RuntimeError(f"missing history entry for snapshot checkpoint step={step}")
        snapshots.append(
            {
                "step": step,
                "path": str(checkpoint.resolve()),
                "elapsed_seconds": max(0.0, float(elapsed_seconds)),
                "train_elapsed_seconds": max(0.0, float(elapsed_seconds)),
            }
        )
    if not snapshots:
        latest_checkpoint = checkpoint_dir / "latest.pt"
        if not latest_checkpoint.exists():
            stage_latest_candidates = sorted(checkpoint_dir.glob("latest_*.pt"))
            if stage_latest_candidates:
                latest_checkpoint = max(
                    stage_latest_candidates,
                    key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
                )
            else:
                latest_checkpoint = checkpoint_dir / "best.pt"
        if not latest_checkpoint.exists():
            raise RuntimeError(f"no step checkpoints found under {checkpoint_dir}")

        latest_record = max(history_records, key=lambda record: int(record["step"]))
        latest_step = int(latest_record["step"])
        latest_elapsed_seconds = resolve_train_elapsed_seconds(
            latest_record,
            context=f"history step={latest_step}",
        )
        snapshots.append(
            {
                "step": latest_step,
                "path": str(latest_checkpoint.resolve()),
                "elapsed_seconds": max(0.0, float(latest_elapsed_seconds)),
                "train_elapsed_seconds": max(0.0, float(latest_elapsed_seconds)),
            }
        )
    return snapshots
