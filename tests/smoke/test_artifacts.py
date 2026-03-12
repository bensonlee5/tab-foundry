from __future__ import annotations

import json
from pathlib import Path

from tab_foundry.bench.artifacts import checkpoint_snapshots_from_history


def test_checkpoint_snapshots_from_history_prefers_train_elapsed_seconds(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
    history_path = tmp_path / "train_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "step": 25,
                "stage": "stage1",
                "train_loss": 0.5,
                "lr": 1.0e-3,
                "elapsed_seconds": 9.0,
                "train_elapsed_seconds": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = checkpoint_snapshots_from_history(history_path, checkpoint_dir)

    assert snapshots[0]["elapsed_seconds"] == 3.0
    assert snapshots[0]["train_elapsed_seconds"] == 3.0
