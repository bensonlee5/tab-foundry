from __future__ import annotations

from pathlib import Path

import tab_foundry.bench.iris as iris_module


def test_iris_main_prints_ranked_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    resolved_checkpoint = checkpoint_path.resolve()

    monkeypatch.setattr(
        iris_module,
        "evaluate_iris_checkpoint",
        lambda checkpoint_path, device, seeds: iris_module.IrisEvalSummary(
            checkpoint=checkpoint_path.resolve(),
            results={
                "tab_foundry": [0.82, 0.80],
                "LogReg": [0.70, 0.69],
                "RF": [0.91, 0.90],
            },
        ),
    )

    exit_code = iris_module.main(
        ["--checkpoint", str(checkpoint_path), "--device", "cpu", "--seeds", "4"]
    )

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert exit_code == 0
    assert lines[0] == f"Iris evaluation for checkpoint={resolved_checkpoint}"
    assert lines[1] == "Method           ROC AUC      Std"
    assert lines[2] == "-" * 33
    assert lines[3].startswith("RF")
    assert lines[4].startswith("tab_foundry")
    assert lines[5].startswith("LogReg")
