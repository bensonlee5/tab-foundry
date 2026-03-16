from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys

from omegaconf import OmegaConf
import pytest

from tab_foundry.training import wandb as wandb_module


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logged.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


def _wandb_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "logging": {
                "use_wandb": True,
                "project": "test-project",
                "run_name": f"run-{tmp_path.name}",
            }
        }
    )


def test_init_wandb_run_uses_api_key_file_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _wandb_cfg(tmp_path)
    key_path = tmp_path / "wandb_api_key.txt"
    key_path.write_text("secret-key\n", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def _fake_init(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return dict(kwargs)

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY_FILE", str(key_path))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=_fake_init))

    run = wandb_module.init_wandb_run(cfg, enabled=True)

    assert run is not None
    assert calls[0]["mode"] == "online"
    assert os.environ["WANDB_API_KEY"] == "secret-key"


def test_init_wandb_run_falls_back_to_offline_without_any_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _wandb_cfg(tmp_path)

    calls: list[dict[str, object]] = []

    def _fake_init(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return dict(kwargs)

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=_fake_init))

    run = wandb_module.init_wandb_run(cfg, enabled=True)

    assert run is not None
    assert calls[0]["mode"] == "offline"


def test_wandb_helpers_normalize_metrics_and_summary() -> None:
    run = _FakeRun()

    wandb_module.log_wandb_metrics(
        run,
        {
            "train/loss": 1.25,
            "train/stage": "stage1",
            "train/flag": True,
            "train/skip": None,
            "train/bad": float("inf"),
        },
        step=3,
    )

    assert run.logged == [
        (
            {
                "train/loss": 1.25,
                "train/stage": "stage1",
                "train/flag": True,
            },
            3,
        )
    ]

    wandb_module.update_wandb_summary(
        run,
        {
            "metrics": {"final_loss": 0.5, "bad": float("nan")},
            "run": {"output_dir": Path("/tmp/demo"), "global_step": 3},
            "error": {"message": "boom"},
            "skip": None,
        },
    )

    assert run.summary["metrics/final_loss"] == 0.5
    assert run.summary["run/output_dir"] == str(Path("/tmp/demo").resolve())
    assert run.summary["run/global_step"] == 3
    assert run.summary["error/message"] == "boom"
    assert "metrics/bad" not in run.summary
    wandb_module.finish_wandb_run(run)
    assert run.finished is True
