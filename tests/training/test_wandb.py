from __future__ import annotations

import json
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


class _FakePublicSummary(dict):
    def __init__(self) -> None:
        super().__init__()
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1


class _FakePublicRun:
    def __init__(self) -> None:
        self.summary = _FakePublicSummary()


class _FakeApi:
    def __init__(self, run: _FakePublicRun) -> None:
        self._run = run
        self.requested_path: str | None = None

    def run(self, path: str) -> _FakePublicRun:
        self.requested_path = path
        return self._run


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


def test_training_surface_wandb_summary_payload_keeps_resolved_surface_fields() -> None:
    payload = wandb_module.training_surface_wandb_summary_payload(
        {
            "labels": {"model": "row_cls_pool", "training": "training_default"},
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "row_cls_pool",
                "stage_label": "row_cls_pool",
                "benchmark_profile": "row_cls_pool",
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_base": 2,
                "module_selection": {
                    "row_pool": "row_cls",
                    "context_encoder": "plain",
                },
                "module_hyperparameters": {
                    "row_pool": {"cls_tokens": 4, "n_heads": 8},
                },
            },
        }
    )

    flattened = {}
    run = _FakeRun()
    wandb_module.update_wandb_summary(run, payload)
    flattened.update(run.summary)

    assert flattened["surface/labels/model"] == "row_cls_pool"
    assert flattened["surface/model/arch"] == "tabfoundry_staged"
    assert flattened["surface/model/module_selection/row_pool"] == "row_cls"
    assert flattened["surface/model/module_hyperparameters/row_pool/cls_tokens"] == 4


def test_wandb_identity_payload_reads_string_public_path(
    tmp_path: Path,
) -> None:
    payload = wandb_module.wandb_identity_payload(
        SimpleNamespace(
            path="test-entity/test-project/run-123",
            settings=SimpleNamespace(mode="online"),
        ),
        cfg=_wandb_cfg(tmp_path),
    )

    assert payload == {
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-123",
        "run_name": f"run-{tmp_path.name}",
        "mode": "online",
    }


def test_posthoc_update_wandb_summary_updates_online_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-123",
                    "run_name": "demo-run",
                    "mode": "online",
                }
            }
        ),
        encoding="utf-8",
    )
    fake_public_run = _FakePublicRun()
    fake_api = _FakeApi(fake_public_run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: fake_api))

    updated = wandb_module.posthoc_update_wandb_summary(
        telemetry_path=telemetry_path,
        payload={"benchmark": {"model_size": {"total_params": 1234}}},
    )

    assert updated is True
    assert fake_api.requested_path == "test-entity/test-project/run-123"
    assert fake_public_run.summary["benchmark/model_size/total_params"] == 1234
    assert fake_public_run.summary.update_calls == 1


def test_posthoc_update_wandb_summary_updates_run_without_entity_when_project_and_run_id_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "project": "test-project",
                    "run_id": "run-123",
                    "run_name": "demo-run",
                    "mode": "online",
                }
            }
        ),
        encoding="utf-8",
    )
    fake_public_run = _FakePublicRun()
    fake_api = _FakeApi(fake_public_run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: fake_api))

    updated = wandb_module.posthoc_update_wandb_summary(
        telemetry_path=telemetry_path,
        payload={"benchmark": {"model_size": {"total_params": 1234}}},
    )

    assert updated is True
    assert fake_api.requested_path == "test-project/run-123"
    assert fake_public_run.summary["benchmark/model_size/total_params"] == 1234
    assert fake_public_run.summary.update_calls == 1


def test_posthoc_update_wandb_summary_skips_offline_or_incomplete_metadata(
    tmp_path: Path,
) -> None:
    offline_telemetry_path = tmp_path / "offline_telemetry.json"
    offline_telemetry_path.write_text(
        json.dumps({"wandb": {"project": "test-project", "run_name": "demo-run", "mode": "offline"}}),
        encoding="utf-8",
    )
    missing_telemetry_path = tmp_path / "missing_telemetry.json"
    missing_telemetry_path.write_text(
        json.dumps({"wandb": {"project": "test-project", "mode": "online"}}),
        encoding="utf-8",
    )

    assert (
        wandb_module.posthoc_update_wandb_summary(
            telemetry_path=offline_telemetry_path,
            payload={"benchmark": {"tab_foundry": {"final_roc_auc": 0.8}}},
        )
        is False
    )
    assert (
        wandb_module.posthoc_update_wandb_summary(
            telemetry_path=missing_telemetry_path,
            payload={"benchmark": {"tab_foundry": {"final_roc_auc": 0.8}}},
        )
        is False
    )


def test_posthoc_update_wandb_summary_returns_false_when_api_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-123",
                    "run_name": "demo-run",
                    "mode": "online",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda: (_ for _ in ()).throw(RuntimeError("missing auth"))),
    )

    assert (
        wandb_module.posthoc_update_wandb_summary(
            telemetry_path=telemetry_path,
            payload={"benchmark": {"tab_foundry": {"final_roc_auc": 0.8}}},
        )
        is False
    )
