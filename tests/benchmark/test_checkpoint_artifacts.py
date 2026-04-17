from __future__ import annotations

import json
from pathlib import Path

import pytest

import tab_foundry.bench.checkpoint_artifacts as checkpoint_artifacts_module
import tab_foundry.bench.run_registration as run_registration_module
from tab_foundry.benchmark_registry import default_benchmark_run_registry_path, load_benchmark_run_registry
import tab_foundry.training.wandb as wandb_module
from tests.support.benchmark_run_registry_cases import _prepare_run


def test_publish_checkpoint_artifact_returns_waited_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}

    class FakeArtifact:
        def __init__(self, name: str, type: str, metadata=None, **_: object) -> None:
            captured["artifact_name"] = name
            captured["artifact_type"] = type
            captured["artifact_metadata"] = metadata
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, path: str, name: str | None = None) -> None:
            self.files.append((path, name))
            captured["files"] = list(self.files)

    class FakeLoggedArtifact:
        name = "benchmark-checkpoint-run_001:v7"

        def wait(self, timeout: int | None = None):
            captured["wait_timeout"] = timeout
            return self

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact, aliases=None):
            captured["aliases"] = list(aliases or [])
            captured["logged_artifact"] = artifact
            return FakeLoggedArtifact()

        def finish(self) -> None:
            captured["finished"] = True

    class FakeWandb:
        Artifact = FakeArtifact

        @staticmethod
        def init(**kwargs):
            captured["init_kwargs"] = kwargs
            return FakeRun()

    monkeypatch.setattr(wandb_module, "_require_wandb_sdk", lambda: FakeWandb)

    published = wandb_module.publish_checkpoint_artifact(
        checkpoint_path=checkpoint_path,
        artifact_name="benchmark-checkpoint-run_001",
        entity="bensonlee55-none",
        project="tab-foundry",
        run_id="bbmhj6c4",
        run_name="run_001",
        metadata={"benchmark_run_id": "run_001", "queue_order": 2},
        aliases=["best"],
    )

    assert published.artifact_ref == "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_001:v7"
    assert published.local_path == checkpoint_path.resolve()
    assert captured["init_kwargs"] == {
        "entity": "bensonlee55-none",
        "project": "tab-foundry",
        "id": "bbmhj6c4",
        "resume": "allow",
        "job_type": "benchmark-checkpoint-publish",
        "mode": "online",
        "name": "run_001",
    }
    assert captured["aliases"] == ["best"]
    assert captured["files"] == [(str(checkpoint_path.resolve()), "best.pt")]
    assert captured["finished"] is True


def test_download_checkpoint_artifact_downloads_best_pt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeApiArtifact:
        def download(self, root: str, **_: object) -> str:
            resolved_root = Path(root).resolve()
            resolved_root.mkdir(parents=True, exist_ok=True)
            (resolved_root / "best.pt").write_bytes(b"checkpoint")
            captured["download_root"] = resolved_root
            return str(resolved_root)

    class FakeApi:
        def artifact(self, name: str):
            captured["artifact_ref"] = name
            return FakeApiArtifact()

    class FakeWandb:
        @staticmethod
        def Api():
            return FakeApi()

    monkeypatch.setattr(wandb_module, "_require_wandb_sdk", lambda: FakeWandb)

    downloaded = wandb_module.download_checkpoint_artifact(
        artifact_ref="bensonlee55-none/tab-foundry/benchmark-checkpoint-run_001:v7",
        out_dir=tmp_path / "cache",
    )

    assert downloaded.local_path == (tmp_path / "cache" / "best.pt").resolve()
    assert captured["artifact_ref"] == "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_001:v7"
    assert downloaded.local_path.read_bytes() == b"checkpoint"


def test_register_benchmark_run_attaches_remote_artifact_when_online_wandb_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    run_dir, summary_path = _prepare_run(
        repo_root,
        run_name="wandb_online",
        telemetry_extra_payload={
            "wandb": {
                "entity": "bensonlee55-none",
                "project": "tab-foundry",
                "run_id": "bbmhj6c4",
                "run_name": "wandb_online",
                "mode": "online",
            }
        },
    )
    monkeypatch.setattr(run_registration_module, "repo_root", lambda: repo_root)
    monkeypatch.setattr(
        run_registration_module,
        "publish_benchmark_checkpoint_artifact",
        lambda **_: checkpoint_artifacts_module.PublishedBenchmarkCheckpoint(
            run_id="run_001",
            checkpoint_path=run_dir / "checkpoints" / "best.pt",
            artifact_ref="bensonlee55-none/tab-foundry/benchmark-checkpoint-run_001:v7",
            wandb={
                "entity": "bensonlee55-none",
                "project": "tab-foundry",
                "run_id": "bbmhj6c4",
                "run_name": "wandb_online",
            },
        ),
    )

    result = run_registration_module.register_benchmark_run(
        run_id="run_001",
        track="binary_ladder",
        experiment="cls_benchmark_staged",
        config_profile="cls_benchmark_staged",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        decision="keep",
        conclusion="online wandb run",
        registry_path=registry_path,
    )

    assert result["run"]["wandb"] == {
        "entity": "bensonlee55-none",
        "project": "tab-foundry",
        "run_id": "bbmhj6c4",
        "run_name": "wandb_online",
    }
    assert result["run"]["remote_artifacts"] == {
        "best_checkpoint_wandb_artifact": "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_001:v7"
    }


def test_register_benchmark_run_suppresses_reused_artifact_wandb_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    run_dir, summary_path = _prepare_run(
        repo_root,
        run_name="wandb_reused",
        telemetry_extra_payload={
            "wandb": {
                "entity": "bensonlee55-none",
                "project": "tab-foundry",
                "run_id": "bbmhj6c4",
                "run_name": "wandb_reused",
                "mode": "online",
            }
        },
    )
    monkeypatch.setattr(run_registration_module, "repo_root", lambda: repo_root)
    monkeypatch.setattr(
        run_registration_module,
        "publish_benchmark_checkpoint_artifact",
        lambda **_: (_ for _ in ()).throw(AssertionError("reused rows should skip checkpoint publication")),
    )

    result = run_registration_module.register_benchmark_run(
        run_id="run_001",
        track="binary_ladder",
        experiment="cls_benchmark_staged",
        config_profile="cls_benchmark_staged",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        decision="keep",
        conclusion="reused artifact row",
        registry_path=registry_path,
        suppress_reused_artifact_wandb=True,
    )

    assert result["run"]["wandb"] is None
    assert result["run"].get("remote_artifacts") is None


def test_backfill_benchmark_checkpoint_artifact_updates_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    run_dir, summary_path = _prepare_run(repo_root, run_name="historical")
    monkeypatch.setattr(run_registration_module, "repo_root", lambda: repo_root)
    _ = run_registration_module.register_benchmark_run(
        run_id="run_002",
        track="binary_ladder",
        experiment="cls_benchmark_staged",
        config_profile="cls_benchmark_staged",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        decision="keep",
        conclusion="historical backfill",
        registry_path=registry_path,
    )
    telemetry_path = run_dir / "telemetry.json"
    telemetry_payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry_payload["wandb"] = {
        "entity": "bensonlee55-none",
        "project": "tab-foundry",
        "run_id": "bbmhj6c4",
        "run_name": "historical",
        "mode": "online",
    }
    telemetry_path.write_text(json.dumps(telemetry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint_artifacts_module, "repo_root", lambda: repo_root)
    monkeypatch.setattr(
        checkpoint_artifacts_module,
        "publish_checkpoint_artifact",
        lambda **_: wandb_module.WandbArtifactReference(
            artifact_name="benchmark-checkpoint-run_002",
            artifact_ref="bensonlee55-none/tab-foundry/benchmark-checkpoint-run_002:v4",
            local_path=(run_dir / "checkpoints" / "best.pt").resolve(),
        ),
    )

    published = checkpoint_artifacts_module.backfill_benchmark_checkpoint_artifact(
        run_id="run_002",
        registry_path=registry_path,
    )

    assert published.artifact_ref == "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_002:v4"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_payload["runs"]["run_002"]["wandb"] == {
        "entity": "bensonlee55-none",
        "project": "tab-foundry",
        "run_id": "bbmhj6c4",
        "run_name": "historical",
    }
    assert registry_payload["runs"]["run_002"]["remote_artifacts"] == {
        "best_checkpoint_wandb_artifact": "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_002:v4"
    }


def test_resolve_benchmark_checkpoint_downloads_remote_artifact_when_local_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    run_dir, summary_path = _prepare_run(repo_root, run_name="remote_only")
    monkeypatch.setattr(run_registration_module, "repo_root", lambda: repo_root)
    _ = run_registration_module.register_benchmark_run(
        run_id="run_003",
        track="binary_ladder",
        experiment="cls_benchmark_staged",
        config_profile="cls_benchmark_staged",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        decision="keep",
        conclusion="remote recovery",
        registry_path=registry_path,
    )
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["runs"]["run_003"]["remote_artifacts"] = {
        "best_checkpoint_wandb_artifact": "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_003:v2"
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "checkpoints" / "best.pt").unlink()
    monkeypatch.setattr(checkpoint_artifacts_module, "repo_root", lambda: repo_root)

    def _fake_download_checkpoint_artifact(*, artifact_ref: str, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / "best.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        return wandb_module.WandbArtifactReference(
            artifact_name="benchmark-checkpoint-run_003",
            artifact_ref=artifact_ref,
            local_path=checkpoint_path,
        )

    monkeypatch.setattr(
        checkpoint_artifacts_module,
        "download_checkpoint_artifact",
        _fake_download_checkpoint_artifact,
    )

    resolved = checkpoint_artifacts_module.resolve_benchmark_checkpoint(
        run_id="run_003",
        registry_path=registry_path,
        allow_remote=True,
        cache_root=tmp_path / "cache",
    )

    assert resolved.source == "wandb_artifact"
    assert resolved.artifact_ref == "bensonlee55-none/tab-foundry/benchmark-checkpoint-run_003:v2"
    assert resolved.checkpoint_path == (tmp_path / "cache" / "run_003" / "best.pt").resolve()
    assert resolved.checkpoint_path.read_bytes() == b"checkpoint"


def test_checked_in_live_candidate_has_remote_checkpoint_artifact_metadata() -> None:
    registry = load_benchmark_run_registry(default_benchmark_run_registry_path())
    entry = registry["runs"][
        "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1"
    ]

    assert entry["wandb"] == {
        "entity": "bensonlee55-none",
        "project": "tab-foundry",
        "run_id": "bbmhj6c4",
        "run_name": "sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1",
    }
    assert (
        entry["remote_artifacts"]["best_checkpoint_wandb_artifact"]
        == "bensonlee55-none/tab-foundry/benchmark-checkpoint-sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1:v0"
    )


def test_checked_in_muon_anchor_has_remote_checkpoint_artifact_metadata() -> None:
    registry = load_benchmark_run_registry(default_benchmark_run_registry_path())
    entry = registry["runs"][
        "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
    ]

    assert (
        entry["artifacts"]["best_checkpoint_path"]
        == "outputs/staged_ladder/research/tf_rd_009_muon_width_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_v1/"
        "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1/train/checkpoints/best.pt"
    )
    assert (
        entry["remote_artifacts"]["best_checkpoint_wandb_artifact"]
        == "bensonlee55-none/tab-foundry/benchmark-checkpoint-sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1:v0"
    )
