from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tab_foundry.provenance as provenance_module


def test_resolve_current_producer_clean_git_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(provenance_module, "_package_version", lambda: "0.6.8")
    monkeypatch.setattr(provenance_module, "_git_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        provenance_module,
        "_git_snapshot",
        lambda _repo_root: ("abc123", False, None),
    )

    producer = provenance_module.resolve_current_producer(
        artifact_dir=tmp_path / "artifacts",
        patch_path_mode="absolute",
    )

    assert producer.to_dict() == {
        "name": "tab-foundry",
        "version": "0.6.8",
        "git_sha": "abc123",
        "git_dirty": False,
    }
    assert not (tmp_path / "artifacts" / provenance_module.SOURCE_PATCH_FILENAME).exists()


def test_resolve_current_producer_dirty_git_state_writes_patch_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_bytes = b"diff --git a/foo b/foo\n"
    artifact_dir = tmp_path / "artifacts"
    patch_path = artifact_dir / provenance_module.SOURCE_PATCH_FILENAME
    monkeypatch.setattr(provenance_module, "_package_version", lambda: "0.6.8")
    monkeypatch.setattr(provenance_module, "_git_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        provenance_module,
        "_git_snapshot",
        lambda _repo_root: ("def456", True, patch_bytes),
    )

    producer = provenance_module.resolve_current_producer(
        artifact_dir=artifact_dir,
        patch_path_mode="absolute",
    )

    assert patch_path.read_bytes() == patch_bytes
    assert producer.to_dict() == {
        "name": "tab-foundry",
        "version": "0.6.8",
        "git_sha": "def456",
        "git_dirty": True,
        "source_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "source_patch_path": str(patch_path.resolve()),
    }


def test_resolve_current_producer_git_unavailable_returns_nullable_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance_module, "_package_version", lambda: "0.6.8")
    monkeypatch.setattr(provenance_module, "_git_repo_root", lambda: None)

    producer = provenance_module.resolve_current_producer()

    assert producer.to_dict() == {
        "name": "tab-foundry",
        "version": "0.6.8",
        "git_sha": None,
        "git_dirty": None,
    }


def test_resolve_current_producer_rejects_incomplete_repo_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(provenance_module, "_package_version", lambda: "0.6.8")
    monkeypatch.setattr(provenance_module, "_git_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        provenance_module,
        "_git_snapshot",
        lambda _repo_root: ("abc123", None, None),
    )

    with pytest.raises(RuntimeError, match="failed to resolve exact git provenance"):
        provenance_module.resolve_current_producer(artifact_dir=tmp_path / "artifacts")
