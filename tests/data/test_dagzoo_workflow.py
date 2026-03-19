from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from tab_foundry.data.dagzoo_handoff import (
    DAGZOO_HANDOFF_SCHEMA_NAME,
    DAGZOO_HANDOFF_SCHEMA_VERSION,
    load_dagzoo_handoff_info,
)
from tab_foundry.data.dagzoo_workflow import (
    DagzooGenerateManifestConfig,
    run_dagzoo_generate_manifest,
)
from tab_foundry.data.manifest import build_manifest

from . import manifest_and_dataset_cases as cases


def _write_handoff_manifest(
    handoff_root: Path,
    *,
    generated_dir_rel: str = "generated",
    schema_name: str = DAGZOO_HANDOFF_SCHEMA_NAME,
    schema_version: int = DAGZOO_HANDOFF_SCHEMA_VERSION,
    include_generated_dir: bool = True,
    include_generated_corpus_id: bool = True,
    run_root: str = ".",
) -> Path:
    handoff_root.mkdir(parents=True, exist_ok=True)
    identity: dict[str, Any] = {
        "source_family": "dagzoo.fixed_layout_scm",
        "generate_run_id": "run-id-123",
    }
    if include_generated_corpus_id:
        identity["generated_corpus_id"] = "corpus-id-456"
    artifacts_relative: dict[str, Any] = {
        "run_root": run_root,
    }
    if include_generated_dir:
        artifacts_relative["generated_dir"] = generated_dir_rel
    payload = {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "identity": identity,
        "artifacts_relative": artifacts_relative,
        "defaults": {
            "recommended_training_corpus": "generated",
            "recommended_training_artifact_key": "generated_dir",
            "curation_policy": "none",
        },
    }
    handoff_manifest_path = handoff_root / "handoff_manifest.json"
    handoff_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return handoff_manifest_path


def test_load_dagzoo_handoff_info_accepts_minimal_consumed_subset(tmp_path: Path) -> None:
    handoff_manifest_path = _write_handoff_manifest(
        tmp_path / "handoff",
        generated_dir_rel="nested/generated",
    )

    info = load_dagzoo_handoff_info(handoff_manifest_path)

    assert info.source_family == "dagzoo.fixed_layout_scm"
    assert info.generate_run_id == "run-id-123"
    assert info.generated_corpus_id == "corpus-id-456"
    assert info.generated_dir == (handoff_manifest_path.parent / "nested" / "generated").resolve()
    assert info.to_summary_dict()["recommended_training_artifact_key"] == "generated_dir"


@pytest.mark.parametrize(
    ("schema_name", "schema_version", "match"),
    [
        ("wrong_schema", DAGZOO_HANDOFF_SCHEMA_VERSION, "schema_name"),
        (DAGZOO_HANDOFF_SCHEMA_NAME, 99, "schema_version"),
    ],
)
def test_load_dagzoo_handoff_info_rejects_schema_drift(
    tmp_path: Path,
    schema_name: str,
    schema_version: int,
    match: str,
) -> None:
    handoff_manifest_path = _write_handoff_manifest(
        tmp_path / "handoff",
        schema_name=schema_name,
        schema_version=schema_version,
    )

    with pytest.raises(RuntimeError, match=match):
        _ = load_dagzoo_handoff_info(handoff_manifest_path)


def test_load_dagzoo_handoff_info_rejects_missing_identity_field(tmp_path: Path) -> None:
    handoff_manifest_path = _write_handoff_manifest(
        tmp_path / "handoff",
        include_generated_corpus_id=False,
    )

    with pytest.raises(RuntimeError, match="generated_corpus_id"):
        _ = load_dagzoo_handoff_info(handoff_manifest_path)


@pytest.mark.parametrize(
    ("include_generated_dir", "run_root", "match"),
    [
        (False, ".", "generated_dir"),
        (True, "not-dot", "run_root"),
    ],
)
def test_load_dagzoo_handoff_info_rejects_invalid_generated_dir_metadata(
    tmp_path: Path,
    include_generated_dir: bool,
    run_root: str,
    match: str,
) -> None:
    handoff_manifest_path = _write_handoff_manifest(
        tmp_path / "handoff",
        include_generated_dir=include_generated_dir,
        run_root=run_root,
    )

    with pytest.raises(RuntimeError, match=match):
        _ = load_dagzoo_handoff_info(handoff_manifest_path)


def test_run_dagzoo_generate_manifest_uses_handoff_generated_dir_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dagzoo_root = tmp_path / "dagzoo"
    dagzoo_root.mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")

    handoff_root = tmp_path / "handoff"
    generated_dir = handoff_root / "nested" / "generated"
    _ = cases._write_dataset(
        generated_dir / "shard_00000",
        filter_status="accepted",
        filter_accepted=True,
    )
    handoff_manifest_path = _write_handoff_manifest(
        handoff_root,
        generated_dir_rel="nested/generated",
    )
    out_manifest = tmp_path / "outputs" / "manifest.parquet"

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tab_foundry.data.dagzoo_workflow.subprocess.run", _fake_run)

    result = run_dagzoo_generate_manifest(
        DagzooGenerateManifestConfig(
            dagzoo_root=dagzoo_root,
            dagzoo_config=Path("configs/default.yaml"),
            handoff_root=handoff_root,
            out_manifest=out_manifest,
            num_datasets=32,
            seed=7,
            rows="1024",
            device="cpu",
            diagnostics=True,
            diagnostics_out_dir=tmp_path / "diag",
            filter_policy="accepted_only",
        )
    )

    assert captured["cwd"] == dagzoo_root
    assert captured["check"] is True
    assert captured["cmd"] == [
        "uv",
        "run",
        "dagzoo",
        "generate",
        "--config",
        str((dagzoo_root / "configs" / "default.yaml").resolve()),
        "--handoff-root",
        str(handoff_root.resolve()),
        "--num-datasets",
        "32",
        "--hardware-policy",
        "none",
        "--seed",
        "7",
        "--rows",
        "1024",
        "--device",
        "cpu",
        "--diagnostics",
        "--diagnostics-out-dir",
        str((tmp_path / "diag").resolve()),
    ]
    assert result.handoff.handoff_manifest_path == handoff_manifest_path.resolve()
    assert result.handoff.generated_dir == generated_dir.resolve()
    assert result.summary.total_records == 1
    persisted_summary = cases._manifest_summary_metadata(out_manifest)
    assert persisted_summary["dagzoo_handoff"]["generate_run_id"] == "run-id-123"
    assert persisted_summary["dagzoo_handoff"]["generated_corpus_id"] == "corpus-id-456"
    assert persisted_summary["dagzoo_handoff"]["generated_dir"] == str(generated_dir.resolve())
    assert persisted_summary["dagzoo_handoff"]["handoff_manifest_path"] == str(
        handoff_manifest_path.resolve()
    )


def test_run_dagzoo_generate_manifest_resolves_relative_paths_against_dagzoo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dagzoo_root = tmp_path / "dagzoo"
    dagzoo_root.mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")

    handoff_root = dagzoo_root / "handoffs" / "tab_foundry"
    generated_dir = handoff_root / "generated"
    _ = cases._write_dataset(
        generated_dir / "shard_00000",
        filter_status="accepted",
        filter_accepted=True,
    )
    handoff_manifest_path = _write_handoff_manifest(handoff_root, generated_dir_rel="generated")
    out_manifest = tmp_path / "outputs" / "manifest.parquet"

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tab_foundry.data.dagzoo_workflow.subprocess.run", _fake_run)

    result = run_dagzoo_generate_manifest(
        DagzooGenerateManifestConfig(
            dagzoo_root=dagzoo_root,
            dagzoo_config=Path("configs/default.yaml"),
            handoff_root=Path("handoffs/tab_foundry"),
            out_manifest=out_manifest,
            num_datasets=4,
            diagnostics=True,
            diagnostics_out_dir=Path("diagnostics/coverage"),
        )
    )

    assert captured["cwd"] == dagzoo_root
    assert captured["check"] is True
    assert captured["cmd"] == [
        "uv",
        "run",
        "dagzoo",
        "generate",
        "--config",
        str((dagzoo_root / "configs" / "default.yaml").resolve()),
        "--handoff-root",
        str(handoff_root.resolve()),
        "--num-datasets",
        "4",
        "--hardware-policy",
        "none",
        "--diagnostics",
        "--diagnostics-out-dir",
        str((dagzoo_root / "diagnostics" / "coverage").resolve()),
    ]
    assert result.handoff.handoff_manifest_path == handoff_manifest_path.resolve()
    assert result.handoff.generated_dir == generated_dir.resolve()


def test_run_dagzoo_generate_manifest_rejects_missing_handoff_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dagzoo_root = tmp_path / "dagzoo"
    dagzoo_root.mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tab_foundry.data.dagzoo_workflow.subprocess.run", _fake_run)

    with pytest.raises(RuntimeError, match="handoff manifest not found"):
        _ = run_dagzoo_generate_manifest(
            DagzooGenerateManifestConfig(
                dagzoo_root=dagzoo_root,
                dagzoo_config=Path("configs/default.yaml"),
                handoff_root=tmp_path / "handoff",
                out_manifest=tmp_path / "manifest.parquet",
            )
        )


def test_run_dagzoo_generate_manifest_rejects_non_directory_dagzoo_root(tmp_path: Path) -> None:
    dagzoo_root = tmp_path / "dagzoo.txt"
    dagzoo_root.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dagzoo root must be a directory"):
        _ = run_dagzoo_generate_manifest(
            DagzooGenerateManifestConfig(
                dagzoo_root=dagzoo_root,
                dagzoo_config=Path("configs/default.yaml"),
                handoff_root=tmp_path / "handoff",
                out_manifest=tmp_path / "manifest.parquet",
            )
        )


def test_build_manifest_omits_dagzoo_handoff_summary_when_not_supplied(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = cases._write_dataset(shard_dir, filter_status="accepted", filter_accepted=True)

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)

    persisted_summary = cases._manifest_summary_metadata(manifest_path)
    assert "dagzoo_handoff" not in persisted_summary
