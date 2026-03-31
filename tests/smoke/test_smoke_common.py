from __future__ import annotations

from pathlib import Path

from tab_foundry.bench.smoke_common import (
    build_cls_smoke_eval_config,
    build_cls_smoke_train_config,
    build_manifest_payload,
)
from tab_realdata_hub.manifest import ManifestSummary


def test_build_manifest_payload_matches_existing_telemetry_shape() -> None:
    summary = ManifestSummary(
        out_path=Path("/tmp/manifest.parquet"),
        filter_policy="accepted_only",
        discovered_records=10,
        excluded_records=2,
        total_records=8,
        train_records=5,
        val_records=1,
        test_records=2,
        warnings=["warning-a"],
    )

    assert build_manifest_payload(summary) == {
        "discovered_records": 10,
        "excluded_records": 2,
        "total_records": 8,
        "train_records": 5,
        "val_records": 1,
        "test_records": 2,
        "filter_policy": "accepted_only",
        "warnings": ["warning-a"],
    }


def test_build_cls_smoke_train_config_omits_removed_row_caps(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    cfg = build_cls_smoke_train_config(
        manifest_path=manifest_path,
        output_dir=tmp_path / "train_outputs",
        history_path=tmp_path / "history.jsonl",
        device="cpu",
        checkpoint_every=25,
        schedule_stages=[{"name": "stage1", "steps": 250, "lr_max": 8.0e-4}],
    )

    assert str(cfg.data.manifest_path) == str(manifest_path)
    assert "train_row_cap" not in cfg.data
    assert "test_row_cap" not in cfg.data
    assert str(cfg.runtime.device) == "cpu"
    assert int(cfg.runtime.checkpoint_every) == 25
    assert int(cfg.model.feature_group_size) == 1
    assert int(cfg.schedule.stages[0]["steps"]) == 250


def test_build_cls_smoke_configs_share_row_cap_free_surface_for_iris(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    train_cfg = build_cls_smoke_train_config(
        manifest_path=manifest_path,
        output_dir=tmp_path / "train_outputs",
        history_path=tmp_path / "history.jsonl",
        device="cpu",
        checkpoint_every=2,
        schedule_stages=[
            {"name": "stage1", "steps": 4, "lr_max": 8.0e-4},
            {"name": "stage2", "steps": 2, "lr_max": 1.0e-4},
        ],
    )
    eval_cfg = build_cls_smoke_eval_config(
        manifest_path=manifest_path,
        checkpoint_path=tmp_path / "best.pt",
        device="cpu",
    )

    assert str(train_cfg.data.manifest_path) == str(manifest_path)
    assert str(eval_cfg.data.manifest_path) == str(manifest_path)
    assert "train_row_cap" not in train_cfg.data
    assert "test_row_cap" not in train_cfg.data
    assert "train_row_cap" not in eval_cfg.data
    assert "test_row_cap" not in eval_cfg.data
    assert int(train_cfg.model.feature_group_size) == 1
    assert int(eval_cfg.model.feature_group_size) == 1
    assert [str(stage["name"]) for stage in train_cfg.schedule.stages] == ["stage1", "stage2"]
    assert [int(stage["steps"]) for stage in train_cfg.schedule.stages] == [4, 2]
    assert str(eval_cfg.eval.split) == "test"
