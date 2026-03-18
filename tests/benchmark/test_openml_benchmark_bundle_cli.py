from __future__ import annotations

from pathlib import Path

import pytest

import tab_foundry.bench.openml_benchmark_bundle as bundle_module


def test_build_openml_benchmark_bundle_main_parses_task_source_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_write(path: Path, config: bundle_module.OpenMLBenchmarkBundleConfig) -> Path:
        captured["path"] = path
        captured["config"] = config
        return path

    monkeypatch.setattr(bundle_module, "write_openml_benchmark_bundle", _fake_write)

    exit_code = bundle_module.main(
        [
            "--out-path",
            str(tmp_path / "bundle.json"),
            "--bundle-name",
            "binary_medium",
            "--version",
            "1",
            "--task-source",
            "binary_expanded_v1",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert isinstance(config, bundle_module.OpenMLBenchmarkBundleConfig)
    assert config.task_source == "binary_expanded_v1"
    assert "wrote benchmark bundle:" in capsys.readouterr().out


def test_build_openml_benchmark_bundle_main_parses_discovery_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        bundle_module,
        "build_openml_benchmark_bundle_result",
        lambda config: bundle_module.OpenMLBenchmarkBundleBuildResult(
            bundle={
                "name": str(config.bundle_name),
                "version": int(config.version),
                "selection": {
                    "new_instances": int(config.new_instances),
                    "task_type": str(config.task_type),
                    "max_features": int(config.max_features),
                    "max_missing_pct": float(config.max_missing_pct),
                    "max_classes": int(config.max_classes or 2),
                    "min_minority_class_pct": float(config.min_minority_class_pct),
                },
                "task_ids": [10],
                "tasks": [
                    {
                        "task_id": 10,
                        "dataset_name": "one",
                        "n_rows": 200,
                        "n_features": 3,
                        "n_classes": 2,
                    }
                ],
            },
            report_entries=(
                bundle_module.OpenMLBenchmarkCandidateReportEntry(
                    task_id=10,
                    dataset_id=1,
                    dataset_name="one",
                    estimation_procedure="10-fold Crossvalidation",
                    status="accepted",
                    reason="validated via prepare_openml_benchmark_task",
                ),
            ),
        ),
    )

    def _fake_write(
        path: Path,
        config: bundle_module.OpenMLBenchmarkBundleConfig,
        *,
        bundle: dict[str, object] | None = None,
    ) -> Path:
        captured["path"] = path
        captured["config"] = config
        captured["bundle"] = bundle
        return path

    monkeypatch.setattr(bundle_module, "write_openml_benchmark_bundle", _fake_write)

    exit_code = bundle_module.main(
        [
            "--out-path",
            str(tmp_path / "bundle.json"),
            "--bundle-name",
            "binary_large_no_missing",
            "--version",
            "1",
            "--discover-from-openml",
            "--min-instances",
            "200",
            "--min-task-count",
            "50",
            "--max-features",
            "50",
            "--max-classes",
            "2",
            "--max-missing-pct",
            "0.0",
            "--min-minority-class-pct",
            "2.5",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert isinstance(config, bundle_module.OpenMLBenchmarkBundleConfig)
    assert config.discover_from_openml is True
    assert config.min_instances == 200
    assert config.min_task_count == 50
    assert captured["bundle"] is not None
    stdout = capsys.readouterr().out
    assert "OpenML discovery candidate report:" in stdout
    assert "wrote benchmark bundle:" in stdout
