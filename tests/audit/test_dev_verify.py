from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dev_verify = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "dev_verify.py",
    "dev_verify_script",
)


def test_load_dev_index_defaults_to_origin_main() -> None:
    index = dev_verify.load_dev_index()

    assert index.default_base_ref == "origin/main"
    assert index.full_verify_checks == ("mdformat", "audit", "ruff", "mypy", "pytest_all")


def test_docs_only_diff_selects_mdformat_and_audit() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(["README.md"], index)

    assert plan.escalated_to_full is False
    assert plan.check_ids == ("mdformat", "audit")
    assert plan.scope.subsystem_paths == {"docs": ("README.md",)}


def test_system_delta_catalog_diff_includes_research_checks() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(["reference/system_delta_catalog.yaml"], index)

    assert plan.escalated_to_full is False
    assert {"mdformat", "audit", "ruff", "mypy", "pytest_research", "pytest_benchmark"}.issubset(
        plan.check_ids
    )
    assert plan.scope.subsystem_paths["research"] == ("reference/system_delta_catalog.yaml",)


def test_system_delta_sweep_queue_diff_includes_research_checks() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(
        ["reference/system_delta_sweeps/input_norm_followup/queue.yaml"],
        index,
    )

    assert plan.escalated_to_full is False
    assert {"mdformat", "audit", "ruff", "mypy", "pytest_research", "pytest_benchmark"}.issubset(
        plan.check_ids
    )
    assert plan.scope.subsystem_paths["research"] == (
        "reference/system_delta_sweeps/input_norm_followup/queue.yaml",
    )


def test_verify_paths_parser_accepts_explicit_paths() -> None:
    args = dev_verify.parse_args(
        [
            "verify",
            "paths",
            "src/tab_foundry/model/factory.py",
            "tests/model/test_factory.py",
        ]
    )

    assert args.command == "verify"
    assert args.verify_command == "paths"
    assert args.paths == [
        "src/tab_foundry/model/factory.py",
        "tests/model/test_factory.py",
    ]


def test_training_only_diff_selects_training_slice() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(["src/tab_foundry/training/trainer.py"], index)

    assert plan.escalated_to_full is False
    assert plan.check_ids == ("ruff", "mypy", "pytest_training", "pytest_runtime", "pytest_smoke")


def test_audit_diff_selects_audit_scripts_and_tests() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(["scripts/audit/dev_verify.py"], index)

    assert plan.escalated_to_full is False
    assert plan.check_ids == ("audit", "pytest_audit")
    assert plan.scope.subsystem_paths == {"audit": ("scripts/audit/dev_verify.py",)}


def test_multi_subsystem_diff_escalates_to_full_verification() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(
        [
            "src/tab_foundry/data/manifest.py",
            "src/tab_foundry/model/factory.py",
        ],
        index,
    )

    assert plan.escalated_to_full is True
    assert plan.check_ids == index.full_verify_checks
    assert "multiple core subsystems" in " ".join(plan.escalation_reasons)


def test_unknown_path_diff_escalates_to_full_verification() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(["assets/logo.txt"], index)

    assert plan.escalated_to_full is True
    assert plan.check_ids == index.full_verify_checks
    assert plan.scope.unmatched_paths == ("assets/logo.txt",)


def test_review_report_warns_about_missing_version_and_changelog() -> None:
    index = dev_verify.load_dev_index()
    plan = dev_verify.build_verification_plan(["src/tab_foundry/training/trainer.py"], index)

    report = dev_verify.render_review_report(
        plan,
        base_ref=index.default_base_ref,
        merge_base="abc123",
    )

    assert "Warnings:" in report
    assert "pyproject.toml" in report
    assert "CHANGELOG.md" in report
    assert "./scripts/dev verify affected --base-ref origin/main" in report


def test_verify_affected_plan_covers_each_rule_minimum_checks() -> None:
    index = dev_verify.load_dev_index()
    sample_paths = {
        "docs": "README.md",
        "audit": "scripts/audit/dev_verify.py",
        "data": "src/tab_foundry/data/manifest.py",
        "model": "src/tab_foundry/model/factory.py",
        "training": "src/tab_foundry/training/trainer.py",
        "export": "src/tab_foundry/export/exporter.py",
        "bench": "src/tab_foundry/bench/compare.py",
        "research": "src/tab_foundry/research/system_delta_execute.py",
        "cli-config": "configs/config.yaml",
    }
    rules_by_name = {rule.name: rule for rule in index.path_rules}

    for rule_name, sample_path in sample_paths.items():
        plan = dev_verify.build_verification_plan([sample_path], index)
        expected_rule = rules_by_name[rule_name]

    assert plan.escalated_to_full is False
    assert set(expected_rule.checks).issubset(plan.check_ids)


def test_verify_paths_reuses_affected_scope_mapping_for_explicit_paths() -> None:
    index = dev_verify.load_dev_index()

    plan = dev_verify.build_verification_plan(
        [
            "src/tab_foundry/model/factory.py",
            "tests/model/test_factory.py",
        ],
        index,
    )

    assert plan.escalated_to_full is False
    assert plan.check_ids == (
        "ruff",
        "mypy",
        "pytest_model",
        "pytest_training",
        "pytest_smoke",
        "pytest_property",
    )


def test_should_live_stream_only_for_pytest_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAB_FOUNDRY_LIVE_PYTEST", "1")
    monkeypatch.setattr(dev_verify.sys.stdout, "isatty", lambda: False)

    assert dev_verify._should_live_stream((str(dev_verify.VENV_PYTHON), "-m", "pytest", "-q", "tests/cli")) is True
    assert dev_verify._should_live_stream((str(dev_verify.VENV_RUFF), "check", "src")) is False


def test_run_check_command_prefers_live_stream_runner_for_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setenv("TAB_FOUNDRY_LIVE_PYTEST", "1")
    monkeypatch.setattr(dev_verify.sys.stdout, "isatty", lambda: False)

    def _fake_live(argv):
        calls.append(("live", tuple(argv)))
        return 0

    def _fake_run(argv, *, cwd, check):
        calls.append(("run", tuple(argv)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_verify, "_run_live_streamed", _fake_live)
    monkeypatch.setattr(dev_verify.subprocess, "run", _fake_run)

    result = dev_verify.run_check_command((str(dev_verify.VENV_PYTHON), "-m", "pytest", "-q", "tests/cli"))

    assert result == 0
    assert calls == [("live", (str(dev_verify.VENV_PYTHON), "-m", "pytest", "-q", "tests/cli"))]


def test_run_check_command_uses_plain_subprocess_for_non_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setenv("TAB_FOUNDRY_LIVE_PYTEST", "1")
    monkeypatch.setattr(dev_verify.sys.stdout, "isatty", lambda: False)

    def _fake_live(argv):
        calls.append(("live", tuple(argv)))
        return 0

    def _fake_run(argv, *, cwd, check):
        calls.append(("run", tuple(argv)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_verify, "_run_live_streamed", _fake_live)
    monkeypatch.setattr(dev_verify.subprocess, "run", _fake_run)

    result = dev_verify.run_check_command((str(dev_verify.VENV_RUFF), "check", "src"))

    assert result == 0
    assert calls == [("run", (str(dev_verify.VENV_RUFF), "check", "src"))]
