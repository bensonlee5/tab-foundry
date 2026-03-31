from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


check_repo_paths = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "check_repo_paths.py",
    "check_repo_paths_script",
)
check_markdown_links = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "check_markdown_links.py",
    "check_markdown_links_script",
)
check_docs_consistency = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "check_docs_consistency.py",
    "check_docs_consistency_script",
)
module_graph = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "module_graph.py",
    "module_graph_script",
)
check_version_bump = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "check_version_bump.py",
    "check_version_bump_script",
)
bump_version = _load_script_module(
    REPO_ROOT / "scripts" / "bump_version.py",
    "bump_version_script",
)


def test_check_repo_paths_reports_missing_reference(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "exists.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "See `docs/exists.md` and `docs/missing.md`.\n",
        encoding="utf-8",
    )

    errors = check_repo_paths.scan_markdown_path_references(tmp_path, ["README.md"])

    assert errors == [
        (tmp_path / "README.md", 1, "docs/missing.md (missing path)"),
    ]


def test_check_repo_paths_ignores_template_placeholders(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "See `reference/system_delta_sweeps/<sweep_id>/queue.yaml`.\n",
        encoding="utf-8",
    )

    errors = check_repo_paths.scan_markdown_path_references(tmp_path, ["README.md"])

    assert errors == []


def test_check_repo_paths_accepts_line_annotated_repo_paths(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "exists.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "See `docs/exists.md:12-14`.\n",
        encoding="utf-8",
    )

    errors = check_repo_paths.scan_markdown_path_references(tmp_path, ["README.md"])

    assert errors == []


def test_check_markdown_links_reports_missing_local_target(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "exists.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "[ok](docs/exists.md)",
                "[bad](docs/missing.md)",
                "[external](https://example.com)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_markdown_links.scan_markdown_links(tmp_path, ["README.md"])

    assert errors == [
        (tmp_path / "README.md", 2, "docs/missing.md"),
    ]


def test_module_graph_smoke_includes_training_to_model_edge() -> None:
    report = module_graph.build_module_graph_report(REPO_ROOT)

    assert ("tab_foundry.training", "tab_foundry.model") in report.top_level_edges
    assert "tab_foundry.training" in report.top_level_modules


def test_repo_path_audit_passes_on_repo_docs() -> None:
    errors = check_repo_paths.scan_markdown_path_references(
        REPO_ROOT,
        check_repo_paths.DEFAULT_ROOTS,
    )

    assert errors == []


def test_markdown_link_audit_passes_on_repo_docs() -> None:
    errors = check_markdown_links.scan_markdown_links(
        REPO_ROOT,
        check_markdown_links.DEFAULT_ROOTS,
    )

    assert errors == []


def test_docs_consistency_audit_passes_on_repo_docs() -> None:
    errors = check_docs_consistency.scan_docs_consistency(
        REPO_ROOT,
        check_docs_consistency.DEFAULT_ROOTS,
    )

    assert errors == []


def test_docs_consistency_reports_stale_python_module_entrypoint(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "Run `python -m tab_foundry.bench.instability_audit --staged-ladder-root outputs/staged_ladder`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert errors[0][2].startswith("stale direct module entrypoint")


def test_docs_consistency_reports_unsupported_python_script_entrypoint(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "custom_helper.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "Run `.venv/bin/python scripts/custom_helper.py --demo`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert errors[0][2] == "unsupported Python script entrypoint in docs: `scripts/custom_helper.py`"


def test_docs_consistency_reports_nonexistent_tab_foundry_command(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "Use `tab-foundry research sweep create --sweep-id demo`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert "unknown tab-foundry command token" in errors[0][2]
    assert "create" in errors[0][2]


def test_docs_consistency_reports_removed_module_reference(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "Import `tab_foundry.research.system_delta_execute` for sweep execution.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert "removed research compatibility wrapper reference" == errors[0][2]


def test_docs_consistency_reports_missing_repo_script_entrypoint(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "Run `./scripts/not_real.sh` before opening a PR.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert errors[0][2] == "documented repo-local script entrypoint is missing: `scripts/not_real.sh`"


def test_docs_consistency_reports_disallowed_readme_cli_tree(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "<details>",
                "<summary>Full CLI tree</summary>",
                "",
                "```text",
                "tab-foundry",
                "├── train",
                "│   └── prior",
                "│       └── staged",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert len(errors) == 1
    assert errors[0][2] == (
        "README must not duplicate a hand-maintained CLI tree; use "
        "packaged CLI `--help` for live commands and `docs/workflows.md` for examples"
    )


def test_docs_consistency_requires_agents_doc_contract(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "# Development Patterns\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["AGENTS.md"])

    messages = [error[2] for error in errors]
    assert any("missing required docs contract statement in `AGENTS.md`" in message for message in messages)


def test_docs_consistency_rejects_agent_markers_in_human_docs(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# README",
                "",
                "**Owns**",
                "- overview",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["README.md"])

    assert errors == [
        (
            tmp_path / "README.md",
            3,
            "agent-facing docs marker must not appear in human-facing docs: `**Owns**`",
        )
    ]


def test_docs_consistency_reports_static_command_inventory_outside_workflows(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text(
        "\n".join(
            [
                "# Guide",
                "- `tab-foundry train run`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["docs/guide.md"])

    assert errors == [
        (
            tmp_path / "docs" / "guide.md",
            2,
            "static command inventory must live in `docs/workflows.md` or CLI --help: `tab-foundry train run`",
        )
    ]


def test_docs_consistency_reports_program_only_heading_outside_program(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "workflows.md").write_text(
        "\n".join(
            [
                "# Workflows",
                "## Objective",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = check_docs_consistency.scan_docs_consistency(tmp_path, ["docs/workflows.md"])

    assert errors == [
        (
            tmp_path / "docs" / "workflows.md",
            2,
            "sweep policy heading must live in `program.md`: `## Objective`",
        )
    ]


def test_module_dependency_doc_matches_observed_graph() -> None:
    report = module_graph.build_module_graph_report(REPO_ROOT)

    assert report.undocumented_edges == []
    assert report.stale_documented_edges == []


@pytest.mark.parametrize(
    ("base_version", "candidate_version"),
    [
        ("0.6.0", "0.6.0"),
        ("0.6.0", "0.6.1"),
        ("0.6.0", "0.7.0"),
        ("0.6.0", "1.0.0"),
        ("1.2.3", "1.2.4"),
        ("1.2.3", "1.3.0"),
        ("1.2.3", "2.0.0"),
    ],
)
def test_version_bump_check_allows_same_or_single_semver_step(
    base_version: str,
    candidate_version: str,
) -> None:
    assert check_version_bump.validate_version_bump(base_version, candidate_version) is None


@pytest.mark.parametrize(
    ("base_version", "candidate_version"),
    [
        ("0.6.0", "0.6.2"),
        ("0.6.0", "0.7.1"),
        ("0.6.0", "0.8.0"),
        ("1.2.3", "1.2.2"),
        ("1.2.3", "2.0.1"),
        ("1.2.3", "2.1.0"),
    ],
)
def test_version_bump_check_rejects_skipped_or_lower_versions(
    base_version: str,
    candidate_version: str,
) -> None:
    message = check_version_bump.validate_version_bump(base_version, candidate_version)

    assert message is not None
    assert base_version in message
    assert candidate_version in message


def test_version_bump_check_rejects_non_semver_versions() -> None:
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        check_version_bump.validate_version_bump("0.6.0", "0.6")


def test_read_version_from_git_ref_uses_requested_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return check_version_bump.subprocess.CompletedProcess(
            args=["git", "show", "main:pyproject.toml"],
            returncode=0,
            stdout='[project]\nversion = "1.2.3"\n',
            stderr="",
        )

    monkeypatch.setattr(check_version_bump.subprocess, "run", fake_run)

    version = check_version_bump.read_version_from_git_ref(REPO_ROOT, ref="main")

    assert version == "1.2.3"
    assert calls == [
        (
            (["git", "show", "main:pyproject.toml"],),
            {
                "cwd": REPO_ROOT,
                "capture_output": True,
                "text": True,
                "check": False,
            },
        )
    ]


def test_read_version_from_git_ref_reports_git_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return check_version_bump.subprocess.CompletedProcess(
            args=["git", "show", "missing:pyproject.toml"],
            returncode=128,
            stdout="",
            stderr="fatal: invalid object name 'missing'.",
        )

    monkeypatch.setattr(check_version_bump.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="invalid object name"):
        check_version_bump.read_version_from_git_ref(REPO_ROOT, ref="missing")


def test_version_bump_check_skips_when_pyproject_is_not_staged_in_staged_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_version_bump, "is_path_staged", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(check_version_bump, "read_version_from_pyproject_path", lambda _path: "0.9.3")

    exit_code = check_version_bump.main(["--staged-only"])

    assert exit_code == 0


def test_version_bump_check_validates_on_clean_checkouts_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(check_version_bump, "is_path_staged", lambda *_args, **_kwargs: False)

    def _fake_read_version_from_git_ref(*_args, **_kwargs):
        calls.append("git_ref")
        return "0.9.2"

    def _fake_read_version_from_pyproject_path(_path):
        calls.append("pyproject")
        return "0.9.3"

    monkeypatch.setattr(check_version_bump, "read_version_from_git_ref", _fake_read_version_from_git_ref)
    monkeypatch.setattr(check_version_bump, "read_version_from_pyproject_path", _fake_read_version_from_pyproject_path)

    exit_code = check_version_bump.main([])

    assert exit_code == 0
    assert calls == ["git_ref", "pyproject"]


def test_version_bump_check_validates_when_pyproject_is_staged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_version_bump, "is_path_staged", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(check_version_bump, "read_version_from_git_ref", lambda *_args, **_kwargs: "0.9.2")
    monkeypatch.setattr(check_version_bump, "read_version_from_pyproject_path", lambda _path: "0.9.3")

    exit_code = check_version_bump.main([])

    assert exit_code == 0


def test_refresh_uv_lock_uses_uv_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return bump_version.subprocess.CompletedProcess(
            args=["uv", "lock"],
            returncode=0,
            stdout="locked",
            stderr="",
        )

    monkeypatch.setattr(bump_version.subprocess, "run", fake_run)

    bump_version.refresh_uv_lock(REPO_ROOT)

    assert calls == [
        (
            (["uv", "lock"],),
            {
                "cwd": REPO_ROOT,
                "capture_output": True,
                "text": True,
                "check": False,
            },
        )
    ]


def test_refresh_uv_lock_reports_subprocess_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return bump_version.subprocess.CompletedProcess(
            args=["uv", "lock"],
            returncode=2,
            stdout="",
            stderr="network error",
        )

    monkeypatch.setattr(bump_version.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="network error"):
        bump_version.refresh_uv_lock(REPO_ROOT)
