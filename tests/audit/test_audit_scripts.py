from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
module_graph = _load_script_module(
    REPO_ROOT / "scripts" / "audit" / "module_graph.py",
    "module_graph_script",
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


def test_module_dependency_doc_matches_observed_graph() -> None:
    report = module_graph.build_module_graph_report(REPO_ROOT)

    assert report.undocumented_edges == []
    assert report.stale_documented_edges == []
