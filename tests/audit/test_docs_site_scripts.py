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


sync_hugo_content = _load_script_module(
    REPO_ROOT / "scripts" / "docs" / "sync_hugo_content.py",
    "sync_hugo_content_script",
)
check_links = _load_script_module(
    REPO_ROOT / "scripts" / "docs" / "check_links.py",
    "check_docs_links_script",
)
check_built_output_links = _load_script_module(
    REPO_ROOT / "scripts" / "docs" / "check_built_output_links.py",
    "check_built_output_links_script",
)


def test_sync_hugo_content_front_door_specs_use_readme_repo_overview() -> None:
    assert any(
        spec.source_rel == "README.md"
        and spec.route == "getting-started/repo-overview"
        and "/docs/repo-overview/" in spec.aliases
        for spec in sync_hugo_content.PAGE_SPECS
    )
    assert all(
        spec.source_rel != "docs/what-is-tab-foundry.md"
        for spec in sync_hugo_content.PAGE_SPECS
    )
    assert all(
        spec.source_rel not in {
            "docs/getting-started.md",
            "docs/ml-engineering.md",
            "docs/research-contributors.md",
        }
        for spec in sync_hugo_content.PAGE_SPECS
    )


def test_sync_hugo_content_tracks_audience_and_topic_router_pages() -> None:
    required_paths = {
        REPO_ROOT / "site" / "content" / "docs" / "getting-started" / "_index.md",
        REPO_ROOT / "site" / "content" / "docs" / "ml-engineering" / "_index.md",
        REPO_ROOT / "site" / "content" / "docs" / "research-contributors" / "_index.md",
        REPO_ROOT / "site" / "content" / "docs" / "development" / "_index.md",
        REPO_ROOT
        / "site"
        / "content"
        / "docs"
        / "ml-engineering"
        / "artifacts-and-inference.md",
        REPO_ROOT
        / "site"
        / "content"
        / "docs"
        / "research-contributors"
        / "sweeps.md",
    }

    assert all(path.exists() for path in required_paths)


def test_site_shortcodes_for_secondary_topic_tree_were_deleted() -> None:
    assert not (REPO_ROOT / "site" / "layouts" / "shortcodes" / "audience-paths.html").exists()
    assert not (REPO_ROOT / "site" / "layouts" / "shortcodes" / "topic-cards.html").exists()


def test_sync_hugo_content_moves_readme_under_getting_started_with_legacy_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# tab-foundry\n\nOverview.\n", encoding="utf-8")

    monkeypatch.setattr(
        sync_hugo_content,
        "PAGE_SPECS",
        (
            sync_hugo_content.PageSpec(
                "README.md",
                "getting-started/repo-overview",
                10,
                aliases=("/docs/repo-overview/",),
                link_title="Repo Overview",
            ),
        ),
    )

    changed = sync_hugo_content.sync_hugo_content(tmp_path)

    rendered = (
        tmp_path
        / "site"
        / ".generated"
        / "content"
        / "docs"
        / "getting-started"
        / "repo-overview.md"
    )
    assert rendered in changed
    text = rendered.read_text(encoding="utf-8")
    assert 'aliases:\n  - "/docs/repo-overview/"' in text
    assert 'linkTitle: "Repo Overview"' in text
    assert "# tab-foundry" not in text.split("---\n\n", 1)[-1]


def test_sync_hugo_content_writes_front_matter_and_rewrites_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "source.md").write_text(
        "\n".join(
            [
                "# Source",
                "",
                "[linked](linked.md)",
                "[unsynced](other.md)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "linked.md").write_text("# Linked\n", encoding="utf-8")
    (tmp_path / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")

    monkeypatch.setattr(
        sync_hugo_content,
        "PAGE_SPECS",
        (
            sync_hugo_content.PageSpec("docs/source.md", "source", 10),
            sync_hugo_content.PageSpec("docs/linked.md", "linked", 20),
        ),
    )

    changed = sync_hugo_content.sync_hugo_content(tmp_path)

    rendered = tmp_path / "site" / ".generated" / "content" / "docs" / "source.md"
    assert rendered in changed
    text = rendered.read_text(encoding="utf-8")
    assert 'canonical_repo_path: "docs/source.md"' in text
    assert "[linked](/tab-foundry/docs/linked/)" in text
    assert "[unsynced](https://github.com/bensonlee5/tab-foundry/blob/main/docs/other.md)" in text
    assert "# Source" not in text.split("---\n\n", 1)[-1]
    assert 'aliases:\n  - "/docs/source.html"' in text


def test_rewrite_katex_math_normalizes_markdown_escaped_tex() -> None:
    content = (
        "Inline $X\\_{\\\\mathrm{tr}}$.\n\n"
        "$$ J(\\phi) = \\mathbb{E}_{F \\sim \\\\mathcal{F}_{\\\\mathrm{val}}} "
        "\\left\\[ \\operatorname{Perf}(F; \\theta^{\\*}(\\phi)) \\right\\] $$\n"
    )

    rendered = sync_hugo_content._rewrite_katex_math(content)

    assert 'data-katex-source="X_{\\mathrm{tr}}"' in rendered
    assert 'data-katex-source="J(\\phi) = \\mathbb{E}_{F \\sim \\mathcal{F}_{\\mathrm{val}}} \\left[ \\operatorname{Perf}(F; \\theta^{*}(\\phi)) \\right]"' in rendered
    assert "\\left\\[" not in rendered
    assert "\\theta^{\\*}" not in rendered
    assert "\\\\mathcal" not in rendered


def test_sync_hugo_content_check_mode_reports_stale_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    source = tmp_path / "docs" / "source.md"
    source.write_text("# Source\n", encoding="utf-8")

    monkeypatch.setattr(
        sync_hugo_content,
        "PAGE_SPECS",
        (sync_hugo_content.PageSpec("docs/source.md", "source", 10),),
    )

    assert sync_hugo_content.sync_hugo_content(tmp_path) != []

    source.write_text("# Source\n\nUpdated.\n", encoding="utf-8")
    changed = sync_hugo_content.sync_hugo_content(tmp_path, check=True)

    assert changed == [
        tmp_path / "site" / ".generated" / "content" / "docs" / "source.md",
    ]


def test_check_links_flags_root_absolute_site_links_without_base_path(tmp_path: Path) -> None:
    (tmp_path / "site" / "content" / "docs").mkdir(parents=True)
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "content" / "docs" / "page.md").write_text(
        "[bad](/docs/getting-started/)\n",
        encoding="utf-8",
    )

    errors = check_links.scan_links(tmp_path, ["site/content"])

    assert errors == [
        (
            tmp_path / "site" / "content" / "docs" / "page.md",
            1,
            "/docs/getting-started/ (root-absolute internal links in site/content must include base path '/tab-foundry/' or use relref)",
        )
    ]


def test_check_links_accepts_valid_relref_shortcode_target(tmp_path: Path) -> None:
    (tmp_path / "site" / "content" / "docs" / "getting-started").mkdir(parents=True)
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "content" / "docs" / "getting-started" / "_index.md").write_text(
        "# Getting Started\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "content" / "docs" / "page.md").write_text(
        '[ok]({{< relref "/docs/getting-started/_index.md" >}})\n',
        encoding="utf-8",
    )

    errors = check_links.scan_links(tmp_path, ["site/content"])

    assert errors == []


def test_check_links_reports_invalid_relref_shortcode_target(tmp_path: Path) -> None:
    (tmp_path / "site" / "content" / "docs" / "getting-started").mkdir(parents=True)
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "content" / "docs" / "getting-started" / "_index.md").write_text(
        "# Getting Started\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "content" / "docs" / "page.md").write_text(
        '[bad]({{< relref "/docs/getting-started/\\_index.md" >}})\n',
        encoding="utf-8",
    )

    errors = check_links.scan_links(tmp_path, ["site/content"])

    assert errors == [
        (
            tmp_path / "site" / "content" / "docs" / "page.md",
            1,
            "/docs/getting-started/\\_index.md",
        )
    ]


def test_check_built_output_links_reports_missing_routes_and_generated_source_links(tmp_path: Path) -> None:
    (tmp_path / "site" / "public").mkdir(parents=True)
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/tab-foundry/\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "public" / "index.html").write_text(
        "\n".join(
            [
                '<a href="https://github.com/bensonlee5/tab-foundry/blob/main/site/.generated/content/docs/foo.md">generated</a>',
                '<a href="/tab-foundry/docs/missing/">missing</a>',
            ]
        ),
        encoding="utf-8",
    )

    errors = check_built_output_links.validate_built_output(tmp_path)

    assert errors["prefix"] == []
    assert errors["escape"] == []
    assert errors["generated_source"] == [
        (
            tmp_path / "site" / "public" / "index.html",
            "https://github.com/bensonlee5/tab-foundry/blob/main/site/.generated/content/docs/foo.md",
        )
    ]
    assert errors["missing"] == [
        (
            tmp_path / "site" / "public" / "index.html",
            "/tab-foundry/docs/missing/",
        )
    ]
