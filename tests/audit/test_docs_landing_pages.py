from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_static_hugo_landing_pages_use_valid_front_matter_and_links() -> None:
    for rel_path in (
        "site/content/_index.md",
        "site/content/docs/_index.md",
        "site/content/docs/getting-started/_index.md",
        "site/content/docs/ml-engineering/_index.md",
        "site/content/docs/research-contributors/_index.md",
        "site/content/docs/development/_index.md",
        "site/content/docs/ml-engineering/artifacts-and-inference.md",
        "site/content/docs/research-contributors/sweeps.md",
    ):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "## title:" not in text
        assert "\\[" not in text
