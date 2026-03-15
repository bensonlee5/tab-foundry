#!/usr/bin/env python3
"""Check local Markdown and HTML links in repo-tracked docs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    "README.md",
    "docs",
    "reference",
    "program.md",
)
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)
SKIP_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "javascript:",
    "data:",
    "//",
)


def _iter_doc_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in {".md", ".html"}:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".html"}:
            yield path


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split(" ", 1)[0]
    return target


def _collect_targets(line: str, suffix: str) -> list[str]:
    if suffix == ".md":
        targets = [match.group(1) for match in MD_LINK_RE.finditer(line)]
        for match in HTML_LINK_RE.finditer(line):
            targets.append(match.group(1) or match.group(2) or match.group(3) or "")
        return targets

    targets: list[str] = []
    for match in HTML_LINK_RE.finditer(line):
        targets.append(match.group(1) or match.group(2) or match.group(3) or "")
    return targets


def _exists_target(repo_root: Path, source: Path, target: str) -> bool:
    candidate = target.strip()
    if not candidate or candidate.startswith(("#", "/")):
        return True
    path_only = candidate.split("#", 1)[0].split("?", 1)[0]
    if not path_only:
        return True

    resolved = (source.parent / path_only).resolve(strict=False)
    if not resolved.is_relative_to(repo_root):
        return True
    if resolved.exists():
        return True

    if resolved.suffix:
        return False

    candidates = [
        resolved.with_suffix(".md"),
        resolved.with_suffix(".html"),
        resolved / "index.md",
        resolved / "_index.md",
        resolved / "index.html",
    ]
    return any(path.exists() for path in candidates)


def scan_markdown_links(
    repo_root: Path = REPO_ROOT,
    roots: Iterable[str] = DEFAULT_ROOTS,
) -> list[tuple[Path, int, str]]:
    errors: list[tuple[Path, int, str]] = []
    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            errors.append((root, 0, "missing scan root"))
            continue

        for path in _iter_doc_files(root):
            suffix = path.suffix.lower()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                for raw_target in _collect_targets(line, suffix):
                    target = _normalize_target(raw_target)
                    if not target or target.startswith(SKIP_PREFIXES):
                        continue
                    if not _exists_target(repo_root, path, target):
                        errors.append((path, lineno, target))
    return errors


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Repo-relative Markdown file or directory roots to scan.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    errors = scan_markdown_links(REPO_ROOT, args.roots)
    if errors:
        print("Broken local links found:")
        for path, lineno, target in errors:
            location = path.relative_to(REPO_ROOT) if path.exists() else path
            suffix = f":{lineno}" if lineno else ""
            print(f"- {location}{suffix} -> {target}")
        return 1

    print("Markdown link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
