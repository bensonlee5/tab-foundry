#!/usr/bin/env python3
"""Link checker for canonical docs and Hugo source content."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROOTS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs",
    "reference",
    "program.md",
    "site/content",
    "site/.generated/content",
]

MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)
SHORTCODE_LINK_RE = re.compile(
    r"""
    ^\{\{(?:<|%)
    \s*(?P<name>ref|relref)
    \s+(?:"(?P<double>[^"]+)"|'(?P<single>[^']+)')
    \s*(?:>|%)\}\}$
    """,
    re.VERBOSE,
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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_base_path(repo_root: Path) -> str:
    text = _read_text(repo_root / "site" / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    parsed = urlparse(match.group(1).strip())
    path = parsed.path.rstrip("/")
    return "" if path == "/" else path


def _read_base_url(repo_root: Path) -> str:
    text = _read_text(repo_root / "site" / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().rstrip("/")


def _iter_doc_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file():
        if root.suffix.lower() in {".md", ".html"}:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".html"}:
            yield path


def _normalize_target(raw_target: str) -> tuple[str, bool]:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if target.startswith(("{{<", "{{%")) and target.endswith((">}}", "%}}")):
        match = SHORTCODE_LINK_RE.match(target)
        if match is not None:
            return (match.group("double") or match.group("single") or "", True)
        return (target, False)
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split(" ", 1)[0]
    return (target, False)


def _collect_targets(line: str, suffix: str) -> list[str]:
    if suffix == ".md":
        targets = [match.group(1) for match in MD_LINK_RE.finditer(line)]
        for match in HTML_LINK_RE.finditer(line):
            targets.append(match.group(1) or match.group(2) or match.group(3) or "")
        return targets
    return [match.group(1) or match.group(2) or match.group(3) or "" for match in HTML_LINK_RE.finditer(line)]


def _route_candidates(repo_root: Path, route: str) -> list[Path]:
    route = route.strip("/")
    if not route:
        return [
            repo_root / "site/content/_index.md",
            repo_root / "site/.generated/content/_index.md",
        ]

    candidates: list[Path] = []
    for content_root in (repo_root / "site/content", repo_root / "site/.generated/content"):
        route_path = content_root / route
        candidates.extend(
            [
                route_path.with_suffix(".md"),
                route_path / "index.md",
                route_path / "_index.md",
                route_path.with_suffix(".html"),
                route_path / "index.html",
            ]
        )
    return candidates


def _strip_base_path(target: str, base_path: str) -> str | None:
    if not base_path:
        return target
    if target == base_path or target == f"{base_path}/":
        return "/"
    prefix = f"{base_path}/"
    if target.startswith(prefix):
        return target[len(base_path) :]
    return None


def _normalize_repo_site_target(target: str, base_url: str, base_path: str) -> str:
    parsed = urlparse(target)
    base_parsed = urlparse(base_url) if base_url else None
    if base_parsed is None or not parsed.scheme or not parsed.netloc:
        return target
    if parsed.scheme != base_parsed.scheme or parsed.netloc != base_parsed.netloc:
        return target
    if base_path and parsed.path != base_path and not parsed.path.startswith(f"{base_path}/"):
        return target
    normalized = parsed.path or "/"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    if parsed.fragment:
        normalized = f"{normalized}#{parsed.fragment}"
    return normalized


def _is_authored_site_content(repo_root: Path, path: Path) -> bool:
    try:
        path.relative_to(repo_root / "site/content")
        return True
    except ValueError:
        return False


def _root_absolute_policy_violation(repo_root: Path, path: Path, target: str, base_path: str) -> bool:
    if not _is_authored_site_content(repo_root, path):
        return False
    if not base_path:
        return False
    if not target.startswith("/"):
        return False
    return not (target == base_path or target.startswith(f"{base_path}/"))


def _exists_target(repo_root: Path, source: Path, target: str, base_path: str) -> bool:
    if target.startswith("/"):
        normalized = _strip_base_path(target, base_path)
        if normalized is None:
            normalized = target
        return any(candidate.exists() for candidate in _route_candidates(repo_root, normalized))

    candidate = source.parent / target
    if candidate.exists():
        return True
    if candidate.suffix:
        return False
    candidates = [
        candidate.with_suffix(".md"),
        candidate.with_suffix(".html"),
        candidate / "index.md",
        candidate / "_index.md",
        candidate / "index.html",
    ]
    return any(path.exists() for path in candidates)


def scan_links(repo_root: Path = REPO_ROOT, roots: Iterable[str] = DEFAULT_ROOTS) -> list[tuple[Path, int, str]]:
    base_path = _read_base_path(repo_root)
    base_url = _read_base_url(repo_root)
    errors: list[tuple[Path, int, str]] = []

    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            errors.append((root, 0, "missing scan root"))
            continue
        for path in _iter_doc_files(root):
            suffix = path.suffix.lower()
            for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
                for raw_target in _collect_targets(line, suffix):
                    target, is_shortcode_link = _normalize_target(raw_target)
                    if not target or target.startswith("#") or target.startswith(("{{<", "{{%")):
                        continue
                    target = _normalize_repo_site_target(target, base_url, base_path)
                    if target.startswith(SKIP_PREFIXES):
                        continue
                    target_path = target.split("#", 1)[0].split("?", 1)[0]
                    if not target_path:
                        continue
                    if not is_shortcode_link and _root_absolute_policy_violation(
                        repo_root,
                        path,
                        target_path,
                        base_path,
                    ):
                        errors.append(
                            (
                                path,
                                lineno,
                                (
                                    f"{target} (root-absolute internal links in site/content "
                                    f"must include base path '{base_path}/' or use relref)"
                                ),
                            )
                        )
                        continue
                    if not _exists_target(repo_root, path, target_path, base_path):
                        errors.append((path, lineno, target))
    return errors


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=DEFAULT_ROOTS, help="Repo-relative roots to scan.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    errors = scan_links(REPO_ROOT, args.roots)
    if errors:
        print("Broken source-doc links found:")
        for path, lineno, target in errors:
            location = path.relative_to(REPO_ROOT) if path.exists() else path
            suffix = f":{lineno}" if lineno else ""
            print(f"- {location}{suffix} -> {target}")
        return 1
    print("Source-doc link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
