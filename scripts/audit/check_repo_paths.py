#!/usr/bin/env python3
"""Check repo-root Markdown path references for stale or missing targets."""

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
REPO_PATH_PREFIXES = (
    "src/",
    "docs/",
    "scripts/",
    "configs/",
    "reference/",
    "tests/",
    "data/",
)
REPO_PATH_EXACT = frozenset(
    {
        "README.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "program.md",
        "pyproject.toml",
        "uv.lock",
        ".pre-commit-config.yaml",
        ".python-version",
    }
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".md":
            yield root
        return
    for path in root.rglob("*.md"):
        if path.is_file():
            yield path


def _iter_inline_code_spans(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_CODE_RE.finditer(line):
            yield lineno, match.group(1).strip()


def _iter_markdown_link_targets(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line_without_code = INLINE_CODE_RE.sub("", line)
        for match in MARKDOWN_LINK_RE.finditer(line_without_code):
            yield lineno, match.group(1).strip()


def _is_repo_path(candidate: str) -> bool:
    return candidate in REPO_PATH_EXACT or candidate.startswith(REPO_PATH_PREFIXES)


def _normalize_repo_path_candidate(candidate: str, *, repo_root: Path) -> str | None:
    stripped = candidate.strip().rstrip(".,:;")
    if "<" in stripped and ">" in stripped:
        return None
    stripped = re.sub(r"#L\d+(?:-\d+)?$", "", stripped)
    stripped = re.sub(r":\d+(?:-\d+)?$", "", stripped)
    if any(char in stripped for char in "*?[]{}"):
        return None
    if (repo_root / stripped).exists():
        return stripped
    token = stripped.split()[0]
    if any(char in token for char in "*?[]{}"):
        return None
    return token


def _normalize_markdown_link_candidate(
    candidate: str,
    *,
    repo_root: Path,
    source_path: Path,
) -> str | None:
    stripped = candidate.strip()
    if not stripped:
        return None
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1].strip()
    if not stripped or URI_SCHEME_RE.match(stripped):
        return None
    if "<" in stripped and ">" in stripped:
        return None
    if stripped.startswith(("#", "/")):
        return None

    link_target = stripped.split(maxsplit=1)[0]
    path_only = re.split(r"[?#]", link_target, maxsplit=1)[0]
    if not path_only or any(char in path_only for char in "*?[]{}"):
        return None

    resolved = (source_path.parent / path_only).resolve(strict=False)
    if not resolved.is_relative_to(repo_root):
        return None
    return resolved.relative_to(repo_root).as_posix()


def scan_markdown_path_references(
    repo_root: Path = REPO_ROOT,
    roots: Iterable[str] = DEFAULT_ROOTS,
) -> list[tuple[Path, int, str]]:
    missing_roots: list[tuple[Path, int, str]] = []
    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            missing_roots.append((repo_root / root_rel, 0, "missing scan root"))

    all_errors: list[tuple[Path, int, str]] = []
    all_errors.extend(missing_roots)
    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            continue
        for path in _iter_markdown_files(root):
            for lineno, candidate in _iter_inline_code_spans(path):
                normalized = _normalize_repo_path_candidate(candidate, repo_root=repo_root)
                if normalized is None or not _is_repo_path(normalized):
                    continue
                if not (repo_root / normalized).exists():
                    all_errors.append((path, lineno, f"{normalized} (missing path)"))
            for lineno, candidate in _iter_markdown_link_targets(path):
                normalized = _normalize_markdown_link_candidate(
                    candidate,
                    repo_root=repo_root,
                    source_path=path,
                )
                if normalized is None or not _is_repo_path(normalized):
                    continue
                if not (repo_root / normalized).exists():
                    all_errors.append((path, lineno, f"{normalized} (missing path)"))
    return all_errors


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
    errors = scan_markdown_path_references(REPO_ROOT, args.roots)
    if errors:
        print("Broken repo path references found:")
        for path, lineno, target in errors:
            location = path.relative_to(REPO_ROOT) if path.exists() else path
            suffix = f":{lineno}" if lineno else ""
            print(f"- {location}{suffix} -> {target}")
        return 1

    print("Repo path check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
