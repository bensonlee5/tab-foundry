#!/usr/bin/env python3
"""Validate internal links in built Hugo output."""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]

ATTR_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)
SKIP_PREFIXES = ("mailto:", "tel:", "javascript:", "data:", "//")


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


def _iter_html_files(output_dir: Path) -> Iterable[Path]:
    for path in output_dir.rglob("*.html"):
        if path.is_file():
            yield path


def _extract_targets(html: str) -> list[str]:
    return [match.group(1) or match.group(2) or match.group(3) or "" for match in ATTR_RE.finditer(html)]


def _resolve_internal_path(output_dir: Path, html_file: Path, target: str, base_path: str) -> tuple[str, str]:
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return ("skip", "")

    path = parsed.path
    if not path or path.startswith("#") or path.startswith(SKIP_PREFIXES):
        return ("skip", "")

    if path.startswith("/"):
        if base_path:
            if path == base_path or path == f"{base_path}/":
                return ("ok", "")
            prefix = f"{base_path}/"
            if path.startswith(prefix):
                return ("ok", path[len(base_path) :].lstrip("/"))
            return ("bad_prefix", path)
        return ("ok", path.lstrip("/"))

    current_rel_dir = html_file.relative_to(output_dir).parent.as_posix()
    joined = posixpath.normpath(posixpath.join(current_rel_dir, path))
    if joined == ".":
        joined = ""
    if joined.startswith("../"):
        return ("escape_root", path)
    return ("ok", joined)


def _is_generated_source_link(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme and parsed.netloc and "/.generated/content/" in parsed.path)


def _built_target_exists(output_dir: Path, rel_path: str) -> bool:
    if rel_path == "":
        return (output_dir / "index.html").exists()

    candidate = output_dir / rel_path
    if candidate.exists():
        return True
    if rel_path.endswith("/"):
        return (output_dir / rel_path / "index.html").exists()
    if candidate.suffix:
        return False
    return any(
        path.exists()
        for path in (
            output_dir / rel_path / "index.html",
            (output_dir / rel_path).with_suffix(".html"),
        )
    )


def validate_built_output(
    repo_root: Path = REPO_ROOT,
    output_dir: str = "site/public",
) -> dict[str, list[tuple[Path, str]]]:
    built_root = (repo_root / output_dir).resolve()
    base_path = _read_base_path(repo_root)
    errors = {
        "prefix": [],
        "missing": [],
        "escape": [],
        "generated_source": [],
    }

    if not built_root.exists():
        errors["missing"].append((built_root, "missing output dir"))
        return errors

    for html_file in _iter_html_files(built_root):
        text = _read_text(html_file)
        for raw_target in _extract_targets(text):
            target = raw_target.strip().strip('"').strip("'")
            if not target:
                continue
            if _is_generated_source_link(target):
                errors["generated_source"].append((html_file, target))
                continue

            status, resolved = _resolve_internal_path(built_root, html_file, target, base_path)
            if status == "skip":
                continue
            if status == "bad_prefix":
                errors["prefix"].append((html_file, target))
                continue
            if status == "escape_root":
                errors["escape"].append((html_file, target))
                continue
            if status == "ok" and not _built_target_exists(built_root, resolved):
                errors["missing"].append((html_file, target))

    return errors


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="site/public", help="Directory containing built Hugo output.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    errors = validate_built_output(REPO_ROOT, args.output_dir)

    if any(errors.values()):
        if errors["prefix"]:
            print("Base-path prefix violations:")
            for html_file, target in errors["prefix"]:
                print(f"- {html_file.relative_to(REPO_ROOT)} -> {target}")
            print()
        if errors["escape"]:
            print("Links escaping output root:")
            for html_file, target in errors["escape"]:
                print(f"- {html_file.relative_to(REPO_ROOT)} -> {target}")
            print()
        if errors["generated_source"]:
            print("Generated-source GitHub link violations:")
            for html_file, target in errors["generated_source"]:
                print(f"- {html_file.relative_to(REPO_ROOT)} -> {target}")
            print()
        if errors["missing"]:
            print("Unresolved internal links:")
            for html_file, target in errors["missing"]:
                if html_file.exists():
                    location = html_file.relative_to(REPO_ROOT)
                else:
                    location = html_file
                print(f"- {location} -> {target}")
        return 1

    print("Built-output link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
