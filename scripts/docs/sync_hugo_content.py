#!/usr/bin/env python3
"""Sync canonical repo docs into the Hugo docs site."""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]

GITHUB_REPO = "https://github.com/bensonlee5/tab-foundry"
GITHUB_BLOB_MAIN = f"{GITHUB_REPO}/blob/main"

MD_LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PageSpec:
    source_rel: str
    route: str
    weight: int
    description: str | None = None
    link_title: str | None = None
    aliases: tuple[str, ...] = ()
    toc_hide: bool = False
    hide_summary: bool = False
    no_list: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)


PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec(
        source_rel="docs/getting-started.md",
        route="getting-started/_index",
        weight=10,
        description="Fast route to the right docs path for repo overview, research work, or ML engineering.",
        link_title="Start Here",
        no_list=True,
    ),
    PageSpec(
        source_rel="docs/research-contributors.md",
        route="research-contributors/_index",
        weight=20,
        description="Research-first route through the active architecture, sweep system, synthetic-data lane, and model-breadth work.",
        link_title="Research",
        no_list=True,
    ),
    PageSpec(
        source_rel="docs/ml-engineering.md",
        route="ml-engineering/_index",
        weight=30,
        description="Operational route through artifacts, validation paths, and runtime handoff boundaries.",
        link_title="ML Engineering",
        no_list=True,
    ),
    PageSpec(
        source_rel="program.md",
        route="research-contributors/sweep-contract",
        weight=20,
        link_title="Sweep Contract",
        description="Rules for the active system-delta sweep: locked surface, allowed changes, and required artifacts.",
        aliases=("/docs/sweep-contract/",),
    ),
    PageSpec(
        source_rel="docs/workflows.md",
        route="ml-engineering/workflows",
        weight=20,
        description="Command runbooks for setup, verification, smoke tests, and research execution.",
        aliases=("/docs/workflows/",),
    ),
    PageSpec(
        source_rel="docs/development/roadmap.md",
        route="development/roadmap",
        weight=10,
        description="Current priorities, frozen baselines, and evidence-gated next steps.",
        link_title="Roadmap",
        extra_params={"mermaid": True},
    ),
    PageSpec(
        source_rel="docs/development/model-architecture.md",
        route="development/model-architecture",
        weight=20,
        description="Reference for the active model surface, subsystem roles, and current sandwich forward path.",
        extra_params={"mermaid": True, "katex": True},
    ),
    PageSpec(
        source_rel="docs/development/synthetic-prior-mission.md",
        route="getting-started/problem-formulation",
        weight=15,
        description="Mathematical formulation of the dagzoo prior-search problem and sandwich training objective.",
        link_title="Problem Formulation",
        aliases=(
            "/docs/development/dagzoo-sandwich-mathematical-formulation/",
            "/docs/development/dagzoo-sandwich-technical-formulation/",
        ),
        extra_params={"katex": True},
    ),
    PageSpec(
        source_rel="docs/development/design-decisions.md",
        route="development/design-decisions",
        weight=30,
        description="Durable architecture, repo-structure, and compatibility decisions.",
    ),
    PageSpec(
        source_rel="docs/development/codebase-navigation.md",
        route="development/codebase-navigation",
        weight=40,
        description="Where to make changes and which packages own each workflow surface.",
    ),
    PageSpec(
        source_rel="docs/development/module-dependency-map.md",
        route="development/module-dependency-map",
        weight=50,
        description="Current package graph and intended dependency direction for refactors.",
    ),
    PageSpec(
        source_rel="docs/development/model-config.md",
        route="development/model-config",
        weight=60,
        description="How model settings resolve across training, evaluation, export, and bundle loading.",
    ),
    PageSpec(
        source_rel="docs/development/dataset-curation.md",
        route="development/dataset-curation",
        weight=70,
        description="Policy for admitting real-data datasets into curated benchmark and comparator surfaces.",
    ),
    PageSpec(
        source_rel="reference/README.md",
        route="reference/_index",
        weight=50,
        description="Start here for papers, evidence, and supporting research artifacts behind roadmap decisions.",
        link_title="References",
    ),
    PageSpec(
        source_rel="reference/papers.md",
        route="reference/papers",
        weight=10,
        description="Reading list and adoption guidance for architecture and training ideas that matter in this repo.",
    ),
    PageSpec(
        source_rel="reference/evidence.md",
        route="reference/evidence",
        weight=20,
        description="Map from roadmap claims to papers and repo-local evidence.",
    ),
    PageSpec(
        source_rel="reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/support/README.md",
        route="reference/tf-rd-013-support",
        weight=30,
        link_title="TF-RD-013 Support",
        description="Regeneration assumptions and committed support notes for the current shape-aware dagzoo TF-RD-013 contract.",
    ),
    PageSpec(
        source_rel="CONTRIBUTING.md",
        route="getting-started/contributing",
        weight=30,
        description="How to make bounded changes without losing the repo's architecture and research context.",
        aliases=("/docs/contributing/",),
    ),
    PageSpec(
        source_rel="docs/glossary.md",
        route="getting-started/glossary",
        weight=20,
        description="Shared vocabulary for architecture, sweeps, artifacts, and workflow discussions.",
        aliases=("/docs/glossary/",),
    ),
    PageSpec(
        source_rel="README.md",
        route="repo-overview",
        weight=110,
        description="Top-level repo overview, docs routing, and quickstart.",
        link_title="Repo Overview",
        toc_hide=True,
        hide_summary=True,
    ),
    PageSpec(
        source_rel="docs/inference.md",
        route="ml-engineering/inference",
        weight=30,
        description="Export-bundle schema, validation rules, and runtime handoff boundary.",
        aliases=("/docs/inference/",),
    ),
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


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _slug_title(route: str) -> str:
    last = Path(route).name
    token = last if last != "_index" else Path(route).parent.name
    return token.replace("-", " ").replace("_", " ").title()


def _strip_matching_h1(content: str, title: str) -> str:
    lines = content.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines) and lines[idx].strip().startswith("# "):
        heading = lines[idx].strip()[2:].strip()
        if heading.lower() == title.lower():
            del lines[idx]
            if idx < len(lines) and not lines[idx].strip():
                del lines[idx]
    stripped = "\n".join(lines).rstrip()
    return f"{stripped}\n" if stripped else ""


def _rewrite_katex_math(content: str) -> str:
    def escape_inline_math(body: str) -> str:
        escaped = html.escape(body)
        pieces: list[str] = []
        prev = ""
        for ch in escaped:
            if ch == "_" and prev != "\\":
                pieces.append("&#95;")
            elif ch == "*" and prev != "\\":
                pieces.append("&#42;")
            else:
                pieces.append(ch)
            prev = ch
        return "".join(pieces)

    def display_replacer(match: re.Match[str]) -> str:
        body = html.escape(match.group(1).strip("\n"))
        return f'\n<div class="math-display">\n{body}\n</div>\n\n'

    content = re.sub(
        r"(?ms)^\$\$\s*\n(.*?)\n\$\$\s*$",
        display_replacer,
        content,
    )
    content = re.sub(
        r"(?m)^\$\$\s*(.+?)\s*\$\$\s*$",
        display_replacer,
        content,
    )
    content = re.sub(
        r"(?ms)^\\\[\s*\n(.*?)\n\\\]\s*$",
        display_replacer,
        content,
    )

    def inline_replacer(match: re.Match[str]) -> str:
        body = escape_inline_math(match.group(1))
        return f'<span class="math-inline">{body}</span>'

    content = re.sub(
        r"\\\((.+?)\\\)",
        inline_replacer,
        content,
    )
    return re.sub(
        r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)",
        inline_replacer,
        content,
    )


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def _yaml_dump(data: dict[str, Any], *, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_yaml_dump(value, indent=indent + 2))
            continue
        if isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.extend(_yaml_dump(item, indent=indent + 4))
                else:
                    lines.append(f"{prefix}  - {_yaml_scalar(item)}")
            continue
        lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
    return lines


def _front_matter(title: str, spec: PageSpec, source_rel: str, content: str) -> str:
    payload: dict[str, Any] = {
        "title": title,
        "weight": spec.weight,
        "canonical_repo_path": source_rel,
    }
    if spec.link_title:
        payload["linkTitle"] = spec.link_title
    if spec.description:
        payload["description"] = spec.description
    if spec.aliases:
        payload["aliases"] = list(spec.aliases)
    if spec.toc_hide:
        payload["toc_hide"] = True
    if spec.hide_summary:
        payload["hide_summary"] = True
    if spec.no_list:
        payload["no_list"] = True
    mermaid = "```mermaid" in content
    payload.update(spec.extra_params)
    if mermaid and "mermaid" not in payload:
        payload["mermaid"] = True
    return "---\n" + "\n".join(_yaml_dump(payload)) + "\n---\n\n"


def _output_rel_path(route: str) -> Path:
    route_path = Path(route)
    if route_path.name == "_index":
        return route_path.parent / "_index.md"
    return route_path.with_suffix(".md")


def _route_url(route: str, base_path: str) -> str:
    route_path = Path(route)
    if route_path.name == "_index":
        rel = route_path.parent.as_posix()
    else:
        rel = route_path.as_posix()
    docs_prefix = f"{base_path}/docs".rstrip("/")
    if not rel or rel == ".":
        return f"{docs_prefix}/"
    return f"{docs_prefix}/{rel.strip('/')}/"


def _normalize_repo_target(source_rel: str, target: str) -> str:
    source_dir = posixpath.dirname(source_rel)
    joined = posixpath.join(source_dir, target)
    return posixpath.normpath(joined)


def _repo_blob_url(normalized_rel: str) -> str:
    return f"{GITHUB_BLOB_MAIN}/{normalized_rel}"


def _build_route_map(base_path: str) -> dict[str, str]:
    return {spec.source_rel: _route_url(spec.route, base_path) for spec in PAGE_SPECS}


def _rewrite_target(
    target: str,
    source_rel: str,
    *,
    route_map: dict[str, str],
    repo_root: Path,
) -> str | None:
    if not target or target.startswith(("http://", "https://", "mailto:", "tel:", "#", "/")):
        return None
    if target.startswith(("{{<", "{{%")):
        return None

    path_only, anchor = (target.split("#", 1) + [""])[:2]
    normalized = _normalize_repo_target(source_rel, path_only)
    if "<" in normalized and ">" in normalized:
        return None

    rewritten = route_map.get(normalized)
    if rewritten is None:
        candidate = repo_root / normalized
        if candidate.exists():
            rewritten = _repo_blob_url(normalized)
        else:
            return None

    if anchor:
        rewritten = f"{rewritten}#{anchor}"
    return rewritten


def _rewrite_markdown_links(
    content: str,
    source_rel: str,
    *,
    route_map: dict[str, str],
    repo_root: Path,
) -> str:
    def replacer(match: re.Match[str]) -> str:
        target = (match.group(1) or "").strip()
        rewritten = _rewrite_target(target, source_rel, route_map=route_map, repo_root=repo_root)
        if rewritten is None:
            return match.group(0)
        start = match.start(1) - match.start(0)
        end = match.end(1) - match.start(0)
        full = match.group(0)
        return full[:start] + rewritten + full[end:]

    return MD_LINK_RE.sub(replacer, content)


def _rewrite_html_links(
    content: str,
    source_rel: str,
    *,
    route_map: dict[str, str],
    repo_root: Path,
) -> str:
    def replacer(match: re.Match[str]) -> str:
        raw_target = match.group(1) or match.group(2) or match.group(3) or ""
        target = raw_target.strip()
        rewritten = _rewrite_target(target, source_rel, route_map=route_map, repo_root=repo_root)
        if rewritten is None:
            return match.group(0)
        group_index = 1 if match.group(1) is not None else 2 if match.group(2) is not None else 3
        start = match.start(group_index) - match.start(0)
        end = match.end(group_index) - match.start(0)
        full = match.group(0)
        return full[:start] + rewritten + full[end:]

    return HTML_LINK_RE.sub(replacer, content)


def _sync_text(path: Path, expected: str, *, check: bool, changed: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == expected:
        return
    changed.append(path)
    if not check:
        path.write_text(expected, encoding="utf-8")


def _remove_stale_paths(
    generated_root: Path,
    valid_paths: set[Path],
    *,
    check: bool,
    changed: list[Path],
) -> None:
    if not generated_root.exists():
        return
    for path in generated_root.rglob("*"):
        if path.is_dir():
            continue
        if path not in valid_paths:
            changed.append(path)
            if not check:
                path.unlink()
    if check:
        return
    for path in sorted(generated_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def sync_hugo_content(repo_root: Path = REPO_ROOT, *, check: bool = False) -> list[Path]:
    base_path = _read_base_path(repo_root)
    route_map = _build_route_map(base_path)
    changed: list[Path] = []
    valid_paths: set[Path] = set()
    generated_root = repo_root / "site" / ".generated"
    content_docs_root = repo_root / "site" / ".generated" / "content" / "docs"

    for spec in PAGE_SPECS:
        source_path = repo_root / spec.source_rel
        content = _read_text(source_path)
        title = _title_from_markdown(content, _slug_title(spec.route))
        body = _strip_matching_h1(content, title)
        body = _rewrite_markdown_links(body, spec.source_rel, route_map=route_map, repo_root=repo_root)
        body = _rewrite_html_links(body, spec.source_rel, route_map=route_map, repo_root=repo_root)
        if spec.extra_params.get("katex"):
            body = _rewrite_katex_math(body)
        rendered = _front_matter(title, spec, spec.source_rel, content) + body
        output_path = content_docs_root / _output_rel_path(spec.route)
        valid_paths.add(output_path)
        _sync_text(output_path, rendered, check=check, changed=changed)

    _remove_stale_paths(generated_root, valid_paths, check=check, changed=changed)
    return changed


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated Hugo content is out of date.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    changed = sync_hugo_content(REPO_ROOT, check=args.check)
    if changed and args.check:
        print("Generated Hugo content is out of date:")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
        return 1

    if changed:
        print("Updated generated Hugo content:")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
    else:
        print("Generated Hugo content is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
