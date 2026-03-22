#!/usr/bin/env python3
"""Sync canonical repo docs into the Hugo docs site."""

from __future__ import annotations

import argparse
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
        source_rel="docs/what-is-tab-foundry.md",
        route="getting-started/what-is-tab-foundry",
        weight=10,
        description="Repo overview of what tab-foundry does, what it produces, and how to route into the relevant docs paths.",
        aliases=("/docs/what-is-tab-foundry/",),
    ),
    PageSpec(
        source_rel="docs/getting-started.md",
        route="getting-started/_index",
        weight=10,
        description="General orientation page for repo overview, research contributions, and ML engineering paths.",
        link_title="Start Here",
        no_list=True,
    ),
    PageSpec(
        source_rel="docs/research-contributors.md",
        route="research-contributors/_index",
        weight=20,
        description="Research-first route through architecture, sweeps, synthetic data, and model-breadth work.",
        link_title="Research",
        no_list=True,
    ),
    PageSpec(
        source_rel="docs/ml-engineering.md",
        route="ml-engineering/_index",
        weight=30,
        description="Operational route through manifests, runs, checkpoints, export bundles, and verification paths.",
        link_title="ML Engineering",
        no_list=True,
    ),
    PageSpec(
        source_rel="program.md",
        route="research-contributors/sweep-contract",
        weight=20,
        link_title="Sweep Contract",
        description="Active system-delta sweep contract, locked surface, and execution loop.",
        aliases=("/docs/sweep-contract/",),
    ),
    PageSpec(
        source_rel="docs/workflows.md",
        route="ml-engineering/workflows",
        weight=20,
        description="Operational runbooks and command syntax for the canonical workflows.",
        aliases=("/docs/workflows/",),
    ),
    PageSpec(
        source_rel="docs/development/roadmap.md",
        route="development/roadmap",
        weight=10,
        description="Canonical planning state and TF-RD priority order.",
        link_title="Roadmap",
        extra_params={"mermaid": True},
    ),
    PageSpec(
        source_rel="docs/development/model-architecture.md",
        route="development/model-architecture",
        weight=20,
        description="Detailed architecture reference for the staged and simple model surfaces.",
        extra_params={"mermaid": True},
    ),
    PageSpec(
        source_rel="docs/development/design-decisions.md",
        route="development/design-decisions",
        weight=30,
        description="Enduring architecture direction, repo-structure policy, and publishing decisions.",
    ),
    PageSpec(
        source_rel="docs/development/codebase-navigation.md",
        route="development/codebase-navigation",
        weight=40,
        description="Package layout, workflow surfaces, and entry-point inventory.",
    ),
    PageSpec(
        source_rel="docs/development/module-dependency-map.md",
        route="development/module-dependency-map",
        weight=50,
        description="Observed top-level package graph and intended dependency direction.",
    ),
    PageSpec(
        source_rel="docs/development/model-config.md",
        route="development/model-config",
        weight=60,
        description="Model configuration reference and resolution rules.",
    ),
    PageSpec(
        source_rel="docs/development/architecture-deltas.md",
        route="development/architecture-deltas",
        weight=70,
        description="Diagram-first comparison of the target lane and reference architectures.",
        extra_params={"mermaid": True},
    ),
    PageSpec(
        source_rel="docs/development/dataset-curation.md",
        route="development/dataset-curation",
        weight=80,
        description="Dataset curation and license-review gate for real-data ladders.",
    ),
    PageSpec(
        source_rel="reference/README.md",
        route="reference/_index",
        weight=50,
        description="Index for literature notes, evidence maps, and research surfaces.",
        link_title="References",
    ),
    PageSpec(
        source_rel="reference/papers.md",
        route="reference/papers",
        weight=10,
        description="Curated papers, adjacent-repo borrowing rules, and adoption tiers.",
    ),
    PageSpec(
        source_rel="reference/evidence.md",
        route="reference/evidence",
        weight=20,
        description="Roadmap-to-reference mapping and repo-local evidence notes.",
    ),
    PageSpec(
        source_rel="reference/system_delta_sweeps/tf_rd_013_data_source_contract_v1/support/README.md",
        route="reference/tf-rd-013-support",
        weight=30,
        link_title="TF-RD-013 Support",
        description="Support bundle notes for the current dagzoo synthetic-data contract.",
    ),
    PageSpec(
        source_rel="CONTRIBUTING.md",
        route="getting-started/contributing",
        weight=30,
        description="Contribution workflow and review readiness for research contributors.",
        aliases=("/docs/contributing/",),
    ),
    PageSpec(
        source_rel="docs/glossary.md",
        route="getting-started/glossary",
        weight=20,
        description="Shared vocabulary for architecture and sweep work.",
        aliases=("/docs/glossary/",),
    ),
    PageSpec(
        source_rel="README.md",
        route="repo-overview",
        weight=110,
        description="Top-level repo overview and quickstart.",
        link_title="Repo Overview",
        toc_hide=True,
        hide_summary=True,
    ),
    PageSpec(
        source_rel="docs/inference.md",
        route="ml-engineering/inference",
        weight=30,
        description="Export bundle schema and inference validation contract.",
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
