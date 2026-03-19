#!/usr/bin/env python3
"""Inspect top-level internal package dependencies for tab_foundry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import grimp

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "tab_foundry"
DEPENDENCY_DOC = REPO_ROOT / "docs" / "development" / "module-dependency-map.md"
OBSERVED_GRAPH_START = "<!-- module-graph:start -->"
OBSERVED_GRAPH_END = "<!-- module-graph:end -->"


@dataclass(slots=True)
class ModuleGraphReport:
    top_level_modules: list[str]
    top_level_edges: list[tuple[str, str]]
    top_level_cycles: list[tuple[str, str]]
    documented_edges: list[tuple[str, str]]
    undocumented_edges: list[tuple[str, str]]
    stale_documented_edges: list[tuple[str, str]]


def discover_top_level_modules(source_root: Path = SOURCE_ROOT) -> list[str]:
    modules = ["tab_foundry.__main__"]
    seen = set(modules)
    for path in sorted(source_root.iterdir()):
        if path.name in {"__init__.py", "__main__.py", "__pycache__"}:
            continue
        if path.is_dir():
            module_name = f"tab_foundry.{path.name}"
        elif path.suffix == ".py":
            module_name = f"tab_foundry.{path.stem}"
        else:
            continue
        if module_name in seen:
            continue
        seen.add(module_name)
        modules.append(module_name)
    return modules


def _top_level_name(module: str, *, top_modules: set[str]) -> str | None:
    if module == "tab_foundry.__main__":
        return module
    parts = module.split(".")
    if len(parts) < 2:
        return None
    candidate = ".".join(parts[:2])
    if candidate in top_modules:
        return candidate
    return None


def collect_top_level_edges(graph: grimp.ImportGraph, top_level_modules: list[str]) -> list[tuple[str, str]]:
    top_module_set = set(top_level_modules)
    edges: dict[str, set[str]] = defaultdict(set)
    for module in sorted(graph.modules):
        source = _top_level_name(module, top_modules=top_module_set)
        if source is None:
            continue
        for imported in graph.find_modules_directly_imported_by(module):
            target = _top_level_name(imported, top_modules=top_module_set)
            if target is None or source == target:
                continue
            edges[source].add(target)
    return sorted((source, target) for source, targets in edges.items() for target in sorted(targets))


def collect_top_level_cycles(graph: grimp.ImportGraph, top_level_modules: list[str]) -> list[tuple[str, str]]:
    cycles: list[tuple[str, str]] = []
    for index, source in enumerate(top_level_modules):
        for target in top_level_modules[index + 1 :]:
            if graph.chain_exists(source, target, as_packages=True) and graph.chain_exists(
                target,
                source,
                as_packages=True,
            ):
                cycles.append((source, target))
    return cycles


def parse_documented_edges(doc_path: Path = DEPENDENCY_DOC) -> list[tuple[str, str]]:
    if not doc_path.exists():
        return []

    doc_lines = doc_path.read_text(encoding="utf-8").splitlines()
    try:
        start_index = doc_lines.index(OBSERVED_GRAPH_START) + 1
        end_index = doc_lines.index(OBSERVED_GRAPH_END)
    except ValueError:
        return []

    blocks: list[str] = []
    current: list[str] = []
    for raw_line in doc_lines[start_index:end_index]:
        if raw_line.startswith("- "):
            if current:
                blocks.append(" ".join(line.strip() for line in current))
            current = [raw_line]
            continue
        if current and raw_line.startswith("  "):
            current.append(raw_line)
            continue
        if current:
            blocks.append(" ".join(line.strip() for line in current))
            current = []
    if current:
        blocks.append(" ".join(line.strip() for line in current))

    edges: set[tuple[str, str]] = set()
    for block in blocks:
        if " depends on " not in block:
            continue
        modules = re.findall(r"`([^`]+)`", block)
        if len(modules) < 2:
            continue
        source = modules[0]
        for target in modules[1:]:
            edges.add((source, target))
    return sorted(edges)


def build_module_graph_report(repo_root: Path = REPO_ROOT) -> ModuleGraphReport:
    graph = grimp.build_graph("tab_foundry", include_external_packages=False)
    top_level_modules = discover_top_level_modules(repo_root / "src" / "tab_foundry")
    top_level_edges = collect_top_level_edges(graph, top_level_modules)
    top_level_cycles = collect_top_level_cycles(graph, top_level_modules)
    documented_edges = parse_documented_edges(repo_root / "docs" / "development" / "module-dependency-map.md")
    actual_edge_set = set(top_level_edges)
    documented_edge_set = set(documented_edges)
    undocumented_edges = sorted(actual_edge_set - documented_edge_set)
    stale_documented_edges = sorted(documented_edge_set - actual_edge_set)
    return ModuleGraphReport(
        top_level_modules=top_level_modules,
        top_level_edges=top_level_edges,
        top_level_cycles=top_level_cycles,
        documented_edges=documented_edges,
        undocumented_edges=undocumented_edges,
        stale_documented_edges=stale_documented_edges,
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    parser.add_argument(
        "--fail-on-doc-drift",
        action="store_true",
        help="Exit non-zero when module-dependency-map.md differs from the observed graph.",
    )
    return parser.parse_args(list(argv))


def _emit_text(report: ModuleGraphReport) -> str:
    lines = ["Top-level internal dependency edges:"]
    for source, target in report.top_level_edges:
        lines.append(f"- {source} -> {target}")
    lines.append("")
    lines.append("Top-level cycle candidates:")
    if report.top_level_cycles:
        for source, target in report.top_level_cycles:
            lines.append(f"- {source} <-> {target}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Doc drift vs docs/development/module-dependency-map.md:")
    if report.undocumented_edges:
        lines.append("- undocumented edges:")
        lines.extend(f"  - {source} -> {target}" for source, target in report.undocumented_edges)
    else:
        lines.append("- undocumented edges: none")
    if report.stale_documented_edges:
        lines.append("- stale documented edges:")
        lines.extend(f"  - {source} -> {target}" for source, target in report.stale_documented_edges)
    else:
        lines.append("- stale documented edges: none")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = build_module_graph_report(REPO_ROOT)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(_emit_text(report))

    if args.fail_on_doc_drift and (report.undocumented_edges or report.stale_documented_edges):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
