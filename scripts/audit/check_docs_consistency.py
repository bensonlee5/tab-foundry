#!/usr/bin/env python3
"""Check source docs against the live CLI and current ownership surfaces."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

from tab_foundry.cli.app import build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    "README.md",
    "docs",
    "reference",
    "program.md",
)
CODEBASE_NAVIGATION_PATH = REPO_ROOT / "docs" / "development" / "codebase-navigation.md"
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
COMMAND_INVENTORY_RE = re.compile(r"- `(?P<command>tab-foundry [^`]+)`$")
PYTHON_MODULE_CMD_RE = re.compile(r"\bpython(?:3)?\s+-m\s+tab_foundry\.[^\s`]+")
PYTHON_SCRIPT_CMD_RE = re.compile(r"^(?P<python>(?:\.venv/bin/)?python(?:3)?)\s+(?P<script>scripts/[^\s`]+\.py)\b")
README_CLI_TREE_SUMMARY_RE = re.compile(r"<summary>\s*Full CLI tree\s*</summary>")
STALE_REFERENCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:src/)?tab_foundry/research/sweep/core\.py\b|\bresearch/sweep/core\.py\b"),
        "removed sweep barrel reference",
    ),
    (
        re.compile(r"\btab_foundry\.research\.sweep\.core\b"),
        "removed sweep barrel module reference",
    ),
    (
        re.compile(r"\b(?:src/)?tab_foundry/research/sweep/runner\.py\b|\bresearch/sweep/runner\.py\b"),
        "removed sweep runner reference",
    ),
    (
        re.compile(r"\btab_foundry\.research\.sweep\.runner\b"),
        "removed sweep runner module reference",
    ),
    (
        re.compile(r"\btab_foundry\.research\.system_delta(?:_execute|_promote)?\b"),
        "removed research compatibility wrapper reference",
    ),
    (
        re.compile(r"\b(?:src/)?tab_foundry/research/system_delta(?:_execute|_promote)?\.py\b"),
        "removed research compatibility wrapper path",
    ),
    (
        re.compile(r"\btab_foundry\.bench\.benchmark_run_registry\b"),
        "removed bench registry CLI-wrapper module reference",
    ),
    (
        re.compile(r"\b(?:src/)?tab_foundry/bench/benchmark_run_registry\.py\b|\bbench/benchmark_run_registry\.py\b"),
        "removed bench registry CLI-wrapper path",
    ),
    (
        re.compile(r"\btab_foundry\.bench\.control_baseline\b"),
        "removed control-baseline CLI-wrapper module reference",
    ),
    (
        re.compile(r"\b(?:src/)?tab_foundry/bench/control_baseline\.py\b|\bbench/control_baseline\.py\b"),
        "removed control-baseline CLI-wrapper path",
    ),
    (
        re.compile(r"\btab_foundry\.bench\.prior_train\b"),
        "removed bench prior-training module reference",
    ),
    (
        re.compile(r"\b(?:src/)?tab_foundry/bench/prior_train\.py\b|\bbench/prior_train\.py\b"),
        "removed bench prior-training path",
    ),
)
ALLOWED_STANDALONE_PYTHON_SCRIPTS = frozenset(
    {
        "scripts/materialize_tf_rd_013_support.py",
        "scripts/bench/instability_audit.py",
        "scripts/bench/iris.py",
        "scripts/bench/nanotabpfn_helper.py",
        "scripts/bench/tabiclv2_helper.py",
    }
)
TAB_FOUNDRY_EXECUTABLES = {"tab-foundry", ".venv/bin/tab-foundry"}
README_CLI_TREE_ERROR = (
    "README must not duplicate a hand-maintained CLI tree; use "
    "`docs/development/codebase-navigation.md` for the canonical command inventory"
)


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".md":
            yield root
        return
    for path in root.rglob("*.md"):
        if path.is_file():
            yield path


def _iter_doc_snippets(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            if stripped:
                yield lineno, stripped
            continue
        for match in INLINE_CODE_RE.finditer(line):
            snippet = match.group(1).strip()
            if snippet:
                yield lineno, snippet


def _subparser_action(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if action.__class__.__name__ == "_SubParsersAction":
            return action
    return None


def _build_cli_tree(parser: argparse.ArgumentParser) -> dict[str, dict]:
    action = _subparser_action(parser)
    if action is None:
        return {}
    return {name: _build_cli_tree(child) for name, child in action.choices.items()}


def live_cli_leaf_commands() -> set[str]:
    parser = build_parser()

    def _walk(node: dict[str, dict], prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        if not node:
            return [prefix]
        result: list[tuple[str, ...]] = []
        for name, child in node.items():
            result.extend(_walk(child, prefix + (name,)))
        return result

    return {"tab-foundry " + " ".join(path) for path in _walk(_build_cli_tree(parser)) if path}


def _normalize_snippet(snippet: str) -> str:
    normalized = snippet.strip()
    if normalized.endswith("\\"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _tokenize_snippet(snippet: str) -> list[str]:
    normalized = _normalize_snippet(snippet)
    try:
        return [token for token in shlex.split(normalized) if token != "\\"]
    except ValueError:
        return normalized.split()


def _validate_tab_foundry_command(tokens: list[str], cli_tree: dict[str, dict]) -> str | None:
    if not tokens or tokens[0] not in TAB_FOUNDRY_EXECUTABLES:
        return None
    args = tokens[1:]
    if not args:
        return None
    if args[0] in {"-h", "--help"}:
        return None

    node = cli_tree
    consumed: list[str] = []
    for token in args:
        if token in {"-h", "--help"}:
            return None
        if token == "...":
            return None
        if "<" in token and ">" in token:
            return None
        if token.startswith("-") or "=" in token:
            return None if consumed else None
        if token in node:
            consumed.append(token)
            node = node[token]
            continue
        if not consumed:
            return f"unknown tab-foundry command token {token!r}"
        if node:
            return (
                f"unknown tab-foundry command token {token!r} after "
                f"`tab-foundry {' '.join(consumed)}`"
            )
        return None
    return None if consumed else "missing command after tab-foundry"


def _validate_python_script_command(tokens: list[str], *, repo_root: Path) -> str | None:
    if not tokens:
        return None
    match = PYTHON_SCRIPT_CMD_RE.match(" ".join(tokens))
    if match is None:
        return None
    script_path = match.group("script")
    python_executable = match.group("python")
    if script_path in ALLOWED_STANDALONE_PYTHON_SCRIPTS:
        if python_executable != ".venv/bin/python":
            return f"`{script_path}` must be documented via `.venv/bin/python {script_path} ...`"
        if not (repo_root / script_path).exists():
            return f"documented standalone Python entrypoint is missing: `{script_path}`"
        return None
    return f"unsupported Python script entrypoint in docs: `{script_path}`"


def _validate_repo_script_command(tokens: list[str], *, repo_root: Path) -> str | None:
    if not tokens:
        return None
    script_token = tokens[0]
    if script_token.startswith("./"):
        script_path = script_token[2:]
    else:
        script_path = script_token
    if not script_path.startswith("scripts/"):
        return None
    if not (repo_root / script_path).exists():
        return f"documented repo-local script entrypoint is missing: `{script_path}`"
    return None


def _extract_codebase_navigation_inventory(path: Path) -> set[str]:
    commands: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = COMMAND_INVENTORY_RE.match(line.strip())
        if match is not None:
            commands.add(match.group("command"))
    return commands


def _find_disallowed_readme_cli_tree(path: Path, lines: list[str]) -> tuple[int, str] | None:
    if path.name != "README.md":
        return None

    in_fence = False
    saw_tab_foundry_root = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if README_CLI_TREE_SUMMARY_RE.search(stripped):
            return lineno, README_CLI_TREE_ERROR
        if stripped.startswith("```"):
            in_fence = not in_fence
            if not in_fence:
                saw_tab_foundry_root = False
            continue
        if not in_fence:
            continue
        if stripped == "tab-foundry":
            saw_tab_foundry_root = True
            continue
        if saw_tab_foundry_root and ("├──" in stripped or "└──" in stripped):
            return lineno, README_CLI_TREE_ERROR
    return None


def compare_codebase_navigation_inventory(
    repo_root: Path = REPO_ROOT,
) -> tuple[set[str], set[str]]:
    doc_path = repo_root / "docs" / "development" / "codebase-navigation.md"
    if not doc_path.exists():
        return set(), set()
    live = live_cli_leaf_commands()
    documented = _extract_codebase_navigation_inventory(doc_path)
    return live - documented, documented - live


def scan_docs_consistency(
    repo_root: Path = REPO_ROOT,
    roots: Iterable[str] = DEFAULT_ROOTS,
) -> list[tuple[Path, int, str]]:
    parser = build_parser()
    cli_tree = _build_cli_tree(parser)
    errors: list[tuple[Path, int, str]] = []

    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            errors.append((root, 0, "missing scan root"))
            continue
        for path in _iter_markdown_files(root):
            lines = path.read_text(encoding="utf-8").splitlines()
            readme_cli_tree_error = _find_disallowed_readme_cli_tree(path, lines)
            if readme_cli_tree_error is not None:
                errors.append((path, *readme_cli_tree_error))
            for lineno, line in enumerate(lines, start=1):
                if PYTHON_MODULE_CMD_RE.search(line):
                    errors.append(
                        (
                            path,
                            lineno,
                            "stale direct module entrypoint; use the packaged CLI or a documented repo-local script entrypoint",
                        )
                    )
                for pattern, message in STALE_REFERENCE_PATTERNS:
                    if pattern.search(line):
                        errors.append((path, lineno, message))
            for lineno, snippet in _iter_doc_snippets(path):
                tokens = _tokenize_snippet(snippet)
                if not tokens:
                    continue
                if tokens[0] in TAB_FOUNDRY_EXECUTABLES:
                    message = _validate_tab_foundry_command(tokens, cli_tree)
                    if message is not None:
                        errors.append((path, lineno, message))
                    continue
                if tokens[0].startswith("python") or tokens[0] == ".venv/bin/python":
                    message = _validate_python_script_command(tokens, repo_root=repo_root)
                    if message is not None:
                        errors.append((path, lineno, message))
                    continue
                if tokens[0].startswith("./scripts/") or tokens[0].startswith("scripts/"):
                    message = _validate_repo_script_command(tokens, repo_root=repo_root)
                    if message is not None:
                        errors.append((path, lineno, message))

    codebase_path = repo_root / "docs" / "development" / "codebase-navigation.md"
    if codebase_path.exists():
        missing, extra = compare_codebase_navigation_inventory(repo_root)
        for command in sorted(missing):
            errors.append((codebase_path, 0, f"missing canonical CLI inventory entry: `{command}`"))
        for command in sorted(extra):
            errors.append((codebase_path, 0, f"stale canonical CLI inventory entry: `{command}`"))

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
    errors = scan_docs_consistency(REPO_ROOT, args.roots)
    if errors:
        print("Docs consistency issues found:")
        for path, lineno, message in errors:
            location = path.relative_to(REPO_ROOT) if path.exists() else path
            suffix = f":{lineno}" if lineno else ""
            print(f"- {location}{suffix} -> {message}")
        return 1

    print("Docs consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
