#!/usr/bin/env python3
"""Diff-aware review and verification helper for repo-local workflows."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_INDEX_PATH = Path(__file__).with_name("dev_index.yaml")
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_RUFF = REPO_ROOT / ".venv" / "bin" / "ruff"
VENV_MDFORMAT = REPO_ROOT / ".venv" / "bin" / "mdformat"


@dataclass(frozen=True, slots=True)
class CheckSpec:
    name: str
    description: str
    argv_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class PathRule:
    name: str
    subsystem: str
    globs: tuple[str, ...]
    checks: tuple[str, ...]
    core: bool


@dataclass(frozen=True, slots=True)
class CompanionRule:
    name: str
    when_any: tuple[str, ...]
    suggest_all: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class FullVerifyEscalation:
    any_changed: tuple[str, ...]
    min_core_subsystems: int
    unmatched_paths: bool


@dataclass(frozen=True, slots=True)
class DevIndex:
    default_base_ref: str
    path_rules: tuple[PathRule, ...]
    full_verify_checks: tuple[str, ...]
    escalation: FullVerifyEscalation
    companion_rules: tuple[CompanionRule, ...]


@dataclass(frozen=True, slots=True)
class ReviewScope:
    changed_paths: tuple[str, ...]
    subsystem_paths: dict[str, tuple[str, ...]]
    core_subsystems: tuple[str, ...]
    unmatched_paths: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    check_ids: tuple[str, ...]
    escalated_to_full: bool
    escalation_reasons: tuple[str, ...]
    scope: ReviewScope


CHECK_SPECS: dict[str, CheckSpec] = {
    "mdformat": CheckSpec(
        name="mdformat",
        description="Check markdown formatting",
        argv_groups=(
            (
                str(VENV_MDFORMAT),
                "--check",
                "AGENTS.md",
                "README.md",
                "CHANGELOG.md",
                "program.md",
                "docs",
                "reference",
            ),
        ),
    ),
    "audit": CheckSpec(
        name="audit",
        description="Run repo audit scripts",
        argv_groups=(
            (str(VENV_PYTHON), "scripts/audit/check_repo_paths.py"),
            (str(VENV_PYTHON), "scripts/audit/check_markdown_links.py"),
            (str(VENV_PYTHON), "scripts/audit/module_graph.py", "--fail-on-doc-drift"),
        ),
    ),
    "ruff": CheckSpec(
        name="ruff",
        description="Run Ruff against repo Python sources",
        argv_groups=((str(VENV_RUFF), "check", "src", "tests", "scripts"),),
    ),
    "mypy": CheckSpec(
        name="mypy",
        description="Run mypy against src",
        argv_groups=((str(VENV_PYTHON), "-m", "mypy", "src"),),
    ),
    "pytest_audit": CheckSpec(
        name="pytest_audit",
        description="Run audit tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/audit"),),
    ),
    "pytest_data": CheckSpec(
        name="pytest_data",
        description="Run data tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/data"),),
    ),
    "pytest_model": CheckSpec(
        name="pytest_model",
        description="Run model tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/model"),),
    ),
    "pytest_training": CheckSpec(
        name="pytest_training",
        description="Run training tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/training"),),
    ),
    "pytest_runtime": CheckSpec(
        name="pytest_runtime",
        description="Run runtime tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/runtime"),),
    ),
    "pytest_smoke": CheckSpec(
        name="pytest_smoke",
        description="Run smoke tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/smoke"),),
    ),
    "pytest_property": CheckSpec(
        name="pytest_property",
        description="Run property tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/property"),),
    ),
    "pytest_export": CheckSpec(
        name="pytest_export",
        description="Run export tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/export"),),
    ),
    "pytest_benchmark": CheckSpec(
        name="pytest_benchmark",
        description="Run benchmark tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/benchmark"),),
    ),
    "pytest_research": CheckSpec(
        name="pytest_research",
        description="Run research tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/research"),),
    ),
    "pytest_cli": CheckSpec(
        name="pytest_cli",
        description="Run CLI tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/cli"),),
    ),
    "pytest_config": CheckSpec(
        name="pytest_config",
        description="Run config tests",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q", "tests/config"),),
    ),
    "pytest_all": CheckSpec(
        name="pytest_all",
        description="Run the full pytest suite",
        argv_groups=((str(VENV_PYTHON), "-m", "pytest", "-q"),),
    ),
}
CHECK_ORDER = tuple(CHECK_SPECS)
_PRECOMMIT_SKIPPED_PYTEST_CHECKS = frozenset({"pytest_runtime", "pytest_smoke", "pytest_property"})
_PRECOMMIT_TEST_CHECK_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("tests/audit/", "pytest_audit"),
    ("tests/data/", "pytest_data"),
    ("tests/model/", "pytest_model"),
    ("tests/training/", "pytest_training"),
    ("tests/runtime/", "pytest_runtime"),
    ("tests/smoke/", "pytest_smoke"),
    ("tests/property/", "pytest_property"),
    ("tests/export/", "pytest_export"),
    ("tests/benchmark/", "pytest_benchmark"),
    ("tests/research/", "pytest_research"),
    ("tests/cli/", "pytest_cli"),
    ("tests/config/", "pytest_config"),
)


def _string_list(payload: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(payload, list) or not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError(f"{field_name} must be a non-empty list of strings")
    return tuple(item.strip() for item in payload)


def load_dev_index(index_path: Path = DEV_INDEX_PATH) -> DevIndex:
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dev_index.yaml must contain a mapping at the top level")

    default_base_ref = payload.get("default_base_ref")
    if not isinstance(default_base_ref, str) or not default_base_ref.strip():
        raise ValueError("default_base_ref must be a non-empty string")

    raw_rules = payload.get("path_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("path_rules must be a non-empty list")
    path_rules: list[PathRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"path_rules[{index}] must be a mapping")
        path_rules.append(
            PathRule(
                name=_required_string(raw_rule, "name", context=f"path_rules[{index}]"),
                subsystem=_required_string(raw_rule, "subsystem", context=f"path_rules[{index}]"),
                globs=_string_list(raw_rule.get("globs"), field_name=f"path_rules[{index}].globs"),
                checks=_string_list(raw_rule.get("checks"), field_name=f"path_rules[{index}].checks"),
                core=bool(raw_rule.get("core", False)),
            )
        )

    raw_full_verify = payload.get("full_verify")
    if not isinstance(raw_full_verify, dict):
        raise ValueError("full_verify must be a mapping")
    full_verify_checks = _string_list(raw_full_verify.get("checks"), field_name="full_verify.checks")

    raw_escalation = payload.get("full_verify_escalation")
    if not isinstance(raw_escalation, dict):
        raise ValueError("full_verify_escalation must be a mapping")
    min_core_subsystems = raw_escalation.get("min_core_subsystems")
    unmatched_paths = raw_escalation.get("unmatched_paths")
    if not isinstance(min_core_subsystems, int) or min_core_subsystems < 1:
        raise ValueError("full_verify_escalation.min_core_subsystems must be a positive integer")
    if not isinstance(unmatched_paths, bool):
        raise ValueError("full_verify_escalation.unmatched_paths must be a boolean")
    escalation = FullVerifyEscalation(
        any_changed=_string_list(raw_escalation.get("any_changed"), field_name="full_verify_escalation.any_changed"),
        min_core_subsystems=min_core_subsystems,
        unmatched_paths=unmatched_paths,
    )

    raw_companion_rules = payload.get("companion_rules")
    if not isinstance(raw_companion_rules, list):
        raise ValueError("companion_rules must be a list")
    companion_rules: list[CompanionRule] = []
    for index, raw_rule in enumerate(raw_companion_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"companion_rules[{index}] must be a mapping")
        companion_rules.append(
            CompanionRule(
                name=_required_string(raw_rule, "name", context=f"companion_rules[{index}]"),
                when_any=_string_list(raw_rule.get("when_any"), field_name=f"companion_rules[{index}].when_any"),
                suggest_all=_string_list(
                    raw_rule.get("suggest_all"),
                    field_name=f"companion_rules[{index}].suggest_all",
                ),
                message=_required_string(raw_rule, "message", context=f"companion_rules[{index}]"),
            )
        )

    _validate_check_ids(path_rules, full_verify_checks)
    return DevIndex(
        default_base_ref=default_base_ref.strip(),
        path_rules=tuple(path_rules),
        full_verify_checks=full_verify_checks,
        escalation=escalation,
        companion_rules=tuple(companion_rules),
    )


def _required_string(payload: dict[str, object], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _validate_check_ids(path_rules: Sequence[PathRule], full_verify_checks: Sequence[str]) -> None:
    unknown_checks = {
        check_id for rule in path_rules for check_id in rule.checks if check_id not in CHECK_SPECS
    }
    unknown_checks.update(check_id for check_id in full_verify_checks if check_id not in CHECK_SPECS)
    if unknown_checks:
        unknown_text = ", ".join(sorted(unknown_checks))
        raise ValueError(f"dev_index.yaml references unknown checks: {unknown_text}")


def _path_matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _ordered_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def classify_changed_paths(changed_paths: Sequence[str], index: DevIndex) -> ReviewScope:
    ordered_paths = _ordered_unique(sorted(path.strip() for path in changed_paths if path.strip()))
    subsystem_paths: defaultdict[str, list[str]] = defaultdict(list)
    core_subsystems: list[str] = []
    unmatched_paths: list[str] = []

    for path in ordered_paths:
        matched_rules = [rule for rule in index.path_rules if _path_matches(path, rule.globs)]
        if not matched_rules:
            unmatched_paths.append(path)
            continue
        for rule in matched_rules:
            subsystem_paths[rule.subsystem].append(path)
            if rule.core:
                core_subsystems.append(rule.subsystem)

    warnings = _build_companion_warnings(ordered_paths, index.companion_rules)
    return ReviewScope(
        changed_paths=ordered_paths,
        subsystem_paths={name: _ordered_unique(paths) for name, paths in sorted(subsystem_paths.items())},
        core_subsystems=_ordered_unique(core_subsystems),
        unmatched_paths=tuple(unmatched_paths),
        warnings=warnings,
    )


def _build_companion_warnings(
    changed_paths: Sequence[str],
    companion_rules: Sequence[CompanionRule],
) -> tuple[str, ...]:
    warnings: list[str] = []
    changed_set = set(changed_paths)
    for rule in companion_rules:
        if not any(_path_matches(path, rule.when_any) for path in changed_paths):
            continue
        missing = [path for path in rule.suggest_all if path not in changed_set]
        if missing:
            warnings.append(f"{rule.message} Missing from diff: {', '.join(missing)}.")
    return tuple(warnings)


def build_verification_plan(changed_paths: Sequence[str], index: DevIndex) -> VerificationPlan:
    scope = classify_changed_paths(changed_paths, index)
    escalation_reasons: list[str] = []

    if any(_path_matches(path, index.escalation.any_changed) for path in scope.changed_paths):
        escalation_reasons.append("diff touches repo-wide verification surfaces")
    if index.escalation.unmatched_paths and scope.unmatched_paths:
        escalation_reasons.append("diff includes paths without an affected-scope mapping")
    if len(scope.core_subsystems) >= index.escalation.min_core_subsystems:
        joined = ", ".join(scope.core_subsystems)
        escalation_reasons.append(f"diff spans multiple core subsystems: {joined}")

    if escalation_reasons:
        check_ids = index.full_verify_checks
        escalated_to_full = True
    else:
        selected_checks: list[str] = []
        for path in scope.changed_paths:
            for rule in index.path_rules:
                if _path_matches(path, rule.globs):
                    selected_checks.extend(rule.checks)
        check_ids = tuple(check_id for check_id in CHECK_ORDER if check_id in selected_checks)
        escalated_to_full = False

    return VerificationPlan(
        check_ids=check_ids,
        escalated_to_full=escalated_to_full,
        escalation_reasons=tuple(escalation_reasons),
        scope=scope,
    )


def read_merge_base(repo_root: Path, *, base_ref: str) -> str:
    return _run_git_capture(repo_root, "merge-base", base_ref, "HEAD").strip()


def collect_changed_paths(repo_root: Path, *, base_ref: str) -> tuple[str, ...]:
    merge_base = read_merge_base(repo_root, base_ref=base_ref)
    changed = set()
    changed.update(_run_git_lines(repo_root, "diff", "--name-only", f"{merge_base}..HEAD"))
    changed.update(_run_git_lines(repo_root, "diff", "--name-only", "--cached"))
    changed.update(_run_git_lines(repo_root, "diff", "--name-only"))
    changed.update(_run_git_lines(repo_root, "ls-files", "--others", "--exclude-standard"))
    return _ordered_unique(sorted(path for path in changed if path))


def _run_git_capture(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _run_git_lines(repo_root: Path, *args: str) -> tuple[str, ...]:
    output = _run_git_capture(repo_root, *args)
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def render_review_report(plan: VerificationPlan, *, base_ref: str, merge_base: str) -> str:
    lines = [
        f"Base ref: {base_ref}",
        f"Merge base: {merge_base}",
        f"Changed files: {len(plan.scope.changed_paths)}",
    ]
    if not plan.scope.changed_paths:
        lines.append("- none")
    else:
        for subsystem, paths in sorted(plan.scope.subsystem_paths.items()):
            lines.append(f"{subsystem}:")
            for path in paths:
                lines.append(f"- {path}")
        if plan.scope.unmatched_paths:
            lines.append("unmatched:")
            for path in plan.scope.unmatched_paths:
                lines.append(f"- {path}")

    if plan.scope.warnings:
        lines.append("Warnings:")
        for warning in plan.scope.warnings:
            lines.append(f"- {warning}")

    lines.append("Verification:")
    if not plan.scope.changed_paths:
        lines.append("- No changed files detected against the selected base ref.")
    elif plan.escalated_to_full:
        lines.append("- Escalated to full verification.")
        for reason in plan.escalation_reasons:
            lines.append(f"- Reason: {reason}")
        lines.append("- Command: ./scripts/dev verify full")
    else:
        checks_text = ", ".join(plan.check_ids)
        lines.append(f"- Affected checks: {checks_text}")
        lines.append(f"- Command: ./scripts/dev verify affected --base-ref {base_ref}")
    return "\n".join(lines)


def _is_pytest_command(argv: Sequence[str]) -> bool:
    return len(argv) >= 3 and argv[1] == "-m" and argv[2] == "pytest"


def _live_pytest_progress_requested() -> bool:
    raw = os.environ.get("TAB_FOUNDRY_LIVE_PYTEST", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_live_stream(argv: Sequence[str]) -> bool:
    return _live_pytest_progress_requested() and _is_pytest_command(argv) and not sys.stdout.isatty()


def _run_live_streamed(argv: Sequence[str]) -> int:
    try:
        tty_handle = open("/dev/tty", "wb", buffering=0)
    except OSError:
        result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        return int(result.returncode)

    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        assert process.stdout is not None
        while True:
            chunk = os.read(process.stdout.fileno(), 1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            tty_handle.write(chunk)
            tty_handle.flush()
        return int(process.wait())
    finally:
        if process.stdout is not None:
            process.stdout.close()
        tty_handle.close()


def run_check_command(argv: Sequence[str]) -> int:
    if _should_live_stream(argv):
        return _run_live_streamed(argv)
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


def _precommit_pytest_argv(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(argv)
    if not _is_pytest_command(normalized):
        return normalized
    for index, value in enumerate(normalized):
        if value == "-n" and index + 1 < len(normalized):
            return normalized
        if value.startswith("-n") and value != "-n":
            return normalized
    return (*normalized, "-n", "0")


def execute_check_ids(
    check_ids: Sequence[str],
    *,
    argv_normalizer: Callable[[Sequence[str]], Sequence[str]] | None = None,
) -> int:
    for check_id in check_ids:
        spec = CHECK_SPECS[check_id]
        print(f"[dev] {spec.description}", flush=True)
        for argv in spec.argv_groups:
            resolved_argv = tuple(argv if argv_normalizer is None else argv_normalizer(argv))
            print(f"+ {shlex.join(resolved_argv)}", flush=True)
            result = run_check_command(resolved_argv)
            if result != 0:
                return int(result)
    return 0


def _explicit_pytest_paths(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        path
        for path in _ordered_unique(paths)
        if path.startswith("tests/") and path.endswith(".py")
    )


def _explicit_pytest_check_ids(paths: Sequence[str]) -> frozenset[str]:
    explicit_checks: set[str] = set()
    for path in _explicit_pytest_paths(paths):
        for prefix, check_id in _PRECOMMIT_TEST_CHECK_BY_PREFIX:
            if path.startswith(prefix):
                explicit_checks.add(check_id)
                break
    return frozenset(explicit_checks)


def build_precommit_check_ids(
    changed_paths: Sequence[str],
    index: DevIndex,
) -> tuple[VerificationPlan, tuple[str, ...]]:
    plan = build_verification_plan(changed_paths, index)
    explicit_paths = _explicit_pytest_paths(changed_paths)
    explicit_check_ids = _explicit_pytest_check_ids(changed_paths)
    filtered_check_ids = tuple(
        check_id
        for check_id in plan.check_ids
        if check_id != "pytest_all"
        and check_id not in _PRECOMMIT_SKIPPED_PYTEST_CHECKS
        and check_id not in explicit_check_ids
    )
    return (
        VerificationPlan(
            check_ids=filtered_check_ids,
            escalated_to_full=plan.escalated_to_full,
            escalation_reasons=plan.escalation_reasons,
            scope=plan.scope,
        ),
        explicit_paths,
    )


def execute_precommit_paths(paths: Sequence[str], index: DevIndex) -> int:
    plan, explicit_pytest_paths = build_precommit_check_ids(paths, index)
    if plan.escalated_to_full:
        print("Pre-commit verification reduced an otherwise full verification request:", flush=True)
        for reason in plan.escalation_reasons:
            print(f"- {reason}", flush=True)
    result = execute_check_ids(plan.check_ids, argv_normalizer=_precommit_pytest_argv)
    if result != 0:
        return int(result)
    if not explicit_pytest_paths:
        return 0
    argv = _precommit_pytest_argv((str(VENV_PYTHON), "-m", "pytest", "-q", *explicit_pytest_paths))
    print("[dev] Run changed pytest files", flush=True)
    print(f"+ {shlex.join(argv)}", flush=True)
    return run_check_command(argv)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review-base", help="Summarize the current diff against a base ref")
    review_parser.add_argument("--base-ref", default=None, help="Git ref used as the diff base")

    verify_parser = subparsers.add_parser("verify", help="Run repo-local verification commands")
    verify_subparsers = verify_parser.add_subparsers(dest="verify_command", required=True)

    verify_affected_parser = verify_subparsers.add_parser(
        "affected",
        help="Run the smallest safe verification slice for the current diff",
    )
    verify_affected_parser.add_argument("--base-ref", default=None, help="Git ref used as the diff base")

    verify_paths_parser = verify_subparsers.add_parser(
        "paths",
        help="Run the smallest safe verification slice for explicit repo paths",
    )
    verify_paths_parser.add_argument(
        "--pre-commit",
        action="store_true",
        help="Use a lighter-weight pre-commit profile for pytest execution",
    )
    verify_paths_parser.add_argument("paths", nargs="+", help="Repo-relative paths to verify")

    verify_subparsers.add_parser("audit", help="Run audit scripts only")
    verify_subparsers.add_parser("full", help="Run the full verification suite")

    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    index = load_dev_index()

    if args.command == "review-base":
        base_ref = args.base_ref or index.default_base_ref
        changed_paths = collect_changed_paths(REPO_ROOT, base_ref=base_ref)
        merge_base = read_merge_base(REPO_ROOT, base_ref=base_ref)
        plan = build_verification_plan(changed_paths, index)
        print(render_review_report(plan, base_ref=base_ref, merge_base=merge_base))
        return 0

    if args.verify_command == "audit":
        return execute_check_ids(("audit",))
    if args.verify_command == "full":
        return execute_check_ids(index.full_verify_checks)
    if args.verify_command == "affected":
        base_ref = args.base_ref or index.default_base_ref
        changed_paths = collect_changed_paths(REPO_ROOT, base_ref=base_ref)
        if not changed_paths:
            print(f"No changed files detected against {base_ref}.")
            return 0
        plan = build_verification_plan(changed_paths, index)
        if plan.escalated_to_full:
            print("Affected verification escalated to full verification:")
            for reason in plan.escalation_reasons:
                print(f"- {reason}")
        return execute_check_ids(plan.check_ids)
    if args.verify_command == "paths":
        explicit_paths = [str(path).strip() for path in args.paths if str(path).strip()]
        if args.pre_commit:
            return execute_precommit_paths(explicit_paths, index)
        plan = build_verification_plan(explicit_paths, index)
        if plan.escalated_to_full:
            print("Explicit path verification escalated to full verification:")
            for reason in plan.escalation_reasons:
                print(f"- {reason}")
        return execute_check_ids(plan.check_ids)

    raise AssertionError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
