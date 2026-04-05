#!/usr/bin/env python3
"""Check that all open GitHub issues are linked from the roadmap."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROADMAP_PATH = REPO_ROOT / "docs" / "development" / "roadmap.md"
ISSUE_LINK_RE = re.compile(r"https://github\.com/[^/]+/[^/]+/issues/(?P<number>\d+)")
TITLE_FAMILY_RE = re.compile(r"^\[(TF-RD-\d{3})\]")


@dataclass(frozen=True, slots=True)
class OpenIssue:
    number: int
    title: str


def extract_roadmap_issue_numbers(roadmap_text: str) -> set[int]:
    return {int(match.group("number")) for match in ISSUE_LINK_RE.finditer(roadmap_text)}


def issue_family(issue: OpenIssue) -> str:
    match = TITLE_FAMILY_RE.match(issue.title.strip())
    if match is None:
        return "unscoped"
    return match.group(1)


def group_missing_open_issues(
    *,
    linked_issue_numbers: set[int],
    open_issues: Sequence[OpenIssue],
) -> dict[str, tuple[OpenIssue, ...]]:
    grouped: dict[str, list[OpenIssue]] = {}
    for issue in open_issues:
        if issue.number in linked_issue_numbers:
            continue
        grouped.setdefault(issue_family(issue), []).append(issue)
    return {
        family: tuple(sorted(issues, key=lambda issue: issue.number))
        for family, issues in sorted(grouped.items())
    }


def _parse_repo_slug(raw_remote_url: str) -> tuple[str, str]:
    remote = raw_remote_url.strip()
    ssh_match = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?/?$", remote)
    if ssh_match is not None:
        return ssh_match.group("owner"), ssh_match.group("repo")
    https_match = re.match(
        r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?/?$",
        remote,
    )
    if https_match is not None:
        return https_match.group("owner"), https_match.group("repo")
    raise RuntimeError(f"unsupported GitHub remote URL: {remote!r}")


def infer_repo_slug(repo_root: Path = REPO_ROOT) -> tuple[str, str]:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git remote get-url origin failed"
        raise RuntimeError(detail)
    return _parse_repo_slug(result.stdout.strip())


def fetch_open_issues(*, owner: str, repo: str, repo_root: Path = REPO_ROOT) -> tuple[OpenIssue, ...]:
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                f"{owner}/{repo}",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,title",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to verify roadmap issue sync") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "gh issue list failed"
        raise RuntimeError(detail)
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("gh issue list must return a JSON list")
    issues: list[OpenIssue] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("gh issue list item must be an object")
        issues.append(
            OpenIssue(
                number=int(item["number"]),
                title=str(item["title"]),
            )
        )
    return tuple(sorted(issues, key=lambda issue: issue.number))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roadmap",
        default=str(DEFAULT_ROADMAP_PATH),
        help="Path to the roadmap Markdown file",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Optional GitHub repo slug in owner/repo form. Defaults to origin remote.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    roadmap_path = Path(args.roadmap).expanduser().resolve()
    if not roadmap_path.exists():
        raise SystemExit(f"roadmap does not exist: {roadmap_path}")
    if args.repo is not None:
        owner, repo = str(args.repo).split("/", 1)
    else:
        owner, repo = infer_repo_slug()
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    linked_issue_numbers = extract_roadmap_issue_numbers(roadmap_text)
    open_issues = fetch_open_issues(owner=owner, repo=repo)
    missing = group_missing_open_issues(
        linked_issue_numbers=linked_issue_numbers,
        open_issues=open_issues,
    )
    if not missing:
        print(
            "Roadmap issue sync passed: "
            f"all {len(open_issues)} open GitHub issues are linked from "
            f"{roadmap_path.relative_to(REPO_ROOT)}."
        )
        return 0
    total_missing = sum(len(issues) for issues in missing.values())
    print(
        "Roadmap issue sync failed: "
        f"{total_missing} open GitHub issue(s) are missing from "
        f"{roadmap_path.relative_to(REPO_ROOT)}."
    )
    for family, issues in missing.items():
        rendered = ", ".join(f"#{issue.number} {issue.title}" for issue in issues)
        print(f"- {family}: {rendered}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
