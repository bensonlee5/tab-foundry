---
name: land
description:
  Land a PR by resolving branch drift, addressing review feedback, waiting for
  green checks, and squash-merging when approved.
---

# Land

## Goals

- Ensure the PR is up to date with `origin/main`.
- Ensure review feedback is addressed or explicitly replied to.
- Wait for green checks.
- Squash-merge only when the PR is approved and ready.

## Preconditions

- `gh auth status` succeeds.
- You are on the PR branch with a clean working tree.
- The required validation gate has passed locally:
  - `uv run ruff check .`
  - `uv run mypy src`
  - `uv run pytest -q`

## Workflow

1. Locate the PR for the current branch:
   - `gh pr view --json number,title,body,mergeable,reviewDecision,url`
2. If the tree is dirty, commit with the `commit` skill and publish with the `push` skill before continuing.
3. If the PR is behind `origin/main` or conflicting, run the `pull` skill, rerun validation, and publish again.
4. Sweep review feedback from all channels:
   - `gh pr view --comments`
   - `gh pr view --json reviews`
   - `gh api repos/<owner>/<repo>/pulls/<pr>/comments`
5. Treat actionable comments as blocking until code or replies resolve them.
6. Watch checks:
   - `gh pr checks --watch`
7. If checks fail:
   - inspect the failing run
   - fix the issue
   - rerun local validation
   - commit and push
   - restart the checks loop
8. Merge only when all checks are green and review state is acceptable.
9. Squash-merge with the PR title and body:
   - `gh pr merge --squash --subject "$pr_title" --body "$pr_body"`

## Notes

- Do not enable auto-merge for this repo.
- Do not ignore inline review comments; handle both inline and top-level feedback.
- If `gh` access fails, stop and surface the exact command and error rather than rewriting remotes or inventing auth workarounds.

## Commands

```bash
branch=$(git branch --show-current)
pr_number=$(gh pr view --json number -q .number)
pr_title=$(gh pr view --json title -q .title)
pr_body=$(gh pr view --json body -q .body)
mergeable=$(gh pr view --json mergeable -q .mergeable)

uv run ruff check .
uv run mypy src
uv run pytest -q

if [ "$mergeable" = "CONFLICTING" ]; then
  echo "Run the pull skill, rerun validation, and push again." >&2
  exit 1
fi

gh pr checks --watch
gh pr merge --squash --subject "$pr_title" --body "$pr_body"
```
