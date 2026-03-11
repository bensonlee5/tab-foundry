---
name: push
description:
  Push current branch changes to origin and create or update the corresponding
  pull request; use when asked to publish branch updates for tab-foundry.
---

# Push

## Goals

- Run the required repo validation before publish.
- Push the current branch safely.
- Create or update an open PR for the branch.

## Required Validation

Run these commands before every push:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Workflow

1. Identify the current branch.
2. Run the required validation.
3. Push to `origin`, using `-u` if upstream is not set.
4. If push is rejected because the remote moved, run the `pull` skill, rerun validation, then push again.
5. If push fails due to auth or permissions, stop and surface the exact error.
6. Ensure there is one open PR for the current branch:
   - create a PR if missing
   - update the title and body if it already exists
   - if the current branch is tied to a closed PR, create a fresh branch before trying again
7. Reply with the PR URL.

## PR Body

This repo does not require a template. Use a short, concrete body such as:

```markdown
## Summary
- <main behavior change>

## Testing
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest -q`

## Notes
- <risks, follow-ups, or "None">
```

## Commands

```bash
branch=$(git branch --show-current)

uv run ruff check .
uv run mypy src
uv run pytest -q

git push -u origin HEAD

pr_state=$(gh pr view --json state -q .state 2>/dev/null || true)
if [ "$pr_state" = "MERGED" ] || [ "$pr_state" = "CLOSED" ]; then
  echo "Current branch is tied to a closed PR; create a new branch." >&2
  exit 1
fi

pr_title="<clear title for the shipped change>"
tmp_pr_body=$(mktemp)
cat >"$tmp_pr_body" <<'EOF'
## Summary
- <main behavior change>

## Testing
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest -q`

## Notes
- None
EOF

if [ -z "$pr_state" ]; then
  gh pr create --title "$pr_title" --body-file "$tmp_pr_body"
else
  gh pr edit --title "$pr_title" --body-file "$tmp_pr_body"
fi

rm -f "$tmp_pr_body"
gh pr view --json url -q .url
```
