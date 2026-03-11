---
name: pull
description:
  Pull latest origin/main into the current local branch and resolve merge
  conflicts; use when Codex needs to sync a branch before or during PR work.
---

# Pull

## Workflow

1. Verify the working tree is clean or intentionally committed.
2. Enable rerere locally:
   - `git config rerere.enabled true`
   - `git config rerere.autoupdate true`
3. Fetch latest refs:
   - `git fetch origin`
4. Sync the current branch with its remote counterpart:
   - `git pull --ff-only origin $(git branch --show-current)`
5. Merge `origin/main` into the current branch:
   - `git -c merge.conflictstyle=zdiff3 merge origin/main`
6. If conflicts appear:
   - inspect intent on both sides before editing
   - resolve one file at a time
   - `git add <files>`
   - `git commit` or `git merge --continue`
7. Run the repo validation required for the scope.
8. Summarize the merge result and any assumptions.

## Conflict Rules

- Prefer minimal, intention-preserving resolutions.
- Do not blindly choose `ours` or `theirs` unless one side should clearly win wholesale.
- For generated outputs, resolve source files first and only regenerate when necessary.
- Use `git diff --check` before considering the merge complete.
