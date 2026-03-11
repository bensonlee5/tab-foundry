---
name: commit
description:
  Create a well-formed git commit from current changes using session history for
  rationale and summary; use when asked to commit, prepare a commit message, or
  finalize staged work.
---

# Commit

## Goals

- Produce a commit that matches the staged changes.
- Use a conventional, concise subject line.
- Include rationale and validation in the body.

## Steps

1. Inspect `git status`, `git diff`, and `git diff --staged`.
2. Stage only the intended files.
3. Choose a conventional commit type and optional scope.
4. Write a subject line in imperative mood, <= 72 characters.
5. Write a body that covers:
   - what changed
   - why it changed
   - validation run, or why validation was not run
6. Append `Co-authored-by: Codex <codex@openai.com>` unless the user asked otherwise.
7. Create the commit with `git commit -F <file>` so newlines are preserved exactly.

## Template

```text
<type>(<scope>): <short summary>

Summary:
- <what changed>

Rationale:
- <why>

Tests:
- <command or "not run (reason)">

Co-authored-by: Codex <codex@openai.com>
```
