---
name: linear
description: |
  Use the Linear workspace tools for issue reads, comment updates, state
  changes, and PR link attachment.
---

# Linear

Use this skill when a Codex session needs to read or update Linear directly.
Prefer the typed Linear tools that are available in the current session rather
than raw GraphQL.

## Preferred Tools

- `mcp__linear__get_issue` for full issue context
- `mcp__linear__list_comments` to inspect existing discussion/workpad comments
- `mcp__linear__save_comment` to create or update comments
- `mcp__linear__save_issue` to change state, assignee, labels, or attach links
- `mcp__linear__get_issue_status` or `mcp__linear__list_issue_statuses` when a
  target state name is ambiguous
- `mcp__linear__list_issues` for search
- `mcp__linear__get_project`, `mcp__linear__get_team`, and
  `mcp__linear__get_user` for metadata lookup when needed

## Usage Rules

- Query or update only the fields needed for the current task.
- Reuse an existing workpad/status comment when one exists instead of creating
  duplicates.
- When linking a PR or external URL back to an issue, use
  `mcp__linear__save_issue` with the append-only `links` field.
- Treat append-only fields such as `links`, `blocks`, `blockedBy`, and
  `relatedTo` carefully to avoid duplicate entries.
- Resolve the exact target issue state before changing it if state names are
  not already known.

## Common Operations

### Read an issue

- Call `mcp__linear__get_issue` with the issue identifier.
- Set `includeRelations` or `includeCustomerNeeds` only when the task needs
  them.

### Find or update a workpad comment

- Call `mcp__linear__list_comments` with the issue ID.
- Reuse the existing comment ID with `mcp__linear__save_comment` when updating.

### Move an issue

- Resolve the target state with `mcp__linear__get_issue_status` or
  `mcp__linear__list_issue_statuses` if needed.
- Call `mcp__linear__save_issue` with `id` and `state`.

### Attach a PR

- Call `mcp__linear__save_issue` with `id` and
  `links: [{ title: "<pr title>", url: "<pr url>" }]`.
