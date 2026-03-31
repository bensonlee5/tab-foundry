# Contributing

Use this guide when you want to make a bounded change without reopening the
entire system.

Start with [README.md](README.md), then go directly to the owner docs that
match your question:

- [README.md](README.md) for the repo overview and quickstart
- [docs/workflows.md](docs/workflows.md) for command examples and artifact
  expectations
- [program.md](program.md) for the active system-delta sweep contract
- [docs/development/roadmap.md](docs/development/roadmap.md) for research
  priorities and TF-RD sequencing
- [docs/development/model-architecture.md](docs/development/model-architecture.md)
  for the live model surface
- [docs/development/codebase-navigation.md](docs/development/codebase-navigation.md)
  for package ownership and entrypoint boundaries
- [docs/inference.md](docs/inference.md) for export/runtime handoff details
- [docs/glossary.md](docs/glossary.md) for sweep and architecture vocabulary

If a Markdown file and `.venv/bin/tab-foundry ... --help` disagree, trust the
CLI and update the Markdown file.

## Choose The Right Unit Of Work

Use the smallest unit that matches the question you are trying to answer.

- Docs clarification:
  - Use when the behavior is already settled but discoverability is poor.
  - Update the canonical Markdown file and any routing page that now points to
    it.
- Roadmap item:
  - Use when the work changes priorities, sequencing, or the meaning of a
    TF-RD item.
  - Update [docs/development/roadmap.md](docs/development/roadmap.md).
- Sweep row:
  - Use when you want to isolate one declared change against the current
    anchor.
  - Follow [program.md](program.md) and the sweep package template under
    [reference/system_delta_campaign_template.md](reference/system_delta_campaign_template.md).
- New sweep:
  - Use when the comparison surface, anchor, or question has changed enough
    that continuing the old sweep would blur interpretation.
  - Preserve completed sweeps as historical evidence.
- Code change:
  - Use when the existing architecture, CLI, or research tooling cannot express
    the intended change.
  - Keep shared wiring in the role-based packages under `src/tab_foundry/`
    instead of creating parallel paths.

## Research Contributor Workflow

1. Route yourself through the current canon.
   - Read [README.md](README.md).
   - Then read the smallest owner doc that actually matches the work:
     [program.md](program.md), [docs/workflows.md](docs/workflows.md),
     [docs/development/model-architecture.md](docs/development/model-architecture.md),
     [docs/development/roadmap.md](docs/development/roadmap.md), or
     [docs/development/codebase-navigation.md](docs/development/codebase-navigation.md).
1. Inspect before changing.
   - Prefer narrow inspection surfaces such as:
     - `.venv/bin/tab-foundry dev resolve-config ...`
     - `.venv/bin/tab-foundry dev forward-check ...`
     - `.venv/bin/tab-foundry dev run-inspect --run-dir ...`
     - `.venv/bin/tab-foundry research sweep summarize --include-screened`
     - `.venv/bin/tab-foundry research sweep inspect --order <n> --sweep-id <id>`
   - Only fall back to broad greps or full runs after those surfaces stop
     answering the question.
1. Make the smallest coherent change.
   - Keep architecture, data, training, and sweep changes attributable.
   - Avoid adding parallel implementations of the same logic in different
     layers.
1. Verify the smallest safe slice.
   - `./scripts/dev review-base`
   - `./scripts/dev verify affected`
   - `./scripts/dev verify paths <PATH...>`
1. Before review, compare the branch to `main`.
   - Confirm all intended changes are present.
   - Confirm unrelated changes are not included.

## Sweep And Research Hygiene

- Put `TF-RD-###` ids on sweep names.
- Always log executed sweeps to `wandb`.
- Update [docs/development/roadmap.md](docs/development/roadmap.md) when a sweep
  is complete.
- Update associated GitHub issues when the roadmap or sweep status changes.
- Link issues to roadmap sections and PRs so the evidence chain stays intact.

## User-Facing Breaks

Treat these as user-facing changes and call them out explicitly in the PR:

- CLI flag or command changes
- persisted metadata schema changes
- dataset artifact contract changes
- export bundle contract changes

If behavior or schema changes under `src/tab_foundry`:

- bump the version in [pyproject.toml](pyproject.toml) just before merging to
  `main`
- update [CHANGELOG.md](CHANGELOG.md) in the same PR

Docs-only and tests-only changes do not require a version bump.

## Ready For Review

The branch is ready for review when:

- the intended question or change is explicit
- the canonical source files were edited instead of generated views
- the smallest relevant verification slice was run
- sweep or research changes name the preserved surface and the changed surface
- user-facing breaks are called out if any CLI/schema/artifact contract changed

If the change would close a GitHub issue, say so explicitly in the PR summary
and reference the issue number directly.
