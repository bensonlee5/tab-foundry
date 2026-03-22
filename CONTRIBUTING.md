# Contributing

This repo is optimized for research contributors who need to understand the
architecture, inspect sweep state, and make bounded changes without reopening
the entire system.

Start with [docs/getting-started.md](docs/getting-started.md) for repo
orientation, then choose the path that matches your work:

- [docs/what-is-tab-foundry.md](docs/what-is-tab-foundry.md) for a repo
  overview
- [docs/research-contributors.md](docs/research-contributors.md) for research
  work
- [docs/ml-engineering.md](docs/ml-engineering.md) for ML engineering / infra
  work
- [docs/glossary.md](docs/glossary.md) for sweep and architecture vocabulary

## Canonical Sources Of Truth

Keep changes on the canonical source, not on generated views.

- `README.md`: top-level routing and quickstart
- `docs/getting-started.md`: researcher onboarding
- `docs/workflows.md`: operational runbooks and command syntax
- `program.md`: active system-delta sweep contract
- `docs/development/roadmap.md`: planning state and TF-RD priorities
- `docs/development/model-architecture.md`: architecture truth
- `reference/README.md`: literature and evidence entry point
- `site/`: Hugo publishing shell only; generated content under `site/.generated/`
  is not canonical

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
   - Read [docs/getting-started.md](docs/getting-started.md).
   - Confirm whether your question is about architecture, sweep execution,
     synthetic data, or model breadth.
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
