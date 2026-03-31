# Research Contributors

**Routes To**

- `program.md` for selected sweep execution policy
- `docs/workflows.md` for command examples and artifact expectations
- `docs/development/model-architecture.md` for the current model surface
- `docs/development/roadmap.md` for research priorities and TF-RD sequencing

**Does Not Own**

- live commands or flags
- the detailed sweep policy owned by `program.md`
- package ownership or dependency direction

**If Stale vs Code**
Trust `.venv/bin/tab-foundry research ... --help` for commands,
`program.md` for sweep policy, and the code-owning docs for architecture and
package boundaries.

## Use This Route When

Use this page when the question is about architecture, sweeps, synthetic data,
or broader model capability. It is intentionally a routing page, not a second
runbook.

The research loop in this repo is attribution-first: isolate one question,
compare it against a locked anchor, and keep the evidence chain readable.

## Route By Question

- What architecture is active?
  Read [Design Decisions](development/design-decisions.md),
  [Model Architecture](development/model-architecture.md),
  [Codebase Navigation](development/codebase-navigation.md), and
  [Module Dependency Map](development/module-dependency-map.md).
- What sweep row should run next, and what artifacts should it leave behind?
  Read [program.md](../program.md), [Workflows](workflows.md), and
  [Roadmap](development/roadmap.md).
- How does synthetic data fit relative to real-data ladders?
  Read [Dataset Curation](development/dataset-curation.md),
  [Roadmap](development/roadmap.md), and [Workflows](workflows.md).
- How should broader capability work be framed?
  Read [Roadmap](development/roadmap.md),
  [Model Architecture](development/model-architecture.md), and
  [reference/README.md](../reference/README.md).

## Inspect-First Entry Points

```bash
.venv/bin/tab-foundry dev resolve-config experiment=cls_smoke
.venv/bin/tab-foundry dev forward-check experiment=cls_smoke
.venv/bin/tab-foundry research sweep summarize --sweep-id <sweep_id> --include-screened
.venv/bin/tab-foundry research sweep inspect --sweep-id <sweep_id> --order <order>
.venv/bin/tab-foundry research sweep diff --sweep-id <sweep_id> --order <order> --against-order <anchor_order>
.venv/bin/tab-foundry research sweep graph --sweep-id <sweep_id> --anchor
```

## Read In This Order

1. Read [docs/glossary.md](glossary.md) if the sweep vocabulary is not fresh.
1. Read the owner docs for your question instead of grepping the repo first.
1. Use the inspect-first surfaces above before broad searches or full runs.
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before editing code, docs, or
   sweep state.

## Common Mistakes

- treating [Workflows](workflows.md) as the sweep policy owner instead of
  [program.md](../program.md)
- treating [Roadmap](development/roadmap.md) as queue-level execution policy
- inferring package ownership from an arbitrary helper file instead of
  [Codebase Navigation](development/codebase-navigation.md)
- adding a parallel data or benchmark path when an existing corpus, manifest,
  or sweep surface already expresses the question
