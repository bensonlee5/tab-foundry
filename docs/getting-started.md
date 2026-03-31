# Getting Started

**Routes To**

- `README.md` for repo overview and quickstart
- `docs/workflows.md` for command examples and artifact expectations
- `docs/research-contributors.md` for research-first routing
- `docs/development/codebase-navigation.md` for package and entrypoint ownership

**Does Not Own**

- live commands or flags
- sweep execution policy
- roadmap state or package-boundary policy

**If Stale vs Code**
Trust `.venv/bin/tab-foundry ... --help` for commands and flags,
`docs/workflows.md` for examples, and the owning docs over this router.

## Pick Your Question

- What does the repo do? Start with [README.md](../README.md).
- Need to run or inspect something? Go to [Workflows](workflows.md).
- Need to work on sweeps or architecture? Go to
  [Research Contributors](research-contributors.md) and
  [program.md](../program.md).
- Need to find the owning package or entrypoint? Go to
  [Codebase Navigation](development/codebase-navigation.md).

## Minimal Mental Model

`tab-foundry` turns corpus- or manifest-backed data plus model configs into run
directories, checkpoints, benchmark comparisons, export bundles, and sweep
artifacts. `dagzoo` is the synthetic-data lane, and `tab-realdata-hub` owns
the upstream manifest contract.

## If You Only Have 15 Minutes

1. Read [README.md](../README.md).
1. Choose [Research Contributors](research-contributors.md) or
   [ML Engineering And Infra](ml-engineering.md).
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before editing code, docs, or
   sweep state.
