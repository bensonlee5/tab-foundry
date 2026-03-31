# ML Engineering And Infra

**Routes To**

- `docs/workflows.md` for command examples and artifact expectations
- `docs/inference.md` for the export-bundle and runtime handoff contract
- `docs/development/codebase-navigation.md` for package ownership
- `CONTRIBUTING.md` for verification and review expectations

**Does Not Own**

- live commands or flags
- architecture policy or sweep policy
- the export schema details owned by `docs/inference.md`

**If Stale vs Code**
Trust `.venv/bin/tab-foundry ... --help` for commands,
`docs/workflows.md` for runbook examples, and `docs/inference.md` for
export/runtime details.

## Use This Route When

- you care about manifests, runs, checkpoints, benchmark artifacts, or export
  bundles
- you need the shortest route from a file on disk to the command that produces
  or validates it
- you are changing verification, packaging, benchmarking, or export wiring

## Questions This Page Routes

- How do I materialize data, train, evaluate, or export? Use
  [Workflows](workflows.md).
- What is the downstream runtime handoff contract? Use
  [Inference Contract](inference.md).
- Which package owns benchmark, export, or CLI wiring? Use
  [Codebase Navigation](development/codebase-navigation.md).
- What review slice should I run before opening a PR? Use
  [CONTRIBUTING.md](../CONTRIBUTING.md).

## Artifact Mental Model

Most operational work reduces to a small set of stable artifacts:

- manifests or corpus refs select data
- run directories hold histories, checkpoints, and summaries
- benchmark outputs hold comparison evidence
- export bundles are the cross-repo handoff surface
