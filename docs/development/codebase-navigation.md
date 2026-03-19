# Codebase Navigation

This document describes the current `tab-foundry` layout on the active branch
and the intended landing zones for future work. It complements
`docs/development/module-dependency-map.md`, which records the observed
top-level package graph plus the intended dependency direction policy.

## 1. Entry Points And Orchestration

These are the user-facing or repo-local control surfaces that bridge commands
into the canonical library modules.

- `src/tab_foundry/__main__.py`: packaged CLI entrypoint for `tab-foundry`.
- `src/tab_foundry/cli/`: nested CLI registration and dispatch for
  `data`, `dev`, `train`, `eval`, `export`, `bench`, and `research`
  workflows.
- `scripts/`: shell convenience helpers plus audit tooling only. Python
  workflow entrypoints have been retired in favor of the packaged CLI.
- `scripts/dev`: diff-aware repo-local review and verification wrapper that
  delegates to the audit tooling and packaged CLI.

## 2. Canonical Library Areas

- `src/tab_foundry/config.py`: shared Hydra composition helpers and config root
  discovery.
- `src/tab_foundry/data/` and `src/tab_foundry/data/sources/`: manifest-backed
  dataset abstractions, surface selection, source provenance wiring, and data
  construction helpers.
- `src/tab_foundry/preprocessing/`: explicit preprocessing-surface resolution
  and fitted-state helpers. This is now a live first-class package area rather
  than an implicit training detail.
- `src/tab_foundry/model/`: model namespace package. Direct imports should
  target `tab_foundry.model.factory`, `tab_foundry.model.spec`, or concrete
  family modules under `tab_foundry.model.architectures`. Shared developer
  inspection helpers such as resolved-surface rendering and synthetic
  forward-check batches now live here too.
- `src/tab_foundry/model/components/`: reusable blocks, QASS primitives, and
  many-class helpers shared across families.
- `src/tab_foundry/model/architectures/`: the current two-family model
  surface:
  - `tabfoundry_simple`: the frozen exact nanoTabPFN-style benchmark anchor
  - `tabfoundry_staged`: the staged classification family and the only active
    architecture-development surface
- `src/tab_foundry/training/`: family-agnostic training loops, batching,
  schedules, optimizers, runtime policy, evaluation helpers, and telemetry
  health summaries.
- `src/tab_foundry/export/`: export bundle construction, loading, and
  validation contracts.
- `src/tab_foundry/bench/`: benchmark bundles, comparison flows, benchmark
  env/bootstrap helpers, smoke harnesses, prior-dump wiring, and shared
  artifact helpers.
- `src/tab_foundry/research/`: system-delta sweep state, queue/matrix
  rendering, sweep-result summaries, and research-package path conventions.

## 3. Workflow Surfaces

The repo uses three stable workflow layers:

- The packaged CLI as the canonical user-facing surface for manifest-backed
  data/build, training, evaluation, export, smoke, tuning, benchmarking,
  registry, and research-sweep flows.
- Shell helpers under `scripts/*.sh` plus `scripts/audit/` as repo-local
  convenience and verification surfaces only.
- `scripts/dev` as the canonical repo-local entrypoint for branch review,
  affected-scope verification, explicit-path verification, full verification,
  and Iris smoke delegation.
- Reference YAML/Markdown artifacts for the active system-delta sweep.

Current canonical CLI namespaces:

- `tab-foundry data build-manifest`
- `tab-foundry data manifest-inspect`
- `tab-foundry dev resolve-config`
- `tab-foundry dev forward-check`
- `tab-foundry dev diff-config`
- `tab-foundry dev export-check`
- `tab-foundry dev health-check`
- `tab-foundry dev run-inspect`
- `tab-foundry train run`
- `tab-foundry train prior simple`
- `tab-foundry train prior staged`
- `tab-foundry eval checkpoint`
- `tab-foundry export bundle`
- `tab-foundry export validate`
- `tab-foundry bench smoke iris`
- `tab-foundry bench smoke dagzoo`
- `tab-foundry bench tune`
- `tab-foundry bench compare`
- `tab-foundry bench env bootstrap`
- `tab-foundry bench bundle build-openml`
- `tab-foundry bench registry register-run`
- `tab-foundry bench registry freeze-baseline`
- `tab-foundry bench diagnose bounce`
- `tab-foundry research sweep create`
- `tab-foundry research sweep list`
- `tab-foundry research sweep next`
- `tab-foundry research sweep render`
- `tab-foundry research sweep validate`
- `tab-foundry research sweep set-active`
- `tab-foundry research sweep execute`
- `tab-foundry research sweep graph`
- `tab-foundry research sweep promote`
- `tab-foundry research sweep summarize`
- `tab-foundry research sweep inspect`
- `tab-foundry research sweep diff`

Shell helpers such as `scripts/build_manifest.sh`, `scripts/train_smoke.sh`,
and `scripts/eval_smoke.sh` are repo-local convenience entrypoints and should
not absorb new orchestration logic.

## 4. Reference And Planning Surfaces

- `program.md` is the active execution contract for agent-driven system-delta
  work.
- `reference/system_delta_catalog.yaml` and
  `reference/system_delta_sweeps/<sweep_id>/` are the canonical system-delta
  sources of truth.
- `reference/system_delta_queue.yaml` and
  `reference/system_delta_matrix.md` are generated active-sweep aliases. They
  exist for convenience and compatibility, but they are not the canonical files
  to edit directly.
- `docs/workflows.md` and `docs/inference.md` stay top-level because they are
  stable operator-facing runbooks.
- `docs/development/` is the canonical home for planning, rationale,
  navigation, dependency docs, and audit artifacts.

## 5. Current Structural Watchpoints

- `src/tab_foundry/bench/` is the canonical home for benchmark and harness
  logic. Core training/model/data packages should not start depending on it.
- `src/tab_foundry/research/` is the canonical home for sweep queue/matrix
  management; do not recreate parallel queue logic in shell helpers or
  docs-only tooling.
- `tabfoundry_staged` is the only active architecture surface. Shared logic
  should continue to move into `model/components/`, `model/spec.py`, and
  family-neutral helpers instead of reintroducing parallel model pathways.
- The active system-delta aliases are generated views. Docs and scripts should
  describe them as such and should resolve canonical state through the sweep
  index and per-sweep sources.

## 6. Tests And Docs

- `tests/data/`, `tests/model/`, `tests/training/`, `tests/export/`,
  `tests/runtime/`, and `tests/config/` cover the library roles.
- `tests/smoke/` and `tests/benchmark/` cover end-to-end and benchmark/harness
  flows.
- `tests/research/` and `tests/audit/` cover the system-delta workflow and the
  repeatable audit tooling.
- `README.md`, `docs/workflows.md`, and `docs/inference.md` are the stable
  operator-facing entrypoints.
- `docs/development/roadmap.md`,
  `docs/development/design-decisions.md`,
  `docs/development/codebase-navigation.md`, and
  `docs/development/module-dependency-map.md` are the internal repo-evolution
  canon.
