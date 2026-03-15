# Codebase Navigation

This document describes the current `tab-foundry` layout on the active branch
and the intended landing zones for future work. It complements
`docs/development/module-dependency-map.md`, which records the observed
top-level package graph plus the intended dependency direction policy.

## 1. Entry Points And Orchestration

These are the user-facing or repo-local control surfaces that bridge commands
into the canonical library modules.

- `src/tab_foundry/__main__.py`: packaged CLI entrypoint for `tab-foundry`.
- `src/tab_foundry/cli/`: parser assembly and command dispatch for
  manifest-backed build, train, eval, export, and validate-export flows.
- `scripts/`: thin repo-local wrappers for smoke, tuning, benchmarking,
  benchmark-env bootstrap, and system-delta queue operations. Python wrappers
  here should stay import-only shims into `src/tab_foundry/` modules rather
  than accumulating business logic.

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
  family modules under `tab_foundry.model.architectures`.
- `src/tab_foundry/model/components/`: reusable blocks, QASS primitives, and
  many-class helpers shared across families.
- `src/tab_foundry/model/architectures/`: the current three-family model
  surface:
  - `tabfoundry`: the main grouped-token backbone family
  - `tabfoundry_simple`: the exact nanoTabPFN-style benchmark anchor
  - `tabfoundry_staged`: the staged research family with atomic surface
    resolution and queue-driven overrides
- `src/tab_foundry/training/`: family-agnostic training loops, batching,
  schedules, optimizers, runtime policy, and evaluation helpers.
- `src/tab_foundry/export/`: export bundle construction, loading, and
  validation contracts.
- `src/tab_foundry/bench/`: benchmark bundles, comparison flows, benchmark
  env/bootstrap helpers, smoke harnesses, prior-dump wiring, and shared
  artifact helpers.
- `src/tab_foundry/research/`: system-delta sweep state, queue/matrix
  rendering, and research-package path conventions.

## 3. Workflow Surfaces

The repo uses three stable workflow layers:

- CLI as the canonical user-facing surface for manifest-backed build, train,
  eval, export, and validate-export.
- Thin script wrappers for repo-local operational flows that are intentionally
  not modeled as long-lived public CLI surfaces yet.
- Reference YAML/Markdown artifacts for the active system-delta sweep.

Current wrapper-to-library mapping:

- `scripts/iris_smoke.py` -> `tab_foundry.bench.iris_smoke`
- `scripts/dagzoo_smoke.py` -> `tab_foundry.bench.dagzoo_smoke`
- `scripts/eval_iris_checkpoint.py` -> `tab_foundry.bench.iris`
- `scripts/tune_tab_foundry.py` -> `tab_foundry.bench.tune`
- `scripts/benchmark_nanotabpfn.py` -> `tab_foundry.bench.compare`
- `scripts/bootstrap_benchmark_envs.py` -> `tab_foundry.bench.envs`
- `scripts/run_nanotabpfn_benchmark_helper.py` ->
  `tab_foundry.bench.nanotabpfn_helper`
- `scripts/system_delta_queue.py` -> `tab_foundry.research.system_delta`
- `scripts/train_tabfoundry_staged_prior.py` -> staged prior-training harness
  under `tab_foundry.training`

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
  management; do not recreate parallel queue logic in `scripts/` or docs-only
  tooling.
- The three model families are intentional. Shared logic should continue to
  move into `model/components/`, `model/spec.py`, and family-neutral helpers
  instead of forking duplicate pathways.
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
