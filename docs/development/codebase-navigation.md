# Codebase Navigation

This document describes the current `tab-foundry` layout and the intended
landing zones for future work. It complements
`docs/development/module-dependency-map.md` with a maintained dependency view.

## 1. Entry Points And Orchestration

The high-level logic that bridges user-visible commands to the canonical
manifest-backed training and benchmark workflows.

- `src/tab_foundry/__main__.py`: packaged CLI entrypoint for `tab-foundry`.
- `src/tab_foundry/cli/app.py`: parser assembly and command dispatch.
- `src/tab_foundry/cli/commands/`: manifest build, train, eval, export, and
  validate-export command registration.
- `scripts/`: thin repo-local wrappers for smoke, tuning, benchmark, and env
  bootstrap workflows. Python wrappers here should stay import-only shims into
  library or `bench/` modules rather than accumulating business logic.

## 2. Canonical Library Areas

- `src/tab_foundry/config.py`: shared Hydra composition helpers and config root
  discovery.
- `src/tab_foundry/data/` and `src/tab_foundry/data/sources/`: manifest-backed
  dataset abstractions, source selection, and data construction helpers.
  Direct imports should target modules such as `tab_foundry.data.manifest`,
  `tab_foundry.data.dataset`, and `tab_foundry.data.factory`.
- `src/tab_foundry/model/`: model namespace package. Direct imports should
  target `tab_foundry.model.factory`, `tab_foundry.model.spec`, or family
  modules under `tab_foundry.model.architectures`.
- `src/tab_foundry/model/components/`: reusable blocks, QASS primitives, and
  many-class helpers shared across families.
- `src/tab_foundry/model/architectures/`: family-specific assemblies such as
  `TabFoundry`.
- `src/tab_foundry/training/`: family-agnostic training loops, batching,
  schedules, optimizers, runtime policy, and evaluation helpers.
- `src/tab_foundry/export/`: export namespace package. Direct imports should
  target `tab_foundry.export.exporter`, `tab_foundry.export.loader_ref`, and
  `tab_foundry.export.contracts`.
- `src/tab_foundry/bench/`: smoke harnesses, tuning, checkpoint benchmarking,
  comparison flows, benchmark env bootstrap, and shared artifact helpers.

## 3. Workflow Surfaces

The repo currently uses two stable workflow layers:

- CLI as the canonical user-facing surface for manifest build, train, eval,
  export, and validate-export.
- Thin script wrappers for repo-local operational flows that are intentionally
  not modeled as long-lived public CLI surfaces yet.

Current wrapper-to-library mapping:

- `scripts/iris_smoke.py` -> `tab_foundry.bench.iris_smoke`
- `scripts/dagzoo_smoke.py` -> `tab_foundry.bench.dagzoo_smoke`
- `scripts/eval_iris_checkpoint.py` -> `tab_foundry.bench.iris`
- `scripts/tune_tab_foundry.py` -> `tab_foundry.bench.tune`
- `scripts/benchmark_nanotabpfn.py` -> `tab_foundry.bench.compare`
- `scripts/bootstrap_benchmark_envs.py` -> `tab_foundry.bench.envs`
- `scripts/run_nanotabpfn_benchmark_helper.py` ->
  `tab_foundry.bench.nanotabpfn_helper`

Shell helpers such as `scripts/build_manifest.sh`, `scripts/train_smoke.sh`,
and `scripts/eval_smoke.sh` are repo-local convenience entrypoints and should
not absorb new orchestration logic.

## 4. Current Structural Watchpoints

- `src/tab_foundry/model/` now separates reusable components from
  family-specific assemblies, but the repo still has only one named family and
  no neutral architecture registry yet.
- `src/tab_foundry/bench/` is the canonical home for benchmark and harness
  logic. Core packages should not start depending on it.
- `docs/development/` is now the canonical home for planning, rationale,
  navigation, and dependency docs.
- `docs/workflows.md` and `docs/inference.md` stay top-level because they are
  operational surfaces and stable links for users and downstream repos.
- `reference/README.md` indexes the `reference/` area for literature and
  evidence notes.

## 5. Tests And Docs

- `tests/data/`, `tests/model/`, `tests/training/`, `tests/export/`,
  `tests/runtime/`, and `tests/config/` cover library roles.
- `tests/smoke/` and `tests/benchmark/` cover end-to-end and benchmark/harness
  flows.
- `README.md`, `docs/workflows.md`, and `docs/inference.md` are the stable
  operator-facing entrypoints.
- `docs/development/roadmap.md`,
  `docs/development/design-decisions.md`,
  `docs/development/codebase-navigation.md`, and
  `docs/development/module-dependency-map.md` are the internal repo-evolution
  canon.
