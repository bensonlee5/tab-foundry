# Module Dependency Map

This file records the current top-level package graph under `src/tab_foundry`
and the intended dependency-direction policy for future refactors.

The observed graph is synchronized against
`scripts/audit/module_graph.py --fail-on-doc-drift`. Keep the current-state
section factual and keep design intent in the policy section below it.

## Observed Current Top-Level Package Graph

<!-- module-graph:start -->

- `tab_foundry.__main__` depends on `tab_foundry.cli`.
- `tab_foundry.bench` depends on `tab_foundry.config`,
  `tab_foundry.data`, `tab_foundry.input_normalization`,
  `tab_foundry.model`, `tab_foundry.training`, and `tab_foundry.types`.
- `tab_foundry.cli` depends on `tab_foundry.bench`,
  `tab_foundry.config`, `tab_foundry.data`, `tab_foundry.export`,
  `tab_foundry.research`, and `tab_foundry.training`.
- `tab_foundry.data` depends on `tab_foundry.preprocessing` and
  `tab_foundry.types`.
- `tab_foundry.export` depends on `tab_foundry.input_normalization`,
  `tab_foundry.model`, `tab_foundry.preprocessing`, and `tab_foundry.types`.
- `tab_foundry.model` depends on `tab_foundry.input_normalization` and
  `tab_foundry.types`.
- `tab_foundry.research` depends on `tab_foundry.bench`,
  `tab_foundry.config`, and `tab_foundry.model`.
- `tab_foundry.training` depends on `tab_foundry.data`,
  `tab_foundry.model`, `tab_foundry.preprocessing`, and `tab_foundry.types`.

<!-- module-graph:end -->

Observed cycle status:

- no top-level cycle candidates

## Intended Dependency-Direction Policy

- `tab_foundry.config`, `tab_foundry.types`, and
  `tab_foundry.input_normalization` should remain dependency-light helpers.
- `tab_foundry.model` should stay independent of `bench`, `research`,
  `training`, and `export`.
- `tab_foundry.preprocessing` should remain a leaf-style utility package that
  can be used by `data`, `training`, and `export` without growing orchestration
  logic of its own.
- `tab_foundry.data` may depend on `preprocessing` helpers and shared types,
  but it should not depend on `training`, `bench`, or `research`.
- `tab_foundry.training` may depend on `data`, `model`, `preprocessing`, and
  shared helpers, but it should not depend on `bench` or `research`.
- `tab_foundry.export` may depend on `model`, `preprocessing`, and shared
  helpers, but it should not depend on `bench`, `research`, or `training`.
- `tab_foundry.bench` is the benchmark and harness layer. It may depend on
  `config`, `data`, `model`, `training`, and shared helpers, but lower layers
  should not depend on it.
- `tab_foundry.research` is the sweep-management layer. It may depend on
  `bench`, `config`, and `model`, but lower layers should not depend on it.
- Python workflow entrypoints should live under the packaged nested CLI rather
  than being duplicated under `scripts/`.
- `scripts/` should stay limited to shell convenience helpers and audit tooling
  instead of reintroducing parallel Python workflow surfaces.

## Change-Impact Hotspots

### `src/tab_foundry/model/factory.py`

- Shared model construction surface used by training, evaluation, export, and
  checkpoint loading.
- Changes here ripple into CLI flows, `bench/` harnesses, research sweeps, and
  export compatibility.

### `src/tab_foundry/training/trainer.py`

- Central training loop and artifact-emission surface.
- Changes here affect smoke, tuning, checkpoint benchmarking, and staged prior
  workflows.

### `src/tab_foundry/export/contracts.py`

- Export bundle contract boundary.
- Changes here are user-facing and must be treated as artifact-schema changes.

### `src/tab_foundry/bench/compare.py` And `src/tab_foundry/bench/checkpoint.py`

- Benchmark-facing comparison and checkpoint evaluation logic.
- Changes here affect external-baseline comparison and benchmark registry
  records.

### `src/tab_foundry/research/system_delta.py`

- Canonical sweep manager, queue materializer, and rendered-matrix surface.
- Changes here affect the active research contract and the generated alias
  views under `reference/`.
