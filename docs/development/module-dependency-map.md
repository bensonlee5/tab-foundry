# Module Dependency Map

Use this map when you need the current package graph or want to plan a refactor
without reopening dependency cycles.

The observed graph is synchronized against
`scripts/audit/module_graph.py --fail-on-doc-drift`. Keep the current-state
section factual and keep design intent in the policy section below it.

## Observed Current Top-Level Package Graph

<!-- module-graph:start -->

- `tab_foundry.__main__` depends on `tab_foundry.cli`.
- `tab_foundry.bench` depends on `tab_foundry.config`,
  `tab_foundry.control_baseline_registry`, `tab_foundry.data`,
  `tab_foundry.device`, `tab_foundry.external_benchmarks`,
  `tab_foundry.benchmark_registry`,
  `tab_foundry.input_normalization`, `tab_foundry.model`,
  `tab_foundry.preprocessing`, `tab_foundry.repo_paths`,
  `tab_foundry.timestamps`, `tab_foundry.training`, and
  `tab_foundry.types`.
- `tab_foundry.benchmark_registry` depends on `tab_foundry.repo_paths`.
- `tab_foundry.cli` depends on `tab_foundry.bench`,
  `tab_foundry.benchmark_registry`, `tab_foundry.config`,
  `tab_foundry.control_baseline_registry`, `tab_foundry.data`,
  `tab_foundry.device`, `tab_foundry.export`, `tab_foundry.model`,
  `tab_foundry.preprocessing`, `tab_foundry.research`, and
  `tab_foundry.training`.
- `tab_foundry.config` depends on `tab_foundry.repo_paths`.
- `tab_foundry.control_baseline_registry` depends on
  `tab_foundry.repo_paths`.
- `tab_foundry.data` depends on `tab_foundry.benchmark_registry`,
  `tab_foundry.preprocessing`, `tab_foundry.repo_paths`,
  `tab_foundry.timestamps`, and `tab_foundry.types`.
- `tab_foundry.export` depends on `tab_foundry.input_normalization`,
  `tab_foundry.model`, `tab_foundry.preprocessing`,
  `tab_foundry.repo_paths`, and `tab_foundry.types`.
- `tab_foundry.model` depends on `tab_foundry.input_normalization` and
  `tab_foundry.types`.
- `tab_foundry.research` depends on `tab_foundry.bench`,
  `tab_foundry.benchmark_registry`, `tab_foundry.config`,
  `tab_foundry.control_baseline_registry`,
  `tab_foundry.external_benchmarks`, `tab_foundry.model`,
  `tab_foundry.repo_paths`, and `tab_foundry.training`.
- `tab_foundry.training` depends on `tab_foundry.config`,
  `tab_foundry.data`, `tab_foundry.device`, `tab_foundry.model`,
  `tab_foundry.preprocessing`, `tab_foundry.repo_paths`,
  `tab_foundry.timestamps`, and `tab_foundry.types`.

<!-- module-graph:end -->

Observed cycle status:

- none

## Intended Dependency-Direction Policy

- `tab_foundry.config`, `tab_foundry.repo_paths`,
  `tab_foundry.device`, `tab_foundry.benchmark_registry`,
  `tab_foundry.control_baseline_registry`,
  `tab_foundry.external_benchmarks`, `tab_foundry.types`,
  `tab_foundry.input_normalization`, and `tab_foundry.timestamps`
  should remain dependency-light helpers.
- `tab_foundry.model` should stay independent of `bench`, `research`,
  `training`, and `export`.
- `tab_foundry.preprocessing` should remain a leaf-style utility package that
  can be used by `data`, `training`, and `export` without growing orchestration
  logic of its own.
- `tab_foundry.data` may depend on `preprocessing` helpers, timestamps, shared
  types, `tab_foundry.repo_paths`, and `tab_foundry.benchmark_registry` when
  corpus-result linkage needs it, but it should not depend on `bench`,
  `training`, or `research`.
- `tab_foundry.training` may depend on `data`, `model`, `preprocessing`, and
  shared helpers such as `tab_foundry.config`, `tab_foundry.device`, and
  `tab_foundry.repo_paths`, but it should not depend on `bench` or `research`.
- `tab_foundry.export` may depend on `model`, `preprocessing`, and shared
  helpers, but it should not depend on `bench`, `research`, or `training`.
- `tab_foundry.bench` is the benchmark and harness layer. It may depend on
  `config`, `data`, `model`, `preprocessing`, `training`, and shared helpers
  such as `tab_foundry.benchmark_registry`, `tab_foundry.device`,
  `tab_foundry.external_benchmarks`, and
  `tab_foundry.control_baseline_registry`, but lower layers should not depend
  on it.
- `tab_foundry.research` is the sweep-management layer. It may depend on
  `bench`, `config`, `model`, dependency-light helper contracts such as
  `tab_foundry.external_benchmarks`,
  `tab_foundry.control_baseline_registry`, `tab_foundry.repo_paths`, and
  read-only `training` inspection helpers, but lower layers should not depend
  on it.
- Execute/promote and sweep-management ownership inside `tab_foundry.research`
  should live under `research/sweep/`; higher layers should import
  `research.sweep.core`, `research.sweep.execute`, and
  `research.sweep.promote` directly instead of reintroducing wrapper modules.
- Research CLI parser ownership should stay under `src/tab_foundry/cli/`; the
  `research/sweep` library modules should stay parser-free.
- Sweep row execution should stay decomposed across dedicated helper modules
  such as `curve_reuse`, `training_state`, `row_dependencies`, and `row_sync`
  instead of regrowing helper logic inside `row_execution.py`.
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

### `src/tab_foundry/bench/comparison_runtime.py`, `src/tab_foundry/bench/compare.py`, And `src/tab_foundry/bench/checkpoint.py`

- Benchmark-facing comparison runtime, CLI/manual orchestration, and checkpoint
  evaluation logic.
- Changes here affect external-baseline comparison and benchmark registry
  records.

### `src/tab_foundry/research/sweep/core.py`

- Canonical sweep manager, queue materializer, and rendered-matrix surface.
- Changes here affect the active research contract and the generated alias
  views under `reference/`.
