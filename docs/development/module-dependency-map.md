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
  `tab_foundry.device`, `tab_foundry.export`,
  `tab_foundry.external_benchmarks`,
  `tab_foundry.benchmark_registry`,
  `tab_foundry.hardware_architecture_registry`,
  `tab_foundry.hardware_profiles`,
  `tab_foundry.checkpoint_state`,
  `tab_foundry.input_normalization`, `tab_foundry.model`,
  `tab_foundry.preprocessing`, `tab_foundry.registry`,
  `tab_foundry.repo_paths`, `tab_foundry.task_batching`,
  `tab_foundry.timestamps`, `tab_foundry.training`, and
  `tab_foundry.types`.
- `tab_foundry.benchmark_registry` depends on `tab_foundry.registry` and
  `tab_foundry.repo_paths`.
- `tab_foundry.hardware_architecture_registry` depends on
  `tab_foundry.registry` and `tab_foundry.repo_paths`.
- `tab_foundry.cli` depends on `tab_foundry.bench`,
  `tab_foundry.benchmark_registry`, `tab_foundry.config`,
  `tab_foundry.config_inspection`, `tab_foundry.control_baseline_registry`,
  `tab_foundry.data`, `tab_foundry.device`, `tab_foundry.export`,
  `tab_foundry.external_benchmarks`,
  `tab_foundry.hardware_architecture_registry`, `tab_foundry.model`,
  `tab_foundry.repo_paths`, `tab_foundry.research`,
  `tab_foundry.task_batching`, and `tab_foundry.training`.
- `tab_foundry.config` depends on `tab_foundry.repo_paths`.
- `tab_foundry.config_inspection` depends on `tab_foundry.config`,
  `tab_foundry.data`, `tab_foundry.model`,
  `tab_foundry.preprocessing`, and `tab_foundry.training`.
- `tab_foundry.control_baseline_registry` depends on
  `tab_foundry.registry` and `tab_foundry.repo_paths`.
- `tab_foundry.data` depends on `tab_foundry.benchmark_registry`,
  `tab_foundry.feature_types`, `tab_foundry.hashing`,
  `tab_foundry.preprocessing`, `tab_foundry.repo_paths`,
  `tab_foundry.task_batching`, `tab_foundry.timestamps`, and
  `tab_foundry.types`.
- `tab_foundry.export` depends on `tab_foundry.checkpoint_state`,
  `tab_foundry.device`, `tab_foundry.feature_types`,
  `tab_foundry.hardware_profiles`, `tab_foundry.hashing`,
  `tab_foundry.model`, `tab_foundry.preprocessing`,
  `tab_foundry.repo_paths`, `tab_foundry.task_batching`, and
  `tab_foundry.types`.
- `tab_foundry.model` depends on `tab_foundry.feature_types`,
  `tab_foundry.input_normalization`, `tab_foundry.likelihoods`,
  `tab_foundry.task_batching`, and `tab_foundry.types`.
- `tab_foundry.preprocessing` depends on `tab_foundry.feature_types`.
- `tab_foundry.registry` depends on `tab_foundry.repo_paths`.
- `tab_foundry.research` depends on `tab_foundry.bench`,
  `tab_foundry.benchmark_registry`, `tab_foundry.config`,
  `tab_foundry.control_baseline_registry`, `tab_foundry.data`,
  `tab_foundry.device`, `tab_foundry.external_benchmarks`,
  `tab_foundry.hashing`, `tab_foundry.model`,
  `tab_foundry.repo_paths`, `tab_foundry.training`, and
  `tab_foundry.types`.
- `tab_foundry.task_batching` depends on `tab_foundry.feature_types`
  and `tab_foundry.types`.
- `tab_foundry.training` depends on `tab_foundry.data`,
  `tab_foundry.checkpoint_state`, `tab_foundry.device`,
  `tab_foundry.feature_types`, `tab_foundry.hardware_profiles`,
  `tab_foundry.hashing`, `tab_foundry.likelihoods`,
  `tab_foundry.model`, `tab_foundry.preprocessing`,
  `tab_foundry.repo_paths`, `tab_foundry.task_batching`,
  `tab_foundry.timestamps`, and `tab_foundry.types`.

<!-- module-graph:end -->

Observed cycle status:

- no top-level cycle candidates are currently present.

## Intended Dependency-Direction Policy

- `tab_foundry.config`, `tab_foundry.repo_paths`,
  `tab_foundry.device`, `tab_foundry.registry`,
  `tab_foundry.benchmark_registry`,
  `tab_foundry.checkpoint_state`,
  `tab_foundry.control_baseline_registry`,
  `tab_foundry.hardware_architecture_registry`,
  `tab_foundry.hardware_profiles`,
  `tab_foundry.external_benchmarks`, `tab_foundry.hashing`,
  `tab_foundry.types`, `tab_foundry.input_normalization`,
  `tab_foundry.likelihoods`,
  `tab_foundry.feature_types`, `tab_foundry.task_batching`, and
  `tab_foundry.timestamps` should remain dependency-light helpers.
- `tab_foundry.model` should stay independent of `bench`, `research`,
  `training`, and `export`.
- `tab_foundry.preprocessing` should remain a leaf-style utility package that
  can be used by `data`, `training`, and `export` without growing orchestration
  logic of its own.
- `tab_foundry.data` may depend on `preprocessing` helpers, timestamps, shared
  types, `tab_foundry.repo_paths`, `tab_foundry.benchmark_registry`, and
  shared task-batching helpers when manifest packing needs them, but it should
  not depend on `bench`, `training`, or `research`.
- `tab_foundry.training` may depend on `data`, `model`, `preprocessing`, and
  shared helpers such as `tab_foundry.checkpoint_state`,
  `tab_foundry.config`, `tab_foundry.device`,
  `tab_foundry.feature_types`, `tab_foundry.hardware_profiles`,
  `tab_foundry.repo_paths`, and
  `tab_foundry.task_batching`, but it should not depend on `bench` or
  `research`.
- Packaged `train` CLI parser ownership should stay under
  `src/tab_foundry/cli/`; `training/prior_train.py` should stay parser-free.
- `tab_foundry.export` may depend on `model`, `preprocessing`, and shared
  helpers such as `tab_foundry.device`,
  `tab_foundry.hardware_profiles`, and `tab_foundry.task_batching`, but it
  should not depend on `bench`, `research`, or `training`.
- `tab_foundry.bench` is the benchmark and harness layer. It may depend on
  `config`, `data`, `export`, `model`, `preprocessing`, `training`, and shared
  helpers such as `tab_foundry.registry`,
  `tab_foundry.benchmark_registry`, `tab_foundry.device`,
  `tab_foundry.external_benchmarks`,
  `tab_foundry.hardware_architecture_registry`,
  `tab_foundry.hardware_profiles`, and
  `tab_foundry.control_baseline_registry`, but lower layers should not depend
  on it.
- `tab_foundry.research` is the sweep-management layer. It may depend on
  `bench`, `config`, `data`, `model`, dependency-light helper contracts such as
  `tab_foundry.control_baseline_registry`, `tab_foundry.device`,
  `tab_foundry.external_benchmarks`, `tab_foundry.hashing`,
  `tab_foundry.repo_paths`, `tab_foundry.types`, and read-only `training`
  inspection helpers, but lower layers should not depend on it.
- Execute/promote and sweep-management ownership inside `tab_foundry.research`
  should live under `research/sweep/`; higher layers should import the
  canonical owner modules directly (`research.sweep.catalog`,
  `research.sweep.manage`, `research.sweep.materialize`,
  `research.sweep.matrix`, `research.sweep.paths_io`,
  `research.sweep.validation`, `research.sweep.anchor`,
  `research.sweep.execute`, and `research.sweep.promote`) instead of
  reintroducing wrapper or barrel modules.
- Research CLI parser ownership should stay under `src/tab_foundry/cli/`; the
  `research/sweep` library modules should stay parser-free.
- Packaged `bench` CLI parser ownership should stay under
  `src/tab_foundry/cli/`; the packaged bench library modules should stay
  parser-free.
- Standalone internal benchmark helper entrypoints should live under
  `scripts/bench/`; their corresponding `src/tab_foundry/bench/` modules
  should stay parser-free and should not regrow local `argparse` surfaces.
- Sweep row execution should stay decomposed across dedicated helper modules
  such as `curve_reuse`, `training_state`, `row_dependencies`, and `row_sync`
  instead of regrowing helper logic inside `row_execution.py`.
- User-facing workflow entrypoints should live under the packaged nested CLI
  rather than being duplicated under `scripts/`.
- `scripts/` should stay limited to shell convenience helpers, audit tooling,
  and the small set of standalone internal benchmark helper entrypoints under
  `scripts/bench/`; it should not regrow parallel user-facing Python workflow
  surfaces.

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

### `src/tab_foundry/bench/comparison_contract.py`, `src/tab_foundry/bench/comparison_runtime.py`, `src/tab_foundry/bench/comparison_reporting.py`, And `src/tab_foundry/bench/checkpoint.py`

- Benchmark-facing comparison contract/defaults, execution, reporting, and
  checkpoint evaluation logic.
- Changes here affect external-baseline comparison and benchmark registry
  records.

### `src/tab_foundry/research/sweep/manage.py`, `src/tab_foundry/research/sweep/materialize.py`, `src/tab_foundry/research/sweep/matrix.py`, And `src/tab_foundry/research/sweep/catalog.py`

- Canonical sweep lifecycle, queue materialization, matrix rendering, and
  catalog/index loading surfaces.
- Changes here affect the active research contract and the generated alias
  views under `reference/`.
