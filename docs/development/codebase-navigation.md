# Codebase Navigation

Use this map when you need to find the right package, entry point, or workflow
surface before making a change. It complements
`docs/development/module-dependency-map.md`, which records the observed
top-level package graph plus the intended dependency direction policy.

## 1. Entry Points And Orchestration

These are the user-facing or repo-local control surfaces that bridge commands
into the canonical library modules.

- `src/tab_foundry/__main__.py`: packaged CLI entrypoint for `tab-foundry`.
- `src/tab_foundry/cli/`: nested CLI registration and dispatch for
  `data`, `dev`, `train`, `eval`, `export`, `bench`, and `research`
  workflows.
- `scripts/`: shell convenience helpers, audit tooling, and a small set of
  standalone internal Python entrypoints: benchmark helpers under
  `scripts/bench/` plus the TF-RD-013 support materializer.
- `scripts/dev`: repo-local bootstrap, doctor, ready, verification, and smoke
  wrapper that delegates to the audit tooling and packaged CLI. Hook and audit
  tooling now resolve interpreters with a worktree-first `.venv` policy and
  fall back to the primary checkout only when the current worktree is not yet
  bootstrapped.

## 2. Canonical Library Areas

- `src/tab_foundry/config.py`: shared Hydra composition helpers and config root
  discovery.
- `src/tab_foundry/repo_paths.py`: dependency-light repo-root and repo-relative
  path helpers shared across package boundaries.
- `src/tab_foundry/device.py`: dependency-light concrete-device resolution
  helper shared by benchmark and prior-training flows.
- `src/tab_foundry/benchmark_registry.py`: dependency-light read-only
  benchmark-registry loader, entry-lookup, and path-resolution surface for
  non-mutating readers across `bench` and `research`.
- `src/tab_foundry/control_baseline_registry.py`: dependency-light read-only
  control-baseline registry loader and path-resolution surface shared by
  `bench` and `research`.
- `src/tab_foundry/external_benchmarks.py`: dependency-light canonical source
  for external benchmark ids, defaults, labels, and normalization.
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
  schedules, optimizers, runtime policy, evaluation helpers, telemetry health
  summaries, and the canonical exact-prior training surface under
  `training/prior_train.py`, `training/prior_dump.py`, and `training/prior/`.
  Packaged `train legacy-prior` parser ownership lives under
  `src/tab_foundry/cli/train_prior.py`; `training/prior_train.py` is a
  parser-free library surface.
- `src/tab_foundry/export/`: export bundle construction, loading, and
  validation contracts.
- `src/tab_foundry/bench/`: benchmark bundles, comparison flows, benchmark
  env/bootstrap helpers, smoke harnesses, registry write/orchestration
  helpers, and shared artifact helpers. `bench/compare.py` is the canonical
  manual benchmark comparison/orchestration library surface,
  `bench/run_registration.py` and
  `bench/control_baseline_freeze.py` are the canonical programmatic registry
  write APIs, and `bench/comparison_runtime.py` is the canonical programmatic
  benchmark-execution surface used by research. Packaged `bench` parser
  ownership now lives under `src/tab_foundry/cli/bench_*.py`; the packaged
  bench library modules are parser-free. The remaining standalone internal
  helper entrypoints now live under `scripts/bench/`; their corresponding
  `src/tab_foundry/bench/` modules are parser-free library code.
- `src/tab_foundry/research/`: system-delta sweep state, queue/matrix
  rendering, sweep-result summaries, and research-package path conventions.
  The canonical sweep ownership now lives under
  `src/tab_foundry/research/sweep/`: `catalog.py` owns sweep/index/catalog
  loading, `manage.py` owns sweep creation and metadata inheritance, `materialize.py`
  owns queue loading/materialization, `matrix.py` owns validation/rendering,
  `paths_io.py` owns sweep paths plus YAML/text helpers, `validation.py`
  owns sweep-shape validation helpers, and `anchor.py` owns anchor-context
  derivation. `execute.py` and `promote.py` remain the canonical
  execute/promote library entrypoints, and sweep execution internals now live
  under `research/sweep/configuration.py`, `research/sweep/runtime_env.py`,
  `research/sweep/curve_reuse.py`, `research/sweep/training_state.py`,
  `research/sweep/row_dependencies.py`, `research/sweep/row_sync.py`, and
  `research/sweep/row_execution.py`.

## 3. Workflow Surfaces

The repo uses three stable workflow layers:

- The packaged CLI as the canonical user-facing surface for manifest-backed
  data/build, training, evaluation, export, smoke, tuning, benchmarking,
  registry, and research-sweep flows.
- Shell helpers under `scripts/` such as `scripts/build_manifest.sh` and
  `scripts/configure_repo_protection.sh`, plus `scripts/audit/`, as repo-local
  convenience and verification surfaces, along with the small set of
  standalone internal Python entrypoints under `scripts/bench/` and
  `scripts/materialize_tf_rd_013_support.py`.
- `scripts/dev` as the canonical repo-local entrypoint for bootstrap checks,
  branch review, affected-scope verification, explicit-path verification, full
  verification, and Iris smoke delegation.
- Reference YAML/Markdown artifacts for explicit system-delta sweeps.

Current canonical CLI namespaces:

- `tab-foundry data corpus compare`
- `tab-foundry data corpus inspect`
- `tab-foundry data corpus list-recipes`
- `tab-foundry data corpus materialize`
- `tab-foundry data corpus results`
- `tab-foundry data manifest-inspect`
- `tab-foundry dev resolve-config`
- `tab-foundry dev forward-check`
- `tab-foundry dev diff-config`
- `tab-foundry dev export-check`
- `tab-foundry dev health-check`
- `tab-foundry dev run-inspect`
- `tab-foundry dev data build-manifest`
- `tab-foundry dev data generate-manifest`
- `tab-foundry train run`
- `tab-foundry train legacy-prior simple`
- `tab-foundry train legacy-prior staged`
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
- `tab-foundry research sweep create-sweep`
- `tab-foundry research sweep list`
- `tab-foundry research sweep list-sweeps`
- `tab-foundry research sweep materialize-corpora`
- `tab-foundry research sweep next`
- `tab-foundry research sweep render`
- `tab-foundry research sweep validate`
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

- `program.md` is the explicit sweep-selection execution contract for
  agent-driven system-delta work.
- `reference/system_delta_catalog.yaml` and
  `reference/system_delta_sweeps/<sweep_id>/` are the canonical system-delta
  sources of truth.
- `docs/workflows.md` and `docs/inference.md` stay top-level because they are
  stable operator-facing runbooks.
- `docs/development/` is the canonical home for planning, rationale,
  navigation, dependency docs, and audit artifacts.

## 5. Current Structural Watchpoints

- `src/tab_foundry/bench/` is the canonical home for benchmark and harness
  logic. Core training/model/data packages should not start depending on it.
- Research should keep importing the programmatic benchmark runtime from
  `src/tab_foundry/bench/comparison_runtime.py` instead of reaching into the
  manual compare CLI module.
- Packaged `bench` command parser ownership should stay under
  `src/tab_foundry/cli/`; the corresponding library modules under
  `src/tab_foundry/bench/` should stay parser-free.
- Standalone benchmark helper entrypoints should live under `scripts/bench/`;
  the corresponding `src/tab_foundry/bench/` modules should stay parser-free
  and importable as libraries.
- Packaged `train` command parser ownership should stay under
  `src/tab_foundry/cli/`; `training/prior_train.py` should stay parser-free.
- Shared repo-root and read-only benchmark-registry helpers should continue to
  live in `src/tab_foundry/repo_paths.py` and
  `src/tab_foundry/benchmark_registry.py`; the equivalent control-baseline and
  external-benchmark contracts should continue to live in
  `src/tab_foundry/control_baseline_registry.py`,
  `src/tab_foundry/device.py`, and
  `src/tab_foundry/external_benchmarks.py` instead of being reimplemented in
  lower layers.
- `src/tab_foundry/research/` is the canonical home for sweep queue/matrix
  management; do not recreate parallel queue logic in shell helpers or
  docs-only tooling.
- `research/sweep/execute.py` and `research/sweep/promote.py` should remain
  the canonical execute/promote library surfaces, with CLI parser ownership
  staying under `src/tab_foundry/cli/`; higher layers should import the
  canonical sweep owner modules directly (`catalog.py`, `manage.py`,
  `materialize.py`, `matrix.py`, `paths_io.py`, `validation.py`, and
  `anchor.py`) instead of recreating a barrel module, and row-level helper
  logic should stay factored under the dedicated sweep helper modules instead
  of regrowing a monolithic `row_execution.py`.
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
