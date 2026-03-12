# Module Dependency Map

This file is a maintained baseline for imports under `src/tab_foundry`.
It is hand-authored for now; there is no generator for this document yet.

Update it when major package boundaries move or when a new top-level package
area becomes part of the canonical architecture.

## Package Dependency DAG

- `tab_foundry.__main__` depends on `tab_foundry.cli`.
- `tab_foundry.cli` depends on `tab_foundry.cli.commands`.
- `tab_foundry.cli.commands` depends on `tab_foundry.bench`,
  `tab_foundry.config`, `tab_foundry.data`, `tab_foundry.export`, and
  `tab_foundry.training`.
- `tab_foundry.bench` depends on `tab_foundry.config`, `tab_foundry.data`,
  `tab_foundry.export`, `tab_foundry.model`, and `tab_foundry.training`.
- `tab_foundry.training` depends on `tab_foundry.data` and
  `tab_foundry.model`.
- `tab_foundry.export` depends on `tab_foundry.model`.
- `tab_foundry.model.factory` depends on `tab_foundry.model.architectures`.
- `tab_foundry.model.architectures` depends on
  `tab_foundry.model.components`.
- `tab_foundry.data.sources` depends on `tab_foundry.data`.
- `tab_foundry.model` currently depends on small root helpers such as
  `tab_foundry.input_normalization` and `tab_foundry.types`, but it should
  avoid depending on `bench`, `training`, or `export`.
- `tab_foundry.config`, `tab_foundry.types`, and
  `tab_foundry.input_normalization` are shared helpers and should remain
  dependency-light.

## Change-Impact Hotspots

### `src/tab_foundry/model/factory.py`

- Shared model construction surface used by training, evaluation, export, and
  checkpoint loading.
- Changes here ripple into CLI flows, `bench/` harnesses, and export
  compatibility.

### `src/tab_foundry/training/trainer.py`

- Central training loop and artifact-emission surface.
- Changes here affect smoke, tuning, checkpoint benchmarking, and shortlist
  evaluation workflows.

### `src/tab_foundry/export/contracts.py`

- Export bundle contract boundary.
- Changes here are user-facing and must be treated as artifact-schema changes.

### `src/tab_foundry/bench/compare.py` And `src/tab_foundry/bench/checkpoint.py`

- Benchmark-facing comparison and checkpoint evaluation logic.
- Changes here affect external-baseline comparison and control-promotion
  workflows.

### `src/tab_foundry/data/manifest.py` And `src/tab_foundry/data/sources/manifest.py`

- Manifest-backed data is the canonical source path today.
- Changes here ripple through train, eval, smoke, and export workflows.

## Current Gaps Versus Target Layout

- The `model/components` / `model/architectures` split is now implemented, but
  the repo still lacks a neutral multi-family registry above the current
  `TabICLv2` path.
- `scripts/` remain thin wrapper entrypoints over `bench/`; this is deliberate
  for now and not a sign that workflow logic should move back out of `bench/`.
- This file is descriptive, not generated. Tooling-backed dependency docs can
  be added later if repo churn makes manual upkeep too costly.
