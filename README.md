# tab-foundry

Modular tabular prior-data fitted network training on `dagzoo` packed shard outputs.

## Environment

- Python `3.14` (pinned in `.python-version`)
- `uv` workflow for sync, tooling, and commands

## Setup

```bash
./scripts/dev bootstrap
```

`./scripts/dev bootstrap` wraps the canonical repo-local setup:

```bash
uv sync
pre-commit install
```

After setup, activate the virtual environment so you can run commands directly
without the `uv run` prefix:

```bash
source .venv/bin/activate
```

Repo-local `uv sync` includes the benchmark helper dependencies plus Muon
through the dev environment. For a minimal non-dev install, opt into the extra
surfaces explicitly:

```bash
uv sync --no-dev --extra benchmark --extra muon
```

## CLI Navigation

`tab-foundry` is the canonical packaged CLI for data, dev, train, eval,
export, bench, and research workflows. Treat `./scripts/dev` as a repo-local
convenience wrapper only for bootstrap, verification, and Iris smoke.

For command discovery and execution, prefer `.venv/bin/tab-foundry ...` or an
activated `.venv`. Use help in this order:

```bash
.venv/bin/tab-foundry --help
.venv/bin/tab-foundry <group> --help
.venv/bin/tab-foundry <group> <command> --help
```

Use [codebase navigation](docs/development/codebase-navigation.md) for the full
namespace inventory and [workflows](docs/workflows.md) for operational
runbooks.

Summarize the current diff against `origin/main` and run the smallest safe
verification slice:

```bash
./scripts/dev review-base
./scripts/dev verify affected
./scripts/dev verify paths src/tab_foundry/model/architectures/tabfoundry_staged/subsystems.py
```

Run the full local quality gate:

```bash
./scripts/dev verify full
```

Inspect one resolved config or run a forward-only construction smoke check:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
```

Summarize one run's instability telemetry or one sweep's local results:

```bash
tab-foundry dev health-check --run-dir outputs/cls_smoke
tab-foundry dev run-inspect --run-dir outputs/cls_smoke
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
tab-foundry research sweep summarize --include-screened
tab-foundry research sweep inspect --order 6 --sweep-id binary_md_v1
tab-foundry research sweep diff --order 7 --against-order 6 --sweep-id binary_md_v1
```

## Quickstart

Build a manifest:

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
tab-foundry data build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

Train a smoke profile:

```bash
tab-foundry train run experiment=cls_smoke
```

Evaluate a checkpoint:

```bash
tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke
```

Export and validate an inference bundle:

```bash
tab-foundry export bundle \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v3

tab-foundry export validate \
  --bundle-dir outputs/exports/cls_smoke_v3
```

Run the Iris smoke harness:

```bash
./scripts/dev smoke iris
```

Repo-local `uv sync` includes Muon. If you are using a minimal install without
the `muon` extra, override the optimizer explicitly:

```bash
tab-foundry train run experiment=cls_smoke optimizer=adamw
```

## Docs

- `docs/workflows.md`: setup, manifest build, train/eval/export, smoke flows, tuning, benchmarking, and CI
- `docs/inference.md`: export bundle schema and validation contract
- `docs/development/roadmap.md`: canonical planning state and ranked roadmap
- `docs/development/design-decisions.md`: architecture direction, repo-structure policy, and compatibility guidance
- `docs/development/model-architecture.md`: detailed architecture reference for the current staged/simple model surfaces
- `docs/development/architecture-deltas.md`: diagram-first comparison of the current anchor against TabPFN and TabICLv2
- `docs/development/model-config.md`: model configuration reference, defaults, and resolution rules
- `docs/development/codebase-navigation.md`: current package layout and workflow entry surfaces
- `docs/development/module-dependency-map.md`: maintained baseline dependency view for repo evolution
- `reference/README.md`: index for literature notes, evidence maps, and future adjacent-repo summaries
- `reference/papers.md`: curated papers, typed-column-encoder references, and external baseline borrowing rules
- `reference/evidence.md`: roadmap-to-reference mapping and evidence notes
