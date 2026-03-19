# tab-foundry

Modular tabular prior-data fitted network training on `dagzoo` packed shard outputs.

## Environment

- Python `3.14` (pinned in `.python-version`)
- `uv` workflow for sync, tooling, and commands

## Setup

```bash
uv sync
uv run pre-commit install
```

`uv sync` is the canonical repo-local setup and includes the benchmark helper
dependencies plus Muon through the dev environment. For a minimal non-dev
install, opt into the extra surfaces explicitly:

```bash
uv sync --no-dev --extra benchmark --extra muon
```

Run the local quality gate:

```bash
uv run pre-commit run --all-files
```

## Quickstart

Build a manifest:

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
uv run tab-foundry data build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

Train a smoke profile:

```bash
uv run tab-foundry train run experiment=cls_smoke
```

Evaluate a checkpoint:

```bash
uv run tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke
```

Export and validate an inference bundle:

```bash
uv run tab-foundry export bundle \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v3

uv run tab-foundry export validate \
  --bundle-dir outputs/exports/cls_smoke_v3
```

Run the Iris smoke harness:

```bash
uv run tab-foundry bench smoke iris
```

Repo-local `uv sync` includes Muon. If you are using a minimal install without
the `muon` extra, override the optimizer explicitly:

```bash
uv run tab-foundry train run experiment=cls_smoke optimizer=adamw
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
