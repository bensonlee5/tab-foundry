# tab-foundry

Modular tabular prior-data fitted network training on `dagzoo` packed shard outputs.

## Environment

- Python `3.13` (pinned in `.python-version`)
- `uv` workflow for sync, tooling, and commands

## Setup

```bash
uv sync
uv run pre-commit install
```

Run the local quality gate:

```bash
uv run pre-commit run --all-files
```

## Quickstart

Build a manifest:

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
uv run tab-foundry build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

Train a smoke profile:

```bash
uv run tab-foundry train experiment=cls_smoke
```

Evaluate a checkpoint:

```bash
uv run tab-foundry eval \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke
```

Export and validate an inference bundle:

```bash
uv run tab-foundry export \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v1

uv run tab-foundry validate-export \
  --bundle-dir outputs/exports/cls_smoke_v1
```

Run the Iris smoke harness:

```bash
uv run python scripts/iris_smoke.py
```

If Muon is not installed locally, override the optimizer explicitly:

```bash
uv run tab-foundry train experiment=cls_smoke optimizer=adamw
```

## Docs

- `docs/workflows.md`: setup, manifest build, train/eval/export, smoke flows, tuning, benchmarking, and CI
- `docs/architecture.md`: architecture direction, repo layout, dependency direction, and compatibility guidance
- `docs/inference.md`: export bundle schema and validation contract
- `docs/roadmap.md`: canonical planning state and ranked roadmap
- `reference/papers.md`: curated papers, typed-column-encoder references, and external baseline borrowing rules
- `reference/evidence.md`: roadmap-to-reference mapping and evidence notes
