# tab-foundry

TabICLv2-style tabular prior-data fitted network training on `cauchy-generator` outputs.

## Environment

- Python `3.13` (pinned in `.python-version`)
- `uv` workflow (lock/sync/run)

## Setup

```bash
uv sync
```

## Build Manifest

Set `CAUCHY_DATA_ROOT` once (optional, but recommended for portability):

```bash
export CAUCHY_DATA_ROOT="$HOME/dev/cauchy-generator/data"
```

```bash
uv run tab-foundry build-manifest \
  --data-root "${CAUCHY_DATA_ROOT:-$HOME/dev/cauchy-generator/data}" \
  --out-manifest data/manifests/default.parquet
```

Or:

```bash
./scripts/build_manifest.sh
```

## Train

Classification smoke:

```bash
uv run tab-foundry train experiment=cls_smoke
```

Regression smoke:

```bash
uv run tab-foundry train experiment=reg_smoke
```

Workstation defaults:

```bash
uv run tab-foundry train experiment=cls_workstation
uv run tab-foundry train experiment=reg_workstation
```

Default training requires Muon to be installed. To run without Muon, explicitly override the
optimizer:

```bash
uv run tab-foundry train experiment=cls_smoke optimizer=adamw
```

## Evaluate

```bash
uv run tab-foundry eval --checkpoint outputs/cls_smoke/checkpoints/best.pt experiment=cls_smoke
```

## Export Inference Bundle

Export a training checkpoint to a versioned inference handoff bundle:

```bash
uv run tab-foundry export \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v1
```

Validate a bundle:

```bash
uv run tab-foundry validate-export --bundle-dir outputs/exports/cls_smoke_v1
```

Contract details:

- `docs/INFERENCE_CONTRACT.md`

## Notes

- Data is consumed from packed `cauchy-generator` shard outputs (`train.parquet`, `test.parquet`, `metadata.ndjson`).
- Many-class classification (`>10` classes) uses mixed-radix + hierarchical probabilities.
- Default `feature_group_size=32` is a scalability choice; use `feature_group_size=1` to recover paper-style per-feature tokenization.
- `digit_position_embed` for mixed-radix views is an intentional enhancement to encode digit significance.
- Default optimizer selection is Muon, and Muon availability is required by default (`optimizer.require_requested=true`).
- Training checkpoints remain unchanged for resume; inference bundles are additive cross-repo artifacts.
