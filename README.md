# tab-foundry

TabICLv2-style tabular prior-data fitted network training on `dagzoo` packed shard outputs.

## Environment

- Python `3.13` (pinned in `.python-version`)
- `uv` workflow (lock/sync/run)

## Setup

```bash
uv sync
```

## Build Manifest

Set `DAGZOO_DATA_ROOT` once (optional, but recommended for portability):

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
```

```bash
uv run tab-foundry build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

By default this includes raw `dagzoo` outputs and warns if the manifest contains datasets with
`filter.status=not_run`, `rejected`, or missing filter metadata.

Accepted-only flow:

```bash
dagzoo filter --in data/run1 --out data/run1_filter --curated-out data/run1_curated
uv run tab-foundry build-manifest \
  --data-root data/run1_curated \
  --filter-policy accepted_only \
  --out-manifest data/manifests/accepted_only.parquet
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

## dagzoo Smoke Harness

Run a repo-local end-to-end smoke harness against a sibling `dagzoo` checkout:

```bash
uv run python scripts/dagzoo_smoke.py
```

Default profile:

- `128` generated datasets
- `1024` rows per dataset
- classification on `cpu`
- `250` training steps with evaluation every step
- checkpoint snapshots every `25` steps

Artifacts are written under a timestamped `/tmp/tab_foundry_dagzoo_smoke_*` directory:

- `generated/`
- `manifest.parquet`
- `train_outputs/checkpoints/*.pt`
- `train_outputs/train_history.jsonl`
- `train_outputs/loss_curve.png`
- `telemetry.json`

## Benchmark Env Bootstrap

Create or refresh sibling repo envs before running the notebook-style comparison:

```bash
uv run python scripts/bootstrap_benchmark_envs.py
```

This bootstraps:

- `~/dev/nanoTabPFN/.venv` using a local `pyproject.toml`
- `~/dev/TabPFN/.venv`
- `~/dev/tabicl/.venv`

## nanoTabPFN-Aligned Fast Training

Train tab-foundry directly on the same nanoTabPFN prior dump instead of on `dagzoo`:

```bash
uv run python scripts/train_nanoprior_tab_foundry.py \
  --out-root /tmp/tab_foundry_nanoprior_run
```

Default profile:

- prior dump `~/dev/nanoTabPFN/300k_150x5_2.h5`
- classification on `auto` device selection
- schedule-free AdamW at `4e-3`
- `2500` max steps
- `330s` train-only time budget
- evaluation and checkpoint snapshots every `25` steps

Artifacts are written under a timestamped `/tmp/tab_foundry_nanoprior_*` directory:

- `train_outputs/checkpoints/*.pt`
- `train_outputs/train_history.jsonl`
- `train_outputs/loss_curve.png`
- `telemetry.json`

## nanoTabPFN Comparison

Run either the smoke harness or the nano-prior fast training path first, then compare its
checkpoint snapshots against a sibling `nanoTabPFN` checkout:

```bash
uv run python scripts/train_nanoprior_tab_foundry.py --out-root /tmp/tab_foundry_nanoprior_bench
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir /tmp/tab_foundry_nanoprior_bench \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

Comparison artifacts are written under a timestamped `/tmp/tab_foundry_nanotabpfn_benchmark_*`
directory:

- `benchmark_tasks.json`
- `tab_foundry_curve.jsonl`
- `nanotabpfn_curve.jsonl`
- `comparison_summary.json`
- `comparison_curve.png`

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

- Data is consumed from packed `dagzoo` shard outputs (`train.parquet`, `test.parquet`, `metadata.ndjson`).
- Manifest rows include byte-range + SHA-256 for each `metadata.ndjson` record; loaders verify checksum before JSON parse.
- Manifest building is filter-aware but does not call `dagzoo filter` yet; raw unfiltered training remains supported.
- Many-class classification (`>10` classes) uses mixed-radix + hierarchical probabilities.
- `many_class_train_mode` is the model behavior switch (`path_nll` or `full_probs`); `many_class_inference_mode` in export bundles is the runtime contract and is currently fixed to `full_probs`.
- Default `feature_group_size=32` is a scalability choice; use `feature_group_size=1` to recover paper-style per-feature tokenization.
- `digit_position_embed` is configurable (`model.use_digit_position_embed`) and is applied before the many-class path branch, so it affects both training modes.
- Default optimizer selection is Muon, and Muon availability is required by default (`optimizer.require_requested=true`).
- Training checkpoints remain unchanged for resume; inference bundles are additive cross-repo artifacts.
