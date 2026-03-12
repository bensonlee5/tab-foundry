# Research Workflow

This document explains how to run the current short-run research loop.

## Workflow Surfaces

There are three separate workflows:

1. `dagzoo` smoke: end-to-end pipeline validation
1. internal tuning: tab-foundry-only sweeps on fixed manifests
1. confirmatory benchmarking: compare completed tab-foundry runs against `nanoTabPFN`

Keep these separate. The sweep loop should not call the OpenML benchmark.

## Artifact Contract

Standard artifacts used by the research workflow:

- `train_history.jsonl`
- `telemetry.json`
- `sweep_summary.json`
- `sweep_results.csv`
- `comparison_summary.json`
- `comparison_curve.png`

These are the minimum handoff artifacts for reviewing a run.

## Control Baseline

The current benchmark-facing control is:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_linear \
  data.manifest_path=data/manifests/default.parquet
```

This profile is intended for short-run comparison work, not for the smoke harness.

## Internal Tuning

Use the internal sweep runner on a fixed manifest:

```bash
uv run python scripts/tune_tab_foundry.py \
  --manifest-path data/manifests/default.parquet
```

Current default sweep dimensions:

- `lr_max`
- `warmup_ratio`
- `grad_clip`

The sweep ranks runs by internal metrics only:

1. lowest `best_val_loss`
1. lowest `final_val_loss`
1. lowest post-warmup train-loss variance

Gradient norm is logged as a stability diagnostic, not as the primary ranking target.

## Confirmatory Benchmarking

Use benchmark comparison only after a run has already been selected internally:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir <run_dir> \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

Comparison summaries should be reviewed using:

- `best_step`
- `best_training_time`
- `best_roc_auc`
- `final_step`
- `final_training_time`
- `final_roc_auc`

If best internal validation and best external benchmark diverge materially, treat that as a selection-quality problem rather than silently trusting internal validation.

## Review Loop

Recommended order for each research cycle:

1. run or reuse the current control baseline
1. run an internal sweep against the same manifest and budget class
1. shortlist the best internal candidates
1. run confirmatory nanoTabPFN comparisons on the shortlist
1. append the results to the shared leaderboard
1. write a one-line conclusion for each candidate: keep, reject, or defer

## Scope Boundaries

Use the smoke harness for plumbing checks:

```bash
uv run python scripts/dagzoo_smoke.py
```

Do not treat smoke-run metrics as the research leaderboard by default. Smoke exists to validate the end-to-end path, while the benchmark profile and sweep workflow exist to drive model decisions.
