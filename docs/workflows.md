# Workflows

This document is the operational runbook for `tab-foundry`: setup, local quality checks, core commands, smoke flows, tuning, and benchmark-adjacent comparisons.

Related docs:

- quickstart: `README.md`
- design decisions and repo structure: `docs/development/design-decisions.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- inference/export contract: `docs/inference.md`
- canonical roadmap: `docs/development/roadmap.md`

Planning and repo-shape docs now live under `docs/development/`.
This file stays top-level because it is the stable operational runbook.

## Environment And Quality Gate

- Python `3.13` is pinned in `.python-version`.
- Use the repo-local `uv` workflow for sync and command execution.

Setup:

```bash
uv sync
uv run pre-commit install
```

Run the full local quality gate:

```bash
uv run pre-commit run --all-files
```

Format markdown directly:

```bash
uv run mdformat README.md docs reference CHANGELOG.md AGENTS.md
```

CI workflow `test` runs two required jobs on pull requests and `main`:

- `quality-and-unit`: `mdformat --check`, `ruff`, `mypy`, `pytest -q`
- `iris-smoke`: the Iris-backed smoke harness with uploaded artifacts and markdown summary

Apply matching branch protection with:

```bash
./scripts/configure_repo_protection.sh
```

## Canonical Workflow Surfaces

Keep these workflows separate:

1. manifest-backed train/eval/export
1. smoke harnesses for end-to-end plumbing checks
1. internal tuning on a fixed manifest
1. confirmatory benchmarking against `nanoTabPFN`

The sweep loop should not call the OpenML benchmark directly.

## Manifest Build

Set `DAGZOO_DATA_ROOT` once if you want a stable sibling data path:

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
```

Build the default manifest:

```bash
uv run tab-foundry build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

By default this includes raw `dagzoo` outputs and warns if the manifest contains datasets with `filter.status=not_run`, `rejected`, or missing filter metadata.

Accepted-only flow:

```bash
dagzoo filter --in data/run1 --out data/run1_filter --curated-out data/run1_curated
uv run tab-foundry build-manifest \
  --data-root data/run1_curated \
  --filter-policy accepted_only \
  --out-manifest data/manifests/accepted_only.parquet
```

Helper script:

```bash
./scripts/build_manifest.sh
```

## Train, Evaluate, And Export

Common training profiles:

```bash
uv run tab-foundry train experiment=cls_smoke
uv run tab-foundry train experiment=reg_smoke
uv run tab-foundry train experiment=cls_workstation
uv run tab-foundry train experiment=reg_workstation
uv run tab-foundry train \
  experiment=cls_benchmark_linear \
  data.manifest_path=data/manifests/default.parquet
uv run tab-foundry train \
  experiment=cls_benchmark_linear_simple \
  data.manifest_path=<binary_manifest.parquet>
uv run tab-foundry train \
  experiment=cls_benchmark_staged \
  data.manifest_path=<binary_manifest.parquet>
```

`cls_benchmark_linear_simple` now targets the exact nanoTabPFN-style binary
debug model. It is benchmark-focused rather than a general small-class
classifier, so it requires binary tasks, internal train-split z-score clipping,
and `many_class_base=2`.

`cls_benchmark_staged` is the staged research counterpart. It defaults to
`model.arch=tabfoundry_staged` and `model.stage=nano_exact`, so the first run
starts from the frozen repro contract and then promotes forward by overriding
`model.stage`.

Example promotion step:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_staged \
  data.manifest_path=<binary_manifest.parquet> \
  model.stage=label_token
```

Prior-dump training for the staged family uses the same harness with the staged
experiment default:

```bash
uv run python scripts/train_tabfoundry_staged_prior.py
```

Default training expects Muon to be installed. To run without Muon:

```bash
uv run tab-foundry train experiment=cls_smoke optimizer=adamw
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
  --out-dir outputs/exports/cls_smoke_v3

uv run tab-foundry validate-export \
  --bundle-dir outputs/exports/cls_smoke_v3
```

## Standard Workflow Artifacts

These artifacts are the minimum handoff surface for reviewable runs:

- `train_history.jsonl`
- `telemetry.json`
- `summary.md`
- `loss_curve.png`
- `sweep_summary.json`
- `sweep_results.csv`
- `comparison_summary.json`
- `comparison_curve.png`

Smoke and benchmark-style runs may also persist generated datasets, manifests, and checkpoint snapshots.

## Smoke Workflows

### Iris Smoke

Run the repo-local Iris-backed smoke harness:

```bash
uv run python scripts/iris_smoke.py
```

This generates manifest-compatible packed Iris tasks, trains a small classification checkpoint on CPU by default, evaluates it on the manifest test split, then runs the binary-Iris checkpoint benchmark.

Artifacts are written under a timestamped `/tmp/tab_foundry_iris_smoke_*` directory.

### dagzoo Smoke

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

Artifacts are written under a timestamped `/tmp/tab_foundry_dagzoo_smoke_*` directory.

Smoke exists to validate the end-to-end path. Do not treat smoke-run metrics as the research leaderboard by default.

## Internal Tuning

Run an internal-only sweep on a fixed manifest:

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

Bootstrap sibling benchmark envs:

```bash
uv run python scripts/bootstrap_benchmark_envs.py
```

This bootstraps sibling envs for:

- `~/dev/nanoTabPFN`
- `~/dev/TabPFN`
- `~/dev/tabicl`

Run benchmark comparison only after a run has already been selected internally:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir <run_dir> \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

The comparison flow is pinned to the repo-tracked benchmark bundle at
`src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`. Runs fail fast if
the live OpenML-resolved selection thresholds or task metadata drift from that
bundle. Older ad hoc bundle files without the full `selection` schema now fail
to load.

The benchmark-profile training config now writes `train_history.jsonl` directly
under `runtime.output_dir`, so benchmark comparison can consume plain
`tab-foundry train experiment=cls_benchmark_linear ...` outputs without a smoke
wrapper.

One simple preparation path is:

```bash
uv run python scripts/dagzoo_smoke.py --out-root /tmp/tab_foundry_dagzoo_smoke_bench
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir /tmp/tab_foundry_dagzoo_smoke_bench \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

The canonical control-baseline path is:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_linear \
  data.manifest_path=data/manifests/default.parquet \
  runtime.output_dir=outputs/control_baselines/cls_benchmark_linear_v1/train

uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir outputs/control_baselines/cls_benchmark_linear_v1/train \
  --out-root outputs/control_baselines/cls_benchmark_linear_v1/benchmark \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5

uv run python scripts/freeze_control_baseline.py \
  --run-dir outputs/control_baselines/cls_benchmark_linear_v1/train \
  --comparison-summary outputs/control_baselines/cls_benchmark_linear_v1/benchmark/comparison_summary.json
```

Keep `outputs/control_baselines/cls_benchmark_linear_v1/train` empty before rerunning the
training command; `tab-foundry train` now fails fast if `runtime.output_dir` already contains a
non-empty history file or checkpoint `.pt` artifacts.

Frozen control baselines are tracked in
`src/tab_foundry/bench/control_baselines_v1.json`. Registry entries store the
baseline id, config profile, budget class, manifest path, seed set, preserved
run path, comparison summary path, benchmark bundle metadata, and compact
tab-foundry best/final ROC AUC metrics.

Later comparison runs can copy one of those registry entries into
`comparison_summary.json` via:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir <run_dir> \
  --out-root <benchmark_out_root> \
  --control-baseline-id cls_benchmark_linear_v1 \
  --control-baseline-registry src/tab_foundry/bench/control_baselines_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

Review comparison summaries using:

- `best_step`
- `best_training_time`
- `best_roc_auc`
- `final_step`
- `final_training_time`
- `final_roc_auc`
- `benchmark_bundle.name`
- `benchmark_bundle.version`
- `benchmark_bundle.task_ids`
- `control_baseline.baseline_id` when the run is explicitly compared against a
  frozen control

If best internal validation and best external benchmark diverge materially, treat that as a selection-quality problem rather than silently trusting internal validation.

## Research Review Loop

Recommended order for each research cycle:

1. run or reuse the current control baseline
1. run an internal sweep against the same manifest and budget class
1. shortlist the best internal candidates
1. run confirmatory `nanoTabPFN` comparisons on the shortlist
1. append the results to the shared leaderboard
1. write a one-line conclusion for each candidate: keep, reject, or defer

## Scope Boundaries

- Use the smoke harnesses for plumbing checks, not for the canonical leaderboard.
- Keep benchmark-facing work in the short-run budget class.
- Use internal sweeps to prune candidates before broader benchmark runs.
- Keep export/inference compatibility intact until a dedicated migration effort is planned.
