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
For agent-driven architecture search, use `program.md` as the execution
contract; this file stays focused on commands, artifacts, and operational
policy.

## Environment And Quality Gate

- Python `3.13` is pinned in `.python-version`.
- Use the repo-local `uv` workflow for sync and command execution.

Setup:

```bash
uv sync
uv run pre-commit install
```

`uv sync` is the canonical repo-local setup and includes the benchmark helper
dependencies plus Muon through the dev environment. For a minimal non-dev
install, opt into those optional surfaces explicitly:

```bash
uv sync --no-dev --extra benchmark --extra muon
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
`model.arch=tabfoundry_staged` and `model.stage=nano_exact`. Treat that as the
base recipe only. The live staged workflow now records isolated changes through
the active system-delta sweep using `model.stage_label` and
`model.module_overrides`, not through a promotion ladder in this runbook.

Prior-dump training for the staged family uses the same harness with the staged
experiment default:

```bash
uv run python scripts/train_tabfoundry_staged_prior.py
```

Use the queue row plus `reference/system_delta_campaign_template.md` to decide
the staged labels, module overrides, and research-package paths for each
benchmark-facing delta.

Repo-local `uv sync` includes Muon. If you are using a minimal install without
the `muon` extra, run without Muon explicitly:

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
- `gradient_history.jsonl`
- `telemetry.json`
- `summary.md`
- `loss_curve.png`
- `sweep_summary.json`
- `sweep_results.csv`
- `comparison_summary.json`
- `benchmark_run_record.json`
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

The benchmark and plotting helpers live under `src/tab_foundry/bench/`, but
their third-party dependencies are modeled as the optional `benchmark` extra
for non-dev installs. Repo-local `uv sync` already includes them.

Run benchmark comparison only after a run has already been selected internally:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir <run_dir> \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

The comparison flow is pinned to the repo-tracked benchmark bundle at
`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`. Runs fail fast
if the live OpenML-resolved selection thresholds or task metadata drift from
that bundle. Older ad hoc bundle files without the full `selection` schema now
fail to load.

The pinned medium-size binary bundle is now the canonical binary benchmark
surface for current sweep runs. The canonical control baseline is
`cls_benchmark_linear_v2`. The legacy 3-task binary bundle remains available at
`src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json` for historical
reproduction only. Benchmark results are not directly comparable across those
two binary surfaces unless the bundle path and control-baseline id match
exactly. The staged multiclass branch uses the companion bundle at
`src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json`, which
widens the TabArena v0.1 source set from binary-only to small multiclass while
preserving the same row-count, feature-count, missingness, and minority-class
filters.

Regenerate the canonical medium binary bundle with:

```bash
uv run python scripts/build_openml_benchmark_bundle.py \
  --out-path src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json \
  --bundle-name nanotabpfn_openml_binary_medium \
  --version 1 \
  --task-source binary_expanded_v1 \
  --new-instances 200 \
  --max-features 10 \
  --max-classes 2 \
  --max-missing-pct 0.0 \
  --min-minority-class-pct 2.5
```

Regenerate the multiclass companion bundle with:

```bash
uv run python scripts/build_openml_benchmark_bundle.py \
  --out-path src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json \
  --bundle-name nanotabpfn_openml_classification_small \
  --version 1 \
  --task-source tabarena_v0_1 \
  --new-instances 200 \
  --max-features 10 \
  --max-classes auto \
  --max-missing-pct 0.0 \
  --min-minority-class-pct 2.5
```

Benchmark against a non-default repo-tracked bundle with:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir <run_dir> \
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

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

The canonical control-baseline surface is:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_linear \
  data.manifest_path=data/manifests/default.parquet \
  runtime.output_dir=outputs/control_baselines/cls_benchmark_linear_v2/train

uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir outputs/control_baselines/cls_benchmark_linear_v2/train \
  --out-root outputs/control_baselines/cls_benchmark_linear_v2/benchmark \
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5

uv run python scripts/freeze_control_baseline.py \
  --baseline-id cls_benchmark_linear_v2 \
  --run-dir outputs/control_baselines/cls_benchmark_linear_v2/train \
  --comparison-summary outputs/control_baselines/cls_benchmark_linear_v2/benchmark/comparison_summary.json
```

Keep `outputs/control_baselines/cls_benchmark_linear_v2/train` empty before rerunning the
training command; `tab-foundry train` now fails fast if `runtime.output_dir` already contains a
non-empty history file or checkpoint `.pt` artifacts.

`cls_benchmark_linear_v2` is the live canonical control surface for the medium
binary bundle. `cls_benchmark_linear_v1` remains in the registry unchanged as a
historical 3-task surface and should only be used when reproducing earlier
comparisons.

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
  --control-baseline-id cls_benchmark_linear_v2 \
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
- `benchmark_bundle.source_path`
- `benchmark_bundle.task_ids`
- `control_baseline.baseline_id` when the run is explicitly compared against a
  frozen control

If best internal validation and best external benchmark diverge materially, treat that as a selection-quality problem rather than silently trusting internal validation.

Benchmark-facing research runs can also be registered in the canonical
benchmark-run ledger at
`src/tab_foundry/bench/benchmark_run_registry_v1.json`:

```bash
uv run python scripts/register_benchmark_run.py \
  --run-id 01_nano_exact \
  --track binary_ladder \
  --run-dir outputs/staged_ladder/01_nano_exact/train \
  --comparison-summary outputs/staged_ladder/01_nano_exact/benchmark/comparison_summary.json \
  --experiment cls_benchmark_staged \
  --config-profile cls_benchmark_staged \
  --decision keep \
  --conclusion "Exact staged repro matches the frozen anchor contract."
```

This registry is the canonical historical record for benchmark-facing runs.
Use `wandb` for live observation and debugging, not as the benchmark system of
record.

#### Benchmark Cost Policy

Use three benchmark tiers:

- Tier 0: for every architecture change, run tests plus one short local
  training run on a fixed manifest and inspect `train_history.jsonl`,
  validation loss, gradient norms, and checkpoint integrity. Do not pay the
  full `nanoTabPFN` comparison cost here.
- Tier 1: for shortlisted candidates, run the pinned benchmark bundle and
  judge the result against the parent run and the frozen anchor first, not
  against `nanoTabPFN`.
- Tier 2: pay the full fresh `nanoTabPFN` helper cost only for milestone
  results you want to treat as canonical, such as prior-trained
  `model.stage=nano_exact`, the final selected compact base, or any run that
  changes the benchmark bundle, prior dump, helper settings, checkout, or
  device class.

Claims about `nanoTabPFN` closeness should come from the prior-trained
`nano_exact` path, not from the short-run staged benchmark configs. A short-run
`nano_exact` benchmark only establishes parity with the frozen
`tabfoundry_simple` anchor under the same short-run training surface.

If the benchmark bundle path, `nanoTabPFN` prior dump, helper settings,
checkout, and device class are unchanged, reusing the current
`nanotabpfn_curve.jsonl` is acceptable for Tier 1 comparisons. Refresh it for
Tier 2 milestones or whenever any of those inputs changes.

### System-Delta Sweep Runbook

Treat `program.md` as the canonical staged-research contract. The live staged
workflow is an anchor-only system-delta sweep, not a promotion ladder.

Canonical sources of truth:

- `reference/system_delta_catalog.yaml`
- `reference/system_delta_sweeps/index.yaml`
- `reference/system_delta_sweeps/<sweep_id>/queue.yaml`
- `reference/system_delta_sweeps/<sweep_id>/matrix.md`
- `reference/system_delta_queue.yaml` and `reference/system_delta_matrix.md`
  as generated active-sweep aliases

Recommended loop:

1. Inspect the active sweep and the next runnable row:

   ```bash
   uv run python scripts/system_delta_queue.py list
   uv run python scripts/system_delta_queue.py next
   ```

1. Execute the active sweep's `ready` rows with the generic executor:

   ```bash
   uv run python scripts/system_delta_execute.py
   ```

1. Re-run explicit rows, including already completed rows, when you need a fresh
   canonical CUDA pass:

   ```bash
   uv run python scripts/system_delta_execute.py \
     --sweep-id <sweep_id> \
     --order <order> \
     --include-completed
   ```

1. Promote the first executed row to the sweep anchor during an execution pass
   when you are intentionally re-baselining the sweep:

   ```bash
   uv run python scripts/system_delta_execute.py \
     --sweep-id <sweep_id> \
     --order <order> \
     --include-completed \
     --promote-first-executed-row-to-anchor
   ```

1. Promote a completed run after review when you want to update the sweep's
   canonical anchor without rerunning the queue:

   ```bash
   uv run python scripts/system_delta_promote.py \
     --sweep-id <sweep_id> \
     --order <order>
   ```

1. Validate the rendered sweep metadata after execution or promotion:

   ```bash
   uv run python scripts/system_delta_queue.py render --sweep-id <sweep_id>
   uv run python scripts/system_delta_queue.py validate --sweep-id <sweep_id>
   ```

Manual train, benchmark, and registry commands remain the advanced fallback when
`system_delta_execute.py` is not flexible enough for a one-off debugging pass.
Use the existing staged-prior, benchmark, and benchmark-registration scripts in
that case, then rerender and validate the sweep metadata afterward.

Every completed benchmark-facing row should leave behind:

- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`
- checkpoint snapshots
- `comparison_summary.json`
- `benchmark_run_record.json`
- `comparison_curve.png`
- `training_surface_record.json`
- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/result_card.md`

For queue reruns used to debug instability, `train_history.jsonl` now includes
additive `train_loss_delta`, `train_loss_ema`, `grad_clip_threshold`, and
`grad_clip_triggered` fields. `comparison_summary.json["artifacts"]` also
surfaces additive `gradient_history_jsonl` and `telemetry_json` paths when the
run directory contains those files.

Historical note:

- The legacy 3-task bundle and older staged-ladder run ids remain in the repo
  for historical reproduction and registry interpretation only.
- New staged research should use the active sweep queue/matrix and the current
  canonical binary benchmark surface.

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
