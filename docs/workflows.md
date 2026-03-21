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

- Python `3.14` is pinned in `.python-version`.
- Use the repo-local `uv` workflow for sync and command execution.

Setup:

```bash
./scripts/dev bootstrap
```

`./scripts/dev bootstrap` wraps the canonical repo-local setup:

```bash
uv sync
pre-commit install
```

After setup, activate the virtual environment so all commands below can be run
directly:

```bash
source .venv/bin/activate
```

Repo-local `uv sync` includes the benchmark helper dependencies plus Muon
through the dev environment. For a minimal non-dev install, opt into those
optional surfaces explicitly:

```bash
uv sync --no-dev --extra benchmark --extra muon
```

Review the current diff against `origin/main` and run the smallest safe
verification slice:

```bash
./scripts/dev review-base
./scripts/dev verify affected
./scripts/dev verify paths src/tab_foundry/model/factory.py
```

Run the full local quality gate:

```bash
./scripts/dev verify full
```

Fast developer-facing inspection commands:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
tab-foundry dev health-check --run-dir outputs/cls_smoke
tab-foundry dev run-inspect --run-dir outputs/cls_smoke
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
tab-foundry research sweep inspect --order 6 --sweep-id binary_md_v1
tab-foundry research sweep diff --order 7 --against-order 6 --sweep-id binary_md_v1
```

Format markdown directly:

```bash
./.venv/bin/mdformat AGENTS.md README.md CHANGELOG.md program.md docs reference
```

CI workflow `test` runs two required jobs on pull requests and `main`:

- `quality-and-unit`: `./scripts/dev verify full`
- `iris-smoke`: `./scripts/dev smoke iris --out-root benchmarks/results/ci_iris_smoke`
  plus uploaded artifacts and markdown summary

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
tab-foundry data build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

By default this includes raw `dagzoo` outputs and warns if the manifest contains datasets with `filter.status=not_run`, `rejected`, or missing filter metadata.

Accepted-only flow:

```bash
dagzoo filter --in data/run1 --out data/run1_filter --curated-out data/run1_curated
tab-foundry data build-manifest \
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
tab-foundry train run experiment=cls_smoke
tab-foundry train run experiment=cls_workstation
tab-foundry train run \
  experiment=cls_benchmark_linear \
  data.manifest_path=data/manifests/default.parquet
tab-foundry train run \
  experiment=cls_benchmark_linear_simple \
  data.manifest_path=<binary_manifest.parquet>
tab-foundry train run \
  experiment=cls_benchmark_staged \
  data.manifest_path=<binary_manifest.parquet>
```

`cls_benchmark_linear_simple` now targets the exact nanoTabPFN-style binary
debug model. It is benchmark-focused rather than a general small-class
classifier, so it requires binary tasks, internal train-split z-score clipping,
and `many_class_base=2`.

`cls_benchmark_staged` is the staged research counterpart. It defaults to
`model.arch=tabfoundry_staged` and `model.stage=nano_exact`. Treat that as the
base recipe only. The intended architecture path is the public staged ladder in
`docs/development/roadmap.md` and
`docs/development/model-architecture.md`. The system-delta sweep records
isolated evidence for that ladder using `model.stage_label` and
`model.module_overrides`, including bounded control rows when a public stage
bundles more than one mechanism. It now enables activation tracing by default
so architecture-screen runs emit `gradient_history.jsonl` and `telemetry.json`
without extra runtime overrides.

Regression is intentionally removed in the current repo state. Future
regression work will be rebuilt on top of `tabfoundry_staged`, not the retired
legacy `tabfoundry` family.

Prior-dump training for the staged family uses the same harness with the staged
experiment default:

```bash
tab-foundry train prior staged
```

`~/dev/nanoTabPFN/300k_150x5_2.h5` is an input to `tab-foundry train prior ...`
and `tab-foundry bench compare`. Plain `tab-foundry train run ...` commands
still consume a packed parquet manifest instead.

Use the queue row plus `reference/system_delta_campaign_template.md` to decide
the staged labels, any bounded module overrides, and the research-package paths
for each benchmark-facing delta.

Repo-local `uv sync` includes Muon. If you are using a minimal install without
the `muon` extra, run without Muon explicitly:

```bash
tab-foundry train run experiment=cls_smoke optimizer=adamw
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

For model-surface work, prefer the fast dev commands before launching a full
smoke or training loop:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
```

`resolve-config` prints the resolved model/data/preprocessing/training surface,
including staged module selection and parameter counts. `forward-check` builds
the resolved model and runs one deterministic synthetic forward pass without
starting training. `diff-config` compares two fully resolved surfaces and only
prints effective deltas. `export-check` wraps bundle export, bundle validation,
and one deterministic reference-consumer smoke. `manifest-inspect` summarizes a
manifest parquet and can preflight one resolved experiment against its task,
missing-value, and class-count contract.

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
./scripts/dev smoke iris
```

This generates manifest-compatible packed Iris tasks, trains a small classification checkpoint on CPU by default, evaluates it on the manifest test split, then runs the binary-Iris checkpoint benchmark.

Artifacts are written under a timestamped `/tmp/tab_foundry_iris_smoke_*` directory.

### dagzoo Smoke

Run a repo-local end-to-end smoke harness against a sibling `dagzoo` checkout:

```bash
tab-foundry bench smoke dagzoo
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
tab-foundry bench tune \
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
tab-foundry bench env bootstrap
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
tab-foundry bench compare \
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

The missing-permitting large binary bundle at
`src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json` is now the
missing-data generalization surface for the open `TF-RD-008` promotion gate. It
is not reserved exclusively for `missingness_followup`; that older sweep remains
useful hybrid-diagnostic evidence on the prenorm foundation, but it is not the
closure path for the row-first anchor decision. The opt-in no-missing larger
binary surface at
`src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json` is the
completed validator that narrowed the row-first choice to `row_cls + qass + no tfcol` versus `row_cls + qass + tfcol_heads4`, while
`nanotabpfn_openml_binary_large_v1.json` remains the pending missing-data gate
before the repo locks either a promoted default or an explicit split
recommendation.

Regenerate the canonical medium binary bundle with:

```bash
tab-foundry bench bundle build-openml \
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
tab-foundry bench bundle build-openml \
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

Discover a fresh no-missing large binary candidate surface with:

```bash
tab-foundry bench bundle build-openml \
  --out-path /tmp/nanotabpfn_openml_binary_large_no_missing_candidate.json \
  --bundle-name nanotabpfn_openml_binary_large_no_missing_candidate \
  --version 1 \
  --discover-from-openml \
  --task-type supervised_classification \
  --new-instances 200 \
  --min-instances 200 \
  --min-task-count 50 \
  --max-features 50 \
  --max-classes 2 \
  --max-missing-pct 0.0 \
  --min-minority-class-pct 2.5
```

Rebuild the checked-in reviewed no-missing large bundle exactly with:

```bash
tab-foundry bench bundle build-openml \
  --out-path src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json \
  --bundle-name nanotabpfn_openml_binary_large_no_missing \
  --version 1 \
  --task-source binary_large_no_missing_v1 \
  --task-type supervised_classification \
  --new-instances 200 \
  --max-features 50 \
  --max-classes 2 \
  --max-missing-pct 0.0 \
  --min-minority-class-pct 2.5
```

Benchmark against a non-default repo-tracked bundle with:

```bash
tab-foundry bench compare \
  --tab-foundry-run-dir <run_dir> \
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

The benchmark-profile training config now writes `train_history.jsonl` directly
under `runtime.output_dir`, so benchmark comparison can consume plain
`tab-foundry train run experiment=cls_benchmark_linear ...` outputs without a
smoke harness.

One simple preparation path is:

```bash
tab-foundry bench smoke dagzoo --out-root /tmp/tab_foundry_dagzoo_smoke_bench
tab-foundry bench compare \
  --tab-foundry-run-dir /tmp/tab_foundry_dagzoo_smoke_bench \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

The checked-in `cls_benchmark_linear_v2` control baseline id remains the
canonical medium-bundle comparator. Its repo-tracked registry entry now freezes
the prior-trained staged `nano_exact` anchor run
`01_nano_exact_md_prior_parity_fix_binary_medium_v1`, using:

- `outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`
- `outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1/comparison_summary.json`

That keeps the historical baseline id while aligning the canonical control
surface with the existing prior-trained PFN-facing medium-bundle anchor.
Refresh that control baseline through the freeze flow when you need a new run;
do not hand-edit the registry entry.

To re-freeze the checked-in entry from the current frozen anchor:

```bash
tab-foundry bench registry freeze-baseline \
  --baseline-id cls_benchmark_linear_v2 \
  --experiment cls_benchmark_staged_prior \
  --config-profile cls_benchmark_staged_prior \
  --run-dir outputs/staged_ladder/01_nano_exact_md/prior_parity_fix \
  --comparison-summary outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1/comparison_summary.json
```

If you intentionally refresh the underlying control run, keep the new artifacts
inside the repo and use the prior-trained staged path:

```bash
tab-foundry train prior staged \
  --prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5 \
  runtime.output_dir=outputs/control_baselines/cls_benchmark_linear_v2/train

tab-foundry bench compare \
  --tab-foundry-run-dir outputs/control_baselines/cls_benchmark_linear_v2/train \
  --out-root outputs/control_baselines/cls_benchmark_linear_v2/benchmark \
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5

tab-foundry bench registry freeze-baseline \
  --baseline-id cls_benchmark_linear_v2 \
  --experiment cls_benchmark_staged_prior \
  --config-profile cls_benchmark_staged_prior \
  --run-dir outputs/control_baselines/cls_benchmark_linear_v2/train \
  --comparison-summary outputs/control_baselines/cls_benchmark_linear_v2/benchmark/comparison_summary.json
```

Keep the chosen refresh train directory empty before rerunning the training
command; `tab-foundry train run` and `tab-foundry train prior staged` fail fast
if `runtime.output_dir` already contains a non-empty history file or checkpoint
`.pt` artifacts.

`cls_benchmark_linear_v2` is the canonical control baseline id for the medium
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
tab-foundry bench compare \
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
tab-foundry bench registry register-run \
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

`tab-foundry research sweep execute` now applies that Tier 1 reuse path
automatically for `benchmark_full` rows when the locked anchor or frozen
control baseline exposes a compatible `nanotabpfn_curve.jsonl`. If no
compatible curve is available, the runner falls back to a fresh helper run on
the same canonical medium-bundle surface.

### System-Delta Sweep Runbook

Treat `program.md` as the canonical staged-research contract. The live staged
workflow is an anchor-only system-delta sweep: it measures rows against a
locked anchor and does not by itself define the long-term promotion path. Use
the roadmap and architecture docs for the intended public stage progression.

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
   tab-foundry research sweep list
   tab-foundry research sweep next
   tab-foundry research sweep summarize --include-screened
   ```

1. Render architecture graphs for the anchor or selected rows when you need a
   structural view of the current sweep surface:

   ```bash
   uv sync
   brew install graphviz
   tab-foundry research sweep graph --anchor
   tab-foundry research sweep graph --sweep-id <sweep_id> --order <order>
   ```

   The graph command writes SVGs and an `index.md` summary under
   `outputs/staged_ladder/research/<sweep_id>/architecture_graphs` by default.
   Run `tab-foundry research sweep graph --anchor` for the current canonical
   surface. The command requires the Graphviz `dot` binary on `PATH`.

1. Summarize completed and screened rows into one compact local table when you
   need to answer "how is this sweep going?" without reopening queue YAML by
   hand:

   ```bash
   tab-foundry research sweep summarize --sweep-id <sweep_id> --include-screened
   ```

1. Execute the active sweep's `ready` rows with the generic executor:

   ```bash
   tab-foundry research sweep execute
   ```

1. Re-run explicit rows, including already completed rows, when you need a fresh
   canonical CUDA pass:

   ```bash
   tab-foundry research sweep execute \
     --sweep-id <sweep_id> \
     --order <order> \
     --include-completed
   ```

   `--include-completed` also applies to previously `screened` train-only rows.

1. Promote the first executed row to the sweep anchor during an execution pass
   when you are intentionally re-baselining the sweep:

   ```bash
   tab-foundry research sweep execute \
     --sweep-id <sweep_id> \
     --order <order> \
     --include-completed \
     --promote-first-executed-row-to-anchor
   ```

1. Promote a completed run after review when you want to update the sweep's
   canonical anchor without rerunning the queue:

   ```bash
   tab-foundry research sweep promote \
     --sweep-id <sweep_id> \
     --order <order>
   ```

1. Validate the rendered sweep metadata after execution or promotion:

   ```bash
   tab-foundry research sweep render --sweep-id <sweep_id>
   tab-foundry research sweep validate --sweep-id <sweep_id>
   ```

Manual train, benchmark, and registry commands remain the advanced fallback when
`tab-foundry research sweep execute` is not flexible enough for a one-off
debugging pass. Use the queue-selected training experiment, benchmark, and
benchmark-registration commands in that case, then rerender and validate the
sweep metadata afterward.

Current lane contract:

- PFN control lane: `tabfoundry_simple` plus `tabfoundry_staged` with `stage=nano_exact`
- Hybrid diagnostic lane: `cls_benchmark_staged_prior`
- Canonical architecture-screen surface for future benchmark-facing sweeps: `cls_benchmark_staged`

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

Train-only `screen_only` rows still need:

- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`
- `training_surface_record.json`

They intentionally skip benchmark registration and do not write
`result_card.md`.

`screen_only` rows are diagnostic only. Benchmark-facing conclusions must come
from `benchmark_full` rows and should cite the locked bundle path,
`cls_benchmark_linear_v2`, `training_surface_record.json`, `research_card.md`,
`campaign.yaml`, and `result_card.md`.

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
