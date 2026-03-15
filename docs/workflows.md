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

The pinned 3-task binary bundle remains the canonical promotion gate through
`model.stage=qass_context`. The staged multiclass branch uses the companion
bundle at `src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json`,
which widens the same TabArena v0.1 notebook source set from binary-only to
small multiclass while preserving the same row-count, feature-count, missingness,
and minority-class filters.

Regenerate a pinned bundle with:

```bash
uv run python scripts/build_openml_benchmark_bundle.py \
  --out-path src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json \
  --bundle-name nanotabpfn_openml_classification_small \
  --version 1 \
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
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json \
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

### Staged Ladder Runbook

Use the staged ladder to separate architecture additions from transformer-size
scaling. The main rule is: change one axis at a time.

- Stage ladder first at a fixed small transformer size.
- Size scaling second on a short list of selected stages.
- LR checks only after a fixed-schedule failure is observed, benchmarked, and
  registered.

Use these size profiles:

- `xs`: `d_icl=32`, `tficl_n_heads=4`, `tficl_n_layers=2`, `head_hidden_dim=64`
- `sm`: `d_icl=64`, `tficl_n_heads=4`, `tficl_n_layers=2`, `head_hidden_dim=128`
- `md`: `d_icl=96`, `tficl_n_heads=4`, `tficl_n_layers=3`, `head_hidden_dim=192`

Keep this optimization surface fixed for the initial ladder:

- `optimizer=adamw`
- `schedule.stages[0].lr_max=8.0e-4`
- `schedule.stages[0].warmup_ratio=0.05`
- `runtime.max_steps=400`
- `runtime.grad_clip=1.0`
- `runtime.eval_every=25`
- `runtime.checkpoint_every=25`
- `input_normalization=train_zscore_clip`

The staged architecture milestone order is:

- `00_simple_anchor`
- `01_nano_exact`
- `02_label_token`
- `03_shared_norm`
- `04_prenorm_block`
- `05_small_class_head`
- `06_test_self`
- `07_grouped_tokens`
- `08_row_cls_pool`
- `09_column_set`
- `10_qass_context`

Treat `06_test_self` as the explicit “old approximate simple rebuilt”
milestone. `07_grouped_tokens` onward is the later fuller-feature path.

Use these benchmark-run registry tracks:

- `binary_arch_xs`
- `binary_full_xs`
- `size_scaling_simple_rebuild`
- `multiclass_branch`

Pass A is the architecture-only ladder at `xs` using one fixed binary manifest
and the default binary benchmark bundle.

- Run and register:
  - `00_simple_anchor_xs`
  - `01_nano_exact_xs`
  - `02_label_token_xs`
  - `03_shared_norm_xs`
  - `04_prenorm_block_xs`
  - `05_small_class_head_xs`
  - `06_test_self_xs`
- Optional later extension:
  - `07_grouped_tokens_xs`
  - `08_row_cls_pool_xs`
  - `09_column_set_xs`
  - `10_qass_context_xs`
- `00_simple_anchor_xs` is the anchor for every `xs` staged run.
- `parent_run_id` is always the immediately previous stage.
- `anchor_run_id` is always the matching anchor for that size.
- `01_nano_exact_xs` still needs the exact-parity unit tests, but prior-dump
  parity is only required at `md`.

Pass B is exactness signoff at `md`.

- Run only:
  - `00_simple_anchor_md`
  - `01_nano_exact_md`
- Then run the prior-dump parity harness for `01_nano_exact_md`.
- Do not claim staged exact parity from the `xs` pass alone.

Pass C is size scaling on the selected stages:

- `00_simple_anchor`
- `01_nano_exact`
- `04_prenorm_block`
- `06_test_self`
- Optional later extension: `10_qass_context`

Use these run ids:

- `00_simple_anchor_xs`, `00_simple_anchor_sm`, `00_simple_anchor_md`
- `01_nano_exact_xs`, `01_nano_exact_sm`, `01_nano_exact_md`
- `04_prenorm_block_xs`, `04_prenorm_block_sm`, `04_prenorm_block_md`
- `06_test_self_xs`, `06_test_self_sm`, `06_test_self_md`

This matrix answers whether exact repro, the first major architecture delta,
and the rebuilt-simple milestone all scale cleanly before broader expansion.

Do not tune LR during Pass A. Only open an LR sanity branch if a larger-size
run shows an undertraining pattern:

- lower `best_roc_auc` and `final_roc_auc`
- higher `best_val_loss` and `final_val_loss`
- low or falling gradient norms
- no instability spike in `post_warmup_train_loss_var`

If that happens, run one small LR sweep on the failing `01_nano_exact_<size>`
run with:

- `4.0e-4`
- `8.0e-4`
- `1.6e-3`

If one LR clearly fixes the size issue, freeze that LR for the rest of the size
band and rerun only the selected scaling stages.

Pass D is the multiclass branch after the binary rebuild ladder is stable.

- `11_qass_context_multiclass_xs`
- `12_many_class_multiclass_xs`

Use:

- a multiclass manifest with `n_classes > many_class_base`
- `many_class_base=10`
- `src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json`

Train template:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_staged \
  data.manifest_path=<manifest.parquet> \
  model.stage=<stage> \
  model.d_icl=<d_icl> \
  model.tficl_n_heads=4 \
  model.tficl_n_layers=<n_layers> \
  model.head_hidden_dim=<head_hidden_dim> \
  runtime.output_dir=outputs/staged_ladder/<run_id>/train \
  logging.run_name=staged-<run_id>
```

Frozen anchor template:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_linear_simple \
  data.manifest_path=<binary_manifest.parquet> \
  model.d_icl=<d_icl> \
  model.tficl_n_heads=4 \
  model.tficl_n_layers=<n_layers> \
  model.head_hidden_dim=<head_hidden_dim> \
  runtime.output_dir=outputs/staged_ladder/<run_id>/train \
  logging.run_name=staged-<run_id>
```

Benchmark template:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir outputs/staged_ladder/<run_id>/train \
  --out-root outputs/staged_ladder/<run_id>/benchmark \
  --control-baseline-id cls_benchmark_linear_v1 \
  --control-baseline-registry src/tab_foundry/bench/control_baselines_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

Multiclass benchmark template:

```bash
uv run python scripts/benchmark_nanotabpfn.py \
  --tab-foundry-run-dir outputs/staged_ladder/<run_id>/train \
  --out-root outputs/staged_ladder/<run_id>/benchmark \
  --benchmark-bundle-path src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json \
  --nanotab-prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5
```

Register template:

```bash
uv run python scripts/register_benchmark_run.py \
  --run-id <run_id> \
  --track <track> \
  --run-dir outputs/staged_ladder/<run_id>/train \
  --comparison-summary outputs/staged_ladder/<run_id>/benchmark/comparison_summary.json \
  --experiment <experiment> \
  --config-profile <config_profile> \
  --parent-run-id <previous_run_id> \
  --anchor-run-id <anchor_run_id> \
  --decision <keep|reject|defer> \
  --conclusion "<one-line conclusion>"
```

Prior-dump signoff:

```bash
uv run python scripts/train_tabfoundry_staged_prior.py \
  model.stage=nano_exact \
  model.d_icl=96 \
  model.tficl_n_heads=4 \
  model.tficl_n_layers=3 \
  model.head_hidden_dim=192 \
  runtime.output_dir=outputs/staged_ladder/01_nano_exact_md/prior
```

Keep this preflight in the runbook:

```bash
./.venv/bin/python -m pytest \
  tests/model/test_tabfoundry_staged.py \
  tests/benchmark/test_prior_train.py \
  tests/benchmark/test_checkpoint_classifier.py \
  tests/benchmark/test_nanotabpfn_compare.py -q
```

Every benchmark-facing run should produce:

- `train_history.jsonl`
- checkpoint snapshots
- `comparison_summary.json`
- `benchmark_run_record.json`
- `comparison_curve.png`

Review after every run:

- `comparison_summary.json`
- `benchmark_run_record.json`
- `src/tab_foundry/bench/benchmark_run_registry_v1.json`
- a one-line `keep` / `reject` / `defer` conclusion

If no local binary dagzoo manifest is available yet, a fixed binary Iris-derived
manifest is acceptable for the initial `xs` dry run, but keep that manifest
constant within the pass and rerun the kept stages later on the real binary
manifest before drawing stronger conclusions.

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
