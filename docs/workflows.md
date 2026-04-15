# Workflows

## Overview

Use this runbook when you need command syntax, artifact expectations, or the
smallest safe verification slice.

Use these alongside this runbook:

- [README.md](../README.md) for repo overview and quickstart
- [CONTRIBUTING.md](../CONTRIBUTING.md) for review expectations
- [Codebase Navigation](development/codebase-navigation.md) for ownership
- [Dataset Curation](development/dataset-curation.md) for license policy
- [Inference Contract](inference.md) for export/runtime details
- [program.md](../program.md) for sweep policy

## Environment And Verification

- Python `3.14` is pinned in `.python-version`.
- Use `./scripts/dev` for bootstrap, doctor, verification, and Iris smoke.
- Use the packaged CLI for everything else.

Bootstrap the repo-local environment:

```bash
./scripts/dev bootstrap
source .venv/bin/activate
./scripts/dev doctor
```

Use the packaged CLI discovery order for live commands and flags:

1. `.venv/bin/tab-foundry --help`
1. `.venv/bin/tab-foundry <group> --help`
1. `.venv/bin/tab-foundry <group> <command> --help`

Review the current diff and run the smallest safe verification slice:

```bash
./scripts/dev ready --base-ref origin/main
./scripts/dev review-base
./scripts/dev verify affected
./scripts/dev verify paths src/tab_foundry/model/factory.py
```

Run the full local quality gate when you need it:

```bash
./scripts/dev verify full
```

Fast inspection surfaces before broad greps or full runs:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
tab-foundry dev run-inspect --run-dir outputs/cls_smoke
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt --device cuda
tab-foundry dev checkpoint-publish --run-id sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1
tab-foundry dev checkpoint-resolve --run-id sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1 --allow-remote
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
```

`tab-foundry dev run-inspect` now reports `runtime_summary`,
`benchmark_timing`, and `inference_timing` when those payloads exist on the
selected run. `tab-foundry dev export-check` now accepts `--device` and emits a
reference-consumer latency sample for the selected hardware surface.

Docs and reference changes are covered by the audit checks in `./scripts/dev verify affected`
and `./scripts/dev verify paths`.

## Dataset Curation Gate

Real-data additions require a license review before they enter a curated OpenML
bundle, a manifest-backed external dataset set, or a benchmark ladder.

- record approvals in `reference/dataset_license_reviews.csv`
- follow [Dataset Curation](development/dataset-curation.md)
- treat `dagzoo` as synthetic-only rather than as an external real-data source
- do not add new loader paths for external real-data ingestion when an existing
  manifest-backed surface already covers the workflow

## Corpus Materialization

For recurring synthetic corpora, use the first-class corpus recipe workflow:

```bash
tab-foundry data corpus list-recipes
tab-foundry data corpus materialize \
  --recipe tf_rd_013_current_corpus_default_v1 \
  --dagzoo-root ../dagzoo \
  --force
tab-foundry data corpus inspect \
  --corpus-ref tf_rd_013_current_corpus_default_v1
```

This writes local corpus artifacts under
`outputs/corpora/<recipe_id>/<corpus_id>/`, including a manifest and
`corpus_record.json`.

For sweep-local corpus recipes, use the sweep-aware surfaces:

```bash
tab-foundry data corpus list-recipes \
  --sweep-id tf_rd_020_harder_dagzoo_ladder_v1
tab-foundry research sweep materialize-corpora \
  --sweep-id tf_rd_020_harder_dagzoo_ladder_v1 \
  --dagzoo-root ../dagzoo
tab-foundry data corpus materialize \
  --recipe tf_rd_020_local_probe_v1 \
  --sweep-id tf_rd_020_harder_dagzoo_ladder_v1 \
  --dagzoo-root ../dagzoo
```

For one-off manifests, use the lower-level dev surface instead of creating a
new recurring corpus path:

```bash
tab-foundry dev data build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:?set DAGZOO_DATA_ROOT to your DagZoo data root}" \
  --out-manifest data/manifests/default.parquet
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
  experiment=cls_workstation_sandwich \
  data.corpus_ref=tf_rd_013_current_corpus_default_v1
```

Short profiler runs write TensorBoard traces and a `key_averages.txt` summary.
Run them through the checked-out module entrypoint so the repo-local `profile`
subcommand is used:

```bash
PYTHONPATH=src .venv/bin/python -m tab_foundry train profile \
  experiment=cls_benchmark_sandwich_speedrun_cached_packed_v1 \
  runtime.output_dir=outputs/profile_speedrun_cached_packed
```

`cls_workstation_sandwich` is the default sandwich training surface for new
development work. The carried packed speed lane remains the Muon variant after
the 1000-step, family-block-2 optimizer A/B on the TF-RD-010 v5 medium corpus;
use `cls_benchmark_sandwich_speedrun_cached_packed_v1` as the
`schedulefree_adamw` comparator and
`cls_benchmark_sandwich_speedrun_cached_packed_muon_v1` as the kept packed
benchmark surface. Regression remains intentionally removed in the current repo
state.

For the benchmark-safe TF-RD-009 heads-1 replay lane, use the explicit
comparison configs instead of the cached speedrun surfaces:

```bash
tab-foundry train run \
  experiment=cls_benchmark_sandwich_tf_rd_009_heads1_benchmark_safe_baseline_v1
tab-foundry train run \
  experiment=cls_benchmark_sandwich_tf_rd_009_heads1_benchmark_safe_packed_v1
tab-foundry train run \
  experiment=cls_benchmark_sandwich_tf_rd_009_heads1_benchmark_safe_packed_cuda_graph_v1
```

Those three configs keep the heads-1 + compile-eager-dynamic replay surface
fixed and differ only in packed attention and the optional signature-family
CUDA-graph lane. They also keep `profile_step_timing=true` and
`module_grad_norm_every=25` so the dominant `forward_backward` bucket can be
compared directly from per-step telemetry without changing checkpoint density.

The prior-trained staged control surface is still available:

```bash
tab-foundry train legacy-prior staged \
  --prior-dump <path-to-nanoTabPFN>/300k_150x5_2.h5
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

Prefer inspect-first surfaces before full training or smoke reruns:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt --device cuda
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
```

## Standard Workflow Artifacts

These are the common handoff artifacts for reviewable runs:

- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`
- `summary.md`
- `loss_curve.png`
- checkpoint `.pt` files
- `comparison_summary.json`
- `benchmark_run_record.json`
- `comparison_curve.png`
- `training_surface_record.json`

Smoke and benchmark-style runs may also persist generated datasets, manifests,
and exported bundles.

## Smoke Workflows

Run the repo-local Iris smoke:

```bash
./scripts/dev smoke iris
```

This writes artifacts under a timestamped `/tmp/tab_foundry_iris_smoke_*`
directory.

Run the dagzoo end-to-end smoke against a sibling checkout:

```bash
tab-foundry bench smoke dagzoo --dagzoo-root <path-to-dagzoo>
```

This writes artifacts under a timestamped `/tmp/tab_foundry_dagzoo_smoke_*`
directory. Smoke is for plumbing validation, not the research leaderboard.

## Internal Tuning

Run an internal-only sweep on a fixed manifest:

```bash
tab-foundry bench tune \
  --manifest-path data/manifests/default.parquet
```

The default sweep ranks runs by internal metrics only:

1. lowest `best_val_loss`
1. lowest `final_val_loss`
1. lowest post-warmup train-loss variance

Gradient norm remains a stability diagnostic, not the primary ranking target.

## Confirmatory Benchmarking

Bootstrap sibling benchmark environments:

```bash
tab-foundry bench env bootstrap \
  --nanotabpfn-root <path-to-nanoTabPFN> \
  --tabpfn-root <path-to-TabPFN> \
  --tabicl-root <path-to-tabicl>
```

Run the default comparison flow after a candidate run is already selected:

```bash
tab-foundry bench compare \
  --tab-foundry-run-dir <run_dir> \
  --tabicl-root <path-to-tabicl>
```

Opt into nanoTabPFN explicitly only when you want the legacy comparator or a
secondary no-missing control lane:

```bash
tab-foundry bench compare \
  --tab-foundry-run-dir <run_dir> \
  --external-benchmark tabiclv2 \
  --external-benchmark nanotabpfn \
  --nanotabpfn-root <path-to-nanoTabPFN> \
  --nanotabpfn-prior-dump <path-to-nanoTabPFN>/300k_150x5_2.h5 \
  --tabicl-root <path-to-tabicl>
```

The canonical medium benchmark surface is
`data/manifests/bench/openml_classification_medium_v1/manifest.parquet`. The
canonical frozen control baseline id is `cls_benchmark_linear_v2`.

Benchmark comparison and sweep execution consume manifest paths only. Use the
repo-tracked bundle JSON only as a materialization input:

```bash
tab-foundry bench materialize-openml-bundle \
  --bundle-path src/tab_foundry/bench/openml_binary_medium_v1.json \
  --out-root data/manifests/bench/openml_classification_medium_v1
```

The checked-in `cls_benchmark_linear_v2` entry freezes the prior-trained staged
anchor run at:

- `outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`
- `outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1/comparison_summary.json`

Re-freeze that control baseline from the current anchor when needed:

```bash
tab-foundry bench registry freeze-baseline \
  --baseline-id cls_benchmark_linear_v2 \
  --experiment cls_benchmark_staged_prior \
  --config-profile cls_benchmark_staged_prior \
  --run-dir outputs/staged_ladder/01_nano_exact_md/prior_parity_fix \
  --comparison-summary outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1/comparison_summary.json
```

Register a benchmark-facing run in the historical registry with:

```bash
tab-foundry bench registry register-run \
  --run-id 01_nano_exact \
  --track binary_ladder \
  --run-dir outputs/staged_ladder/01_nano_exact/train \
  --comparison-summary outputs/staged_ladder/01_nano_exact/benchmark/comparison_summary.json \
  --experiment cls_benchmark_staged_corpus \
  --config-profile cls_benchmark_staged_corpus \
  --decision keep \
  --conclusion "Exact staged repro matches the frozen anchor contract."
```

Freeze a preferred architecture for one hardware surface with:

```bash
tab-foundry bench registry freeze-hardware-baseline \
  --baseline-id tf_rd_009_rtx8000_44gb_classification_medium_v1 \
  --preferred-run-id sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1 \
  --formal-anchor-run-id sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2 \
  --baseline-run-id sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1 \
  --evidence-run-id sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2 \
  --evidence-run-id sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1 \
  --surface-role classification_scaling_law \
  --decision keep \
  --rationale "Retain the healthiest matched-budget architecture for the retained RTX 8000 medium classification surface."
```

Use `wandb` for live observation and debugging. Use the benchmark registries
for the repo's historical system of record. Use the hardware architecture
registry to persist the preferred architecture for a specific hardware profile
plus benchmark surface, including the compact constraint model that records the
effective-size, parameter, VRAM, and timing fits used to justify that choice.

### Benchmark Cost Policy

- Tier 0: tests plus one short local training run on a fixed manifest.
- Tier 1: run the pinned benchmark manifest for shortlisted candidates and
  judge against the parent run plus the frozen control.
- Tier 2: pay the full nanoTabPFN helper cost only for milestone results or
  when the manifest provenance, helper settings, prior dump, or device class
  changes.

## System-Delta Sweep Runbook

Treat [program.md](../program.md) as the policy owner. This section only covers
commands and artifact expectations for the selected sweep.

Canonical sweep files:

- `reference/system_delta_catalog.yaml`
- `reference/system_delta_sweeps/index.yaml`
- `reference/system_delta_sweeps/<sweep_id>/queue.yaml`
- `reference/system_delta_sweeps/<sweep_id>/matrix.md`

Inspect the selected sweep before execution:

```bash
tab-foundry research sweep list-sweeps
tab-foundry research sweep list --sweep-id <sweep_id>
tab-foundry research sweep next --sweep-id <sweep_id>
tab-foundry research sweep summarize --sweep-id <sweep_id> --include-screened
tab-foundry research sweep inspect --sweep-id <sweep_id> --order <order>
tab-foundry research sweep diff \
  --sweep-id <sweep_id> \
  --order <order> \
  --against-order <anchor_order>
```

Render architecture graphs when you need a structural view:

```bash
brew install graphviz
tab-foundry research sweep graph --sweep-id <sweep_id> --anchor
tab-foundry research sweep graph --sweep-id <sweep_id> --order <order>
```

The graph command writes outputs under
`outputs/staged_ladder/research/<sweep_id>/architecture_graphs`. It requires
Graphviz `dot` on `PATH`.

Execute, rerun, promote, and validate from the packaged sweep surface:

```bash
tab-foundry research sweep execute --sweep-id <sweep_id>
tab-foundry research sweep execute \
  --sweep-id <sweep_id> \
  --nanotabpfn-root <path-to-nanoTabPFN> \
  --order <order> \
  --include-completed
tab-foundry research sweep promote \
  --sweep-id <sweep_id> \
  --order <order>
tab-foundry research sweep render --sweep-id <sweep_id>
tab-foundry research sweep validate --sweep-id <sweep_id>
```

Pass `--nanotabpfn-root` only when the selected sweep's
`external_benchmarks` include `nanotabpfn`.

Manual train, benchmark, and registry commands remain the advanced fallback
when the generic executor is not flexible enough for a one-off debug pass.

Benchmark-facing rows should leave behind:

- `research_card.md`
- `campaign.yaml`
- `result_card.md`
- `training_surface_record.json`
- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`
- `comparison_summary.json`
- `benchmark_run_record.json`

Train-only `screen_only` rows still need:

- `training_surface_record.json`
- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`

`screen_only` rows intentionally skip benchmark registration and
`result_card.md`. `screen_only` rows are diagnostic only.

Benchmark-facing writeups should cite the locked manifest path,
`cls_benchmark_linear_v2`, `training_surface_record.json`,
`research_card.md`, `campaign.yaml`, and `result_card.md`.

For scaling studies whose benchmark-only runs were executed with
`runtime.val_batches=0`, do not create a fresh validation-training sweep first.
Backfill strict validation losses from completed checkpoints:

```bash
tab-foundry research scaling backfill-validation \
  --study tf_rd_009_phase2 \
  --launch-gcs-root gs://tab-foundry-execution/tab-foundry/launches/<launch-id> \
  --preseed-gcs-root gs://tab-foundry-execution/tab-foundry/launches/<preseed-launch-id>/preseed \
  --val-batches 16 \
  --device cpu
```

This downloads only `latest.pt` plus small train metadata into a bounded local
overlay, writes `validation_backfill_v1.json` sidecars, and lets strict scaling
fits use validation-backed losses without substituting benchmark loss. Use the
emitted `registry_overlay_path` with `research scaling inspect` and `research
scaling fit` so the fitter resolves the local overlay sidecars. Fresh validation
sweeps are the fallback only when the completed checkpoint artifacts or their
checkpoint configs cannot be validated posthoc.

For the April 12, 2026 TF-RD-009 completed Phase-2 state, the repo-tracked
`tf_rd_009_phase2` study carries 24 completed `tf_rd_009_ns_medium_v1`
validation-backed NS-core rows, 20 completed
`tf_rd_009_batch_critical_medium_v1` validation-backed batch-critical rows, and
a compact validation overlay. Inspect and fit that full state with:

```bash
tab-foundry research scaling inspect --study tf_rd_009_phase2
tab-foundry research scaling fit --study tf_rd_009_phase2
tab-foundry research scaling audit --study tf_rd_009_phase2
```

The default full fit reports `L(N)`, `L(D)`, `L(C)`, `L(N,D)`, `L(N,S)`,
`Bcrit(L)`, `L(Cmin)`, and derived `Cmin` relations. `--fit-scope ns-only`
remains available for interim recovery work when batch-critical rows are
pending, but is no longer the canonical TF-RD-009 Phase-2 path. The `L(C)` path
requires observed training-shape summaries and same-model monotone NS compute
accounting, so legacy fallback FLOP estimates do not silently enter the C axis.
Treat the current `Bcrit(L)` output as weak diagnostic evidence: the completed
lower envelope has only two points.

For the corrected one-epoch TF-RD-009 rerun, materialize
`tf_rd_010_dagzoo_medium_control_curated_v6` remotely first rather than
generating it locally. A Vast `research sweep materialize-corpora` job syncs
`outputs/corpora` and `data/manifests` to the launch artifacts and uploads the
corpus cache to GCS; later `research sweep execute` launches restore that cache
before training. Use larger-than-default Vast disks for these launches because
v6 is about 9.33x the v5 corpus size.

Use `research scaling audit` before interpreting the Phase-2 fit as a
compute-optimal law. It writes `audit/audit_summary.json` and `audit/audit.md`
under the study artifact root by default, compares validation-loss and
benchmark-loss targets, runs leave-one-geometry and leave-one-step residual
checks for joint `N,D` / `N,S` fits, bootstraps parameter intervals, adds
diagnostic broken-power-law univariate checks, and gates `Cmin` interpretation
on a redesigned iso-loss `Bcrit(L)` analysis with at least four contour
estimates across the multi-geometry batch sweep.

## Scope Boundaries

- Use smoke for plumbing checks, not the canonical leaderboard.
- Use internal tuning to prune candidates before confirmatory benchmark runs.
- Regenerate obsolete export bundles, benchmark manifests, and prior dumps
  instead of adding compatibility backfills for removed contracts.
