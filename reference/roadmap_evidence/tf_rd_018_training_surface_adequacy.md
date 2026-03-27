# TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-018](../../docs/development/roadmap.md#tf-rd-018-training-surface-adequacy-on-the-promoted-anchor).

- Status: `closed incomplete`
- Milestone: `Historical`
- Dependency position: follows TF-RD-013 and records partial staged-control
  training-surface evidence that now remains historical input rather than an
  active blocker for sandwich dagzoo, many-class, steering, runtime, or
  scaling work

## External Evidence

- [Training-Surface Adequacy And Batch/LR Scaling](../papers.md#training-surface-adequacy-and-batchlr-scaling):
  the current TF-RD-018 bibliography for batch-size saturation, LR coupling,
  conservative optimizer baselines, and large-batch failure modes, now kept as
  historical context rather than the next execution lane
- [Compact Transformers And Training Recipes](../papers.md#compact-transformers-and-training-recipes):
  shared schedule and optimizer context that remains relevant once the batch
  rung is fixed

## Repo-Local Evidence

- TF-RD-013 settled `tf_rd_013_dagzoo_shape_aware_size_medium_v1` as the
  representative post-008 training-data surface on 2026-03-23
- [#107](https://github.com/bensonlee5/tab-foundry/issues/107) is the tracking
  issue for the adequacy epic, and
  [#109](https://github.com/bensonlee5/tab-foundry/issues/109) is the completed
  first execution issue
- `row_first_training_adequacy_v1` completed the first manifest-backed
  `task_batch_size` ladder on the medium surface
- the current medium-surface singleton runtime is about `227s`, so the roadmap
  gates for `4/8/16/32` remain iterative rather than overnight by default
- [#109](https://github.com/bensonlee5/tab-foundry/issues/109) executed the first
  ladder on 2026-03-24: `task_batch_size=4` finished in `699.4s` with `0.0%`
  singleton fallback, while `task_batch_size=8` finished in `1109.3s` with
  `0.0%` singleton fallback and reused the saved nanoTabPFN curve from row 1
- `task_batch_size=4` is now the preferred TF-RD-018 batch rung on the settled
  medium surface, and `task_batch_size=16` plus `32` remain blocked by the row-2
  runtime miss
- [#146](https://github.com/bensonlee5/tab-foundry/issues/146) carried the
  harder dagzoo synthetic front to closure, and TF-RD-020 now records kept
  winners for missingness, shift or drift, and mechanism or noise before
  TF-RD-018 resumes optimizer-family, LR-shape, clipping, or step-budget
  follow-up
- [#147](https://github.com/bensonlee5/tab-foundry/issues/147) now records the
  canonical harder-front ladder under
  [tf_rd_020_harder_dagzoo_ladder_v1](../system_delta_sweeps/tf_rd_020_harder_dagzoo_ladder_v1/matrix.md)
- `tf_rd_020_shift_noise_drift_v1` is now the default TF-RD-018 harder
  carry-forward surface because it leads the kept TF-RD-020 winners on final
  log loss and final Brier while preserving a positive final ROC delta and the
  shortest runtime among the kept set
- [#137](https://github.com/bensonlee5/tab-foundry/issues/137) is now closed on
  completed sweep
  [tf_rd_018_optimizer_family_v1](../system_delta_sweeps/tf_rd_018_optimizer_family_v1/matrix.md),
  which kept `schedulefree_adamw` as the primary optimizer family and left
  both `adamw` and `muon` deferred on the inherited TF-RD-020 noise-drift
  winner
- `tf_rd_020_noise_mixture_v1` remains documented fallback context, but the
  completed optimizer-family read on `tf_rd_020_shift_noise_drift_v1` was not
  close or unstable enough to activate that surface
- the TF-RD-020 noise-drift winner is now both the data-surface handoff and
  the locked optimizer anchor for issue `#137`, carrying forward the uncapped
  `task_batch_size=1`, `grad_accum_steps=4`, `max_steps=400` runtime

## Current Interpretation

- keep the completed batch ladder and optimizer-family read as historical
  staged-control evidence rather than as the next open TF-RD-018 question
- retain `task_batch_size=4` on the medium surface and
  `schedulefree_adamw` on the inherited TF-RD-020 noise-drift control as the
  explicit partial closeout record for this lane
- do not continue TF-RD-018 into LR, clipping, or step-budget follow-up while
  the roadmap is prioritizing the sandwich family
- keep architecture, steering, runtime, and scaling work on the sandwich path
  under [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md),
  [TF-RD-010](tf_rd_010_many_class_promotion.md),
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md), and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) instead of reopening
  TF-RD-018

## Open Evidence Gaps

- no further TF-RD-018 evidence is blocking the active roadmap path
- if later optimizer or runtime tuning becomes necessary, it should be reopened
  on the carried sandwich classification family rather than on this staged
  control note

## Exit Signals

- one explicit partial training-surface record exists for the staged-control
  line, starting from the completed dataset-batch ladder on the TF-RD-013
  medium surface and the carried harder dagzoo winners from TF-RD-020
- the repo has a clear closeout rule: unfinished TF-RD-018 LR or clipping work
  is not a blocker for the sandwich-first classification path
