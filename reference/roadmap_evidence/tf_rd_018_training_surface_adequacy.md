# TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-018](../../docs/development/roadmap.md#tf-rd-018-training-surface-adequacy-on-the-promoted-anchor).

- Status: `research`
- Milestone: `Next`
- Dependency position: follows TF-RD-013, now carries the settled batch-ladder
  recipe into [TF-RD-020](tf_rd_020_harder_dagzoo_corpus_fronts.md), which now
  records three synthetic harder-front family winners before the remaining
  optimizer or LR or clipping continuation, and then sets the default training
  surface for
  [TF-RD-014](tf_rd_014_missingness_robustness.md),
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md), and the scaling handoff
  into [TF-RD-009](tf_rd_009_scaling_law_measurement.md)

## External Evidence

- [Training-Surface Adequacy And Batch/LR Scaling](../papers.md#training-surface-adequacy-and-batchlr-scaling):
  the current TF-RD-018 bibliography for batch-size saturation, LR coupling,
  conservative optimizer baselines, and large-batch failure modes
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
  canonical pre-filter harder-front ladder under
  [tf_rd_020_harder_dagzoo_ladder_v1](../system_delta_sweeps/tf_rd_020_harder_dagzoo_ladder_v1/matrix.md)
- `tf_rd_020_shift_noise_drift_v1` is now the default TF-RD-018 harder
  carry-forward surface because it leads the kept TF-RD-020 winners on final
  log loss and final Brier while preserving a positive final ROC delta and the
  shortest runtime among the kept set
- `tf_rd_020_noise_mixture_v1` remains the named fallback harder surface if the
  first optimizer-family read on `tf_rd_020_shift_noise_drift_v1` is too close
  or unstable to collapse cleanly
- [#137](https://github.com/bensonlee5/tab-foundry/issues/137) now executes as
  active sweep
  [tf_rd_018_optimizer_family_v1](../system_delta_sweeps/tf_rd_018_optimizer_family_v1/matrix.md)
  with direct `adamw` and `muon` comparisons against the inherited TF-RD-020
  noise-drift winner
- the TF-RD-020 noise-drift winner is now both the data-surface handoff and
  the locked optimizer anchor for issue `#137`, carrying forward the uncapped
  `task_batch_size=1`, `grad_accum_steps=4`, `max_steps=400` runtime

## Current Interpretation

- `task_batch_size=4` is the current default training-surface rung on the
  settled medium surface because it satisfied the runtime gate that stopped the
  first ladder
- the inherited harder-surface optimizer anchor now comes from TF-RD-020 row
  `06`, so issue `#137` no longer replays schedulefree on top of
  `tf_rd_020_shift_noise_drift_v1`
- `task_batch_size=8` is now negative gate evidence rather than the new default:
  it preserved clean batching, reused the row-1 nanoTabPFN curve, but still
  missed the `<=900s` gate and regressed final benchmark-facing metrics
- treat the completed batch ladder as the settled first adequacy spine rather
  than as the next open TF-RD-018 question
- carry that settled batch rung onto TF-RD-020 before reopening optimizer or
  schedule-family follow-up
- use the recorded `tf_rd_020_harder_dagzoo_ladder_v1` ladder as the fixed
  pre-filter handoff for issues `#148`, `#149`, and `#150` rather than
  reopening harder-front design inside TF-RD-018
- use `tf_rd_020_shift_noise_drift_v1` as the default harder carry-forward
  surface for issues `#137`, `#138`, and `#139`
- use `tf_rd_018_optimizer_family_v1` as the active execution sweep for issue
  `#137`: compare `adamw` and `muon` directly against the locked TF-RD-020 row
  `06` noise-drift anchor
- retain `tf_rd_020_noise_mixture_v1` as the named fallback harder surface only
  if the first optimizer-family read on noise drift is too confounded to
  collapse to a single carry-forward front
- after the full uncapped harder dagzoo blocker closed, retune LR and schedule
  on the settled rung rather than jointly searching batch and LR across the
  whole ladder
- issues `#137`, `#138`, and `#139` should now rebase onto the inherited
  TF-RD-020 noise-drift runtime (`task_batch_size=1`, `grad_accum_steps=4`,
  `max_steps=400`) plus `tf_rd_020_shift_noise_drift_v1` instead of reopening
  singleton updates or leaving the harder surface implicit
- compare strong Adam-family baselines before treating `muon` or other
  specialized optimizers as necessary
- keep architecture changes out of TF-RD-018; they belong later under
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md)

## Open Evidence Gaps

- optimizer-family, LR-shape, clipping, and step-budget evidence are still
  open, but they should now be read on top of the inherited TF-RD-020
  noise-drift runtime and the documented `tf_rd_020_shift_noise_drift_v1`
  carry-forward surface
- the repo still needs an explicit handoff rule for how much of the TF-RD-018
  recipe should stay fixed when TF-RD-020 closes and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) starts
- the repo still needs an explicit stop rule for when TF-RD-020 should fall
  back from `tf_rd_020_shift_noise_drift_v1` to `tf_rd_020_noise_mixture_v1`
  during the first optimizer-family read
- the current medium-surface record still lacks evidence that larger manifest
  task batches are worth reopening without separate runtime work

## Exit Signals

- one explicit default training surface exists for the promoted row-first
  anchor, starting from the completed dataset-batch ladder on the TF-RD-013
  medium surface and the carried harder dagzoo winners from TF-RD-020
- the repo has a clear rule for when optimizer or schedule adequacy must be
  resolved before interpreting harder-surface or scaling outcomes
