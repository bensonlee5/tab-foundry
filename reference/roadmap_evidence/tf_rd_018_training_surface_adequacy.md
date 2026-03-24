# TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-018](../../docs/development/roadmap.md#tf-rd-018-training-surface-adequacy-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows TF-RD-013, sets the default training surface for
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
  [#109](https://github.com/bensonlee5/tab-foundry/issues/109) is the first
  execution issue
- `row_first_training_adequacy_v1` now starts with a manifest-backed
  `task_batch_size` ladder on the medium surface
- the current medium-surface singleton runtime is about `227s`, so the roadmap
  gates for `4/8/16/32` remain iterative rather than overnight by default

## Current Interpretation

- search for the largest useful manifest task batch on the settled medium
  surface before reopening optimizer or schedule-family follow-up
- treat `task_batch_size=4` and `8` as unconditional reads; treat `16` and `32`
  as gated by runtime, OOM, and singleton-fallback behavior
- after the preferred batch rung is chosen, retune LR and schedule on that rung
  rather than jointly searching batch and LR across the whole ladder
- compare strong Adam-family baselines before treating `muon` or other
  specialized optimizers as necessary
- keep architecture changes out of TF-RD-018; they belong later under
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md)

## Open Evidence Gaps

- the repo does not yet have a benchmark-backed preferred batch rung on the
  representative medium surface
- optimizer-family, LR-shape, clipping, and step-budget evidence are still
  contingent on the batch decision
- the repo still needs an explicit handoff rule for how much of the TF-RD-018
  recipe should stay fixed when
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) starts

## Exit Signals

- one explicit default training surface exists for the promoted row-first
  anchor, starting from a documented dataset-batch ladder on the TF-RD-013
  medium surface
- the repo has a clear rule for when optimizer or schedule adequacy must be
  resolved before interpreting harder-surface or scaling outcomes
