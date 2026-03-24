# TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-018](../../docs/development/roadmap.md#tf-rd-018-training-surface-adequacy-on-the-promoted-anchor).

- Status: `research`
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
- [#109](https://github.com/bensonlee5/tab-foundry/issues/109) executed the first
  ladder on 2026-03-24: `task_batch_size=4` finished in `699.4s` with `0.0%`
  singleton fallback, while `task_batch_size=8` finished in `1109.3s` with
  `0.0%` singleton fallback and reused the saved nanoTabPFN curve from row 1
- `task_batch_size=4` is now the preferred TF-RD-018 batch rung on the settled
  medium surface, and `task_batch_size=16` plus `32` remain blocked by the row-2
  runtime miss

## Current Interpretation

- `task_batch_size=4` is the current default training-surface rung on the
  settled medium surface because it satisfied the runtime gate that stopped the
  first ladder
- `task_batch_size=8` is now negative gate evidence rather than the new default:
  it preserved clean batching, reused the row-1 nanoTabPFN curve, but still
  missed the `<=900s` gate and regressed final benchmark-facing metrics
- after the preferred batch rung is chosen, retune LR and schedule on that rung
  rather than jointly searching batch and LR across the whole ladder
- issues `#137`, `#138`, and `#139` should now rebase onto
  `task_batch_size=4` instead of reopening singleton updates
- compare strong Adam-family baselines before treating `muon` or other
  specialized optimizers as necessary
- keep architecture changes out of TF-RD-018; they belong later under
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md)

## Open Evidence Gaps

- optimizer-family, LR-shape, clipping, and step-budget evidence are still open,
  but they should now be read on top of `task_batch_size=4`
- the repo still needs an explicit handoff rule for how much of the TF-RD-018
  recipe should stay fixed when
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) starts
- the current medium-surface record still lacks evidence that larger manifest
  task batches are worth reopening without separate runtime work

## Exit Signals

- one explicit default training surface exists for the promoted row-first
  anchor, starting from a documented dataset-batch ladder on the TF-RD-013
  medium surface
- the repo has a clear rule for when optimizer or schedule adequacy must be
  resolved before interpreting harder-surface or scaling outcomes
