# TF-RD-009: Scaling-Law Measurement On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-measurement-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-018](tf_rd_018_training_surface_adequacy.md), at least one harder
  post-008 ladder, and potentially
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md) if architecture
  separation remains low-signal

## External Evidence

- [Scaling Laws](../papers.md#scaling-laws): Chinchilla, Kaplan, Power Lines,
  and Broken Neural Scaling Laws define the methodology and expected caveats
- [Compact Transformers And Training Recipes](../papers.md#compact-transformers-and-training-recipes):
  muP and related compact-training references only become interpretable once a
  stable recipe exists
- [Training-Surface Adequacy And Batch/LR Scaling](../papers.md#training-surface-adequacy-and-batchlr-scaling):
  TF-RD-018 literature establishes why scaling should start from one settled
  batch and LR starting point rather than reopening recipe search at every size

## Repo-Local Evidence

- the roadmap now requires TF-RD-013, TF-RD-018, and at least one harder
  post-008 benchmark-backed front before scaling becomes architecture evidence
- the promoted parent remains `row_cls + qass + no tfcol`, with TFCol scaling
  still explicitly opt-in
- tuning and benchmark-adjacent tooling already exist, but there is no
  canonical scaling artifact path yet on the promoted row-first anchor

## Current Interpretation

- treat [TF-RD-018](tf_rd_018_training_surface_adequacy.md) as the training
  recipe handoff rather than letting scaling reopen the batch-selection problem
- hold the training surface as steady as possible across size, depth, and width
  reads so scaling comparisons stay attributable
- use scaling as architecture evidence only after the promoted row-first family
  has a harder post-008 surface and a stable-enough recipe to make compute and
  quality tradeoffs interpretable

## Open Evidence Gaps

- there is no canonical compute-accounting, parameter-count, and benchmark
  artifact contract yet for scaling runs on the promoted anchor
- the repo has not yet established how much hyperparameter transfer across size
  should be assumed before rerunning adequacy work
- the harder-surface prerequisite is still unresolved, so current simple-binary
  scaling would remain low-signal

## Exit Signals

- scaling curves are fit on the promoted row-first architecture under a harder
  or broader post-008 surface than the current simple binary regime
- the scaling artifacts reuse one explicit TF-RD-018-derived training recipe
  strongly enough that cross-size conclusions are interpretable
