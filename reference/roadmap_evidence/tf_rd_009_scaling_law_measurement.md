# TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-design-and-measurement-on-the-classification-first-sandwich-target).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md),
  [TF-RD-010](tf_rd_010_many_class_promotion.md), and the simplified-parent
  phase of
  [TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion)

## External Evidence

- [Scaling Laws](../papers.md#scaling-laws): Kaplan and Chinchilla remain the
  main methodology references for matched-budget comparisons and compute-quality
  tradeoffs.
- [Compact Transformers And Training Recipes](../papers.md#compact-transformers-and-training-recipes):
  μP / muTransfer is the strongest prior for width-dependent transfer.
- [Training-Surface Adequacy And Batch/LR Scaling](../papers.md#training-surface-adequacy-and-batchlr-scaling):
  the modern optimizer-scaling literature is the main prior for LR, momentum,
  and batch as a function of budget.
- [Synthetic Data And Curriculum](../papers.md#synthetic-data-and-curriculum):
  synthetic-data scaling and curriculum references remain the best guide for
  treating curriculum or SCM-mixture as a first-class scaling dimension.
- Dedicated references to keep explicit in the law-design note:
  - μP / Tensor Programs V
  - Spectral Condition for μP under Width-Depth Scaling
  - Deriving Hyperparameter Scaling Laws via Modern Optimization Theory
  - Scaling Laws for Neural Language Models
  - Chinchilla
  - Deliberate Practice
  - CAMEL

## Repo-Local Evidence

- the roadmap now treats `tabfoundry_sandwich` as the primary classification
  scaling family, with `tabfoundry_staged` retained as the incumbent reference
  line rather than the scaling parent
- training telemetry and benchmark-registry artifacts now preserve resolved
  sandwich specs, runtime summaries, and regime-budget metadata needed for
  later scaling comparisons
- matched token budget alone is not sufficient once curriculum, SCM mixture,
  or task complexity changes; the repo now needs a matched regime-budget
  contract
- sandwich simplification under
  [#184](https://github.com/bensonlee5/tab-foundry/issues/184) is the required
  pre-scaling step for the family, but it does not satisfy TF-RD-009 by itself
- the first scaling target is now a fixed dagzoo many-class plus missingness
  slice rather than the earlier binary-only regime
- TF-RD-021 now supplies that slice through a two-phase lane: freeze the
  admissible post-RD-002/RD-005 dagzoo candidate surface first, then make one
  bounded carry-forward decision before runtime or scaling work
- regression is explicitly deferred from the first scaling program and is not a
  blocker for the first classification law fit

## Theory-Backed Versus Empirical Dimensions

- High-confidence dimensions:
  - width transfer via `d_icl` with μP-style priors
  - optimizer transfer via LR, momentum, and batch as a function of budget
  - curriculum or SCM-mixture as a real scaling dimension rather than noise
- Medium-confidence dimensions:
  - width and depth should be modeled jointly rather than collapsed to
    parameter count alone
  - matched regime budget should include unique-task budget as well as token
    budget
- Lower-confidence dimensions:
  - exact width-depth exponents from recent theory transfer cleanly to the
    tabular sandwich family without refitting
  - one universal compound knob will transfer cleanly across every later
    classification regime without further validation

## Current Interpretation

- the first scaling-law note should be a design note, not an immediate sweep
  spec
- the first fit should stay classification-only and should not wait on
  regression
- the first law should be conditional over:
  - width via `d_icl`
  - depth via `sandwich_layers`
  - optimizer transfer via LR, momentum, and batch
  - fixed inherited runtime policy
  - fixed curriculum or SCM-mixture slice
- the first carried slice should be dagzoo-backed many-class plus missingness,
  and the primary objective on that slice should be multiclass log loss
- matched token budget remains necessary, but comparisons should be interpreted
  through matched regime budget:
  - token budget
  - unique-task budget
  - fixed curriculum or SCM-mixture slice
  - fixed task-complexity band
- the first public single-knob interface should be derived from the fitted law
  later; it should not be authored up front

## Open Evidence Gaps

- the dedicated literature-synthesis and law-design note is not yet written
- the repo does not yet have one fixed dagzoo many-class plus missingness slice
  on the same simplified sandwich trunk
- the runtime policy is not yet finalized as a hard inherited precondition for
  the scaling ladder
- the TF-RD-021 carry-forward keep/defer decision is not yet finalized on the carried
  post-RD-002/RD-005 dagzoo slice
- sweep/result summaries still need compact presentation of the new
  runtime-summary and regime-budget fields

## Exit Signals

- the repo has a written law-design note that separates theory-backed and
  empirical dimensions explicitly
- width-depth classification laws are fit on the simplified sandwich family
  under one carried dagzoo many-class plus missingness slice and one inherited
  runtime policy
- the scaling artifacts compare rows by matched regime budget with final
  multiclass log loss as the primary objective
- any later `sandwich_scale` interface is explicitly derived from those law
  fits and remains internal until cross-surface validation is complete
