# TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-design-and-measurement-on-the-classification-first-sandwich-target).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md),
  [TF-RD-024](tf_rd_024_post_performance_architecture_knob_sweep.md),
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
- the first scaling target is now the closed TF-RD-010 medium/large
  classification benchmark contract rather than the earlier binary-only regime
- TF-RD-022 now owns the three required pre-scaling performance follow-ups on
  the settled runtime surface: training throughput, benchmark throughput, and
  corpus materialization throughput
- TF-RD-024 now owns the bounded non-dynamics sandwich knob sweep that should
  finish before TF-RD-009 freezes the remaining architecture knobs
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
  - fixed benchmark contract
- the first carried slice should be the closed TF-RD-010 classification
  benchmark contract, and the primary objective on that slice should be
  multiclass log loss
- matched token budget remains necessary, but comparisons should be interpreted
  through matched regime budget:
  - token budget
  - unique-task budget
  - fixed curriculum or SCM-mixture slice
  - fixed task-complexity band
- TF-RD-024, not TF-RD-009, owns the bounded non-dynamics sandwich knob sweep;
  TF-RD-009 should reserve `d_icl`, `sandwich_layers`, and optimizer transfer
  as the main live dimensions
- the first public single-knob interface should be derived from the fitted law
  later; it should not be authored up front

## Open Evidence Gaps

- the dedicated literature-synthesis and law-design note is not yet written
- TF-RD-022 performance closeout is not yet finalized as a hard inherited
  precondition for the scaling ladder
- the TF-RD-024 bounded architecture keep/defer decision is not yet finalized
- the repo still does not have one canonical TF-RD-009 artifact path on the
  inherited benchmark and runtime contract

## Exit Signals

- the repo has a written law-design note that separates theory-backed and
  empirical dimensions explicitly
- width-depth classification laws are fit on the simplified sandwich family
  under the closed TF-RD-010 benchmark contract, one inherited runtime policy,
  and one fixed post-TF-RD-024 non-scaling architecture surface
- the scaling artifacts compare rows by matched regime budget with final
  multiclass log loss as the primary objective
- any later `sandwich_scale` interface is explicitly derived from those law
  fits and remains internal until cross-surface validation is complete
