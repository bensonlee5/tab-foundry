# TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-design-and-measurement-on-the-classification-first-sandwich-target).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-010](tf_rd_010_many_class_promotion.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md),
  [TF-RD-024](tf_rd_024_post_performance_architecture_knob_sweep.md), and the
  simplified-parent phase of
  [TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion)
- GitHub issue chain: umbrella
  [#51](https://github.com/bensonlee5/tab-foundry/issues/51), design-note
  child [#229](https://github.com/bensonlee5/tab-foundry/issues/229), then
  first execution child [#140](https://github.com/bensonlee5/tab-foundry/issues/140)

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
- The law-design note should keep these references explicit:
  - μP / Tensor Programs V
  - Spectral Condition for μP under Width-Depth Scaling
  - Deriving Hyperparameter Scaling Laws via Modern Optimization Theory
  - Scaling Laws for Neural Language Models
  - Chinchilla
  - Deliberate Practice
  - CAMEL

## Repo-Local Evidence

- `tabfoundry_sandwich` is now the primary classification scaling family, with
  `tabfoundry_staged` retained only as the incumbent reference line rather than
  the scaling parent
- sandwich simplification under
  [#184](https://github.com/bensonlee5/tab-foundry/issues/184) is the required
  pre-scaling step for the family, but it does not satisfy TF-RD-009 by itself
- TF-RD-010 is now closed on the carried classification benchmark contract, so
  the first scaling target is the benchmark-defined medium/large multiclass
  family rather than the older binary-only regime
- TF-RD-022 is now closed on one inherited runtime policy, and training
  telemetry plus benchmark-registry artifacts preserve resolved sandwich specs,
  `runtime_summary`, and `regime_budget` metadata needed for later scaling
  comparisons
- TF-RD-024 is now closed on medium-only evidence under
  [#233](https://github.com/bensonlee5/tab-foundry/issues/233), with
  `sandwich_heads=1` carried forward as the bounded non-dynamics architecture
  winner for the first scaling pass
- TF-RD-021 remains sidecar corpus context under
  [#165](https://github.com/bensonlee5/tab-foundry/issues/165) rather than a
  blocker on the TF-RD-009 critical path
- regression is explicitly deferred from the first scaling program and is not a
  blocker for the first classification law fit

## Locked Starting Surface

- Benchmark contract:
  - inherit the closed TF-RD-010 classification benchmark contract only
  - keep the active classification metric as
    `final_log_loss_at_matched_regime_budget`, interpreted as label-target log
    loss per test cell
  - keep the first fitting slice on the carried medium rung that TF-RD-024 used
    for its closeout; hold the large rung as follow-on validation on the same
    benchmark family once a coherent first fit exists
- Runtime policy:
  - inherit the closed TF-RD-022 runtime surface unchanged:
    `mixed_precision=bf16`, `trace_activations=false`,
    `activation_checkpointing=true`, `compile_model=true`,
    `compile_backend=eager`, and `compile_dynamic=true`
- Architecture freeze:
  - inherit the TF-RD-024 compile-eager-dynamic anchor values for every
    non-scaling sandwich knob
  - substitute the TF-RD-024 winner `sandwich_heads=1`
  - keep `head_hidden_dim`, `sandwich_summary_tokens_per_axis`,
    `sandwich_latents`, `sandwich_ff_expansion`,
    `sandwich_self_attention_per_cross`, and
    `sandwich_pre_row_attention_layers` frozen for the first fit

## First-Pass Law Design Contract

- Live theory-backed dimensions:
  - width via `d_icl`
  - depth via `sandwich_layers`
  - optimizer transfer via learning-rate scale, Adam-style momentum or beta
    settings, and batch-size transfer on the inherited runtime surface
- Empirical dimensions that stay explicit but secondary:
  - curriculum or SCM-mixture choice is a real scaling dimension, but the first
    fit should hold one carried slice fixed rather than mix multiple curricula
  - width and depth should be modeled jointly instead of collapsing the first
    fit to parameter count alone
- Frozen dimensions for this branch of TF-RD-009:
  - no reopening of TF-RD-024 bounded non-dynamics knob work
  - no reopening of TF-RD-022 runtime-policy or kernel-acceleration work
  - no TF-RD-021 corpus change as a prerequisite to the first fit
  - no regression, missingness, or imbalance expansion in the first law fit

## Matched Regime Budget Contract

- Matched token budget remains necessary but is not sufficient on its own.
- Treat rows as the same regime-budget comparison only when they keep the
  carried benchmark slice fixed through the same benchmark manifest and
  complexity rung, and preserve the same:
  - `token_budget`
  - `unique_task_budget`
  - `curriculum_id`
  - `objective_metric`
  - inherited runtime policy
- Treat any change to benchmark rung, curriculum or SCM mixture, or task-family
  mix as a new empirical slice rather than another point on the same first law.

## Ranking, Guardrails, And Non-Goals

- Primary ranking objective:
  - use `final_log_loss_at_matched_regime_budget` as the only ranking key for
    the first classification scaling fit
- Guardrails:
  - keep calibration, stability, runtime, and clipped-step or instability
    summaries as explicit guardrails and tie-break context rather than folding
    them into one composite score
- Non-goals:
  - do not create a public `sandwich_scale` interface yet
  - do not author a TF-RD-009 sweep id or system-delta scaffold in this note
  - do not reinterpret historical BPC-era comparisons as the live ranking rule

## Handoff To #140

- [#229](https://github.com/bensonlee5/tab-foundry/issues/229) is satisfied
  when this design note gives
  [#140](https://github.com/bensonlee5/tab-foundry/issues/140) one explicit
  first-sweep contract.
- The first execution issue should:
  - create the first `tf_rd_009_*` sweep scaffold on the inherited medium
    classification rung
  - vary `d_icl`, `sandwich_layers`, and optimizer-transfer settings together
    on the locked runtime and architecture surface rather than turning them into
    separate epics
  - keep all comparisons on the matched regime-budget contract defined above
  - hold the large rung for follow-on validation after the first fit is
    coherent on medium
  - keep any later single-knob or public scaling interface internal-only until
    cross-surface validation is complete

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
