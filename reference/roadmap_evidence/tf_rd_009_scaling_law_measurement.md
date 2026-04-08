# TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-design-and-measurement-on-the-classification-first-sandwich-target).
It is the design-note contract for
[#229](https://github.com/bensonlee5/tab-foundry/issues/229) and the literature
handoff into the sweep-program design issue
[#140](https://github.com/bensonlee5/tab-foundry/issues/140).

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
  sweep-program design child [#140](https://github.com/bensonlee5/tab-foundry/issues/140)

## Core Reading Path

TF-RD-009 should treat the following exact primary sources as the core reading
path:

- [Scaling Laws](../papers.md#scaling-laws):
  - *Scaling Laws for Neural Language Models*
  - *Training Compute-Optimal Large Language Models*
  - *Spectral Condition for μP under Width-Depth Scaling*
  - *Deriving Hyperparameter Scaling Laws via Modern Optimization Theory*
  - *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training*
  - *Scaling Data-Constrained Language Models*
- [Compact Transformers And Training Recipes](../papers.md#compact-transformers-and-training-recipes):
  - *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)*
- [Synthetic Data And Curriculum](../papers.md#synthetic-data-and-curriculum):
  - *Improving the Scaling Laws of Synthetic Data with Deliberate Practice*

CAMEL is intentionally out of scope for TF-RD-009. It is not a scaling-law
source for this repo and should not appear in the core sweep-design reading path
for [#140](https://github.com/bensonlee5/tab-foundry/issues/140).

## Repo-Local Inputs

TF-RD-009 does not start from a blank scaling surface. It inherits a specific
classification-first benchmark, runtime, and architecture contract that later
sweeps must respect unless a follow-on issue explicitly reopens them.

- `tabfoundry_sandwich` is now the primary classification scaling family, with
  `tabfoundry_staged` retained only as the incumbent reference line rather than
  the scaling parent.
- TF-RD-010 is closed on the carried classification benchmark contract. The
  live ranking metric is `final_log_loss_at_matched_regime_budget`,
  interpreted explicitly as label-target log loss per test cell.
- TF-RD-022 is closed on one inherited runtime policy. Queue and registry
  artifacts now persist `runtime_summary` and `regime_budget`, including
  `token_budget`, `unique_task_budget`, `objective_metric`, and
  `curriculum_id`.
- TF-RD-024 is closed on medium-rung architecture evidence under
  [#233](https://github.com/bensonlee5/tab-foundry/issues/233), with
  `sandwich_heads=1` carried forward as the first-pass bounded non-dynamics
  winner.
- TF-RD-021 remains sidecar corpus context rather than a blocker on the
  TF-RD-009 critical path.
- Regression, missingness expansion, and public interface work remain out of
  scope for the first law-design branch.

## Executive Prescriptions

The literature does not imply one monolithic sweep. It implies three core
contracts plus one explicit empirical-slice axis, and TF-RD-009 should not
conflate them.

### Dimensions That Must Scale Together

- Fixed-budget architecture-law fits:
  - `model.d_icl`
  - `model.sandwich_layers`
  - interpretation: these are the only live architecture axes for the first
    TF-RD-009 law family
- Compute-optimal frontier fits:
  - architecture size proxy
  - `regime_budget.token_budget`
  - interpretation: Chinchilla-style claims apply when model size and training
    exposure are co-designed rather than when data is held constant
- Optimizer-transfer bundles:
  - `schedule.stages[].lr_max`
  - `optimizer.min_lr`
  - `optimizer.betas`
  - `training.task_batch_size`
  - `runtime.grad_accum_steps`
  - `optimizer.weight_decay` if later unfrozen from the inherited `0.0`
  - interpretation: TF-RD-009 should not sweep these as unrelated single knobs

### Dimensions That Stay Frozen For The First Law Family

- Benchmark slice:
  - `data.surface_label=tf_rd_010_dagzoo_medium_control`
  - `data.corpus_ref=tf_rd_010_dagzoo_medium_control_curated_v5`
  - `regime_budget.objective_metric=final_log_loss_at_matched_regime_budget`
- Runtime policy:
  - `runtime.mixed_precision=bf16`
  - `runtime.activation_checkpointing=true`
  - `runtime.compile_model=true`
  - `runtime.compile_backend=eager`
  - `runtime.compile_dynamic=true`
- Inherited optimizer family:
  - `optimizer.name=schedulefree_adamw`
- Non-scaling architecture knobs:
  - `model.sandwich_heads=1`
  - `model.head_hidden_dim=96`
  - `model.sandwich_latents=24`
  - `model.sandwich_ff_expansion=2`
  - `model.sandwich_summary_tokens_per_axis=3`
  - `model.sandwich_self_attention_per_cross=4`
  - `model.sandwich_pre_row_attention_layers=1`
  - `model.sandwich_pre_column_attention_layers=1`
  - `model.sandwich_pre_column_inducing_tokens=16`
  - `model.feature_type_conditioning=film`

### Variables That Remain Empirical Slice Axes Rather Than Core Law Axes

- `regime_budget.curriculum_id`
- repeated-data rate, approximated by the ratio of `token_budget` to
  `unique_task_budget`
- SCM or curriculum mixture
- benchmark rung promotion from medium to large

TF-RD-009 adoption decision: the first law fit should hold these slice
variables fixed. Later sweeps may vary them, but those rows should be reported
as different empirical slices rather than merged into one width-depth law.

## Variable Mapping: Paper Notation To Repo Notation

| Paper concept | Common paper notation | Repo-local interpretation |
|---------------|-----------------------|---------------------------|
| Model size / parameter scale | `N`, `P`, or parameter count | first-class proxy is the pair `model.d_icl` and `model.sandwich_layers`; report parameter count as a derived scalar, not as the only design axis |
| Width | width, hidden size, `n`, or `d` | `model.d_icl` |
| Depth | `L` | `model.sandwich_layers` |
| Training tokens / data exposure | `D`, tokens, training tokens | `regime_budget.token_budget` |
| Unique data supply | dataset size, unique examples, effective dataset size | `regime_budget.unique_task_budget` |
| Batch size | `B` | `training.task_batch_size`; if accumulation matters, treat effective optimization batch as `training.task_batch_size * runtime.grad_accum_steps` before any later distributed multiplier |
| Learning rate | `eta` | `schedule.stages[].lr_max` plus `optimizer.min_lr` and the applied schedule |
| Momentum / Adam beta terms | momentum, `beta`, `beta_1`, `beta_2` | `optimizer.betas` |
| Weight decay | `lambda`, WD | `optimizer.weight_decay` |
| Training horizon | steps, iterations, token horizon | `runtime.max_steps` together with `regime_budget.token_budget` |
| Objective under matched budget | loss, validation loss | `regime_budget.objective_metric=final_log_loss_at_matched_regime_budget` |
| Curriculum or data-mixture identity | data mixture, curriculum, recipe id | `regime_budget.curriculum_id` plus `data.surface_label` and `data.corpus_ref` |

## Paper Matrix

| Paper | Reported variables | Reported relationship | Direct TF-RD-009 takeaway | Caveat or non-transfer warning |
|-------|--------------------|-----------------------|---------------------------|-------------------------------|
| *Scaling Laws for Neural Language Models* | model size, dataset size, compute | loss follows smooth power laws in model size, data size, and compute; larger models are more sample-efficient; compute-optimal training undertrains larger models relative to full convergence | measure width-depth quality at fixed matched budget, but do not assume a fixed-token fit is the compute-optimal frontier | the paper is about language-model loss, not tabular multiclass log loss; do not import its fitted exponents as repo constants |
| *Training Compute-Optimal Large Language Models* | model parameters, training tokens, fixed compute budget | under fixed compute, model size and training tokens should scale equally; the abstract states that for every doubling of model size, training tokens should also double | when TF-RD-009 later designs compute-optimal sweeps, model growth and `token_budget` must be co-designed | this is not permission to vary `token_budget` inside one matched-budget law slice |
| *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)* | width, parametrization, hyperparameters | many optimal hyperparameters transfer across width when parametrization is correct | treat width transfer through `model.d_icl` as the cleanest first theory-backed transfer axis | width-only transfer does not automatically justify depth changes |
| *Spectral Condition for μP under Width-Depth Scaling* | width, depth, spectral scaling conditions, update norms | stable feature learning and transfer under joint width-depth movement requires width-depth scaling conditions, not naive width-only assumptions | once `model.sandwich_layers` moves, TF-RD-009 must treat joint width-depth scaling as a distinct design problem | this is a theoretical condition paper; it does not directly hand over a tabular sweep grid |
| *Deriving Hyperparameter Scaling Laws via Modern Optimization Theory* | learning rate, momentum, batch size, training horizon | optimizer hyperparameters follow budget-dependent power-law prescriptions when model size is held fixed | use this as the main prior for coupled LR, momentum, and batch retuning inside a fixed model family | do not mix these optimizer laws with model-size laws without stating that the paper held model size fixed |
| *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training* | weight decay, batch size, model size, dataset size | optimal weight decay scales linearly with batch size; optimal batch size and critical batch size scale with dataset size rather than model size | if TF-RD-009 later unfreezes weight decay, batch and weight decay must be co-designed rather than swept independently | the result is for LLM pre-training; treat it as an optimizer prior to test, especially because the inherited TF-RD-022 surface keeps `weight_decay=0.0` |
| *Scaling Data-Constrained Language Models* | data supply, repeated epochs, scaling curves | repeated data is tolerable in a modest regime, then marginal value decays; the paper reports repetition up to roughly four epochs without noticeable loss change in its setting | keep repeated-data rate or curriculum slice fixed within a first law fit; do not mix low-repetition and heavy-repetition rows on one curve | the safe repetition range is setting-dependent and should not be hard-coded into tabular policy |
| *Improving the Scaling Laws of Synthetic Data with Deliberate Practice* | synthetic-data quality, curriculum difficulty, sample efficiency | more informative or challenging synthetic examples can improve sample efficiency | treat curriculum difficulty as an empirical slice variable worth testing after the first law fit | this is not evidence for width, depth, or optimizer exponents |

## Locked Starting Surface

The inherited TF-RD-022 benchmark experiment is the repo-local baseline surface:

- `model.arch=tabfoundry_sandwich`
- `model.d_icl=60`
- `model.sandwich_layers=2`
- `training.task_batch_size=16`
- `runtime.grad_accum_steps=4`
- `optimizer.name=schedulefree_adamw`
- `optimizer.weight_decay=0.0`
- `optimizer.betas=[0.9, 0.999]`
- `optimizer.min_lr=1.0e-5`
- `schedule.stages[0].lr_max=1.0e-3`
- `runtime.max_steps=2500`

TF-RD-009 adoption decision: this exact surface is the inherited anchor for the
first law-design branch, except that TF-RD-024 carries `model.sandwich_heads=1`
instead of the earlier `4`.

## Matched Regime Budget Contract

Kaplan-style and Chinchilla-style claims answer different questions. TF-RD-009
must preserve that distinction in the sweep tree.

- Same-law fixed-budget comparisons require the same:
  - `regime_budget.token_budget`
  - `regime_budget.unique_task_budget`
  - `regime_budget.curriculum_id`
  - `regime_budget.objective_metric`
  - benchmark slice
  - inherited runtime policy
- Chinchilla-style compute-optimal comparisons require model size and
  `token_budget` to move together under an explicit fixed-compute design.
- Therefore one TF-RD-009 report should not mix:
  - width-depth rows evaluated at one matched `token_budget`
  - and separate parameter-token frontier rows where `token_budget` grows with
    model size

TF-RD-009 adoption decision: the first law fit should be a fixed-budget law on
the medium classification rung. Any later compute-optimal parameter-token
frontier belongs to a distinct sweep family designed by
[#140](https://github.com/bensonlee5/tab-foundry/issues/140).

## Literature Synthesis

### Size, Data, And Compute Laws

#### *Scaling Laws for Neural Language Models*

Paper-reported relationship:

- validation loss follows approximate power laws in model size, dataset size,
  and compute
- larger models are more sample-efficient than smaller ones
- compute-optimal training does not require full convergence; larger models are
  relatively undertrained when compute is fixed

TF-RD-009 adoption decision:

- measure the first architecture law under matched `token_budget` and matched
  benchmark slice so that width-depth quality differences are interpretable
- report parameter count, but do not collapse the first fit to one scalar model
  size because width and depth may not be interchangeable on the sandwich
  surface

#### *Training Compute-Optimal Large Language Models*

Paper-reported relationship:

- the abstract states that under a fixed compute budget, model size and
  training tokens should be scaled equally
- the paper's headline example is that every doubling of model size should be
  accompanied by a doubling of training tokens

TF-RD-009 adoption decision:

- do not design later TF-RD-009 frontier sweeps that increase model size while
  holding `token_budget` fixed and then call that compute-optimal
- separate "fixed-budget architecture law" work from "compute-optimal
  parameter-token frontier" work

### Width And Depth Transfer

#### *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)*

Paper-reported relationship:

- with the correct parametrization, many optimal hyperparameters transfer across
  width changes
- this makes width the cleanest first transfer axis for hyperparameter reuse

TF-RD-009 adoption decision:

- use `model.d_icl` as the first transfer-backed axis when designing the sweep
  tree
- treat width-only transfer as a stronger prior than naive parameter-count
  transfer

#### *Spectral Condition for μP under Width-Depth Scaling*

Paper-reported relationship:

- once width and depth move together, stable feature learning and transfer need
  width-depth scaling conditions on norms and update scales
- plain width-only μP intuition is not enough for joint width-depth movement

TF-RD-009 adoption decision:

- any sweep family that changes `model.sandwich_layers` must be designed as a
  joint width-depth family rather than as "width transfer plus a depth tweak"
- optimizer transfer claims should be treated as lower-confidence whenever depth
  also changes, and should be validated empirically rather than assumed

### Optimizer And Batch Coupling

#### *Deriving Hyperparameter Scaling Laws via Modern Optimization Theory*

Paper-reported relationship:

- learning rate, momentum, and batch-size prescriptions can be derived as
  power-law functions of training horizon or token budget
- the setup holds model size fixed while deriving those optimizer-budget laws

TF-RD-009 adoption decision:

- when TF-RD-009 retunes the optimizer surface, retune LR, beta terms, and
  batch as one coupled bundle against the budget axis
- document explicitly when a recommendation comes from a model-fixed optimizer
  theory result rather than from a model-scaling law

#### *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training*

Paper-reported relationship:

- optimal weight decay scales linearly with batch size
- optimal batch size and critical batch size scale with dataset size rather
  than model size
- the paper also motivates timescale-style invariants as a useful organizing
  prior

TF-RD-009 adoption decision:

- if later sweeps reopen `optimizer.weight_decay`, they should pair batch and
  weight decay in the same design bundle
- because TF-RD-022 inherits `optimizer.weight_decay=0.0`, the first law family
  should keep weight decay frozen rather than partially opening the optimizer
  surface
- treat timescale-style invariants as priors to test, not as universal tabular
  truths

Supporting caution from the batch/LR literature:

- *Don't Decay the Learning Rate, Increase the Batch Size* and *An Empirical
  Model of Large-Batch Training* support the broader claim that batch and
  schedule are coupled
- *Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling* is a
  caution that Adam-style optimizers need not obey naive linear LR scaling

### Data Scarcity And Curriculum

#### *Scaling Data-Constrained Language Models*

Paper-reported relationship:

- repeated data is tolerable in a modest regime and then shows declining
  marginal value
- the paper reports little loss change up to roughly four epochs of repetition
  in its studied setting

TF-RD-009 adoption decision:

- keep repeated-data rate fixed inside the first law fit
- use the ratio between `token_budget` and `unique_task_budget` as the repo
  marker of whether a later sweep has entered a materially different
  repetition regime

#### *Improving the Scaling Laws of Synthetic Data with Deliberate Practice*

Paper-reported relationship:

- more informative or deliberately challenging synthetic examples can improve
  sample efficiency relative to weaker synthetic curricula

TF-RD-009 adoption decision:

- treat curriculum quality as a later empirical slice variable
- do not treat deliberate-practice results as evidence that width, depth, or
  optimizer exponents have changed

## Non-Transferable Or Low-Confidence Claims

The following are useful priors, but TF-RD-009 should not hard-code them as if
they were already proven on the sandwich classification surface.

- Do not import Kaplan or Chinchilla fitted exponents as tabular constants.
- Do not treat a fixed-token width-depth fit as proof of a compute-optimal
  frontier.
- Do not assume width-only μP transfer remains valid when
  `model.sandwich_layers` changes.
- Do not assume Power Lines' weight-decay law applies unchanged while the repo
  still uses `optimizer.weight_decay=0.0` on the inherited surface.
- Do not merge distinct `curriculum_id` or repeated-data regimes into one law
  curve.
- Do not cite *Improving the Scaling Laws of Synthetic Data with Deliberate
  Practice* as evidence for width/depth exponents.

## Ranking, Guardrails, And Non-Goals

- Primary ranking objective:
  - use `final_log_loss_at_matched_regime_budget` as the only ranking key for
    the first classification scaling fit
- Guardrails:
  - keep calibration, stability, runtime, reserved VRAM, and clipped-step or
    instability summaries as explicit guardrails and tie-break context
- Non-goals:
  - do not create a public `sandwich_scale` interface yet
  - do not author a `tf_rd_009_*` sweep id or system-delta scaffold in this
    note
  - do not reinterpret historical BPC-era comparisons as the live ranking rule
  - do not reopen TF-RD-021, regression, or missingness as prerequisites

## Handoff To #140

This note should give
[#140](https://github.com/bensonlee5/tab-foundry/issues/140) a much sharper
design input than "run some scaling sweeps."

The sweep-program design issue should preserve these separations explicitly:

- Sweep family 1: fixed-budget width transfer
  - vary `model.d_icl`
  - keep `model.sandwich_layers` fixed
  - keep matched `token_budget`, `unique_task_budget`, and `curriculum_id`
- Sweep family 2: fixed-budget joint width-depth scaling
  - vary `model.d_icl` and `model.sandwich_layers`
  - treat optimizer transfer as a lower-confidence empirical question
  - keep non-scaling sandwich knobs frozen
- Sweep family 3: compute-optimal parameter-token frontier
  - co-design model size and `regime_budget.token_budget`
  - do not mix these rows into the fixed-budget law fit
- Sweep family 4: curriculum or repeated-data slices
  - hold one architecture family fixed
  - vary `curriculum_id` or repetition regime explicitly

The issue tree designed by [#140](https://github.com/bensonlee5/tab-foundry/issues/140)
should also preserve these rules:

- dimensions that must be co-designed:
  - `model.d_icl` and `model.sandwich_layers` for architecture laws
  - LR, beta terms, batch, and any reopened weight decay for optimizer laws
  - model size and `token_budget` for compute-optimal frontier work
- dimensions that must stay frozen in the first law family:
  - runtime policy
  - benchmark slice
  - objective metric
  - non-scaling sandwich knobs
- paper-backed laws that are priors to test rather than assumptions to
  hard-code:
  - Chinchilla parameter-token coupling
  - μP width transfer
  - Spectral width-depth transfer conditions
  - Power Lines batch-weight-decay coupling
  - deliberate-practice curriculum effects

## Exit Signals

- the repo has a written law-design note that separates paper-reported claims
  from TF-RD-009 adoption decisions
- width and depth scaling on the simplified sandwich family are defined as a
  distinct fixed-budget law family on the closed TF-RD-010 benchmark contract
- any later compute-optimal frontier is explicitly separated from the first
  fixed-budget law fit
- optimizer transfer is defined as a coupled LR/beta/batch, and optionally
  weight-decay, design problem rather than a set of unrelated single-knob
  sweeps
- any later `sandwich_scale` interface remains internal until cross-surface
  validation is complete
