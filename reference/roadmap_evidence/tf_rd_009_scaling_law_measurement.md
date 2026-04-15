# TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-009](../../docs/development/roadmap.md#tf-rd-009-scaling-law-design-and-measurement-on-the-classification-first-sandwich-target).
It is the design-note contract for
[#229](https://github.com/bensonlee5/tab-foundry/issues/229) and the literature
handoff into the sweep-program design issue
[#140](https://github.com/bensonlee5/tab-foundry/issues/140).

- Status: `in_progress`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-010](tf_rd_010_many_class_promotion.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md),
  [TF-RD-024](tf_rd_024_post_performance_architecture_knob_sweep.md), and the
  simplified-parent phase of
  [TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion)
- GitHub issue chain: umbrella
  [#51](https://github.com/bensonlee5/tab-foundry/issues/51), completed
  design-note child [#229](https://github.com/bensonlee5/tab-foundry/issues/229),
  completed sweep-program design child
  [#140](https://github.com/bensonlee5/tab-foundry/issues/140), active
  fixed-budget family epic [#253](https://github.com/bensonlee5/tab-foundry/issues/253),
  historical schedulefree width-transfer child
  [#254](https://github.com/bensonlee5/tab-foundry/issues/254), completed
  historical schedulefree joint width-depth child
  [#255](https://github.com/bensonlee5/tab-foundry/issues/255), completed
  historical schedulefree Kaplan-exact Phase-2 fit/report child
  [#256](https://github.com/bensonlee5/tab-foundry/issues/256), follow-on
  historical large-rung validation / hardware-freeze child
  [#257](https://github.com/bensonlee5/tab-foundry/issues/257), active Muon
  Phase-1 child [#275](https://github.com/bensonlee5/tab-foundry/issues/275),
  active Muon Phase-2 child
  [#274](https://github.com/bensonlee5/tab-foundry/issues/274), later
  upper-family child [#269](https://github.com/bensonlee5/tab-foundry/issues/269),
  compute-frontier child [#259](https://github.com/bensonlee5/tab-foundry/issues/259),
  and curriculum/repetition slice child
  [#260](https://github.com/bensonlee5/tab-foundry/issues/260)

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

## Historical Schedulefree Family Vs Fresh Muon Family

TF-RD-009 now has two distinct branches in repo state, and the distinction is
operational rather than cosmetic.

- Historical schedulefree family:
  - `tf_rd_009_width_transfer_medium_v1`
  - `tf_rd_009_width_depth_medium_v1`
  - `tf_rd_009_ns_medium_v1`
  - `tf_rd_009_batch_critical_medium_v1`
  - corrected one-epoch rerun
    `tf_rd_009_ns_one_epoch_medium_v1`,
    `tf_rd_009_batch_critical_one_epoch_medium_v1`, and
    `tf_rd_009_phase2_one_epoch_v1`
  - interpretation: preserved diagnostic history only; keep
    [#256](https://github.com/bensonlee5/tab-foundry/issues/256) closed and do
    not reuse those ids in place
- Fresh Muon family:
  - experiment alias
    `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
  - completed width screen `tf_rd_009_muon_width_screen_medium_v1`
  - diagonal scaffold `tf_rd_009_muon_width_depth_medium_v1`
  - one-epoch Phase-2 scaffolds
    `tf_rd_009_muon_ns_one_epoch_medium_v1`,
    `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`, and
    `tf_rd_009_muon_phase2_one_epoch_v1`
  - upper-extension scaffolds
    `tf_rd_009_muon_width_depth_upper_extension_one_epoch_medium_v1`,
    `tf_rd_009_muon_ns_upper_extension_one_epoch_medium_v1`, and
    `tf_rd_009_muon_phase2_upper_extension_one_epoch_v1`
  - interpretation: active reboot branch under
    [#275](https://github.com/bensonlee5/tab-foundry/issues/275) then
    [#274](https://github.com/bensonlee5/tab-foundry/issues/274), frozen to
    `tf_rd_010_dagzoo_medium_control_curated_v6` plus the post-PR
    [#271](https://github.com/bensonlee5/tab-foundry/pull/271) packed Muon
    runtime stack on `main`; the landed Muon width screen now records
    `60x2=0.4172`, `48x2=0.4147`, `96x2=0.4128`, and `128x2=0.3951`, with
    `60x2` retained as the formal external anchor and `128x2` carried forward
    as the current in-family baseline for the diagonal derivation

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

Historical schedulefree TF-RD-009 and the active Muon reboot do not freeze the
same starting contract. Keep the two families separate when reading or writing
later sweep ids.

- Historical schedulefree family:
  - `data.surface_label=tf_rd_010_dagzoo_medium_control`
  - `data.corpus_ref=tf_rd_010_dagzoo_medium_control_curated_v5`
  - `regime_budget.objective_metric=final_log_loss_at_matched_regime_budget`
  - `runtime.mixed_precision=bf16`
  - `runtime.activation_checkpointing=true`
  - `runtime.compile_model=true`
  - `runtime.compile_backend=eager`
  - `runtime.compile_dynamic=true`
  - `optimizer.name=schedulefree_adamw`
- Fresh Muon reboot:
  - `data.surface_label=tf_rd_010_dagzoo_medium_control`
  - `data.corpus_ref=tf_rd_010_dagzoo_medium_control_curated_v6`
  - `regime_budget.objective_metric=final_log_loss_at_matched_regime_budget`
  - `runtime.mixed_precision=bf16`
  - `runtime.activation_checkpointing=true`
  - `runtime.compile_model=true`
  - `runtime.compile_backend=eager`
  - `runtime.compile_dynamic=true`
  - `runtime.loader_task_batch_cache_mode=bounded_streaming`
  - `model.sandwich_packed_attention=true`
  - `optimizer.name=muon`
  - `optimizer.weight_decay=0.01`
  - `optimizer.betas=[0.9,0.95]`
  - `optimizer.min_lr=1.0e-6`
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

## Phase Split

TF-RD-009 now has two explicit reporting layers and they must stay separate in
the code, the sweep tree, and the interpretation prose.

- Phase 1:
  - keep the running width-depth queue on
    `tf_rd_009_width_depth_medium_v1`
  - use the repo-local mixed-depth bridge only to derive executable integer
    rows on the dense diagonal
  - report the first fixed-budget law against measured
    `model_size.total_params` from completed in-family rows only
- Phase 2:
  - run the paper-faithful loss study on
    `tf_rd_009_ns_medium_v1` and
    `tf_rd_009_batch_critical_medium_v1`
  - fit `L(N)`, `L(D)`, `L(C)`, `L(N,D)`, `L(N,S)`, `Bcrit(L)`, and the
    derived `L(Cmin)` frontier from completed benchmark-backed rows only
  - persist per-run inspected parameter and compute accounting so the study
    never estimates `N` or `C` by reading source code heuristics

TF-RD-009 adoption decision: the queue-construction bridge and the final loss
fits are now separate surfaces. Phase 1 keeps the bridge for row selection,
whereas Phase 2 fits the laws on inspected run metadata only.

## Inspected Parameter And Compute Accounting

Phase 2 uses direct model inspection plus measured training telemetry to define
all canonical axes.

- strict embedding params:
  - parameters owned by explicit `nn.Embedding` modules only
- strict non-embedding params:
  - every remaining trainable parameter after removing explicit embeddings
  - this is the canonical paper-style `N` axis
- expanded embedding-like params:
  - strict embeddings plus learned lookup/query/seed tensors such as
    `test_token`, summary queries, latent seeds, BOS/CLS-like tokens, and
    inducing seeds
  - this split is diagnostic only and must not replace the canonical `N`
    axis in the reported fits
- expanded non-embedding params:
  - complement of the expanded embedding-like partition

Canonical Phase-2 variables are:

- `N = parameter_accounting.canonical_non_embedding_params`
- `S = tab_foundry_metrics.final_step`
- `B_eff = regime_budget.tokens_per_step`
- `D = regime_budget.tokens_seen = B_eff * S`
- `C = compute_accounting.total_train_flops = train_flops_per_token * D`

The inspected compute contract is training-only:

- embedding tables count toward parameter accounting
- embedding lookups do not count as dense matmul FLOPs
- learned embedding-adjacent seeds/queries count toward parameters under the
  documented strict/expanded rules, but their compute is charged only through
  downstream attention and MLP operations
- `train_flops_per_token`, `train_flops_per_step`, and `total_train_flops`
  are derived from instantiated-module shapes plus measured training-shape
  telemetry

TF-RD-009 adoption decision: Phase 2 must reject rows that do not satisfy the
measured identity `D = B_eff * S` within telemetry tolerance, and it must treat
strict non-embedding params as the canonical `N` axis even when the expanded
diagnostic partition is also reported.

## Phase-2 Functional Forms

The literature-backed Phase-2 study uses the paper family as the default fit
hypothesis, while treating every fitted scale and exponent as repo-specific
empirical quantities rather than imported constants.

The historical schedulefree branch tracked its Kaplan-exact Phase-2 study in
[#256](https://github.com/bensonlee5/tab-foundry/issues/256), with the
canonical study config in `reference/scaling_studies/tf_rd_009_phase2.yaml`
and executable sweep surfaces `tf_rd_009_ns_medium_v1` and
`tf_rd_009_batch_critical_medium_v1`.

As of April 12, 2026, the current repo-tracked Phase-2 result is a complete
fit over 44 validation-backed points: 24 `family=ns_core` rows from
`tf_rd_009_ns_medium_v1` and 20 `family=batch_critical` rows from
`tf_rd_009_batch_critical_medium_v1`. The artifact root is
`outputs/research_scaling/tf_rd_009_phase2`, with inspected `N`, measured `S`,
derived `D = B_eff * S`, training-only `C`, and posthoc CPU validation sidecars
materialized through `reference/scaling_studies/tf_rd_009_phase2_validation_backfill_v1.json`.
This is the current Phase-2 fit evidence payload for
[#256](https://github.com/bensonlee5/tab-foundry/issues/256).

Correction note, April 13, 2026 PT: the above `tf_rd_009_phase2` payload is now
historical/superseded for scaling-law and `Cmin` interpretation. The
higher-budget `NS` and `batch_critical` rows cycled the same 143,976-task train
manifest, so the corrected path is to rerun the original `{625,1250,2500,5000}`
and `{1,2,4,8,16}` ladders on
`tf_rd_010_dagzoo_medium_control_curated_v6`, using
`tf_rd_009_ns_one_epoch_medium_v1`,
`tf_rd_009_batch_critical_one_epoch_medium_v1`, and
`tf_rd_009_phase2_one_epoch_v1`. The original artifacts remain preserved as
diagnostic history and must not be mixed into the corrected one-epoch fit.

Muon reboot note, April 14, 2026 PT: the active Phase-2 branch is now the fresh
Muon family under [#274](https://github.com/bensonlee5/tab-foundry/issues/274),
with study config
`reference/scaling_studies/tf_rd_009_muon_phase2_one_epoch_v1.yaml` and sweep
scaffolds `tf_rd_009_muon_ns_one_epoch_medium_v1` and
`tf_rd_009_muon_batch_critical_one_epoch_medium_v1`. Those Muon ids stay empty
until the fresh diagonal derivation lands under
[#275](https://github.com/bensonlee5/tab-foundry/issues/275). The fresh Muon
width screen has now landed on the v6 contract with `60x2=0.4172`,
`48x2=0.4147`, `96x2=0.4128`, and `128x2=0.3951`, so the immediate remaining
Phase-1 work is to materialize `tf_rd_009_muon_width_depth_medium_v1` from the
carried `128x2` baseline while keeping `60x2` as the formal external anchor.

The current C axis has also been audited. Five reused 2,500-step NS rows
(`07`, `11`, `15`, `19`, and `23`) and the reused batch-critical 96x2 row
(`11`) originally carried `compute_accounting.training_shape_summary: null`,
which triggered fallback accounting and made the reused rows appear more
expensive than their same-model higher-step fresh continuations. The compact
registry now carries shape-summary-backed compute accounting for those reused
rows, and `research scaling fit` rejects C-axis fits if shape summaries are
missing for non-reuse rows or same-model NS `C` is non-monotone across steps.

- one-dimensional fits:
  - `L(N) = E + (N_c / N)^alpha_N`
  - `L(D) = E + (D_c / D)^alpha_D`
  - `L(C) = E + (C_c / C)^alpha_C`
- joint fits:
  - `L(N,D) = E + ((N_c / N)^(alpha_N / alpha_D) + D_c / D)^alpha_D`
  - `L(N,S) = E + ((N_c / N)^(alpha_N / alpha_S) + S_c / S)^alpha_S`
- batch-critical law:
  - `Bcrit(L) = B_* / L^(1 / alpha_B)`
- compute frontier relation:
  - `Cmin = C / (1 + B_eff / Bcrit(L))`

Phase-2 reporting targets are:

- benchmark-backed matched-budget log loss as the repo-facing canonical target
  for `L(N)`, `L(D)`, `L(C)`, `L(N,D)`, and `L(Cmin)`
- validation loss as the paper-faithful companion target and the primary
  checkpointwise target for `L(N,S)` and `Bcrit(L)`

The study must emit inspectable artifacts for every reported fit:

- JSON payloads with raw points, chosen subsets, fitted scales, alphas,
  residuals, and direct-vs-implied diagnostics
- PNG graphs for every one-dimensional law, residual surface, joint-law plot,
  and the compute frontier
- a Markdown summary rooted under `outputs/research_scaling/tf_rd_009_phase2/`
- posthoc W&B summary payloads that surface the fitted alphas and artifact root

Current complete Phase-2 fit values:

| Fit | Target | Points | Key parameters | log-space R2 | RMSE |
| --- | --- | ---: | --- | ---: | ---: |
| `L(N)` | benchmark log loss | 6 | `alpha_n=0.0223655`, `Nc=0.00200950` | 0.196187 | 0.027509 |
| `L(D)` | benchmark log loss | 4 | `alpha_d=0.0368629`, `Dc=12671.9517` | 0.205136 | 0.039687 |
| `L(C)` | benchmark log loss | 44 | `alpha_c=0.521774`, `Cc=5.456582059841496e11` | 0.237521 | 0.036255 |
| `L(N,D)` | benchmark log loss | 24 | `alpha_n=0.0111205`, `alpha_d=663.875`, `Nc=1.0e-12`, `Dc=49864.2548` | 0.484884 | 0.031717 |
| `L(N,S)` | validation loss | 24 | `alpha_n=0.0302565`, `alpha_s=0.331430`, `Nc=258222760.6`, `Sc=608.501` | 0.820915 | 0.033284 |
| `Bcrit(L)` | validation loss | 2 | `alpha_b=0.00459242`, `B_star=5.144890799137182e34` | -0.064950 | 664615.416 |
| `L(Cmin)` | benchmark log loss | 12 | `alpha_cmin=0.123823`, `Ccmin=2.1079076205300552e10` | 0.915117 | 0.014289 |

Interpretation: the validation-backed `L(N,S)` surface is the useful primary
signal. The one-dimensional benchmark-loss slices plus `L(N,D)` remain noisier
and partly degenerate on this small matrix, so carry them as diagnostics. The
batch-critical data is complete, but `Bcrit(L)` is weak: its lower envelope has
only two points and a negative log-space R2, so `L(Cmin)` is a derived
diagnostic rather than a high-confidence operating law.

After the one-epoch correction, this `Bcrit(L)` and its derived `L(Cmin)` are
not admissible for the corrected compute-optimal story. Refit both only from
the corrected one-epoch batch-critical artifact once its validation backfill
lands.

Axis audit for the complete matrix:

- `N` is structurally valid: six increasing strict non-embedding parameter
  values from `666542` through `11335950`. It is not a clean monotone loss axis
  in this grid because each row changes width and depth together, and the
  highest-step benchmark slice has `log_space_r2=0.196187`.
- `S` is structurally valid: four completed step values
  `{625, 1250, 2500, 5000}` per model row. Loss mostly improves with `S`, but
  several larger rows regress at `5000` versus `2500`, so single-row step slices
  should be treated as diagnostics rather than strong standalone laws.
- `D` is internally consistent with the measured runtime budget:
  `D = B_eff * S` holds to floating tolerance. Because `B_eff` varies by only
  about 5% and only with the step ladder in the NS matrix, `D` is nearly
  collinear with `S`; the current `L(D)` uses only four points from the largest
  `N` row and has `log_space_r2=0.205136`.
- `B_eff` is independently sampled by the completed batch-critical sweep, but
  the observed lower envelope is only two points after validation-loss
  selection; this is why `Bcrit(L)` is recorded as weak.
- `C` is now structurally valid after the 2,500-step accounting correction: all
  44 points have observed or explicitly reused shape-summary-backed compute
  accounting, and same-model NS `C` is monotone in `S`. Its benchmark-loss fit
  remains weak at `log_space_r2=0.237521`.
- Target axes differ materially: validation loss has the cleanest structure and
  supports the reported `L(N,S)` surface; benchmark log loss is noisier across
  `N`, `D`, and `C`, which is why the one-dimensional benchmark-loss fits are
  carried as diagnostics.

### Stronger Fit Audit And Follow-On Law Design

Treat [#256](https://github.com/bensonlee5/tab-foundry/issues/256) as a
completed Phase-2 diagnostic, not as a settled compute-optimal law. The useful
signal is validation-backed `L(N,S)`. The weak signals are `L(N)`, `L(D)`,
`L(C)`, `L(N,D)`, and especially `Bcrit(L)` because the current axes are
partly entangled and the completed batch envelope has only two points.

The repo now exposes the stronger fit-audit surface:

```bash
tab-foundry research scaling audit --study tf_rd_009_phase2
```

The audit writes `audit/audit_summary.json` and `audit/audit.md` under the
study artifact root by default. It compares validation-loss versus
benchmark-loss targets, runs leave-one-geometry and leave-one-step residual
checks on joint laws, bootstraps parameter intervals, adds diagnostic
broken-power-law univariate checks for knees and non-monotone slices, and gates
any `Cmin` interpretation on iso-loss `Bcrit(L)` readiness. The audit policy is
to fit repo telemetry directly: validation loss is the primary law-fitting
target, benchmark log loss is external transfer validation and repo-facing
ranking evidence, and Kaplan/Chinchilla exponents are not imported from the
papers.

Next sweep ordering:

- first: run the audit and seed/noise checks before interpreting benchmark-loss
  fits; add two extra seeds for `{96x2,128x4,152x5}` at `{2500,5000}` steps so
  residuals can be separated into target noise versus real scaling deviation
- second: redesign `Bcrit(L)` as an iso-loss crossing analysis; run `96x2`,
  `152x5`, and `176x6` over `grad_accum_steps={1,2,4,8}` to `5000` steps with
  validation checkpoints at `{625,1250,2500,5000}`, then fit
  McCandlish-style `Bcrit = Emin / Smin` from equal-validation-loss contours
  rather than a final-only lower envelope
- third: use [#259](https://github.com/bensonlee5/tab-foundry/issues/259) for
  the medium compute-frontier sweep; choose steps from measured
  `train_flops_per_step(N)` across `{72x1,96x2,112x3,128x4,152x5,176x6}`,
  bound steps to `625..10000`, and compare Chinchilla-style
  `L(N,D)=E+A/N^alpha+B/D^beta` against the existing Kaplan-style `L(N,S)`
- last: keep [#260](https://github.com/bensonlee5/tab-foundry/issues/260)
  separate for repetition/curriculum slices with fixed architecture plus
  explicit `unique_task_budget` and `curriculum_id`; fit data-constrained
  effective-data laws instead of merging those rows into the base `N,S,C`
  curve

TF-RD-009 adoption decision: do not use `Bcrit(L)` to derive `Cmin` until the
iso-loss analysis has at least four contour estimates across the redesigned
multi-geometry batch sweep. Broken neural scaling laws are diagnostic tools for
knees or non-monotone slices, not the default compute-optimal frontier.

## Joint Width-Depth Derivation For [#255](https://github.com/bensonlee5/tab-foundry/issues/255)

The first joint width-depth family should be derived in two layers: paper
constraints first, then a repo-local integer-row bridge. The papers constrain
the family shape, but they do not provide a closed-form integer schedule for
the sandwich architecture used here.

### Paper Constraints

- *Scaling Laws for Neural Language Models*:
  - use a smooth, approximately power-law size axis
  - TF-RD-009 adoption decision: use a log-spaced effective-size family rather
    than an ad hoc width-depth grid
- *Training Compute-Optimal Large Language Models*:
  - under fixed compute, parameters and tokens should scale together
  - TF-RD-009 adoption decision: keep `token_budget` fixed in
    [#255](https://github.com/bensonlee5/tab-foundry/issues/255), so this
    branch is explicitly not the compute-optimal frontier
- *Tensor Programs V (muP)*:
  - width is the cleanest first transfer axis
  - TF-RD-009 adoption decision: carry the width-only winner `96x2` forward as
    the in-family baseline rather than re-deriving the whole family from
    parameter count alone
- *Spectral Condition for μP under Width-Depth Scaling*:
  - once depth moves, width and depth must be co-designed together
  - TF-RD-009 adoption decision: use a diagonal co-scaling family instead of
    independent width and depth sweeps

### Repo-Local Size Bridge

The papers do not provide a closed-form sandwich width-depth exchange rule. For
the frozen TF-RD-009 sandwich surface, use the following queue-construction
bridge:

- planning axis:
  - `S(d, L) = L * d^2`
  - interpretation: a monotone family index only, not the final reported
    scaling-law x-axis
- repo-local parameter bridge:
  - `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2`

This bridge is empirical, not paper-claimed. It is for integer row selection
only, not the final reported law fit. It is derived from two sources:

- benchmark-backed width evidence already on `main`:
  - formal external anchor `60x2`:
    - `d_icl=60`
    - `sandwich_layers=2`
    - `total_params=646,970`
  - carried width baseline `96x2`:
    - `d_icl=96`
    - `sandwich_layers=2`
    - `total_params=1,618,286`
  - upper width evidence `128x2`:
    - `d_icl=128`
    - `sandwich_layers=2`
    - `total_params=2,849,422`
- the previously materialized draft joint-family rows that exposed the mixed-depth
  drift:
  - `88x1`: `986,886`
  - `104x3`: `2,419,862`
  - `112x4`: `3,410,046`
  - `128x5`: `5,234,830`
  - `144x6`: `7,615,262`

Why the old bridge changed:

- the width-only `L=2` evidence fits `P_local(d, L) ≈ 88.20 * L * d^2`
- once depth varies, that width-only bridge breaks materially:
  - `88x1` is only `+7.6%` in `L * d^2` versus `60x2`, but `+52.5%` in params
  - `104x3` is only `-1.0%` in `L * d^2` versus `128x2`, but `-15.1%` in params
- TF-RD-009 adoption decision:
  - keep `S(d, L)` only as a planning axis
  - use the affine depth-aware bridge for parameter and VRAM targeting while
    constructing the queue
  - exclude the historical draft points from the final reported scaling-law fit;
    they remain queue-construction context only
  - treat the older `89 * L * d^2` rule as a width-only diagnostic that should
    not be reused for mixed-depth sweep design

### Dense Diagonal Locked For [#255](https://github.com/bensonlee5/tab-foundry/issues/255)

Keep the formal external TF-RD-009 anchor at `60x2`, carry `96x2` as the
in-family baseline, and widen the fixed-budget width-depth family to the dense
diagonal:

- formal external anchor for lane interpretation: `60x2`
- in-family ladder for [#255](https://github.com/bensonlee5/tab-foundry/issues/255):
  - `72x1`
  - `96x2`
  - `112x3`
  - `128x4`
  - `152x5`
  - `176x6`

Derivation:

- formal external anchor:
  - `60x2`
  - parameter target:
    `P_anchor = 646,970`
- lower joint seed:
  - choose `L = 1`
  - solve `P_local(d, 1) = 646,970`, giving `d = 70.6`
  - round to the nearest practical `d_icl` rung: `72`
  - resulting row: `72x1`
  - predicted parameter count: `0.672M`
- carried in-family baseline:
  - `96x2`
  - observed parameter count: `1.618M`
- upper joint seed:
  - match the width-only upper evidence row `128x2` in parameter scale rather
    than in `L * d^2`
  - choose `L = 3`
  - solve `P_local(d, 3) = 2,849,422`, giving `d = 113.0`
  - round to the nearest practical `d_icl` rung: `112`
  - resulting row: `112x3`
  - predicted parameter count: `2.798M`
- ceiling probe:
  - current RTX 8000 planning fit:
    `reserved_gb ≈ 6.47 + 2.36e-6 * params`
  - TF-RD-009 adoption decision: use the retained `rtx8000_44gb` surface more
    meaningfully by targeting an upper row near `32-33 GB` reserved rather than
    stopping around the already-observed `13.23 GB` width-only `128x2` row
  - using the affine parameter bridge, a `32-33 GB` reserved target implies
    roughly `11.1M` parameters
  - choose `L = 6`
  - solve `P_local(d, 6) = 11.1M`, giving `d = 173.9`
  - round to the nearest practical `d_icl` rung: `176`
  - resulting row: `176x6`
  - predicted parameter count: `11.366M`
  - predicted reserved VRAM: `33.29 GB`
- dense upper interpolation:
  - keep the family one-dimensional by interpolating in log-space in predicted
    parameter count between `112x3` and `176x6`
  - for `L = 4`:
    - interpolated parameter target: `4.519M`
    - solve `P_local(d, 4) = 4.519M`, giving `d = 129.2`
    - round to the nearest practical `d_icl` rung: `128`
    - resulting row: `128x4`
    - predicted parameter count: `4.439M`
  - for `L = 5`:
    - interpolated parameter target: `7.055M`
    - solve `P_local(d, 5) = 7.055M`, giving `d = 148.7`
    - round to the nearest practical `d_icl` rung: `152`
    - resulting row: `152x5`
    - predicted parameter count: `7.366M`

The old `144x6` draft no longer survives the corrected fit: the materialized row
lands at only `7.615M` parameters, which maps to roughly `24.44 GB` reserved
under the same VRAM fit, so it is not a genuine ceiling probe.

This keeps [#255](https://github.com/bensonlee5/tab-foundry/issues/255)
theory-constrained but large enough to fit an actual curve:

- external anchor for interpretation: `60x2`
- lower seed from anchor-matched parameter scale: `72x1`
- carried width-transfer baseline: `96x2`
- upper seed matched to the width-only upper evidence row: `112x3`
- near-saturating upper diagonal extension: `128x4`, `152x5`, `176x6`

It also remains honest about what is paper-backed versus repo-local:

- the papers constrain the family shape and what must scale together
- the repo-local bridge and hardware budget model choose the integer rows

### Reported Law-Fit Policy

Now that the dense diagonal has landed benchmark-backed rows, the first
reported fixed-budget TF-RD-009 law fit should use:

- in-family completed rows only:
  - `{72x1, 96x2, 112x3, 128x4, 152x5, 176x6}`
- measured x-axis:
  - benchmark-registry `model_size.total_params`
- ranking metric:
  - matched-budget `final_log_loss_at_matched_regime_budget`
- first fit family:
  - a Kaplan-style power law on the measured parameter counts, with the fitted
    exponent and intercept reported as repo-specific empirical quantities

Do not use the queue-construction bridge, historical draft rows, or incomplete
pending rows as direct inputs to the reported law fit.

## Constraint Budget Table

These tables now summarize the frozen
`hardware_architecture_baselines_v1.json` entry for the retained
`rtx8000_44gb` classification surface. The width-depth family is complete, and
the hardware baseline froze only after corrected [#257](https://github.com/bensonlee5/tab-foundry/issues/257)
rerun `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2`
filtered telemetry-only missing late numbered checkpoints, consumed terminal
`latest.pt` at `global_step=2500`, and beat the carried TF-RD-010 large
clean-control anchor `0.8974410961` at
`final_log_loss_at_matched_regime_budget=0.7436636568`. The large-rung result
remains a gate rather than an additional fitted constraint-model point.

Current frozen formulas:

- planning-axis expression:
  - `S(d, L) = L * d^2`
- current fitted mixed-depth parameter bridge:
  - `P_local(d, L) ≈ 18638.80 + 77.94 * d^2 + 47.93 * L * d^2`
- deprecated width-only diagnostic bridge:
  - `P_local(d, L) ≈ 88.20 * L * d^2` on the historical `L=2` family only
- current RTX 8000 reserved-memory fit from the frozen medium evidence:
  - `reserved_vram_gb ≈ 8.69 + 9.271e-07 * params`
- current RTX 8000 train-wall fit from the frozen medium evidence:
  - `train_wall_seconds ≈ 8298.45 + 2.275e-04 * params`

The historical width-family rows predate first-class `benchmark_timing` and
`inference_timing`, so those timing columns remain intentionally pending for
`60x2` and `96x2`. The completed widened width-depth rows now register those
timings directly.

The reopened upper-family child under
[#269](https://github.com/bensonlee5/tab-foundry/issues/269) keeps these
frozen formulas as the design surface. The checked-in selector under
`reference/system_delta_sweeps/tf_rd_009_width_depth_upper_extension_medium_v1/support/selection_summary.{json,md}`
records three admissible corrected-bridge continuations under the `~40 GB`
target:

- `216x7 -> 272x8`
- `200x7 -> 224x8 -> 256x9`
- `192x7 -> 208x8 -> 224x9 -> 248x10`

It chooses `192x7 -> 208x8 -> 224x9 -> 248x10` because that continuation
maximizes D-optimal information gain on the current validation `L(N,S)` fit
while also reducing projected `alpha_n` and `alpha_s` uncertainty the most.

### Capacity Table

| Row | `d_icl` | `sandwich_layers` | `S = L * d^2` | Predicted `P_local` | Predicted reserved GB | Observed reserved GB | Observed headroom to 44 GB |
|-----|---------|-------------------|---------------|---------------------|-----------------------|----------------------|----------------------------|
| `60x2` | 60 | 2 | 7200 | 0.644M | 9.29 | 8.05 | 35.95 |
| `72x1` | 72 | 1 | 5184 | 0.671M | 9.32 | 8.75 | 35.25 |
| `96x2` | 96 | 2 | 18432 | 1.620M | 10.20 | 10.18 | 33.82 |
| `112x3` | 112 | 3 | 37632 | 2.800M | 11.29 | 11.65 | 32.35 |
| `128x4` | 128 | 4 | 65536 | 4.437M | 12.81 | 14.55 | 29.45 |
| `152x5` | 152 | 5 | 115520 | 7.357M | 15.51 | 16.59 | 27.41 |
| `176x6` | 176 | 6 | 185856 | 11.342M | 19.21 | 17.84 | 26.16 |

### Timing Table

| Row | Predicted train wall seconds | Observed train wall seconds | Observed delta vs `96x2` | Observed benchmark wall seconds | Observed inference mean ms |
|-----|------------------------------|-----------------------------|--------------------------|---------------------------------|----------------------------|
| `60x2` | 8445 | 8486 | -62 | pending | pending |
| `72x1` | 8451 | 8105 | -443 | 1230 | 10.64 |
| `96x2` | 8667 | 8548 | +0 | pending | pending |
| `112x3` | 8936 | 9136 | +587 | 1335 | 20.39 |
| `128x4` | 9308 | 9594 | +1046 | 1391 | 23.91 |
| `152x5` | 9972 | 10154 | +1606 | 1509 | 27.70 |
| `176x6` | 10879 | 10635 | +2087 | 1610 | 24.23 |

### Completed Phase-1 Outcome

- the corrected dense diagonal is now benchmark-backed end to end on
  `tf_rd_009_width_depth_medium_v1`
- `152x5` is the current fixed-budget winner by both final log loss
  (`0.5740`) and final ROC AUC (`0.7351`)
- `176x6` completed cleanly and remains useful upper-family evidence, but it
  did not beat `152x5` on the carried matched-budget objective
- the first registry-backed [#257](https://github.com/bensonlee5/tab-foundry/issues/257)
  rerun stopped at the last retained numbered snapshot because telemetry still
  referenced late numbered checkpoints that were no longer present in the
  reusable artifact tree
- corrected rerun `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2`
  filtered telemetry-only missing late numbered checkpoints, appended terminal
  `latest.pt` at `global_step=2500`, completed `25/25` checkpoint comparisons,
  and finished at `final_log_loss=0.7436636568`,
  `final_brier_score=0.4288940`, and `final_roc_auc=0.7650940`
- the corrected large clean-control comparison is now a keep:
  `delta_final_log_loss=-0.1538` and `delta_final_roc_auc=+0.1327` versus the
  carried anchor `0.8974410961`
- because the large gate passed on the terminal 2500-step artifact,
  `src/tab_foundry/bench/hardware_architecture_baselines_v1.json` now freezes
  `tf_rd_009_rtx8000_44gb_classification_medium_v1` from medium evidence rows
  only; the large validation row remains a gate rather than a constraint-model
  point
- the observed upper-row training VRAM (`16.59 GiB` at `152x5`, `17.84 GiB` at
  `176x6`) still undershot the old width-evidence reserved-memory fit, so the
  frozen baseline now uses the medium-evidence formulas above rather than the
  old conservative queue-construction heuristic
- the old "stop at `176x6`" closeout remains historical evidence for the first
  fixed-budget family only; as of April 13, 2026,
  [#269](https://github.com/bensonlee5/tab-foundry/issues/269) deliberately
  reopens the upper family under the corrected post-`#257` hardware model
- this reopen is law-information-first rather than immediate model-chasing:
  `tf_rd_009_width_depth_upper_extension_medium_v1` runs the selected
  continuation only at the carried fixed-budget gate row first, and
  `tf_rd_009_ns_upper_extension_medium_v1` stays empty until benchmark-backed
  health=`ok` rows earn promotion into the full `{625,1250,2500,5000}` NS
  ladder

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
- keep width and depth as the first-class queue-design axes because they may
  not be interchangeable on the sandwich surface
- fit the first reported law against measured benchmark-registry
  `model_size.total_params` only after completed in-family rows land, and treat
  the fitted exponent/intercept as empirical repo quantities rather than
  imported paper constants

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

## Width-Transfer Execution Summary

The sweep-program design work under
[#140](https://github.com/bensonlee5/tab-foundry/issues/140) is complete, and
[#254](https://github.com/bensonlee5/tab-foundry/issues/254) has now executed
the first fixed-budget TF-RD-009 width-transfer family on the carried medium
multiclass rung.

Execution recap:

- replay requirement:
  - replay the carried TF-RD-024 `sandwich_heads=1` row because the historical
    follow-up queue result was not benchmark-registry-backed on `main`
- formal replay anchor:
  - run id
    `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`
  - `d_icl=60`
  - `final_log_loss=0.6620`
- executed width ladder:
  - `d_icl=48`
  - `d_icl=60` as the replayed anchor
  - `d_icl=96`
  - `d_icl=128`
- row outcomes at matched regime budget:
  - `d_icl=48`: `final_log_loss=0.6939`; clear underperformer versus anchor
  - `d_icl=96`: `final_log_loss=0.6331`; improved log loss, Brier score, and
    ROC AUC with health verdict `ok`
  - `d_icl=128`: `final_log_loss=0.6225`; best raw objective result, but it
    also carried health verdict `warn`, `max_grad_norm=54.6871`, and sharply
    worse legacy BPC/BPF diagnostics
- width-only family conclusion:
  - keep width-only as a live empirical baseline because the upper-width rows
    clearly beat the replay anchor on the matched-regime-budget objective
  - carry `d_icl=96` into
    [#255](https://github.com/bensonlee5/tab-foundry/issues/255) as the explicit
    joint width-depth baseline because it is the cleanest improved row
  - keep `d_icl=128` only as higher-risk upper-width evidence rather than the
    default handoff point

Exact handoff to [#255](https://github.com/bensonlee5/tab-foundry/issues/255):

- keep [#253](https://github.com/bensonlee5/tab-foundry/issues/253) as the
  fixed-budget family epic, but treat
  [#254](https://github.com/bensonlee5/tab-foundry/issues/254) as completed
  execution input rather than the final TF-RD-009 law claim
- keep `60x2` and run id
  `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`
  as the formal external TF-RD-009 anchor for lane-level interpretation; do
  not rerun it inside the joint family
- use `d_icl=96` and run id
  `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`
  as the carried width baseline for the first joint width-depth family
- execute the dense diagonal
  `72x1 -> 96x2 -> 112x3 -> 128x4 -> 152x5 -> 176x6` inside
  [#255](https://github.com/bensonlee5/tab-foundry/issues/255), where:
  - `72x1` comes from matching the formal `60x2` anchor against the empirical
    mixed-depth sandwich parameter bridge
  - `112x3` comes from matching the width-only upper evidence row `128x2`
    against the same empirical bridge
  - `128x4` and `152x5` are intermediate log-spaced parameter-scale rows
    between the `112x3` upper seed and the intended ceiling probe
  - `176x6` is the explicit RTX 8000 capacity-targeted ceiling probe chosen to
    land near the retained `32-33 GB` reserved-memory band under the carried
    repo-local VRAM fit
- keep `token_budget`, `unique_task_budget`, `curriculum_id`, benchmark slice,
  runtime policy, objective metric, and the non-scaling sandwich knobs fixed
- keep `sandwich_heads=1` as the carried TF-RD-024 handoff winner
- treat optimizer transfer as a lower-confidence empirical question that stays
  secondary to the architecture-law read in
  [#255](https://github.com/bensonlee5/tab-foundry/issues/255)
- do not freeze the first
  `src/tab_foundry/bench/hardware_architecture_baselines_v1.json` entry until
  the broadened family is complete enough to choose the best
  matched-regime-budget row among `health=ok` runs on `rtx8000_44gb`
- keep compute-optimal parameter-token work in
  [#259](https://github.com/bensonlee5/tab-foundry/issues/259) and curriculum or
  repeated-data slice work in
  [#260](https://github.com/bensonlee5/tab-foundry/issues/260)

## Hardware-Aware Preferred Architecture State

TF-RD-009 should not leave the "best architecture for this hardware surface" as
an issue comment or W&B memory. It should live in a repo-tracked registry with
the same discipline as the benchmark and control-baseline registries.

Canonical registry:

- `src/tab_foundry/bench/hardware_architecture_baselines_v1.json`

Selection contract:

- hardware identity:
  - `gpu_class + vram_class_gb`
- surface identity:
  - `track`
  - `surface_role`
  - `runtime_profile`
  - benchmark manifest path and associated surface labels
- ranking rule:
  - choose the benchmark-backed row with the best matched-budget objective
    among `health=ok` runs
  - record runtime and VRAM as guardrails and rationale fields, not as the
    ranking metric itself

First TF-RD-009 use:

- [#257](https://github.com/bensonlee5/tab-foundry/issues/257) is now complete:
  corrected rerun `tf_rd_009_large_validation_152x5_v1` consumed the retained
  terminal `latest.pt` checkpoint at step 2500, passed the large-rung gate at
  `final_log_loss_at_matched_regime_budget=0.7436636568` versus the carried
  control `0.8974410961`, and froze the first
  `rtx8000_44gb` medium-classification hardware baseline entry on this branch
- keep the formal external anchor `60x2` for lane interpretation
- keep the carried baseline `96x2` as the fallback preferred architecture if
  the widened family does not produce a cleaner healthy winner
- the canonical large-rung reuse row is
  `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2`,
  benchmarked on `openml_classification_large_v1`; it recorded
  `final_log_loss_at_matched_regime_budget=0.7436636568`, so the freeze gate
  was met on the retained terminal 2500-step artifact
- keep the medium constraint model derived only from the medium evidence rows
  `60x2`, `72x1`, `96x2`, `112x3`, `128x4`, `152x5`, and `176x6`; the large
  validation row is a gate, not an additional fitted point
- include the machine-readable `constraint_model` block so the registry records
  the exact effective-size, parameter, VRAM, train-wall, benchmark-wall, and
  inference-latency formulas together with the evidence rows used to fit them

The issue tree should still preserve these rules:

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
