# Mission-Aligned Roadmap (2026Q1)

This is the canonical roadmap for `tab-foundry`.

The repo-wide plan is now architecture-first:

- keep one frozen PFN-style control lane for trust and comparison
- evolve `tabfoundry_staged` toward a coherent row-first classifier inspired by
  TabICLv2
- stay free to borrow the best components from TabPFN and other tabular models
  rather than aiming for literal TabICLv2 parity
- defer regression, extended modalities, and broader runtime handoff until the
  row-first classification anchor is coherent

Related docs:

- design decisions and repo structure: `docs/development/design-decisions.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- architecture reference: `docs/development/model-architecture.md`
- architecture deltas: `docs/development/architecture-deltas.md`
- workflow runbooks: `docs/workflows.md`
- inference/export contract: `docs/inference.md`
- reference index: `reference/README.md`
- evidence appendix: `reference/evidence.md`

## Status Labels

- `implemented`: available in current code and wired into the canonical
  workflow surface
- `completed`: scoped work finished with a recorded decision or evidence
  package, even if it does not imply a promoted default
- `partial`: meaningful building blocks or evidence exist, but the roadmap
  claim is not yet satisfied end to end
- `planned`: clearly scoped and prioritized, but not yet implemented
- `research`: intentionally deferred or gated behind earlier roadmap work
- `retired`: historical item retained only for traceability

## Canonical Planning Metadata

`docs/development/roadmap.md` is the single source of truth for planning state
in `tab-foundry`.

The canonical planning unit is the local roadmap item `TF-RD-###`. External BL
issues or GitHub issues can track execution work when needed, but missing
external issue chains should not block planning. If another document disagrees
with this file, this file is authoritative.

## Program Statement

The repo now operates with two architecture lanes:

- PFN control lane:
  - `tabfoundry_simple`
  - `tabfoundry_staged` with `stage=nano_exact`
  - used to preserve benchmark comparability and experiment trust
- active architecture target lane:
  - a row-first staged classifier inspired primarily by TabICLv2
  - allowed to borrow components from TabPFN or other references when they fit
    better
  - must remain one coherent `tabfoundry_staged` surface rather than a second
    live model family

Important non-goals for this roadmap:

- do not treat the current `nano_exact + prenorm + row_cls` hybrid line as the
  long-term destination
- do not bundle regression into the first row-first promotion push
- do not make QASS structurally mandatory
- do not rewrite the repo around a second architecture family

## Prioritization Lens

- Close TF-RD-008 first, then move quickly onto harder or broader post-008
  surfaces before committing to a heavy scaling pass.
- Classification remains the anchor workload until the row-first control family
  is stable.
- Architecture conclusions should come from coherent staged surfaces, not from
  piling more overrides onto `nano_exact`.
- Row-first migration work should move one architectural boundary at a time:
  shared surface, small-class/test-self bridge, grouped tokens, row
  embedding, column set reasoning, then row-level context.
- After the binary anchor is coherent, the next deliberate fronts should make
  the research surface less saturating and more realistic: synthetic data,
  missingness, class imbalance, training adequacy, many-class, and regression.
- If those harder or broader surfaces still leave the model hard to separate or
  obviously underfit, open a later architecture-surface adequacy pass before
  relying on scaling-law work as the main next source of evidence.
- Class imbalance is still not sufficiently tested on the current benchmark
  surfaces because the bundles only enforce `min_minority_class_pct = 2.5`
  rather than defining an explicit skew ladder.
- Training adequacy should become explicit before the repo keeps treating
  optimizer and schedule choices as sweep-local notes instead of a cross-cutting
  decision surface.
- Scaling-law work should follow once the promoted anchor has at least one
  harder post-008 surface fixed well enough to make scaling conclusions
  decision-relevant and any needed selective architecture-surface expansion is
  scoped.

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for implemented work
retained for traceability.

| Rank | Roadmap ID | Item | Status | Milestone |
| ---- | ---------- | ---- | ------ | --------- |
| 0 | TF-RD-000 | Repo foundation and staged-family split | implemented | Implemented |
| 1 | TF-RD-001 | Control freeze and experiment trust | implemented | Implemented |
| 2 | TF-RD-002 | Measurement surfaces for architecture migration | implemented | Implemented |
| 3 | TF-RD-003 | Shared-surface unlock | implemented | Implemented |
| 4 | TF-RD-011 | Repo-wide enablers and contract fidelity | implemented | Implemented |
| 5 | TF-RD-004 | Tokenization migration | completed | Now |
| 6 | TF-RD-005 | Row-embedding unlock | completed | Now |
| 7 | TF-RD-006 | Column-set integration | completed | Now |
| 8 | TF-RD-007 | Row-level context and QASS attribution | completed | Now |
| 9 | TF-RD-008 | Coherent classification anchor promotion | partial | Next |
| 10 | TF-RD-013 | Dagzoo synthetic-data efficacy on the promoted anchor | planned | Next |
| 11 | TF-RD-014 | Missingness robustness on the promoted anchor | planned | Next |
| 12 | TF-RD-017 | Class-imbalance robustness on the promoted anchor | planned | Next |
| 13 | TF-RD-018 | Training-surface adequacy on the promoted anchor | planned | Next |
| 14 | TF-RD-010 | Many-class promotion on the row-first base | planned | Next |
| 15 | TF-RD-015 | Regression rebuild on the promoted row-first base | planned | Next |
| 16 | TF-RD-016 | Architecture surface adequacy and selective expansion | research | Later |
| 17 | TF-RD-009 | Scaling-law measurement on the promoted anchor | planned | Next |
| 18 | TF-RD-012 | Inference handoff and later modalities | research | Later |

## Dependency Graph

```mermaid
flowchart TD
    RD000["TF-RD-000 ✅<br/>Repo foundation"]

    RD001["TF-RD-001 ✅<br/>Control freeze &<br/>experiment trust"]
    RD002["TF-RD-002 ✅<br/>Measurement surfaces"]
    RD011["TF-RD-011 ✅<br/>Repo-wide enablers<br/>(independent)"]

    RD003["TF-RD-003 ✅<br/>Shared-surface unlock"]
    RD004["TF-RD-004<br/>Tokenization migration"]
    RD005["TF-RD-005<br/>Row-embedding unlock"]
    RD006["TF-RD-006<br/>Column-set integration"]
    RD007["TF-RD-007<br/>Row-level context & QASS"]
    RD008{{"TF-RD-008<br/>PROMOTION GATE<br/>Coherent anchor"}}
    RD013["TF-RD-013<br/>Dagzoo synthetic<br/>data efficacy"]
    RD014["TF-RD-014<br/>Missingness<br/>robustness"]
    RD017["TF-RD-017<br/>Class-imbalance<br/>robustness"]
    RD018["TF-RD-018<br/>Training-surface<br/>adequacy"]
    RD010["TF-RD-010<br/>Many-class promotion"]
    RD015["TF-RD-015<br/>Regression rebuild"]
    RD016["TF-RD-016<br/>Architecture surface<br/>adequacy"]
    RD009["TF-RD-009<br/>Scaling-law measurement"]
    RD012["TF-RD-012<br/>Inference handoff &<br/>later modalities"]

    RD001 --> RD002
    RD001 --> RD003
    RD002 --> RD003
    RD003 --> RD004
    RD004 --> RD005
    RD005 --> RD006
    RD006 --> RD007
    RD007 --> RD008
    RD008 --> RD013
    RD008 --> RD014
    RD008 --> RD017
    RD013 --> RD018
    RD014 --> RD018
    RD017 --> RD018
    RD008 --> RD010
    RD008 --> RD015
    RD018 --> RD016
    RD016 --> RD009
    RD008 --> RD012

    classDef done fill:#d4edda,stroke:#28a745,color:#155724;
    classDef readyNow fill:#fff3cd,stroke:#ffc107,color:#856404;
    classDef chain fill:#e8f0fe,stroke:#4285f4,color:#1a3d6e;
    classDef gate fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    classDef later fill:#f3e8ff,stroke:#7c3aed,color:#3b1f6e;

    class RD000,RD001,RD002,RD003,RD011 done;
    class RD004,RD005,RD006,RD007 chain;
    class RD008 gate;
    class RD013,RD014,RD017,RD018,RD010,RD015 readyNow;
    class RD016,RD009,RD012 later;
```

Critical path: **003 → 004 → 005 → 006 → 007 → 008**. 000, 001, 002, 003, and
011 are implemented; 004, 005, 006, and 007 are completed evidence steps. 008
remains the active promotion gate. The deliberate post-008 fronts are
TF-RD-013, TF-RD-014, TF-RD-017, TF-RD-018, TF-RD-010, and TF-RD-015. If those
harder or broader surfaces still leave the staged family low-signal, TF-RD-016
is the deliberate follow-on architecture-depth track before the repo treats
TF-RD-009 scaling work as the main next source of architecture evidence.

## Current Capability Matrix

| Objective / Claim | Current State | Evidence In Repo | Current Gap | Roadmap IDs |
| --- | --- | --- | --- | --- |
| Frozen PFN-style control exists | `implemented` | `tabfoundry_simple`, `stage=nano_exact`, benchmark comparison tooling, and prior-trained PFN-facing lanes already exist | The current large-anchor hybrid line is still easy to confuse with the intended destination | `TF-RD-001` |
| Coherent row-first migration ladder exists in code | `partial` | The staged recipe ladder already encodes `shared_norm -> prenorm_block -> small_class_head -> test_self -> grouped_tokens -> row_cls_pool -> column_set -> qass_context -> many_class`; `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1` locks the grouped-token replay, `sd_row_embedding_attribution_v2_01_delta_row_embeddings_no_context_v2_v1` closes the row-embedding unlock, `row_embedding_attribution_v3` completes the TFCol × QASS factorization, `sd_qass_tfcol_adequacy_v1_03_delta_qass_context_tfcol_heads4_v1_v1` wins the medium-bundle adequacy screen, and `qass_tfcol_large_no_missing_validation_v1` passed its large no-missing validator narrowly | The remaining gap is no longer row-first ladder shape but final regime coverage: the missing-data generalization check on `nanotabpfn_openml_binary_large_v1.json` still needs to land before promotion language can fully close | `TF-RD-003`, `TF-RD-004`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007` |
| Architecture comparisons are attributable | `partial` | Grouped-token replay, v2/v3 matched controls, the TFCol adequacy sweep, and the large no-missing validation now separate row embeddings, plain context, TFCol-only, QASS-only, and the selected `qass + tfcol_heads4` calibration line | The remaining open comparison is no longer medium-bundle attribution, but whether the post-008 fronts use harder and broader surfaces that reduce saturation risk | `TF-RD-002`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007`, `TF-RD-008` |
| One promoted row-first classification anchor exists | `partial` | The calibration-first candidate `row_cls + qass + tfcol_heads4` now passed the large no-missing validator narrowly, while `row_cls + qass + no tfcol` remains the ROC-oriented fallback | The repo still needs the missing-data generalization check on `nanotabpfn_openml_binary_large_v1.json` before it can claim a single promoted default anchor across regimes | `TF-RD-008` |
| Harder post-008 data surfaces can be exercised | `partial` | Dagzoo CLI-to-manifest handoff, path-independent corpus identity, and canonical no-missing versus allow-missing binary bundles already exist | There is no post-008 program yet that decides whether dagzoo or explicit missingness should become a canonical harder research surface on the promoted anchor | `TF-RD-011`, `TF-RD-013`, `TF-RD-014` |
| Class-imbalance robustness is meaningfully exercised | `partial` | Current benchmark bundles enforce `min_minority_class_pct = 2.5`, so the repo already excludes degenerate class-balance cases | There is no dedicated imbalance-focused bundle ladder, imbalance-oriented reporting contract, or explicit decision on the promoted anchor under materially skewed priors | `TF-RD-017` |
| Training adequacy is handled coherently across fronts | `partial` | Sweep-local `parameter_adequacy_plan` notes exist throughout the research metadata, and bounded adequacy sweeps such as `qass_tfcol_adequacy_v1` already exist | Optimizer, schedule, step-budget, batch, and clipping adequacy are not yet tracked as one roadmap workstream with a canonical decision surface | `TF-RD-018` |
| Many-class evaluation can start on the row-first base | `partial` | The staged family already includes `many_class`, reusable machinery exists, and `nanotabpfn_openml_classification_small_v1.json` provides a benchmark-facing multiclass bundle | Many-class still lacks a promoted row-first benchmark ladder, adequacy sweeps, and a keep/defer decision | `TF-RD-010` |
| Regression rebuild can start on the staged base | `research` | Regression metrics and benchmark-bundle normalization support already exist in the repo | There is no active staged regression program, canonical regression bundle, or staged regression head/loss contract | `TF-RD-015` |
| The staged surface is broad enough for future adequacy work before adding new knobs | `partial` | Tokenization already includes `scalar_per_feature`, `scalar_per_feature_nan_mask`, and `shifted_grouped`; token count is already adjustable through `feature_group_size`; norms, widths, depths, row CLS count, TFCol inducing count, context FF expansion, dropout, and clipping are already exposed | The repo still needs a deliberate future decision on whether harder-surface adequacy can be done on the existing surface alone or whether selective exposure of hardcoded choices such as activation family, row or column FF expansion, QASS scaler capacity, grouped shift recipe, or many-class threshold is warranted | `TF-RD-016` |
| Scaling-law work targets the right architecture and surface | `planned` | Tuning and benchmark-adjacent tooling already exist | Scaling on the current simple binary regime risks low-signal conclusions until the promoted anchor is paired with at least one harder post-008 surface | `TF-RD-009` |
| Repo-wide data, preprocessing, and export surfaces can support the migration | `implemented` | The dagzoo CLI-to-manifest boundary, canonical dagzoo dataset identity, and export/reference preprocessing fidelity were completed under TF-RD-011 | New dagzoo work is now about efficacy on the promoted anchor, not about reopening the boundary layer | `TF-RD-011` |
| Inference handoff and later modalities are ready | `research` | The repo has clear placeholders and partial bundle/runtime infrastructure | Inference handoff and later modalities should follow the promoted classification base and not absorb regression ownership again | `TF-RD-012` |

## Current Implementation Baseline

This roadmap assumes the following repo truths:

- `tabfoundry_staged` is the only active architecture-development surface.
- `tabfoundry_simple` is the frozen exact PFN-style anchor.
- the staged family already contains the intended migration ladder through
  `shared_norm`, `prenorm_block`, `small_class_head`, `test_self`,
  `grouped_tokens`, `row_cls_pool`, `column_set`, `qass_context`, and
  `many_class`
- grouped-token replay
  `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`
  is the canonical predecessor for the current row-first line
- `row_embedding_attribution_v2` established that row embeddings help on the
  grouped-token replay surface, while plain row-level context does not justify
  promotion on that same base
- `row_embedding_attribution_v3` established that TFCol alone is negative on the
  row-first base, `qass + no tfcol` is near-tied with the row-embedding base,
  and `qass + tfcol` wins on calibration while losing ROC
- `qass_tfcol_adequacy_v1` established that `tfcol_heads4` is the only TFCol
  adequacy winner worth carrying forward, while `inducing64` and `layers1`
  remain negative evidence
- `qass_tfcol_large_no_missing_validation_v1` established a narrow large
  no-missing validation pass for `row_cls + qass + tfcol_heads4`; the
  remaining gate is missing-data generalization on
  `nanotabpfn_openml_binary_large_v1.json`
- closed `TF-RD-011` work already completed the dagzoo CLI-to-manifest
  boundary, path-independent canonical dagzoo identity, and export/reference
  preprocessing fidelity
- `missingness_followup` exists only as a hybrid-diagnostic precursor on the
  old prenorm foundation and should not be treated as the closure path for
  row-first missingness robustness
- current benchmark bundles already enforce `min_minority_class_pct = 2.5`, but
  that is only a floor and should not be treated as a real class-imbalance
  program
- current training adequacy work is fragmented across sweep-local
  `parameter_adequacy_plan` notes and isolated follow-up sweeps rather than a
  single roadmap workstream
- the repo already ships `nanotabpfn_openml_classification_small_v1.json`, so
  many-class benchmark scaffolding exists even though the row-first many-class
  program does not
- regression metrics plumbing and bundle normalization support exist in parts of
  the repo, but there is no active staged regression program yet
- the staged model surface is already broad enough for a future adequacy pass:
  tokenizers include `scalar_per_feature`, `scalar_per_feature_nan_mask`, and
  `shifted_grouped`; token count is adjustable through `feature_group_size`;
  and norms, widths, depths, row CLS count, TFCol inducing count, context FF
  expansion, dropout, and clipping are already exposed
- several architecture choices remain deliberately hardcoded, including feed
  forward activation family, row or column FF expansion, QASS scaler capacity,
  grouped-token shift recipe, and the many-class routing threshold; these
  should only be exposed selectively if harder surfaces still remain low-signal
- `Muon` is already supported in the optimizer surface and belongs in training
  adequacy work rather than architecture-surface expansion
- the current large-anchor `nano_exact + prenorm + row_cls` line is useful as a
  diagnostic bridge, but not yet a promoted architecture target

## Roadmap Items

### TF-RD-000: Repo Foundation And Staged-Family Split

- Status: `implemented`
- Milestone: `Implemented`
- Goal: preserve the current role-based repo organization and the split between
  the frozen PFN control and the active staged family
- Current state:
  - `tabfoundry_staged` is the active family
  - `tabfoundry_simple` is the frozen anchor
  - reusable model pieces already live under `model/components`
- Exit criteria:
  - this remains the stable base for all later roadmap work

### TF-RD-001: Control Freeze And Experiment Trust

- Status: `implemented`
- Milestone: `Implemented`
- Goal: make the PFN control lane and the row-first target lane explicit so the
  roadmap stops mixing benchmark trust with architecture aspiration
- Current state:
  - the PFN control lane is named explicitly as `tabfoundry_simple` plus
    `tabfoundry_staged` with `stage=nano_exact`
  - the current large-anchor hybrid line is documented as diagnostic rather
    than promotable
  - the canonical medium-bundle control-baseline id `cls_benchmark_linear_v2`
    now resolves through the prior-trained staged `nano_exact` anchor
    `01_nano_exact_md_prior_parity_fix_binary_medium_v1`
- Implemented contract:
  - keep `tabfoundry_simple` and `stage=nano_exact` as the frozen PFN control
    lane
  - document the current large-anchor hybrid line as diagnostic rather than
    promotable
  - preserve one canonical control interpretation surface for benchmark claims
- Exit criteria:
  - one named PFN control lane exists
  - one explicitly non-promoted hybrid diagnostic lane exists
  - benchmark-facing interpretation is tied to the control lane rather than the
    hybrid line

### TF-RD-002: Measurement Surfaces For Architecture Migration

- Status: `implemented`
- Milestone: `Implemented`
- Goal: add the telemetry needed to interpret row-first architecture changes
  structurally rather than by end metrics alone
- Current state:
  - the exact-prior diagnostic lane already emits rich module-gradient and
    activation telemetry
  - the canonical architecture-screen surface still lacks that same telemetry
    parity in the regular trainer
  - row-first stage boundaries are partially traced, but `post_context_encoder`
    is still missing
- Required work:
  - emit and persist `post_column_encoder`, `post_row_pool`, and
    `post_context_encoder` on the regular training path
  - write `gradient_history.jsonl` and `telemetry.json` from the regular
    trainer, not only the prior-dump loop
  - sync selected stage-local stability summaries to wandb and expose them in
    sweep artifacts and result cards
  - defer per-stage runtime and memory profiling until later architecture
    tickets prove those costs are decision-critical
- Exit criteria:
  - regular training emits the same class of module-gradient and activation
    telemetry as the exact-prior path
  - row-first rows can be compared on quality and stage-local stability without
    relying only on raw wandb charts
  - runtime and memory are explicitly out of scope for closing TF-RD-002

### TF-RD-003: Shared-Surface Unlock

- Status: `implemented`
- Milestone: `Implemented`
- Goal: move the active architecture program off the PFN-only `nano` encoder
  path and onto the coherent shared staged surface
- Current state:
  - `shared_surface_bridge_v1` established the stage-native architecture-screen
    bridge from `nano_exact` through `shared_norm` and `prenorm_block`
  - the canonical grouped-token predecessor is locked as
    `sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1`,
    registered at `2026-03-20T00:17:09Z` (`2026-03-19` in Los Angeles)
  - `small_class_head` and `test_self` remain explicit historical bridge rows,
    but neither displaced `prenorm_block` as the default grouped-token handoff
- Implemented contract:
  - treat the public shared-surface stages as the primary migration program
  - keep tokenizer work off the old `feature_encoder=nano` lane where it is not
    active
  - carry grouped-token work forward from the locked `prenorm_block` handoff
    rather than reopening optional bridge rows by default
- Exit criteria:
  - the architecture target lane starts from a shared surface
  - one explicit shared-surface handoff row is locked for grouped-token work
  - later tokenization and row-first rows are tested only where they are
    actually active

### TF-RD-004: Tokenization Migration

- Status: `completed`
- Milestone: `Implemented`
- Goal: evaluate grouped tokenization as the first true row-first preparation
  step on the shared surface
- Current state:
  - `grouped_tokens` already exists in the staged recipe ladder
  - compact-ladder evidence showed that tokenizer changes under the nano encoder
    were not isolatable
  - `shared_surface_bridge_v1` closed TF-RD-003 and locked `prenorm_block` as
    the canonical grouped-token predecessor
  - `small_class_head` and `test_self` remain optional historical bridge rows,
    not the default TF-RD-004 handoff
  - the architecture-screen grouped-token benchmark `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2`
    was mixed, so `grouped_token_stability_probe_v1` was executed on March 19,
    2026 against that locked anchor
  - the traced anchor rerun `sd_grouped_token_stability_probe_v1_01_delta_anchor_activation_trace_baseline_v1`
    and the no-trace warmup-decay row
    `sd_grouped_token_stability_probe_v1_03_delta_training_linear_warmup_decay_v1`
    converged to the same grouped-token read: final log loss about `0.4002`,
    final Brier about `0.2618`, final ROC AUC about `0.741`, clipped-step
    fraction `0.0012`, and zero drift
  - the no-warmup decay row
    `sd_grouped_token_stability_probe_v1_02_delta_training_linear_decay_v1`
    improved final ROC AUC to `0.7540`, but it lost on log loss/Brier and pushed
    `max_grad_norm` to `9.99`, so it is not the preferred grouped-token surface
  - TF-RD-004 now has an explicit keep decision: grouped tokens stay on the
    migration path, with `prior_linear_warmup_decay` as the preferred adequacy
    surface for the benchmark-facing replay
  - the benchmark-facing grouped-token replay
    `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`
    is now registered on the architecture-screen lane and lands the probe's
    no-trace warmup-decay surface as the canonical grouped-token predecessor
- Implemented contract:
  - keep `prenorm_block` as the locked TF-RD-004 anchor for attributable
    comparisons, but carry later row-first work forward from
    `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`
  - treat the warmup-decay grouped-token replay, not the old scalar-per-feature
    token path, as the predecessor for TF-RD-005, TF-RD-006, and TF-RD-007
  - preserve the adequacy probe as supporting evidence rather than the
    benchmark-facing handoff itself
- Exit criteria:
  - the grouped-token keep decision is recorded from the mixed
    `tokenization_migration_v1` result plus the warmup-decay stability follow-up
  - the winning grouped-token replay is registered on the benchmark-facing lane
  - later row-first work inherits grouped tokens as the working token surface
    rather than assuming scalar-per-feature PFN tokens

### TF-RD-005: Row-Embedding Unlock

- Status: `completed`
- Milestone: `Now`
- Goal: determine whether the staged family can form useful row embeddings on
  the intended shared/grouped surface
- Current state:
  - `row_cls_pool` exists as a coherent staged recipe on the grouped-token
    replay surface
  - `row_embedding_attribution_v2` closed the grouped-token row-embedding
    question with the no-context row
    `sd_row_embedding_attribution_v2_01_delta_row_embeddings_no_context_v2_v1`
  - the paired plain-context row did not improve that row-embedding base, so
    old compact-surface row-CLS evidence is no longer the main blocker
- Completed evidence:
  - TF-RD-005 was anchored on grouped-token replay
    `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`,
    not on `prenorm_block` or the older scalar-token path
  - row pooling was isolated first on that grouped-token replay before the
    bundled public `row_cls_pool` stage was interpreted
  - the result package now separates row embeddings from plain row-level
    context on the intended migration surface
- Exit criteria:
  - the repo has a direct answer to whether row embeddings help on the intended
    migration surface
  - satisfied: row embeddings help on the grouped-token replay surface
  - satisfied: plain row-level context does not improve that row-embedding base

### TF-RD-006: Column-Set Integration

- Status: `completed`
- Milestone: `Now`
- Goal: decide whether explicit column-set reasoning belongs in the promoted
  row-first line
- Current state:
  - `column_set` already exists as a staged recipe
  - `delta_column_set_no_context_v3` is negative evidence for default TFCol
    alone on the grouped-token row-first base
  - `sd_qass_tfcol_adequacy_v1_03_delta_qass_context_tfcol_heads4_v1_v1` was
    the medium-bundle adequacy winner and the only TFCol row worth carrying
    forward
  - `qass_tfcol_large_no_missing_validation_v1` then validated
    `delta_qass_context_tfcol_heads4_v1` against `delta_qass_no_column_v3` on
    `nanotabpfn_openml_binary_large_no_missing_v1.json` with a narrow pass:
    final log loss `-0.0013818`, final Brier `-0.0004977`, and final ROC AUC
    `-0.0047989` versus the no-TFCol control
- Completed evidence:
  - default TFCol alone is resolved negative evidence on the row-first base and
    should not be reopened as a standalone promotion candidate
  - TFCol is justified only under the validated
    `row_cls + qass + tfcol_heads4` calibration-first line
  - the remaining follow-up is missing-data generalization, not another TFCol
    adequacy or TFCol-alone screen
- Exit criteria:
  - satisfied: TFCol is promoted only under the validated
    `qass + tfcol_heads4` line, not as a standalone default row-first module
  - satisfied: the repo now has an explicit ROC-oriented fallback against
    `qass + no tfcol`

### TF-RD-007: Row-Level Context And QASS Attribution

- Status: `completed`
- Milestone: `Now`
- Goal: determine whether row-level context helps, and whether QASS helps beyond
  plain row-level context
- Current state:
  - `qass_context` already exists as a staged recipe
  - QASS components already exist as reusable modules
  - `row_embedding_attribution_v2` showed that plain row-level context does not
    justify promotion over the no-context row-embedding base
  - `row_embedding_attribution_v3` showed that `qass + no tfcol` is near-tied
    with the row-embedding base, TFCol alone is bad, and `qass + tfcol` wins on
    calibration while losing ROC
  - `qass_tfcol_large_no_missing_validation_v1` then showed that
    `row_cls + qass + tfcol_heads4` keeps the calibration win over
    `row_cls + qass + no tfcol` on the larger no-missing bundle while staying
    inside the `-0.005` ROC guardrail
- Completed evidence:
  - plain row-level context is resolved negative evidence on the row-first path
  - QASS remains optional by construction, but the calibration-first line is
    now `row_cls + qass + tfcol_heads4`
  - `row_cls + qass + no tfcol` remains the explicit ROC-oriented fallback
  - the remaining follow-up is missing-data generalization, not reopening
    medium-bundle attribution
- Exit criteria:
  - satisfied: the repo has an explicit calibration-first recommendation and an
    explicit ROC-oriented fallback for `qass + no tfcol` versus
    `qass + tfcol_heads4`

### TF-RD-008: Coherent Classification Anchor Promotion

- Status: `partial`
- Milestone: `Next`
- Goal: promote one coherent row-first classification anchor and stop treating
  the architecture target as an open set of hybrid lines
- Current state:
  - the staged ladder exists
  - no row-first classification anchor has been promoted
  - the calibration-first candidate is
    `row_cls + qass + tfcol_heads4`
  - `row_cls + qass + no tfcol` remains the ROC-oriented fallback
  - `qass_tfcol_large_no_missing_validation_v1` passed narrowly, so the
    remaining blocker is missing-data generalization rather than large
    no-missing validation
  - the architecture docs still describe the pre-settlement surface and must be
    refreshed once the anchor decision is locked
- Required work:
  - run the missing-data generalization check on
    `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json` using
    `row_cls + qass + no tfcol` as the ROC-oriented control and
    `row_cls + qass + tfcol_heads4` as the calibration-first candidate
  - if the missing-data check keeps the calibration win while the ROC gap stays
    small, promote `row_cls + qass + tfcol_heads4` as the default row-first
    classification anchor
  - if the missing-data check fails or the ROC penalty widens materially, keep
    a split recommendation rather than forcing a single promoted anchor
  - publish the settled architecture reference in
    `docs/development/model-architecture.md`, including the selected blocks,
    their exact inputs and outputs, their purpose, the decision-relevant
    discarded alternatives, and the tradeoffs of the selected design
  - refresh `docs/development/architecture-deltas.md` so it documents the
    settled promoted anchor or explicit split outcome rather than the old
    anchor-under-investigation framing
  - update research defaults and promotion language only after the benchmark
    and documentation gates close
- Exit criteria:
  - one row-first classification anchor, or an explicit split recommendation,
    is named, benchmarked, documented, and treated as the active architecture
    target

### TF-RD-010: Many-Class Promotion On The Row-First Base

- Status: `planned`
- Milestone: `Next`
- Goal: extend the promoted row-first backbone into the existing `many_class`
  path
- Current state:
  - the staged family already contains `many_class`
  - the hierarchical many-class machinery already exists
  - `nanotabpfn_openml_classification_small_v1.json` already exists as a
    benchmark-facing multiclass bundle
  - the many-class path is implemented but still unvalidated on the promoted
    row-first base
- Required work:
  - confirm the canonical multiclass bundle, control baseline, and promoted
    row-first backbone for the first many-class program
  - run the first many-class benchmark and adequacy sweeps on top of the
    promoted row-first base rather than reopening older hybrid lines
  - keep many-class as an extension of the same staged family
  - avoid opening a separate architecture lane for many-class
  - record an explicit keep/defer decision for the row-first many-class path
- Exit criteria:
  - many-class uses the promoted row-first backbone, has benchmark-facing
    evidence, and no longer sits only as untested scaffolding

### TF-RD-011: Repo-Wide Enablers And Contract Fidelity

- Status: `implemented`
- Milestone: `Implemented`
- Goal: keep the repo-wide data, preprocessing, and export surfaces healthy
  enough to support the architecture program without letting them dominate it
- Current state:
  - manifest-backed training and evaluation exist
  - the reusable dagzoo CLI-to-manifest boundary is complete
  - canonical dagzoo dataset identity is path-independent for canonical corpora
  - export and reference-consumer preprocessing fidelity now track the resolved
    preprocessing surface
- Implemented contract:
  - keep dagzoo as a CLI-and-artifact boundary rather than importing dagzoo
    internals into `tab-foundry`
  - preserve path-independent manifest identity for canonical dagzoo corpora
  - keep export and reference-consumer preprocessing policy aligned with the
    resolved training/runtime surface
- Exit criteria:
  - satisfied: the row-first architecture program can rely on trustworthy data
    and export contracts without forcing a second planning track

### TF-RD-012: Inference Handoff And Later Modalities

- Status: `research`
- Milestone: `Later`
- Goal: advance separate-runtime handoff and genuinely later modalities only
  after the promoted row-first classification base is stable
- Current state:
  - classification remains the only active supported prediction mode
  - runtime handoff and later modalities remain deferred
- Required work:
  - advance separate-runtime handoff only after classification/export contracts
    settle
  - keep time series, text-conditioned inputs, and other later modalities out of
    the critical path
- Exit criteria:
  - inference handoff and later modalities build on the promoted staged base
    rather than running ahead of it

### TF-RD-013: Dagzoo Synthetic-Data Efficacy On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: decide whether dagzoo-generated corpora materially improve training
  difficulty, architecture discrimination, or final quality on the promoted
  row-first anchor
- Current state:
  - the dagzoo handoff boundary is complete through closed TF-RD-011 work
  - dagzoo smoke and manifest identity are no longer the main blocker
  - there is no benchmark-facing post-008 data-source comparison program yet
- Required work:
  - define the canonical dagzoo corpus variants, provenance rules, and first
    post-008 data-source sweep ladder
  - compare the current corpus against dagzoo-generated corpora on the promoted
    row-first anchor rather than on older hybrid surfaces
  - record whether dagzoo becomes a canonical harder research surface or stays
    auxiliary
- Exit criteria:
  - the repo has an explicit keep/defer decision on dagzoo as a post-008
    training-data source

### TF-RD-014: Missingness Robustness On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: decide how the promoted row-first family should handle missing-valued
  inputs in training and evaluation
- Current state:
  - `missingness_followup` exists, but it is anchored on the older stabilized
    prenorm hybrid surface rather than the row-first line
  - the repo already has separate no-missing and allow-missing benchmark bundle
    contracts
  - there is no row-first missingness recommendation yet
- Required work:
  - re-anchor missingness work on the promoted row-first base
  - separate missing-token or missingness-mechanism adequacy from synthetic
    missingness training and from benchmark-surface evaluation
  - decide whether explicit missingness handling belongs in the default
    row-first line or remains an optional robustness variant
- Exit criteria:
  - the repo has a benchmark-backed missingness recommendation for the row-first
    family

### TF-RD-017: Class-Imbalance Robustness On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: decide how the promoted row-first family behaves under materially
  skewed class priors
- Current state:
  - current benchmark bundles only enforce `min_minority_class_pct = 2.5`
  - there is no dedicated imbalance-focused bundle ladder yet
  - benchmark-facing reporting is still centered on ROC AUC, log loss, and
    Brier score
- Required work:
  - define the canonical imbalance-focused binary bundle or bundle-selection
    ladder on the promoted row-first anchor
  - define benchmark-facing comparison and reporting that includes PR AUC,
    average precision, and balanced accuracy alongside ROC/log loss/Brier
  - measure the promoted anchor first without class reweighting or focal-style
    loss changes
  - only if the baseline read is weak, run bounded weighted-loss or focal-loss
    follow-up work
- Exit criteria:
  - the repo has a benchmark-backed keep/defer decision on the promoted
    row-first line under class imbalance
  - the repo has an explicit imbalance metric and reporting contract rather
    than relying only on the current general binary bundle metrics

### TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: turn optimizer, schedule, budget, and runtime adequacy from sweep-local
  notes into a deliberate cross-cutting workstream
- Current state:
  - adequacy work already exists in `parameter_adequacy_plan` notes and
    isolated sweeps, but there is no canonical training-surface epic
  - `Muon` is already supported in the optimizer surface but has no clean
    roadmap home
  - current adequacy reads are still scattered across frontier-specific sweeps
- Required work:
  - define the canonical adequacy ladder on the promoted anchor and the first
    harder post-008 fronts
  - start with optimizer family, LR/schedule/warmup shape, step budget, batch
    size or accumulation, and clipping policy
  - include `schedulefree_adamw`, `adamw`, and `muon` in the bounded optimizer
    family comparison
  - keep architecture changes out of this epic and treat device/runtime
    constraints only insofar as they change interpretation of optimizer or
    schedule adequacy
- Exit criteria:
  - the repo has an explicit default training surface for next-tier fronts
  - the repo has a clear rule for when optimizer or schedule adequacy must be
    resolved before interpreting architecture outcomes

### TF-RD-015: Regression Rebuild On The Promoted Row-First Base

- Status: `planned`
- Milestone: `Next`
- Goal: rebuild regression as a staged-family extension on top of the promoted
  row-first base
- Current state:
  - regression support is intentionally removed from the active repo surface
  - regression metrics and bundle normalization support already exist in parts
    of the repo
  - there is no active staged regression program, canonical regression bundle,
    or staged regression head/loss contract
- Required work:
  - define the canonical regression benchmark surface or surfaces for the staged
    family
  - define the staged regression head and loss contract on the promoted
    row-first base
  - build the first architecture and adequacy sweep ladder for regression while
    staying on the same staged family rather than opening a second architecture
    lane
- Exit criteria:
  - regression has a benchmark-facing staged baseline and a bounded roadmap for
    promotion or deferral

### TF-RD-016: Architecture Surface Adequacy And Selective Expansion

- Status: `research`
- Milestone: `Later`
- Goal: revisit architecture depth only if the promoted row-first family still
  proves hard to separate or obviously underpowered on harder post-008
  surfaces, using the existing config surface first and exposing new knobs only
  selectively
- Current state:
  - the staged surface is already broad enough to support meaningful future
    adequacy work without immediately adding more model fields
  - tokenization already includes `scalar_per_feature`,
    `scalar_per_feature_nan_mask`, and `shifted_grouped`
  - token count and grouping are already adjustable through
    `feature_group_size`
  - norms, widths, depths, row CLS count, TFCol inducing count, context FF
    expansion, dropout, and clipping are already exposed
  - several architecture choices remain hardcoded and should be revisited only
    if harder surfaces remain low-signal
  - optimizer adequacy work, including `Muon`, stays out of this epic because
    it already belongs to the training surface rather than the model-config
    surface
- Required work:
  - Phase 1: existing-surface adequacy on harder post-008 surfaces
    - make tokenization the first explicit subtrack, covering
      `feature_group_size`, tokenizer choice, and grouped-token usage on
      coherent shared-surface rows
    - keep tokenizer work off the `feature_encoder=nano` lane where tokenizer
      changes are ineffective
    - evaluate already-exposed row, column, and context width-depth-capacity
      knobs on harder post-008 surfaces before adding new model fields
    - record an explicit decision on whether the current staged surface is
      sufficient for future architecture-depth work
  - Phase 2: selective surface expansion only if Phase 1 remains low-signal
    - consider bounded additions such as `activation_family`,
      `tfcol_ff_expansion`, `tfrow_ff_expansion`,
      `qass_scaler_hidden_dim`, `group_shift_recipe`, and
      `many_class_threshold`
    - expose a hardcoded architecture choice only when it is likely to matter
      across multiple future regimes, can be compared on one coherent staged
      surface, and can be represented with a small bounded option set
    - do not open a generic “everything configurable” program
- Exit criteria:
  - the repo has an explicit keep/defer decision on whether the current staged
    architecture surface is sufficient on harder post-008 regimes
  - any newly exposed architecture knobs are bounded, justified, and tied to a
    coherent staged comparison surface rather than broad config expansion

### TF-RD-009: Scaling-Law Measurement On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: move size, depth, width, and compute scaling work onto the promoted
  row-first classification anchor only after the repo has at least one harder
  post-008 research surface
- Current state:
  - tuning and benchmark-adjacent tooling already exist
  - scaling-law intent is clear, but scaling on the current simple binary regime
    risks low-signal conclusions because recent architecture deltas are already
    close on that surface
  - there is no canonical scaling artifact path yet on a promoted row-first
    anchor paired with a harder post-008 surface
- Required work:
  - close TF-RD-008 and lock the promoted row-first anchor or the explicit
    split recommendation
  - finish at least one harder post-008 front, starting with TF-RD-013 or
    TF-RD-014, before using scaling results as architecture evidence
  - if those harder surfaces still leave architecture separation low-signal,
    finish TF-RD-016 before treating scaling as the main next architecture
    program
  - run size sweeps on the promoted row-first anchor rather than the hybrid
    diagnostic line, and do so on the chosen harder post-008 surface
  - emit comparable compute, parameter-count, and benchmark artifacts
  - use these artifacts as the basis for future architecture decisions
- Exit criteria:
  - the repo can fit scaling trends on the promoted architecture under a
    post-008 surface that is harder or broader than the current simple binary
    no-missing regime

## Acceptance Gates For Architecture Promotion

The architecture target lane should promote rows only when all of the following
hold:

- the row is coherent as a staged surface rather than an ad hoc override pile
- the row is stable on training telemetry, not only acceptable on final metrics
- the row is benchmark-neutral-or-better against the relevant control surface
- added runtime and memory cost are justified by the gain
- the interpretation is attributable against matched controls rather than
  speculative

## Planning Defaults And Assumptions

- TabICLv2 is the strongest directional reference for the active architecture
  target, but not a literal reproduction target.
- TabPFN remains the control lineage and a legitimate donor for specific
  components.
- QASS remains optional.
- Classification remains the anchor workload until the row-first classification
  base is promoted.
- After TF-RD-008, harder or broader surfaces should come before large scaling
  passes whenever the current binary regime risks saturation.
- The current large-anchor hybrid line is diagnostic evidence, not the intended
  architecture destination.
