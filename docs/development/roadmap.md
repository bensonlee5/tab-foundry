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
- `partial`: meaningful building blocks or evidence exist, but the roadmap
  claim is not yet satisfied end to end
- `planned`: clearly scoped and prioritized, but not yet implemented
- `research`: intentionally deferred or gated behind earlier roadmap work
- `retired`: historical item retained only for traceability

## Canonical Planning Metadata

`docs/development/roadmap.md` is the single source of truth for planning state
in `tab-foundry`.

The canonical planning unit is the local roadmap item `TF-RD-###`. External BL
issues can track execution work when needed, but missing BL chains should not
block planning. If another document disagrees with this file, this file is
authoritative.

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

- Scaling predictability comes first.
- Classification remains the anchor workload until the row-first control family
  is stable.
- Architecture conclusions should come from coherent staged surfaces, not from
  piling more overrides onto `nano_exact`.
- Row-first migration work should move one architectural boundary at a time:
  shared surface, grouped tokens, row embedding, column set reasoning, then
  row-level context.
- Many-class, regression, inference handoff, and modality expansion remain
  later unless earlier evidence shows they are blocking the main architecture
  program.

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for implemented work
retained for traceability.

| Rank | Roadmap ID | Item | Status | Milestone |
| ---- | ---------- | ---- | ------ | --------- |
| 0 | TF-RD-000 | Repo foundation and staged-family split | implemented | Implemented |
| 1 | TF-RD-001 | Control freeze and experiment trust | partial | Now |
| 2 | TF-RD-002 | Measurement surfaces for architecture migration | planned | Now |
| 3 | TF-RD-003 | Shared-surface unlock | planned | Now |
| 4 | TF-RD-004 | Tokenization migration | planned | Now |
| 5 | TF-RD-005 | Row-embedding unlock | planned | Now |
| 6 | TF-RD-006 | Column-set integration | planned | Now |
| 7 | TF-RD-007 | Row-level context and QASS attribution | planned | Now |
| 8 | TF-RD-008 | Coherent classification anchor promotion | planned | Next |
| 9 | TF-RD-009 | Scaling-law measurement on the promoted anchor | planned | Next |
| 10 | TF-RD-010 | Many-class promotion on the row-first base | research | Next |
| 11 | TF-RD-011 | Repo-wide enablers and contract fidelity | partial | Next |
| 12 | TF-RD-012 | Regression, inference handoff, and later modalities | research | Later |

## Current Capability Matrix

| Objective / Claim | Current State | Evidence In Repo | Current Gap | Roadmap IDs |
| --- | --- | --- | --- | --- |
| Frozen PFN-style control exists | `implemented` | `tabfoundry_simple`, `stage=nano_exact`, benchmark comparison tooling, and prior-trained PFN-facing lanes already exist | The current large-anchor hybrid line is still easy to confuse with the intended destination | `TF-RD-001` |
| Coherent row-first migration ladder exists in code | `partial` | The staged recipe ladder already encodes `shared_norm -> prenorm_block -> grouped_tokens -> row_cls_pool -> column_set -> qass_context -> many_class` | The roadmap and sweep system do not yet treat that ladder as the primary migration program | `TF-RD-003`, `TF-RD-004`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007` |
| Architecture comparisons are attributable | `partial` | Isolated row-CLS, TFCol, and QASS evidence exists on compact surfaces | Negative evidence is easy to overgeneralize because stage-local telemetry and matched controls are still incomplete | `TF-RD-002`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007` |
| One promoted row-first classification anchor exists | `planned` | The ingredients exist in `tabfoundry_staged`, but no coherent row-first anchor has been promoted | The active architecture target is still split across compact-ladder evidence and the current large hybrid line | `TF-RD-008` |
| Scaling-law work targets the right architecture | `partial` | Tuning and benchmark-adjacent tooling already exist | There is no canonical scaling artifact path on a promoted row-first anchor yet | `TF-RD-009` |
| Many-class extends the active backbone cleanly | `partial` | The staged family already includes `many_class` and the reusable machinery exists | Many-class is not yet tied to a promoted row-first classification base | `TF-RD-010` |
| Repo-wide data, preprocessing, and export surfaces can support the migration | `partial` | Manifest-backed training, shared preprocessing work, and v3 export scaffolding already exist | Corpus provenance, producer-fidelity cleanup, and downstream contract hardening are still unfinished | `TF-RD-011` |
| Regression and downstream runtime handoff are ready | `research` | The repo has clear placeholders and partial bundle/runtime infrastructure | Regression is intentionally removed, and runtime handoff should follow the promoted classification base rather than run ahead of it | `TF-RD-012` |

## Current Implementation Baseline

This roadmap assumes the following repo truths:

- `tabfoundry_staged` is the only active architecture-development surface.
- `tabfoundry_simple` is the frozen exact PFN-style anchor.
- the staged family already contains the intended migration ladder through
  `shared_norm`, `grouped_tokens`, `row_cls_pool`, `column_set`,
  `qass_context`, and `many_class`
- current negative row-CLS, TFCol, and QASS evidence was gathered mostly on
  compact or PFN-adjacent anchor surfaces and should not be treated as a final
  rejection of the row-first direction
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

- Status: `partial`
- Milestone: `Now`
- Goal: make the PFN control lane and the row-first target lane explicit so the
  roadmap stops mixing benchmark trust with architecture aspiration
- Current state:
  - PFN-style controls exist and benchmark comparison tooling is in place
  - the current large-anchor hybrid line still risks being read as the intended
    destination
  - benchmark trust remains bounded by no-missing/control-lane discipline
- Required work:
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

- Status: `planned`
- Milestone: `Now`
- Goal: add the telemetry needed to interpret row-first architecture changes
  structurally rather than by end metrics alone
- Current state:
  - activation tracing exists for the cell-table stack
  - row-first stage boundaries are under-instrumented
  - runtime and memory cost are not yet first-class promotion signals by stage
- Required work:
  - emit and persist `post_column_encoder`, `post_row_pool`, and
    `post_context_encoder`
  - add stage-local runtime and memory summaries for column, row, and context
    stages
  - make these metrics available in sweep artifacts and result cards
- Exit criteria:
  - row-first rows can be compared on quality, stability, runtime, and memory
  - later TFCol and QASS rows are attributable without relying only on wandb
    charts

### TF-RD-003: Shared-Surface Unlock

- Status: `planned`
- Milestone: `Now`
- Goal: move the active architecture program off the PFN-only `nano` encoder
  path and onto the coherent shared staged surface
- Current state:
  - `shared_norm` and `prenorm_block` already exist as public staged recipes
  - tokenizer changes remain ineffective while `feature_encoder=nano`
  - the active large-anchor work still leans on `nano_exact` plus overrides
- Required work:
  - establish `shared_norm` as the first mandatory step out of the PFN control
    lane
  - establish `prenorm_block` as the first shared-surface backbone change
  - stop treating `nano_exact` overrides as the main path for TabICL-inspired
    work
- Exit criteria:
  - the architecture target lane starts from a shared surface
  - later tokenization and row-first rows are tested only where they are
    actually active

### TF-RD-004: Tokenization Migration

- Status: `planned`
- Milestone: `Now`
- Goal: evaluate grouped tokenization as the first true row-first preparation
  step on the shared surface
- Current state:
  - `grouped_tokens` already exists in the staged recipe ladder
  - compact-ladder evidence showed that tokenizer changes under the nano encoder
    were not isolatable
- Required work:
  - evaluate `grouped_tokens` directly on top of the shared/prenorm path
  - treat grouped tokenization as a structural milestone rather than a minor
    ablation
  - use grouped-token evidence to bound later row-CLS and TFCol interpretation
- Exit criteria:
  - the repo has a benchmark and stability read on grouped tokens as part of the
    migration ladder
  - row-first work no longer assumes the old scalar-per-feature PFN token
    surface

### TF-RD-005: Row-Embedding Unlock

- Status: `planned`
- Milestone: `Now`
- Goal: determine whether the staged family can form useful row embeddings on
  the intended shared/grouped surface
- Current state:
  - `row_cls_pool` exists as a coherent staged recipe
  - old compact-surface row-CLS evidence was strongly negative
  - that evidence is entangled with the old PFN-adjacent surface and should not
    be generalized too far
- Required work:
  - evaluate `row_cls_pool` only on top of the shared/grouped migration line
  - keep adequacy checks bounded to row-encoder capacity knobs after the main
    row-first surface is established
  - interpret old row-CLS negative evidence as compact-surface evidence, not as
    a blanket rejection
- Exit criteria:
  - either row embeddings work on the intended migration surface, or they fail
    with enough evidence to redirect the architecture target

### TF-RD-006: Column-Set Integration

- Status: `planned`
- Milestone: `Now`
- Goal: decide whether explicit column-set reasoning belongs in the promoted
  row-first line
- Current state:
  - `column_set` already exists as a staged recipe
  - old compact-surface TFCol evidence was near-neutral, stable, and expensive
  - the old row pool and compact surface limited how much that result could say
- Required work:
  - evaluate `column_set` only after row embeddings are viable
  - keep inducing-count and depth adequacy follow-ups bounded and explicit
  - interpret TFCol as part of the row-first bundle, not as a bolt-on to the PFN
    anchor
- Exit criteria:
  - TFCol is either promoted into the active row-first line or deferred with
    clear cost-versus-value evidence

### TF-RD-007: Row-Level Context And QASS Attribution

- Status: `planned`
- Milestone: `Now`
- Goal: determine whether row-level context helps, and whether QASS helps beyond
  plain row-level context
- Current state:
  - `qass_context` already exists as a staged recipe
  - QASS components already exist as reusable modules
  - old compact-surface QASS evidence was stable but slightly negative and did
    not cleanly separate context value from added depth
- Required work:
  - compare row-level `plain` context against row-level `qass` context on the
    same row-first surface
  - pair each QASS row with a matched non-QASS added-depth control
  - keep QASS optional by construction
- Exit criteria:
  - the repo has a direct answer to whether row-level context is useful
  - the repo separately has a direct answer to whether QASS earns its added cost

### TF-RD-008: Coherent Classification Anchor Promotion

- Status: `planned`
- Milestone: `Next`
- Goal: promote one coherent row-first classification anchor and stop treating
  the architecture target as an open set of hybrid lines
- Current state:
  - the staged ladder exists
  - no row-first classification anchor has been promoted
  - the active architecture story is still split across multiple partial lines
- Required work:
  - choose the first promotable row-first surface from the migration ladder
  - likely candidates are a `column_set`-based line or a `qass_context`-based
    line, not the current hybrid PFN row-CLS line
  - update research defaults and promotion language to point at the selected
    anchor
- Exit criteria:
  - one row-first classification anchor is named, benchmarked, and treated as
    the active architecture target

### TF-RD-009: Scaling-Law Measurement On The Promoted Anchor

- Status: `planned`
- Milestone: `Next`
- Goal: move size, depth, width, and compute scaling work onto the promoted
  row-first classification anchor
- Current state:
  - tuning and benchmark-adjacent tooling already exist
  - scaling-law intent is clear, but there is no canonical artifact path on the
    right architecture yet
- Required work:
  - run size sweeps on the promoted row-first anchor rather than the hybrid
    diagnostic line
  - emit comparable compute, parameter-count, and benchmark artifacts
  - use these artifacts as the basis for future architecture decisions
- Exit criteria:
  - the repo can fit scaling trends on the promoted architecture rather than on
    a transient bridge surface

### TF-RD-010: Many-Class Promotion On The Row-First Base

- Status: `research`
- Milestone: `Next`
- Goal: extend the promoted row-first backbone into the existing `many_class`
  path
- Current state:
  - the staged family already contains `many_class`
  - the hierarchical many-class machinery already exists
  - the backbone it should inherit from has not yet been promoted
- Required work:
  - re-evaluate `many_class` on top of the promoted row-first base
  - keep many-class as an extension of the same staged family
  - avoid opening a separate architecture lane for many-class
- Exit criteria:
  - many-class uses the promoted row-first backbone and has benchmark-facing
    evidence

### TF-RD-011: Repo-Wide Enablers And Contract Fidelity

- Status: `partial`
- Milestone: `Next`
- Goal: keep the repo-wide data, preprocessing, and export surfaces healthy
  enough to support the architecture program without letting them dominate it
- Current state:
  - manifest-backed training and evaluation exist
  - shared preprocessing and v3 export groundwork exist
  - corpus provenance, preprocessing producer fidelity, and downstream contract
    hardening are still incomplete
- Required work:
  - finish dagzoo handoff and path-independent corpus identity
  - stabilize shared preprocessing and export fidelity
  - keep these changes architecture-compatible and family-neutral
- Exit criteria:
  - the row-first architecture program can rely on trustworthy data and export
    contracts without forcing a second planning track

### TF-RD-012: Regression, Inference Handoff, And Later Modalities

- Status: `research`
- Milestone: `Later`
- Goal: rebuild prediction-mode breadth only after the promoted row-first
  classification anchor is stable
- Current state:
  - classification is the only active supported mode
  - regression has been intentionally removed
  - runtime handoff and later modalities remain deferred
- Required work:
  - rebuild regression as a staged-family extension on the promoted row-first
    base
  - advance separate-runtime handoff only after classification/export contracts
    settle
  - keep time series, text-conditioned inputs, and other later modalities out of
    the critical path
- Exit criteria:
  - regression and runtime handoff build on the promoted staged base rather than
    running ahead of it

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
- The current large-anchor hybrid line is diagnostic evidence, not the intended
  architecture destination.
