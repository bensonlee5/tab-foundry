# Roadmap (202604)

Use this roadmap to understand which questions are active now, which surfaces
are frozen, and what evidence the repo still needs before promotion.

The repo-wide plan is now sandwich-first:

- keep one frozen PFN-style control lane for trust and comparison
- treat `tabfoundry_sandwich` as the active classification architecture target
  and scaling-prep family
- use dagzoo-backed many-class plus missingness as the first anti-saturation
  classification regime before the first scaling fit
- test steering-derived corpus fronts only after that first carried dagzoo
  slice exists
- make bounded kernel/runtime and VRAM optimization a hard prerequisite to
  later scaling so every scale ladder inherits one measured runtime policy
- defer regression from the first classification scaling plan

Use these alongside this roadmap:

- design decisions and repo structure: `docs/development/design-decisions.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- dataset curation and license gate: `docs/development/dataset-curation.md`
- architecture reference: `docs/development/model-architecture.md`
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

The roadmap now has one active architecture-development lane:

- frozen control lane:
  - `tabfoundry_simple`
  - `tabfoundry_staged` with `stage=nano_exact`
  - used only for benchmark comparability and experiment trust
- historical incumbent reference:
  - `tabfoundry_staged`
  - useful as a comparison surface, but no longer the focus of roadmap
    sequencing
- active development lane:
  - `tabfoundry_sandwich`
  - fixed-latent `y` / byte-array `x` Perceiver-style classifier
  - owns the current architecture simplification, dagzoo transfer,
    many-class/missingness, runtime, and scaling work

Important non-goals for this roadmap:

- do not treat the current `nano_exact + prenorm + row_cls` hybrid line as the
  long-term destination
- do not assume the incumbent staged reference line is the final architecture
  destination either; it is the current benchmark carry-forward surface
- do not bundle regression into the first row-first promotion push
- do not make QASS structurally mandatory
- do not proliferate many parallel live architecture families beyond the frozen
  control, the incumbent staged reference line, and the current sandwich
  candidate line

## Prioritization Lens

- Classification remains the anchor workload while the sandwich family is
  tested on harder post-008 regimes.
- The next useful architecture evidence should come from coherent sandwich
  surfaces, not from extending the older staged line.
- After TF-RD-016 closeout, the next deliberate front should make the research
  surface less saturating and more realistic: a benchmark-defined
  classification regime where `dagzoo` owns synthetic training fronts,
  `tab-realdata-hub` owns medium and large validation manifests, and
  `tab-foundry` owns the evolved sandwich model and sweep contracts.
- If those harder or broader surfaces still leave the model hard to separate or
  obviously underfit, open a later architecture-surface adequacy pass before
  relying on scaling-law work as the main next source of evidence.
- Class imbalance is still not sufficiently tested on the current benchmark
  surfaces because the bundles still only enforce
  `min_minority_class_pct = 2.5` rather than defining an explicit skew ladder.
- Harder real-data ladders should keep one canonical OpenML baseline where the
  benchmark tooling is already native, but the validation-bundle source of
  truth now lives in `tab-realdata-hub` rather than in local `tab-foundry`
  fixtures. `dagzoo` remains the synthetic-data lane under TF-RD-013 rather
  than an external real-data source.
- The deliberate post-008 execution order is now:
  TF-RD-016 closes on the modest sandwich evolution and benchmark-definition
  handoff; TF-RD-010 then freezes the first medium and large classification
  validation program, with `dagzoo` synthetic training fronts feeding
  `tab-realdata-hub` materialized manifests; TF-RD-017 class-imbalance work
  proceeds as a side robustness lane on the same family; TF-RD-021 then tests
  whether steering-derived dagzoo corpus fronts improve that benchmark-defined
  slice; TF-RD-022 performs the kernel, runtime, and VRAM tuning needed for
  reliable scaling on the kept front; TF-RD-009 then fits the first scaling law
  on that carried classification slice.
- Low-level questions such as norm family or placement, learned special-token
  initialization scale, QASS scaler capacity, and activation family belong
  under TF-RD-016 after the earlier adequacy and harder-surface gates are in
  place, not as free-floating anchor-settlement work.
- Many-class, regression, and runtime handoff should build on that
  deconfounded post-008 base rather than competing with the earlier gates.
- Runtime, VRAM, and any bounded kernel tuning should finish before the first
  scaling fit so classification ladders inherit one measured 80 GB A100-safe
  policy instead of discovering memory limits during the scale study.
- Scaling-law design should start only after the repo has a simplified sandwich
  parent, one carried dagzoo curriculum slice, runtime telemetry and artifact
  integrity, and at least one missingness read on the same classification
  family.

## Canonical Priority Queue

This queue is intentionally sandwich-focused. Historical staged/control work is
summarized later instead of occupying the active queue.

| Rank | Roadmap ID | Item | Status | Milestone |
| ---- | ---------- | ---- | ------ | --------- |
| 1 | TF-RD-021 | Steering-derived dagzoo corpus fronts on the classification-first sandwich target | planned | Next |
| 2 | TF-RD-017 | Class-imbalance robustness on the classification-first sandwich target | planned | Next |
| 3 | TF-RD-022 | Training runtime and VRAM efficiency before classification scaling | planned | Next |
| 4 | TF-RD-009 | Scaling-law design and measurement on the classification-first sandwich target | planned | Next |
| 5 | TF-RD-014 | Missingness robustness on the classification-first sandwich target | planned | Next |
| 6 | TF-RD-015 | Regression rebuild deferred from the classification-first scaling plan | research | Later |
| 7 | TF-RD-012 | Inference handoff and later modalities | research | Later |

## Dependency Graph

```mermaid
flowchart TD
    HIST["Historical baseline<br/>TF-RD-000 through TF-RD-020<br/>control, staged, dagzoo, and closeout evidence"]
    RD016["TF-RD-016 / TF-RD-021B<br/>Freeze simplified<br/>sandwich parent"]
    DAG["Carry frozen sandwich<br/>parent onto dagzoo"]
    RD010["TF-RD-010<br/>Many-class + missingness<br/>dagzoo gate"]
    RD021["TF-RD-021<br/>Steering-derived<br/>corpus fronts"]
    RD017["TF-RD-017<br/>Class-imbalance<br/>side lane"]
    RD022["TF-RD-022<br/>Kernel/runtime & VRAM<br/>pre-scaling gate"]
    RD014["TF-RD-014<br/>Missingness<br/>follow-up"]
    RD015["TF-RD-015<br/>Regression rebuild<br/>(deferred)"]
    RD012["TF-RD-012<br/>Inference handoff &<br/>later modalities"]
    RD009["TF-RD-009<br/>Scaling-law design &<br/>measurement"]

    HIST --> RD016
    RD016 --> DAG
    DAG --> RD010
    RD010 --> RD021
    RD010 --> RD014
    RD010 --> RD017
    RD021 --> RD022
    RD022 --> RD009
    RD016 --> RD015
    RD016 --> RD012

    classDef hist fill:#e5e7eb,stroke:#6b7280,color:#111827;
    classDef done fill:#d1fae5,stroke:#059669,color:#064e3b;
    classDef readyNow fill:#fff3cd,stroke:#ffc107,color:#856404;
    classDef gate fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    classDef later fill:#f3e8ff,stroke:#7c3aed,color:#3b1f6e;

    class HIST hist;
    class RD010,RD016 done;
    class RD009,RD021,RD022,RD014,RD017 readyNow;
    class RD012,RD015 later;
```

Current path: **TF-RD-010 trusted rerun → TF-RD-021 → TF-RD-022 → TF-RD-009**.

- TF-RD-016 is now completed historical context: issue
  [#178](https://github.com/bensonlee5/tab-foundry/issues/178) closes on the
  decision to evaluate the next sandwich phase through benchmark definition and
  modest head evolution rather than more simplification-only evidence.
- TF-RD-010 is active again: issue
  [#52](https://github.com/bensonlee5/tab-foundry/issues/52), issue
  [#99](https://github.com/bensonlee5/tab-foundry/issues/99), and child issues
  [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
  [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
  [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
  [#200](https://github.com/bensonlee5/tab-foundry/issues/200) remain historical
  setup context, while issue [#202](https://github.com/bensonlee5/tab-foundry/issues/202)
  plus issues [#205](https://github.com/bensonlee5/tab-foundry/issues/205),
  [#203](https://github.com/bensonlee5/tab-foundry/issues/203), and
  [#204](https://github.com/bensonlee5/tab-foundry/issues/204) now define the
  trusted rerun and refactor path on the same benchmark contract.
- TF-RD-010 still ranks rows by `final_bpc_at_matched_regime_budget`,
  treating BPC as the normalized log-loss view while calibration, runtime, and
  stability remain guardrails, but the reset state is now canonical until the
  trusted reruns land on the corrected sandwich and training surface.
- TF-RD-021 then decides whether any steering-derived dagzoo corpus front
  replaces the carried sandwich dagzoo training front before runtime/kernel
  tuning.
- TF-RD-022 is the next hard pre-scaling gate after steering: it must hand back
  one measured kernel/runtime policy before broader scaling fits.
- TF-RD-009 only starts after those gates are closed and fits the first law on
  that carried multiclass slice under matched regime budget.

Parallel/later lanes are intentionally off that main path:

- TF-RD-014 is now a follow-on missingness robustness lane after the first
  many-class plus missingness gate rather than a blocker to the first scaling
  fit.
- TF-RD-017 is the preferred side robustness lane during the many-class plus
  missingness push now that TF-RD-010 has fixed the first carried contract,
  but it is still not a blocker for the first scaling fit.
- TF-RD-015 regression and TF-RD-012 inference handoff/later modalities remain
  later work.

## Current Capability Matrix

| Objective / Claim | Current State | Evidence In Repo | Current Gap | Roadmap IDs |
| --- | --- | --- | --- | --- |
| Frozen PFN-style control exists | `implemented` | `tabfoundry_simple`, `stage=nano_exact`, and the prior-trained PFN-facing benchmark lane are all stable | Keep that lane clearly separate from the architecture target | `TF-RD-001` |
| Sandwich is the primary classification candidate | `implemented` | `tabfoundry_sandwich` is landed, the compact hybrid replay is benchmarked, the first knob screen plus bounded width/head follow-up both kept the compact control, the completed removal-first package under [#184](https://github.com/bensonlee5/tab-foundry/issues/184) retained that anchor, and TF-RD-016 now closes on a bounded direct-multiclass head evolution | Judge the evolved sandwich family on the TF-RD-010 benchmark program rather than reopening simplification-first work | `TF-RD-016`, `TF-RD-021A`, `TF-RD-021B`, `TF-RD-010` |
| Harder synthetic classification fronts are runnable | `implemented` | Dagzoo manifest/export fidelity is complete, TF-RD-013 settled the representative medium surface, and TF-RD-020 settled harder-front winners that can seed the new sandwich benchmark program | Freeze the benchmark-defined dagzoo training fronts, then choose whether steering improves that first carried slice | `TF-RD-011`, `TF-RD-013`, `TF-RD-020`, `TF-RD-016`, `TF-RD-010`, `TF-RD-021` |
| Runtime and VRAM are measurable | `partial` | Training and registry artifacts now preserve runtime-summary and regime-budget fields, and the repo already has bf16/checkpointing-capable runtime plumbing | TF-RD-022 still needs to turn that into one explicit 80 GB A100-safe kernel/runtime policy on the carried sandwich dagzoo slice | `TF-RD-022` |
| Benchmark-backed classification validation contract is fixed, but trusted execution is reset | `partial` | `many_class` is implemented, the sandwich evolution config fixes FiLM plus `sandwich_summary_tokens_per_axis=3`, `tab-realdata-hub` issue [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) owns the medium and large validation manifests under `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, TF-RD-010 child issues [#197](https://github.com/bensonlee5/tab-foundry/issues/197), [#198](https://github.com/bensonlee5/tab-foundry/issues/198), [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and [#200](https://github.com/bensonlee5/tab-foundry/issues/200) froze the missing baselines plus corpora, and successor issues [#202](https://github.com/bensonlee5/tab-foundry/issues/202), [#205](https://github.com/bensonlee5/tab-foundry/issues/205), [#203](https://github.com/bensonlee5/tab-foundry/issues/203), and [#204](https://github.com/bensonlee5/tab-foundry/issues/204) now track the trusted rerun path | TF-RD-010 still needs trusted medium and large reruns before later lanes can inherit canonical benchmark evidence again | `TF-RD-010`, `TF-RD-021`, `TF-RD-014`, `TF-RD-017` |
| Follow-on missingness and imbalance robustness remain open | `partial` | Missing-permitting binary bundles exist, and the current bundle policy already excludes degenerate minority-class cases | TF-RD-014 remains later missingness follow-up, while TF-RD-017 still needs an explicit side-lane imbalance ladder on the same sandwich family | `TF-RD-014`, `TF-RD-017` |
| Regression and later modalities are deferred | `research` | Partial bundle/runtime scaffolding exists | They should not absorb attention from the classification-first path | `TF-RD-015`, `TF-RD-012` |
| Scaling-law work has the needed metadata path | `planned` | Artifacts now preserve resolved sandwich specs plus runtime/regime-budget metadata, and TF-RD-010 has fixed the first benchmark-defined classification contract | TF-RD-009 now waits on the TF-RD-010 trusted rerun, the TF-RD-021 steering decision, and the TF-RD-022 runtime gate on that contract | `TF-RD-009`, `TF-RD-010`, `TF-RD-021`, `TF-RD-022` |

## Current Implementation Baseline

This roadmap assumes the following repo truths:

- `tabfoundry_simple` and `tabfoundry_staged` with `stage=nano_exact` remain
  the frozen PFN-style control lane.
- `tabfoundry_sandwich` exists as the primary classification architecture
  candidate; the initial replay, knob screen, bounded width/head follow-up,
  and removal-first package are complete, and the compact hybrid anchor remains
  the kept parent after [#184](https://github.com/bensonlee5/tab-foundry/issues/184).
- dagzoo manifest identity, export/reference preprocessing fidelity, and the
  one-way data boundary are part of the baseline rather than active blockers.
- the representative post-008 synthetic training-data surface is
  `tf_rd_013_dagzoo_shape_aware_size_medium_v1`.
- many-class scaffolding exists, and the next harder carried classification
  target is now a dagzoo-backed many-class plus missingness slice; regression
  and later inference/runtime handoff are still not part of the first
  classification-scaling path.

## Roadmap Items

### Historical Summary

- TF-RD-000 through TF-RD-013 established the control lane, the staged
  row-first reference, the shared repo/data contracts, and the representative
  dagzoo training surface.
- TF-RD-018 and TF-RD-020 remain historical staged-control evidence only:
  partial training-surface closeout plus harder dagzoo corpus-front winners.
- The detailed historical record remains in completed issues, sweep artifacts,
  and [reference/evidence.md](../../reference/evidence.md); the sections below
  focus on active sandwich development and later follow-up lanes.

Legacy wording note:

- some TF-RD item titles still use earlier row-first or promoted-anchor
  phrasing so existing roadmap links remain stable
- the dependency order, current-state bullets, and required-work bullets below
  are the authoritative description of the active sandwich-first path

### TF-RD-010: Benchmark-Defined Multiclass Evolution On The Classification-First Sandwich Target

- Status: `partial`
- Milestone: `Next`
- Goal: define the first benchmark-backed classification evaluation program for
  the evolved sandwich family using explicit hub-owned bundle policy rather
  than local fixture assumptions
- Current state:
  - issue [#52](https://github.com/bensonlee5/tab-foundry/issues/52) and issue
    [#99](https://github.com/bensonlee5/tab-foundry/issues/99) now serve as
    historical umbrella and execution context, while issue
    [#202](https://github.com/bensonlee5/tab-foundry/issues/202) is the active
    trusted-rerun umbrella
  - historical child issues
    [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
    [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
    [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
    [#200](https://github.com/bensonlee5/tab-foundry/issues/200) define the
    TF-RD-010 corpora and freeze the missing baselines; successor issues
    [#205](https://github.com/bensonlee5/tab-foundry/issues/205) and
    [#203](https://github.com/bensonlee5/tab-foundry/issues/203) now own the
    trusted medium and large reruns
  - issue [#204](https://github.com/bensonlee5/tab-foundry/issues/204) is the
    required sandwich refactor follow-up and lands before any trusted TF-RD-010
    rerun is recorded as canonical evidence
  - `tab-realdata-hub` issue
    [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) is now the
    upstream dependency for medium and large classification validation bundles
    and materialized manifests
  - the evolved sandwich benchmark config uses FiLM,
    `sandwich_summary_tokens_per_axis=3`, `many_class_base=10`, and a direct
    multiclass head
  - the synthetic TF-RD-010 corpora and upstream hub bundles now align on
    `min_classes=2`, `max_classes=10`; the upstream hub bundles also fix
    `max_missing_pct=20.0`
  - the repo now has materialized validation manifests under the local
    benchmark-manifest output root, with the legacy local output ids
    `nanotabpfn_openml_classification_medium_v1` and
    `nanotabpfn_openml_classification_large_v1`
- What remains fixed:
  - `dagzoo` now owns the explicit TF-RD-010 control, MCAR, MAR, and MNAR
    corpora through `tf_rd_010_dagzoo_medium_control_v1`,
    `tf_rd_010_missingness_mcar_v1`, `tf_rd_010_missingness_mar_v1`, and
    `tf_rd_010_missingness_mnar_v1`
  - `tab-foundry` froze the legacy baseline ids
    `cls_benchmark_linear_multiclass_medium_v1` and
    `cls_benchmark_linear_multiclass_large_v1` against the hub-backed medium
    and large manifests
  - the `dagzoo -> tab-realdata-hub -> tab-foundry` benchmark contract remains
    fixed and does not need to be redefined
- Reset state:
  - the previously recorded medium and large sweep executions are invalidated by
    later training and sandwich correctness fixes and are no longer canonical
    evidence
  - on-disk run directories, W&B runs, and old closed execution issues remain as
    historical artifacts only
  - the trusted rerun path now flows through issues
    [#202](https://github.com/bensonlee5/tab-foundry/issues/202),
    [#205](https://github.com/bensonlee5/tab-foundry/issues/205),
    [#203](https://github.com/bensonlee5/tab-foundry/issues/203), and
    [#204](https://github.com/bensonlee5/tab-foundry/issues/204)
- Exit criteria:
  - satisfied: the repo has one explicit medium-plus-large classification benchmark
    contract on the evolved sandwich family
  - satisfied: the hub-backed classification contract is no longer only
    untested scaffolding on the first scaling path
  - open: trusted medium and large reruns must re-establish canonical keep/defer
    evidence on the corrected sandwich and training surface
  - open: later lanes can inherit TF-RD-010 again only after those trusted
    reruns land

### TF-RD-012: Inference Handoff And Later Modalities

- Status: `research`
- Milestone: `Later`
- Goal: advance separate-runtime handoff and genuinely later modalities only
  after the classification-first sandwich base is stable
- Current state:
  - classification remains the only active supported prediction mode
  - runtime handoff and later modalities remain deferred
- Required work:
  - advance separate-runtime handoff only after the classification base settles
  - keep time series, text-conditioned inputs, and other later modalities out
    of the current path
- Exit criteria:
  - inference handoff and later modalities build on the classification-first
    sandwich base
    rather than running ahead of it

### TF-RD-014: Missingness Robustness On The Classification-First Sandwich Target

- Status: `planned`
- Milestone: `Next`
- Goal: deepen missingness robustness after the first many-class plus
  missingness gate is established on the carried sandwich family
- Current state:
  - `missingness_followup` exists, but it is anchored on the older stabilized
    prenorm hybrid surface rather than the carried sandwich family
  - the repo already has separate no-missing and allow-missing benchmark bundle
    contracts
  - issue [#97](https://github.com/bensonlee5/tab-foundry/issues/97) remains
    the missingness epic, and completed TF-RD-010 now records the first
    anti-saturation many-class plus missingness gate
  - issue [#146](https://github.com/bensonlee5/tab-foundry/issues/146) now
    occupies the adjacent synthetic harder-dagzoo slot and does not replace
    this benchmark-front missingness lane
  - there is no explicit carried-sandwich missingness recommendation yet; the
    older TF-RD-008 row-first settlement remains historical context only
- Required work:
  - re-anchor missingness work on the carried sandwich family after the carried
    many-class plus missingness slice is established under TF-RD-010
  - keep one pinned OpenML missingness ladder as the canonical benchmark
    baseline and allow license-cleared manifest-backed external augmentations
    when they add missingness regimes OpenML does not cover cleanly
  - use a review ledger for both OpenML datasets and vetted external real-data
    candidates rather than relying on source names alone
  - keep regime identity in task-source names, bundle names, manifest names,
    and curation reports rather than changing benchmark bundle schema in this
    pass
  - focus this lane on deeper missingness mechanism, severity, and benchmark
    coverage rather than on establishing the first anti-saturation regime
  - decide whether explicit missingness handling belongs in the default carried
    sandwich family or remains an optional robustness variant after TF-RD-010
- Exit criteria:
  - the repo has a benchmark-backed follow-on missingness recommendation for
    the carried sandwich family beyond the first many-class plus missingness
    gate

### TF-RD-017: Class-Imbalance Robustness On The Classification-First Sandwich Target

- Status: `planned`
- Milestone: `Next`
- Goal: decide how the carried sandwich family behaves under materially
  skewed class priors
- Current state:
  - current benchmark bundles only enforce `min_minority_class_pct = 2.5`
  - there is no dedicated imbalance-focused bundle ladder yet
  - this is now the preferred side robustness lane once TF-RD-010 has
    established the first sandwich many-class plus missingness dagzoo slice
  - issue [#146](https://github.com/bensonlee5/tab-foundry/issues/146) now
    occupies the adjacent synthetic harder-dagzoo slot and does not replace
    this benchmark-front imbalance lane
  - benchmark-facing reporting is still centered on ROC AUC, log loss, and
    Brier score
- Required work:
  - define the canonical imbalance-focused binary bundle or bundle-selection
    ladder on the carried sandwich family
  - keep one pinned OpenML imbalance ladder as the canonical benchmark
    baseline and allow license-cleared manifest-backed external augmentations
    when they add skew regimes OpenML does not cover cleanly
  - use the same review ledger for OpenML datasets and vetted external
    real-data candidates
  - keep regime identity in task-source names, bundle names, manifest names,
    and curation reports rather than changing benchmark bundle schema in this
    pass
  - define benchmark-facing comparison and reporting that includes PR AUC,
    average precision, and balanced accuracy alongside ROC/log loss/Brier
  - measure the carried sandwich family first without class reweighting or
    focal-style loss changes
  - only if the baseline read is weak, run bounded weighted-loss or focal-loss
    follow-up work
- Exit criteria:
  - the repo has a benchmark-backed keep/defer decision on the promoted
    sandwich family under class imbalance
  - the repo has an explicit imbalance metric and reporting contract rather
    than relying only on the current general binary bundle metrics

### TF-RD-018: Training-Surface Adequacy On The Promoted Anchor

- Status: `closed incomplete`
- Milestone: `Historical`
- Goal: preserve the completed staged-control training-surface reads without
  continuing this lane as a blocker now that the repo is prioritizing the
  sandwich family
- Current state:
  - issue [#109](https://github.com/bensonlee5/tab-foundry/issues/109) closed
    the initial dataset-batch ladder on the representative post-008 medium
    dagzoo surface
  - issue [#137](https://github.com/bensonlee5/tab-foundry/issues/137) kept
    `schedulefree_adamw` over `adamw` and `muon` on the inherited
    `tf_rd_020_shift_noise_drift_v1` staged-control runtime
  - TF-RD-020 remains the historical harder-front handoff for that same
    staged-control line
  - the remaining LR, clipping, and step-budget follow-up under issues
    [#138](https://github.com/bensonlee5/tab-foundry/issues/138) and
    [#139](https://github.com/bensonlee5/tab-foundry/issues/139) is no longer
    on the active roadmap path
  - issue [#107](https://github.com/bensonlee5/tab-foundry/issues/107) is now
    a closeout tracker rather than an active research epic
- Closure decision:
  - keep the completed TF-RD-018 artifacts as historical evidence on the
    staged-control line
  - do not treat unfinished TF-RD-018 LR or clipping work as a blocker for
    sandwich-first dagzoo, many-class, steering, runtime, or scaling work
  - if later runtime or optimizer tuning is needed, do it on the carried
    sandwich classification family under TF-RD-021, TF-RD-022, or TF-RD-009
    instead of reopening TF-RD-018
- Exit criteria:
  - satisfied: the repo retains one explicit partial training-surface record on
    the staged-control line
  - satisfied: the active path now moves to sandwich simplification and dagzoo
    carry-forward instead of continuing TF-RD-018

### TF-RD-022: Training Runtime And VRAM Efficiency Before Classification Scaling

- Status: `planned`
- Milestone: `Next`
- Goal: do the bounded kernel, runtime, and VRAM tuning needed so later
  scaling work inherits one measured runtime policy instead of ad hoc
  host-specific tweaks
- Current state:
  - deferred issue [#58](https://github.com/bensonlee5/tab-foundry/issues/58)
    already exists for runtime or VRAM summaries, but it stayed attached to
    the earlier TF-RD-002 measurement chain and never became a full execution
    spine
  - new epic [#168](https://github.com/bensonlee5/tab-foundry/issues/168) now
    tracks this runtime lane, with child issues
    [#169](https://github.com/bensonlee5/tab-foundry/issues/169),
    [#170](https://github.com/bensonlee5/tab-foundry/issues/170), and
    [#171](https://github.com/bensonlee5/tab-foundry/issues/171)
  - the sandwich architecture lane now lives under issue
    [#178](https://github.com/bensonlee5/tab-foundry/issues/178), with issue
    [#184](https://github.com/bensonlee5/tab-foundry/issues/184) recording the
    keep-current-anchor decision before dagzoo carry-forward
  - training telemetry and benchmark-registry records now preserve
    `runtime_summary` and `regime_budget` payloads, including peak VRAM,
    throughput, token-budget fields, objective metric, and curriculum or SCM
    metadata
  - canonical benchmark prior configs still inherit `runtime.mixed_precision: "no"` from `configs/experiment/_shared/compact_binary_prior.yaml` unless a
    higher-level experiment overrides it
  - `tabfoundry_staged` already supports
    `runtime.activation_checkpointing`, but the ordinary benchmark-facing
    defaults keep it disabled
  - benchmark-facing exact-prior runs still default to
    `runtime.trace_activations: true`, which is useful for diagnostics but is
    not yet separated cleanly from ordinary benchmark execution
- this epic now follows the first carried dagzoo many-class slice and the
  steering decision; it should not reopen sandwich-parent selection or earlier
  regime-choice work
- Required work:
  - land the runtime and VRAM measurement dependency from issue
    [#58](https://github.com/bensonlee5/tab-foundry/issues/58) so sweep and
    result artifacts expose peak memory, reserved memory, throughput, and time
    breakdowns directly
  - run the bounded low-risk runtime ladder under issue
    [#169](https://github.com/bensonlee5/tab-foundry/issues/169) on one frozen
    carried classification recipe: treat bf16, benchmark-facing
    activation-trace policy, and activation checkpointing as the first
    runtime-policy knobs
  - include any low-level kernel tuning only to the extent needed to make the
    carried sandwich dagzoo slice reliable and efficient enough for scaling
  - encode the winning runtime policy as a first-class config and sweep surface
    under issue [#170](https://github.com/bensonlee5/tab-foundry/issues/170)
    rather than relying on per-run overrides
  - keep sandwich architecture ownership under historical implementation issue
    [#174](https://github.com/bensonlee5/tab-foundry/issues/174), active
    umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178),
    and the completed anchor-retention decision in
    [#184](https://github.com/bensonlee5/tab-foundry/issues/184) rather than
    reopening this runtime epic as the sandwich owner; MPS OOMs should not be
    part of the quantitative CUDA decision record
  - only after the runtime policy is explicit, reopen harder-surface batching
    under issue [#171](https://github.com/bensonlee5/tab-foundry/issues/171)
    with a conservative 80 GB A100 memory guardrail and a fixed effective
    optimizer batch
  - keep architecture, steering/curriculum choice, and law-fitting changes out
    of this epic except insofar as TF-RD-016, TF-RD-010, and TF-RD-021 have
    already frozen them for the runtime read
- Exit criteria:
  - the repo has one explicit runtime policy for the classification scaling
    target, justified by repo-local time and VRAM evidence
  - sweep outputs, inspect surfaces, and result summaries expose runtime and
    VRAM metrics compactly enough that future runs can be compared without
    manual log inspection
  - later sandwich scaling, deferred CUDA-capacity follow-up, and
    TF-RD-009 preparation can inherit the same runtime policy without
    re-deriving it from scratch

### TF-RD-020: Harder Dagzoo Corpus Fronts On The Promoted Anchor

- Status: `completed`
- Milestone: `Completed`
- Goal: once the first TF-RD-018 dataset-batch ladder is complete, turn harder
  dagzoo-generated corpus fronts into the next explicit synthetic harder-surface
  decision lane on the historical staged-control line
- Current state:
  - TF-RD-020 is now historical handoff material rather than an active lane.
  - It closed on one kept harder-front winner in each family, with
    `tf_rd_020_shift_noise_drift_v1` becoming the default carry-forward
    surface for TF-RD-018 and `tf_rd_020_noise_mixture_v1` retained as named
    fallback context.
  - The completed ladder stayed synthetic-data-only; benchmark-front
    missingness and imbalance remain separate work.
- Completed outcomes:
  - `tf_rd_020_missingness_mcar_v1` is the kept missingness family winner.
  - `tf_rd_020_shift_noise_drift_v1` is the kept default harder carry-forward
    winner.
  - `tf_rd_020_noise_mixture_v1` is the kept mechanism/noise-family winner.
  - Larger-corpus, winner-mix, and steering follow-ups moved to later lanes
    rather than reopening TF-RD-020 itself.
- Exit criteria:
  - satisfied: the repo retains a documented default harder front plus named
    fallback context without reopening the completed ladder
  - satisfied: the relationship to TF-RD-010, TF-RD-017, and TF-RD-021 is
    explicit and non-overlapping

### TF-RD-021: Steering-Derived Dagzoo Corpus Fronts On The Classification-First Sandwich Target

- Status: `planned`
- Milestone: `Next`
- Goal: after the first sandwich dagzoo many-class plus missingness gate is in
  place, test whether steering-derived dagzoo corpus fronts improve that
  carried slice before runtime or scaling work
- Current state:
  - issue [#165](https://github.com/bensonlee5/tab-foundry/issues/165) is the
    successor synthetic-data epic after the first carried sandwich dagzoo
    many-class slice is established
  - TF-RD-020 already records the historical staged-control harder-front
    winners, while TF-RD-010 now records the fixed first sandwich benchmark
    contract with trusted execution still pending
  - dagzoo issue
    [#246](https://github.com/bensonlee5/dagzoo/issues/246) now owns the
    upstream steering implementation, deterministic metadata, and
    coverage-movement diagnostics
- Required work:
  - reuse the TF-RD-010 carried sandwich many-class plus missingness contract
    once the trusted rerun lands, and wait for dagzoo RD-008 to land enough of
    issue `#246` to make steering fixed-seed reproducible and auditable
  - define one bounded first sweep under issue
    [#167](https://github.com/bensonlee5/tab-foundry/issues/167): one control
    row on the carried sandwich dagzoo slice plus `3-4` steering-derived
    corpus rows produced from named steering policies or presets
  - hold architecture, many-class plus missingness regime definition, and
    benchmark contract fixed across every row
  - interpret `final_bpc_at_matched_regime_budget` first, with raw log loss,
    runtime, clipped-step fraction, and stability telemetry as guardrails
  - keep exactly one steering-derived carry-forward surface only if it clearly
    beats the incumbent control; otherwise keep the original carried slice
  - keep this epic synthetic-data-only rather than absorbing imbalance,
    runtime-kernel, or scaling-law conclusions
- Exit criteria:
  - the repo has one explicit keep/defer decision on whether any
    steering-derived corpus front replaces the carried sandwich dagzoo slice
  - the relationship between TF-RD-021 and TF-RD-010, TF-RD-017, TF-RD-022,
    and TF-RD-009 is explicit and non-overlapping

### TF-RD-015: Regression Rebuild Deferred From The Classification-First Scaling Plan

- Status: `research`
- Milestone: `Later`
- Goal: resume regression only after the classification-first sandwich scaling
  plan has landed a stable runtime policy, harder-surface classification
  evidence, and a usable scaling contract
- Current state:
  - regression support is intentionally removed from the active repo surface
  - regression metrics and bundle normalization support already exist in parts
    of the repo
  - there is no active regression program, canonical regression bundle, or
    regression head/loss contract
  - regression is intentionally out of scope as a blocker for the first
    classification scaling program
- Required work:
  - define the canonical regression benchmark surface or surfaces only after
    the classification-first scaling plan closes its core gates
  - if regression curation expands, keep one OpenML baseline where possible and
    use only license-cleared manifest-backed external datasets as augmentations
  - define the eventual regression head and loss contract on the chosen
    post-classification family rather than reopening regression ownership now
- Exit criteria:
  - regression has a benchmark-facing baseline and a bounded roadmap for
    promotion or deferral after the classification-first scaling plan is
    complete

### TF-RD-016: Architecture Surface Adequacy, Sandwich Simplification, And Selective Expansion

- Status: `completed`
- Milestone: `Completed`
- Goal: determine whether the sandwich family needed more simplification-first
  work before broader harder-surface evaluation
- Current state:
  - issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178) now
    closes as a historical handoff issue rather than an active lane
  - the earlier summary-bottleneck replay is closed negative evidence under
    [#179](https://github.com/bensonlee5/tab-foundry/issues/179), while the
    compact hybrid replay and the first two follow-up screens under
    [#181](https://github.com/bensonlee5/tab-foundry/issues/181),
    [#182](https://github.com/bensonlee5/tab-foundry/issues/182), and
    [#183](https://github.com/bensonlee5/tab-foundry/issues/183) all kept the
    compact hybrid control unchanged
  - issue [#184](https://github.com/bensonlee5/tab-foundry/issues/184) ran the
    four-row removal-first package and kept the compact hybrid control
    unchanged as the historical predecessor surface
  - TF-RD-016 now records that the next useful evidence is benchmark definition
    plus a modest head/output evolution, not more simplification-only replay
- Completed outcomes:
  - the repo now has a bounded decision that the sandwich backbone is coherent
    enough to move into medium and large multiclass benchmark formulation
  - the follow-on family is explicitly FiLM-conditioned, uses
    `sandwich_summary_tokens_per_axis=3`, raises `many_class_base` to `10`, and
    uses a direct multiclass head rather than reopening the staged line
  - TF-RD-010 then fixed the first benchmark program contract, with `dagzoo`
    synthetic training fronts feeding `tab-realdata-hub` validation manifests,
    while trusted execution is now being rerun on the corrected surface
- Exit criteria:
  - satisfied: the repo has an explicit keep/defer decision that the sandwich
    architecture surface is adequate enough to move into benchmark-defined
    multiclass evaluation
  - satisfied: later work now reuses one coherent sandwich family instead of
    reopening general simplification or staged-family divergence

### TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

- Status: `planned`
- Milestone: `Next`
- Goal: fit the first classification scaling laws on the simplified sandwich
  family only after the repo has one fixed dagzoo many-class plus missingness
  slice, one steering decision, one runtime policy, and a literature-grounded
  law-design note
- Current state:
  - tuning and benchmark-adjacent tooling already exist
  - scaling-law intent is clear, but scaling on the current simple binary regime
    risks low-signal conclusions because recent architecture deltas are already
    close on that surface
  - training telemetry and benchmark-registry artifacts now preserve resolved
    sandwich specs, runtime summaries, and regime-budget metadata needed for
    later scaling comparisons
  - there is still no canonical scaling artifact path on a fixed dagzoo
    many-class plus missingness slice with matched runtime policy and matched
    regime budget
  - the keep-current-anchor decision under
    [#184](https://github.com/bensonlee5/tab-foundry/issues/184) is the
    required precursor for this family, but it does not satisfy TF-RD-009 by
    itself
- Required work:
  - write the dedicated law-design note before any scaling fit, grounded in
    μP, depth-aware μP follow-ups, optimizer-budget scaling work, Chinchilla,
    and synthetic-data curriculum references
  - separate theory-backed and empirical dimensions explicitly:
    width via `d_icl`, depth via `sandwich_layers`, optimizer transfer via LR,
    momentum, and batch, and curriculum or SCM mixture as an empirical
    higher-order term
  - treat matched token budget as necessary but not sufficient; compare by
    matched regime budget using token budget, unique-task budget, fixed
    curriculum or SCM-mixture slice, and fixed task-complexity band
  - reuse the TF-RD-010 benchmark contract after the trusted rerun lands, then
    finish the steering decision under TF-RD-021 and the runtime policy under
    TF-RD-022 before using scaling results as architecture evidence
  - keep the other sandwich knobs frozen at the retained compact hybrid anchor
    values while fitting the first width-depth classification laws
  - use `final_bpc_at_matched_regime_budget` as the primary ranking objective
    on the carried multiclass slice, treating BPC as the normalized log-loss
    view
  - run optimizer-transfer and model-size scaling together rather than as
    separate programs
  - keep the eventual `sandwich_scale` interface internal-only until the law is
    validated on the carried multiclass slice and later follow-on robustness
    lanes
- Exit criteria:
  - the repo can fit width-depth classification laws on the simplified sandwich
    architecture under a fixed dagzoo many-class plus missingness slice that is
    harder or broader than the current simple binary regime
  - scaling artifacts compare runs by matched regime budget with final BPC as
    the primary objective, raw log loss as supporting context, and stability,
    calibration, and runtime as guardrails
  - any later single-knob scaling interface is explicitly derived from those
    law fits and remains internal until cross-surface validation is complete

## Acceptance Gates For Architecture Promotion

The architecture target lane should promote rows only when all of the following
hold:

- the row is coherent as a named architecture surface rather than an ad hoc
  override pile
- the row is stable on training telemetry, not only acceptable on final metrics
- the row is benchmark-neutral-or-better against the relevant control surface
- added runtime and memory cost are justified by the gain
- the interpretation is attributable against matched controls rather than
  speculative

## Planning Defaults And Assumptions

- TabICLv2 is the strongest directional reference for the active architecture
  target, but not a literal reproduction target.
- Dagzoo synthetic-data efficacy is already established historically; the
  active sandwich path now runs: keep-current-anchor decision, sandwich-on-dagzoo
  carry-forward, first many-class plus missingness gate, steering-derived
  corpus fronts, bounded kernel/runtime tuning, then scaling-law work.
- The current large-anchor hybrid line is diagnostic evidence, not the intended
  architecture destination.
