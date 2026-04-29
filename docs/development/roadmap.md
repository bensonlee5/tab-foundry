# Roadmap (202604)

Use this roadmap to understand which questions are active now, which surfaces
are frozen, and what evidence the repo still needs for follow-up validation.

The repo-wide plan is now grid-anchor first:

- keep one frozen PFN-style control lane for trust and comparison
- treat `grid_sandwich` as the active classification architecture target and
  carried repo anchor after the TF-RD-026 row `10` medium sweep promotion
- preserve `tabfoundry_sandwich` as the previous carried in-family comparison
  and scaling-evidence lane
- use the closed TF-RD-010 classification benchmark contract as the fixed
  anti-saturation regime before the first scaling fit
- treat steering-derived dagzoo fronts and other synthetic-surface expansion
  work as sidecars rather than blockers on the `tab-foundry` critical path
- make bounded kernel/runtime and VRAM optimization a hard prerequisite to
  later scaling so every scale ladder inherits one measured runtime policy
- defer regression from the first classification scaling plan

Use these alongside this roadmap:

- problem formulation: `docs/development/synthetic-prior-mission.md`
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
  - `grid_sandwich`
  - grid-preserving row-feature classifier promoted from the April 20-21, 2026
    medium multiclass sidecar benchmark
  - owns new architecture work; `tabfoundry_sandwich` remains the previous
    carried in-family comparison and scaling-evidence lane

Important non-goals for this roadmap:

- do not treat the current `nano_exact + prenorm + row_cls` hybrid line as the
  long-term destination
- do not assume the incumbent staged reference line is the final architecture
  destination either; it is retained as a historical reference surface
- do not bundle regression into the first row-first promotion push
- do not make QASS structurally mandatory
- do not proliferate many parallel live architecture families beyond the frozen
  control, the historical staged reference line, the `tabfoundry_sandwich`
  comparison surface, and the current grid anchor line

## Prioritization Lens

- Classification remains the anchor workload while the grid family is tested on
  harder post-008 regimes.
- The next useful architecture evidence should come from coherent grid
  surfaces, not from extending the older staged line.
- After TF-RD-016 closeout, the next deliberate front should make the research
  surface less saturating and more realistic: a benchmark-defined
  classification regime where `dagzoo` owns synthetic training fronts,
  `tab-realdata-hub` owns medium and large validation manifests, and
  `tab-foundry` owns the evolved sandwich model and sweep contracts.
- If those harder or broader surfaces still leave the model hard to separate or
  obviously underfit, open a later bounded architecture follow-up before
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
  `tab-realdata-hub` materialized manifests; TF-RD-022 then performs the
  telemetry/read-surface cleanup, carried-runtime validation, and
  kernel-level training acceleration work needed so later work inherits one
  explicit runtime policy plus a profiler-backed training-speed verdict on
  that closed benchmark contract; TF-RD-024 then closes one bounded
  post-performance architecture read on the inherited runtime surface and
  hands off `sandwich_heads=1` as the pre-scaling winner; TF-RD-009 now writes
  the scaling-law design note and fits the first scaling law on the same fixed
  classification contract; TF-RD-021, dagzoo RD-002, dagzoo RD-005, and other
  synthetic-surface expansions remain sidecars rather than blockers on the
  main `tab-foundry` roadmap; TF-RD-014 remains the next follow-on missingness
  lane, while TF-RD-017 moves to later imbalance work outside the current
  critical path.
- Low-level questions such as norm family or placement, learned special-token
  initialization scale, QASS scaler capacity, and activation family belong
  under TF-RD-016 after the earlier adequacy and harder-surface gates are in
  place, not as free-floating anchor-settlement work.
- Many-class, regression, and runtime handoff should build on that
  deconfounded post-008 base rather than competing with the earlier gates.
- Runtime, VRAM, and any kernel-level training acceleration work should finish
  before the first scaling fit so classification ladders inherit one measured
  80 GB A100-safe policy plus a settled training-speed verdict instead of
  discovering performance limits during the scale study.
- Scaling-law design should start only after the repo has a simplified sandwich
  parent, the closed TF-RD-010 benchmark contract, one TF-RD-022 runtime
  policy, and one bounded TF-RD-024 post-performance architecture read on the
  same classification family.

## Canonical Priority Queue

This queue is intentionally sandwich-focused. Historical staged/control work is
summarized later instead of occupying the active queue.

| Rank | Roadmap ID | Item | Status | Milestone |
| ---- | ---------- | ---- | ------ | --------- |
| 1 | TF-RD-009 | Scaling-law design and measurement on the classification-first sandwich target | in_progress | Next |
| 2 | TF-RD-026 | Broad-ML Grid Sandwich performance campaign | implemented | Next |
| 3 | TF-RD-029 | Grid MoE normalized top-2 follow-up | completed | Next |
| 4 | TF-RD-030 | Grid MoE shared-expert follow-up | in_progress | Next |
| 5 | TF-RD-014 | Missingness robustness on the classification-first sandwich target | planned | Next |
| 6 | TF-RD-017 | Class-imbalance robustness on the classification-first sandwich target | planned | Later |
| 7 | TF-RD-021 | Steering-derived dagzoo corpus fronts on the classification-first sandwich target | research | Later |
| 8 | TF-RD-015 | Regression rebuild deferred from the classification-first scaling plan | research | Later |
| 9 | TF-RD-012 | Inference handoff and later modalities | research | Later |

## Dependency Graph

```mermaid
flowchart TD
    HIST["Historical baseline<br/>TF-RD-000 through TF-RD-020<br/>control, staged, dagzoo, and closeout evidence"]
    RD016["TF-RD-016 / TF-RD-021B<br/>Freeze simplified<br/>sandwich parent"]
    RD010["TF-RD-010<br/>Many-class + missingness<br/>dagzoo gate"]
    RD021["TF-RD-021<br/>Steering-derived dagzoo<br/>sidecar lane"]
    DZ002["dagzoo RD-002<br/>Interventional + counterfactual<br/>generation expansion"]
    DZ005["dagzoo RD-005<br/>Robustness stress profiles<br/>and carried regimes"]
    RD017["TF-RD-017<br/>Class-imbalance<br/>side lane"]
    RD022["TF-RD-022<br/>Kernel-level training<br/>runtime & VRAM gate"]
    RD024["TF-RD-024<br/>Bounded post-performance<br/>architecture sweep"]
    RD014["TF-RD-014<br/>Missingness<br/>follow-up"]
    RD015["TF-RD-015<br/>Regression rebuild<br/>(deferred)"]
    RD012["TF-RD-012<br/>Inference handoff &<br/>later modalities"]
    RD009["TF-RD-009<br/>Scaling-law design &<br/>measurement"]

    HIST --> RD016
    RD016 --> RD010
    RD010 --> RD022
    RD022 --> RD024
    RD024 --> RD009
    RD010 --> RD021
    DZ002 --> RD021
    DZ005 --> RD021
    RD010 --> RD014
    RD010 --> RD017
    RD016 --> RD015
    RD016 --> RD012

    classDef hist fill:#e5e7eb,stroke:#6b7280,color:#111827;
    classDef done fill:#d1fae5,stroke:#059669,color:#064e3b;
    classDef readyNow fill:#fff3cd,stroke:#ffc107,color:#856404;
    classDef gate fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    classDef later fill:#f3e8ff,stroke:#7c3aed,color:#3b1f6e;

    class HIST hist;
    class RD010,RD016,RD022,RD024 done;
    class RD009,RD014 readyNow;
    class DZ002,DZ005,RD021,RD012,RD015,RD017 later;
```

Current path: **TF-RD-009** on the closed TF-RD-010 benchmark contract, with TF-RD-022 completed as the inherited runtime and training-speed gate and TF-RD-024 completed as the bounded medium-only architecture handoff to `sandwich_heads=1`.

- TF-RD-016 is now completed historical context: issue
  [#178](https://github.com/bensonlee5/tab-foundry/issues/178) closes on the
  decision to evaluate the next sandwich phase through benchmark definition and
  modest head evolution rather than more simplification-only evidence.
- TF-RD-010 is now completed: issue
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
  trusted rerun and refactor path that closed the benchmark gate on the same
  contract.
- Issue [#205](https://github.com/bensonlee5/tab-foundry/issues/205) now owns
  the completed `tf_rd_010_classification_evolution_medium_v4` package: row 1
  keeps the refreshed medium control anchor by benchmarking the reusable
  `tf_rd_010_synthetic_adequacy_v3` production-control artifact, while rows 2
  through 4 defer as exploratory missingness reads on the same medium manifest.
- That medium read is now the carried TF-RD-010 comparator: the completed
  sorted-order replay did not beat it, so the original `medium_v4` control
  remains the trusted benchmark anchor for later lanes.
- The completed TF-RD-010 medium follow-up under
  [#202](https://github.com/bensonlee5/tab-foundry/issues/202) is now closed
  negative evidence: row 1 retrained
  `tf_rd_010_dagzoo_medium_control_curated_v5` under the current sorted-order
  code path and finished at `final_log_loss=0.6849303354`, which is worse than
  the original `medium_v4` control at `0.6811727401`.
- `large_v2` under [#203](https://github.com/bensonlee5/tab-foundry/issues/203)
  now records the trusted large-rung replay under that kept comparator:
  control `0.8974410961` remains better than `mcar=0.9155278224`,
  `mar=0.9418792099`, and `mnar=0.9411754209`, so TF-RD-010 does not promote
  missingness exposure from the first medium/large package.
- TF-RD-010 continues to rank the active grid benchmark path by
  `final_log_loss_at_matched_regime_budget`, interpreted explicitly as
  label-target log loss per test cell. The older `cell_bpc` / BPC lane is
  preserved only for historical reruns while calibration, runtime, and
  stability remain guardrails on the active CE-based path.
- TF-RD-021, dagzoo RD-002, and dagzoo RD-005 are now sidecar synthetic-data
  work rather than blockers on the main `tab-foundry` architecture path.
- TF-RD-022 is now completed: issues
  [#168](https://github.com/bensonlee5/tab-foundry/issues/168) and
  [#247](https://github.com/bensonlee5/tab-foundry/issues/247) close on one
  kept runtime and training-speed verdict, where
  `compile_eager_dynamic` preserved the carried bf16 plus activation-
  checkpointing runtime surface, improved same-host A100 wall time from
  `3848.0996s` to `3586.6358s`, and improved
  `final_log_loss_at_matched_regime_budget` from `0.6820820744` to
  `0.6810689708`.
- TF-RD-024 is now closed on medium-only evidence by decision, with
  `sandwich_heads=1` beating both the fresh compile anchor and the earlier
  `sandwich_heads=2` medium winner while `sandwich_pre_row_attention_layers=2`
  improved over the anchor but did not beat either head-count winner.
- TF-RD-009 now starts from that inherited runtime plus architecture surface
  and fits the first law on the same fixed multiclass benchmark contract under
  matched regime budget.

Parallel/later lanes are intentionally off that main path:

- TF-RD-014 is now a follow-on missingness robustness lane after the first
  many-class plus missingness gate rather than a blocker to the first scaling
  fit.
- TF-RD-017 remains a later imbalance robustness lane on the same family, but
  it is now explicitly off the current TF-RD-009 critical path.
- TF-RD-015 regression and TF-RD-012 inference handoff/later modalities remain
  later work.

## Current Capability Matrix

| Objective / Claim | Current State | Evidence In Repo | Current Gap | Roadmap IDs |
| --- | --- | --- | --- | --- |
| Frozen PFN-style control exists | `implemented` | `tabfoundry_simple`, `stage=nano_exact`, and the prior-trained PFN-facing benchmark lane are all stable | Keep that lane clearly separate from the architecture target | `TF-RD-001` |
| Grid is the primary classification candidate | `implemented` | `grid_sandwich` is the carried architecture anchor after TF-RD-026 row `10` promoted the two-layer recurrent SwiGLU grid core at final log loss `0.4181767299`; `tabfoundry_sandwich` remains the previous carried comparison family and retains its compact hybrid, knob-screen, width/head, and direct-head evidence | Validate the promoted grid anchor on larger or harder rungs rather than reopening simplification-first sandwich work | `TF-RD-026`, `TF-RD-016`, `TF-RD-021A`, `TF-RD-021B`, `TF-RD-010`, `TF-RD-009` |
| Broad-ML grid follow-ons selected a new anchor | `completed` | TF-RD-026 completed the medium sweep, promoted row `10` (`grid_recurrence_steps=8`, `grid_recurrence_unique_layers=2`, `grid_ffn_mode=swiglu`) as the current `cls_workstation_grid_sandwich` surface, and left row `11` as close comparison evidence | Larger-rung validation remains before replacing external planning assumptions; row `09` is not planned after its early stop | `TF-RD-026`, `TF-RD-009` |
| Expedited FFN and stability grid screens closed without replacing the anchor | `completed` | TF-RD-027 completed under issue [#289](https://github.com/bensonlee5/tab-foundry/issues/289): GELU 4:1 (`0.4254928909`), SwiGLU 8:3 (`0.4261833356`), GEGLU 8:3 (`0.4268446367`), weight decay `0.1` (`0.4263516327`), z-loss (`0.4243303110`), logit softcap (`0.4251839620`), and QK-norm (`0.4741464959`) all lost to the TF-RD-026 row `10` anchor `0.4181767299` | Keep the TF-RD-026 row `10` anchor unchanged; do not stack stability knobs or reopen FFN variants without a stronger hypothesis | `TF-RD-027`, `TF-RD-026` |
| Sparse grid MoE is implemented but not promoted | `in_progress` | TF-RD-028 completed core-only SwiGLU MoE execution under issue [#295](https://github.com/bensonlee5/tab-foundry/issues/295); no row beat the TF-RD-026 row `10` anchor, raw top-2 was healthy but worse at `0.4234291888`; TF-RD-029 issue [#296](https://github.com/bensonlee5/tab-foundry/issues/296) completed normalized top-2 at `0.4218944597`, making it the best MoE row so far but still `+0.0037177297` worse and much slower than the anchor; and TF-RD-030 issue [#297](https://github.com/bensonlee5/tab-foundry/issues/297) now records the always-on shared-expert, router-temperature, and load-balance-schedule follow-up | Run `tf_rd_030_grid_moe_shared_expert_v1` only after its shared-expert smoke row passes on the same Vast/W&B path; promote no MoE variant unless it beats the TF-RD-026 row `10` anchor without route collapse or unacceptable wall-time cost | `TF-RD-028`, `TF-RD-029`, `TF-RD-030`, `TF-RD-026` |
| Harder synthetic classification fronts are runnable | `implemented` | Dagzoo manifest/export fidelity is complete, TF-RD-013 settled the representative medium surface, TF-RD-020 settled harder-front winners that can seed the sandwich benchmark program, and dagzoo epics [#249](https://github.com/bensonlee5/dagzoo/issues/249) and [#247](https://github.com/bensonlee5/dagzoo/issues/247) define later surface-expansion work | Keep TF-RD-021 and dagzoo RD-002/RD-005 as sidecar synthetic-data context while TF-RD-009 executes on the closed TF-RD-010 benchmark contract | `TF-RD-011`, `TF-RD-013`, `TF-RD-020`, `TF-RD-016`, `TF-RD-010`, `TF-RD-021` |
| Runtime and VRAM are measurable | `implemented` | Training and registry artifacts now preserve runtime-summary and regime-budget fields, `tab-foundry dev run-inspect` now exposes compact runtime and regime-budget summaries, sweep summaries now carry compact runtime columns, and TF-RD-022 closed on the kept bf16 plus activation-checkpointing `compile_eager_dynamic` runtime surface with an A100 wall-time improvement from `3848.0996s` to `3586.6358s` and matched-budget log loss improvement from `0.6820820744` to `0.6810689708` | Carry the kept TF-RD-022 runtime surface unchanged into TF-RD-009; any later runtime changes now need a new dedicated lane rather than reopening the closed gate | `TF-RD-022` |
| Benchmark-backed classification validation contract is fixed, `medium_v4` completed the directional medium package, `medium_v5` now records the sorted-control replay, and `large_v2` now records the local large-rung replay | `implemented` | `many_class` is implemented, the sandwich evolution config fixes FiLM plus `sandwich_summary_tokens_per_axis=3`, `tab-realdata-hub` issue [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) owns the medium and large validation manifests under `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, TF-RD-010 child issues [#197](https://github.com/bensonlee5/tab-foundry/issues/197), [#198](https://github.com/bensonlee5/tab-foundry/issues/198), [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and [#200](https://github.com/bensonlee5/tab-foundry/issues/200) froze the missing baselines plus corpora, `medium_v4` now records a kept medium control anchor plus exploratory MCAR, MAR, and MNAR defer rows, `medium_v5` now records the completed sorted-order control replay under [#202](https://github.com/bensonlee5/tab-foundry/issues/202) at `0.6849303354`, and `large_v2` now records the completed local all-rows benchmark-only large replay under [#203](https://github.com/bensonlee5/tab-foundry/issues/203) with control `0.8974410961`, `mcar=0.9155278224`, `mar=0.9418792099`, and `mnar=0.9411754209` | TF-RD-010 now explicitly keeps the original `medium_v4` control (`0.6811727401`) over the worse sorted-order `medium_v5` replay (`0.6849303354`), and the completed `large_v2` replay preserves the same ordering with control best on the harder rung. Later lanes inherit that closed benchmark contract and the no-missingness-promotion read, while the canonical metric key remains `final_log_loss_at_matched_regime_budget`, interpreted as label-target log loss per test cell | `TF-RD-010`, `TF-RD-022`, `TF-RD-024`, `TF-RD-014`, `TF-RD-017` |
| Follow-on missingness and imbalance robustness remain open | `partial` | Missing-permitting binary bundles exist, and the current bundle policy already excludes degenerate minority-class cases | TF-RD-014 remains the next follow-on missingness lane after the first scaling pass, while TF-RD-017 still needs an explicit later imbalance ladder on the same grid family | `TF-RD-014`, `TF-RD-017` |
| Regression and later modalities are deferred | `research` | Partial bundle/runtime scaffolding exists | They should not absorb attention from the classification-first path | `TF-RD-015`, `TF-RD-012` |
| Scaling-law work has the needed metadata path | `implemented` | Artifacts now preserve resolved sandwich specs plus runtime/regime-budget metadata, TF-RD-010 fixed the first benchmark-defined classification contract, TF-RD-022 closed on the carried compile-eager-dynamic runtime surface, and TF-RD-024 closed on medium-only architecture evidence with `sandwich_heads=1` as the handoff winner | The remaining gap is the actual TF-RD-009 design note and scaling execution, not another prerequisite architecture or runtime screen | `TF-RD-009`, `TF-RD-010`, `TF-RD-022`, `TF-RD-024` |

## Current Implementation Baseline

This roadmap assumes the following repo truths:

- `tabfoundry_simple` and `tabfoundry_staged` with `stage=nano_exact` remain
  the frozen PFN-style control lane.
- `grid_sandwich` exists as the primary classification architecture candidate
  and carried repo anchor; `tabfoundry_sandwich` remains the previous carried
  comparison family with initial replay, knob screen, bounded width/head
  follow-up, and removal-first evidence preserved.
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
  focus on active grid development and later follow-up lanes.

Legacy wording note:

- some TF-RD item titles still use earlier row-first or promoted-anchor
  phrasing so existing roadmap links remain stable
- the dependency order, current-state bullets, and required-work bullets below
  are the authoritative description of the active grid-first path

### TF-RD-010: Benchmark-Defined Multiclass Evolution On The Classification-First Sandwich Target

- Status: `completed`
- Milestone: `Completed`
- Goal: define the first benchmark-backed classification evaluation program for
  the evolved sandwich family using explicit hub-owned bundle policy rather
  than local fixture assumptions
- Current state:
  - issue [#52](https://github.com/bensonlee5/tab-foundry/issues/52) and issue
    [#99](https://github.com/bensonlee5/tab-foundry/issues/99) now serve as
    historical umbrella and execution context, while issue
    [#202](https://github.com/bensonlee5/tab-foundry/issues/202) is the closed
    trusted-rerun umbrella
  - historical child issues
    [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
    [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
    [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
    [#200](https://github.com/bensonlee5/tab-foundry/issues/200) define the
    TF-RD-010 corpora and freeze the missing baselines; successor issues
    [#205](https://github.com/bensonlee5/tab-foundry/issues/205) completed the
    trusted medium rerun package, while
    [#203](https://github.com/bensonlee5/tab-foundry/issues/203) now records
    the completed local large-rung benchmark-only replay
  - issue [#204](https://github.com/bensonlee5/tab-foundry/issues/204) is the
    completed sandwich refactor follow-up that landed before the trusted
    TF-RD-010 reruns were recorded as canonical evidence
  - `tab-realdata-hub` issue
    [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) is now the
    upstream dependency for medium and large classification validation bundles
    and materialized manifests
  - the evolved sandwich benchmark config uses FiLM,
    `sandwich_summary_tokens_per_axis=3`, `many_class_base=10`, and a direct
    multiclass head
  - the canonical benchmark metric key remains
    `final_log_loss_at_matched_regime_budget`, interpreted explicitly as
    label-target log loss per test cell
  - the synthetic TF-RD-010 corpora and upstream hub bundles now align on
    `min_classes=2`, `max_classes=10`; the upstream hub bundles also fix
    `max_missing_pct=20.0`
  - the active TF-RD-010 synthetic fronts are explicit balanced grids with
    `144` tasks per front, row totals `128/256/512/1024`, feature counts
    `6/10/14/20`, and class coverage for every integer `2..10`
  - the repo now has materialized validation manifests under the local
    benchmark-manifest output root, with the legacy local output ids
    `openml_classification_medium_v1` and
    `openml_classification_large_v1`
  - issue [#205](https://github.com/bensonlee5/tab-foundry/issues/205) is now
    completed for the medium rung: row 1 reused the finished
    `tf_rd_010_synthetic_adequacy_v3` production-control train artifact and
    kept the medium anchor at `final_log_loss=0.6811727401`, while rows 2
    through 4 finished as exploratory defer reads with `mcar=0.6944203482`,
    `mar=0.7090284828`, and `mnar=0.7095624598`
  - `tf_rd_010_classification_evolution_medium_v5` now records the completed
    sorted-order control replay: row 1 finished at `final_log_loss=0.6849303354`,
    which is worse than the carried `medium_v4` control anchor at
    `0.6811727401`, so TF-RD-010 explicitly keeps the original `medium_v4`
    control as the carried comparator
  - `tf_rd_010_classification_evolution_large_v2` now records the completed
    local all-rows benchmark-only replay: the large control row finished at
    `final_log_loss=0.8974410961`, while the exploratory missingness rows
    finished at `mcar=0.9155278224`, `mar=0.9418792099`, and
    `mnar=0.9411754209`, preserving the same control-best ordering on the
    harder rung
- What remains fixed:
  - `dagzoo` now owns the explicit TF-RD-010 control, MCAR, MAR, and MNAR
    corpora through `tf_rd_010_dagzoo_medium_control_v1`,
    `tf_rd_010_missingness_mcar_v1`, `tf_rd_010_missingness_mar_v1`, and
    `tf_rd_010_missingness_mnar_v1`
  - upcoming synthetic sweeps now default to one pass over corpus manifest
    tasks at `prior_dump_batch_size=64`, and TF-RD-010 adopts that single-epoch
    synthetic budget for its trusted reruns
  - `tab-foundry` froze the legacy baseline ids
    `cls_benchmark_linear_multiclass_medium_v1` and
    `cls_benchmark_linear_multiclass_large_v1` against the hub-backed medium
    and large manifests
  - the `dagzoo -> tab-realdata-hub -> tab-foundry` benchmark contract remains
    fixed and does not need to be redefined
  - the refreshed `dagzoo` factorization now follows equation `(1)`, and the
    closed TF-RD-010 rerun package is recorded on that refreshed `*_v3` corpus
    family
  - medium row 1 is the only promotable row in `medium_v4`; rows 2 through 4
    remain exploratory because curated missingness fronts do not exist yet
  - `medium_v5` is the completed single-row sorted-control follow-up and does
    not reopen the exploratory medium missingness rows
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
  - `medium_v4` is no longer a blocked draft: it is the completed medium rerun
    package on the refreshed contract
  - `medium_v5` is now the completed sorted-order control replay on the medium
    rung
  - `large_v2` is no longer blocked on execution or interpretation: the local
    benchmark-only replay is completed and read against the kept original
    `medium_v4` control
- Exit criteria:
  - satisfied: the repo has one explicit medium-plus-large classification benchmark
    contract on the evolved sandwich family
  - satisfied: the hub-backed classification contract is no longer only
    untested scaffolding on the first scaling path
  - satisfied: `medium_v4` now records one kept medium control anchor plus the
    first exploratory medium missingness read on the corrected sandwich family
  - satisfied: TF-RD-010 explicitly keeps the original `medium_v4` control
    (`0.6811727401`) over the completed `medium_v5` sorted-order replay
    (`0.6849303354`)
  - satisfied: the completed medium and large missingness rows remain deferred
    because none beat the carried control on either rung
  - satisfied: later lanes can now inherit TF-RD-010 again as closed benchmark
    context

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
- Goal: deepen missingness robustness after the first scaling pass has landed
  on the carried sandwich family
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
  - keep this lane off the immediate TF-RD-022 -> TF-RD-024 -> TF-RD-009 path
    so it can inherit the first kept benchmark contract and runtime policy
    rather than compete with the first scaling pass
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
- Milestone: `Later`
- Goal: decide how the carried sandwich family behaves under materially
  skewed class priors
- Current state:
  - current benchmark bundles only enforce `min_minority_class_pct = 2.5`
  - there is no dedicated imbalance-focused bundle ladder yet
  - this now remains a later robustness lane once TF-RD-010 has established
    the first sandwich many-class plus missingness dagzoo slice
  - issue [#146](https://github.com/bensonlee5/tab-foundry/issues/146) now
    occupies the adjacent synthetic harder-dagzoo slot and does not replace
    this benchmark-front imbalance lane
  - benchmark-facing reporting is still centered on ROC AUC, log loss, and
    Brier score
- Required work:
  - keep this lane explicitly behind TF-RD-022, TF-RD-024, and TF-RD-009 so
    the first scaling pass lands before imbalance-specific follow-up work
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
    sandwich classification family under TF-RD-022, TF-RD-024, or TF-RD-009
    instead of reopening TF-RD-018
- Exit criteria:
  - satisfied: the repo retains one explicit partial training-surface record on
    the staged-control line
  - satisfied: the active path now moves to grid-anchor validation and dagzoo
    carry-forward instead of continuing TF-RD-018

### TF-RD-022: Kernel-Level Training Acceleration On The Settled Sandwich Runtime Surface Before Classification Scaling

- Status: `completed`
- Milestone: `Completed`
- Goal: close the remaining carried-runtime and kernel-level training
  performance questions after runtime-policy selection, so later scaling work
  inherits one settled runtime surface and one explicit keep or defer read on
  serious training-speed implementations rather than repeating low-risk
  loader/copy-path tweaks
- Current state:
  - completed historical issues [#58](https://github.com/bensonlee5/tab-foundry/issues/58),
    [#169](https://github.com/bensonlee5/tab-foundry/issues/169), and
    [#170](https://github.com/bensonlee5/tab-foundry/issues/170) now record
    the runtime-summary instrumentation, bounded medium ladder, and named
    runtime-policy surface that made TF-RD-022 explicit enough for downstream
    work; issue [#171](https://github.com/bensonlee5/tab-foundry/issues/171)
    is superseded because TF-RD-022 will not reopen harder-surface batching
  - epic [#168](https://github.com/bensonlee5/tab-foundry/issues/168) now
    closes kernel-level training acceleration on the settled runtime surface,
    with closed low-risk training-throughput defer
    [#239](https://github.com/bensonlee5/tab-foundry/issues/239), closed
    operational sidecars
    [#240](https://github.com/bensonlee5/tab-foundry/issues/240) and
    [#241](https://github.com/bensonlee5/tab-foundry/issues/241), and closed
    kernel-acceleration child
    [#247](https://github.com/bensonlee5/tab-foundry/issues/247)
  - the `tabfoundry_sandwich` comparison lane still lives under issue
    [#178](https://github.com/bensonlee5/tab-foundry/issues/178), with issue
    [#184](https://github.com/bensonlee5/tab-foundry/issues/184) recording the
    keep-current-anchor decision before the current benchmark-only follow-up
  - training telemetry and benchmark-registry records now preserve
    `runtime_summary` and `regime_budget` payloads, including peak VRAM,
    throughput, token-budget fields, objective metric, and curriculum or SCM
    metadata
  - `tab-foundry dev run-inspect`, sweep summaries, and result-card reporting
    now surface compact runtime and regime-budget reads directly, so issue
    [#58](https://github.com/bensonlee5/tab-foundry/issues/58) is no longer
    blocked on manual log inspection
  - the repo now has a named runtime policy surface at
    `configs/runtime/tf_rd_022_policy.yaml` plus the inherited benchmark-facing
    experiment `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
  - sweep `tf_rd_022_runtime_policy_medium_v1` is now registered as the
    benchmark-first TF-RD-022 medium screening ladder: row 1 replays the
    no-AMP control, rows 2 through 4 isolate bf16, benchmark-facing activation
    tracing, and activation checkpointing, and the keep bar requires
    non-worse `final_log_loss_at_matched_regime_budget` before runtime or VRAM
    tie-breakers are considered
  - `tf_rd_022_runtime_policy_medium_v1` is now completed locally from the
    mirrored CUDA run artifacts: row 4
    `sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2`
    is the medium-rung winner with `final_log_loss=0.6765953232`,
    `peak_vram_reserved=3321888768`, and
    `throughput_tokens_per_second=150561.1995`; row 2 bf16 is benchmark-safe
    but deferred, row 3 trace is benchmark-safe but diagnostic-only, and row 1
    remains the no-AMP screening control
  - the inherited validation contract is the closed TF-RD-010 medium and large
    classification benchmark package under issues
    [#202](https://github.com/bensonlee5/tab-foundry/issues/202),
    [#203](https://github.com/bensonlee5/tab-foundry/issues/203),
    [#204](https://github.com/bensonlee5/tab-foundry/issues/204), and
    [#205](https://github.com/bensonlee5/tab-foundry/issues/205)
  - the named TF-RD-022 runtime policy surface now inherits the measured
    medium winner (`mixed_precision=bf16`, `trace_activations=false`,
    `activation_checkpointing=true`)
  - issue [#239](https://github.com/bensonlee5/tab-foundry/issues/239) now
    records a completed same-host CUDA training-throughput decomposition on
    the low-risk loader-overlap and non-blocking-transfer path: a short screen
    advanced `transfer` and `loader_overlap`, then the full replay measured
    `transfer` at `best_training_time=3081.4472`,
    `final_training_time=3110.9282`, `best_roc_auc=0.6584391`,
    `best_log_loss=0.5329700`, `best_brier_score=0.3622467`,
    `best_bpc=2.1101876` and `loader_overlap` at
    `best_training_time=3063.8761`, `final_training_time=3127.7445`,
    `best_roc_auc=0.6608182`, `best_log_loss=0.5358135`,
    `best_brier_score=0.3647582`, `best_bpc=2.1100665` against the same-host
    baseline `best_training_time=5142.3102`,
    `final_training_time=5373.6851`, `best_roc_auc=0.6608562`,
    `best_log_loss=0.5353731`, `best_brier_score=0.3642339`,
    `best_bpc=2.1077142`; both splits preserved large speedups but neither was
    benchmark-safe across the tracked metrics, so `#239` closes as a
    decomposition-backed defer and the carried runtime policy remains
    unchanged
  - issue [#247](https://github.com/bensonlee5/tab-foundry/issues/247) now
    closes the remaining training-speed question as a profiler-backed
    compile-first kernel investigation on the core sandwich path, while
    leaving Inductor as explicit negative evidence and skipping loader or
    benchmark-orchestration reopeners
  - the completed CUDA compile-debug ladder under issue
    [#247](https://github.com/bensonlee5/tab-foundry/issues/247) now records
    explicit negative evidence for Inductor on the carried surface: the first
    short ladder measured `baseline_uncompiled=80.7s`,
    `compile_eager=97.5s`, `compile_aot_eager=129.4s`,
    `compile_inductor_default=299.2s`, and terminated
    `compile_inductor_max_autotune=1315.7s`; after scalar-hoist,
    feature-type-tensorization, and normalization-traceability fixes, the
    second ladder cleared graph breaks and metadata guards, and the CUDA 12.8
    A100 rerun kept `compile_eager_dynamic` as the only viable compile
    candidate while `compile_inductor_default_dynamic` still failed in the
    backward path
  - the first same-host CUDA 12.8 A100 medium replay pair measured
    `compile_eager_dynamic` as the only viable compile candidate but did not
    close the gate by itself because the matched-budget metric was slightly
    worse despite a runtime win
  - posthoc benchmark comparison on compile-enabled checkpoints now works
    through the ordinary loader path because benchmark checkpoint loading
    normalizes leading repeated `_orig_mod.`-prefixed state-dict keys without
    mutating ordinary checkpoints
  - the confirming same-host CUDA 12.8 A100 medium replay pair now closes the
    keep decision for `compile_eager_dynamic`: baseline finished in
    `3848.0996s` at matched-budget
    `final_log_loss_at_matched_regime_budget=0.6820820744`, while
    `compile_eager_dynamic` finished in `3586.6358s` at
    `0.6810689708`; that is a `~6.8%` wall-time win plus a benchmark-better
    matched-budget result on the same host, so TF-RD-022 records compile-first
    eager-plus-dynamic-shape execution as the kept kernel-level
    training-acceleration outcome on the carried runtime surface
  - the follow-up 1000-step, `signature_family_optimizer_step_block_length=2`
    optimizer A/B on the packed TF-RD-010 v5 medium surface now keeps Muon as
    the carried packed optimizer: `cached_packed` seeds 1 and 2 averaged
    `end_to_end_wall_seconds=715.6769`,
    `ma100_train_acc=0.3999542236`, and
    `ma100_train_loss_ema=1.4393207842`, while
    `cached_packed_muon` seeds 1 and 2 averaged
    `end_to_end_wall_seconds=693.7159`,
    `ma100_train_acc=0.5102392578`, and
    `ma100_train_loss_ema=1.2215796496`; the broad packed defaults therefore
    stay on `cached_packed_muon` and `cached_packed` remains the explicit
    `schedulefree_adamw` comparator
  - issues [#240](https://github.com/bensonlee5/tab-foundry/issues/240) and
    [#241](https://github.com/bensonlee5/tab-foundry/issues/241) remain useful
    operational evidence on benchmark and materialization speed, but they no
    longer define the remaining TF-RD-022 gate
- this epic now follows the closed TF-RD-010 benchmark contract directly; it
  should not reopen sandwich-parent selection, TF-RD-021, dagzoo RD-002,
  dagzoo RD-005, or broader regime-choice work
- Completed outcomes:
  - the carried TF-RD-022 runtime policy remains the medium-rung winner:
    `mixed_precision=bf16`, `trace_activations=false`, and
    `activation_checkpointing=true`
  - the carried packed optimizer remains Muon on the bounded-streaming packed
    lane; `schedulefree_adamw` stays as the explicit comparator rather than a
    new default
  - low-risk overlap and transfer tweaks remain closed negative evidence under
    issue [#239](https://github.com/bensonlee5/tab-foundry/issues/239)
  - compile-first kernel investigation now closes under issue
    [#247](https://github.com/bensonlee5/tab-foundry/issues/247) with kept
    `runtime.compile_model=true`, `runtime.compile_backend=eager`, and
    `runtime.compile_dynamic=true` on the same-host medium benchmark contract,
    while Inductor and max-autotune stay explicit negative evidence
  - TF-RD-024 now inherits the same benchmark contract, runtime policy, and
    kept training-acceleration verdict without reopening TF-RD-022
- Exit criteria:
  - satisfied: the repo has one explicit runtime policy for the
    classification scaling target, justified by repo-local time and VRAM
    evidence
  - satisfied: the repo has one explicit measured keep outcome for
    kernel-level training acceleration on that settled runtime surface, with
    the low-risk loader or copy path and operational sidecars preserved as
    historical evidence rather than open blockers
  - satisfied: sweep outputs, inspect surfaces, and result summaries expose
    runtime and timing reads plus profiler attribution compactly enough that
    future runs can be compared without manual log inspection
  - satisfied: later TF-RD-024 architecture work and TF-RD-009 preparation can
    inherit the same runtime policy and training-acceleration verdict without
    re-deriving them from scratch

### TF-RD-024: Post-Performance Architecture-Knob Sweep On The Classification-First Sandwich Target

- Status: `completed`
- Milestone: `Complete`
- Goal: close one bounded post-performance sandwich knob read after TF-RD-022
  so TF-RD-009 inherits a fixed runtime policy and one explicit non-scaling
  architecture handoff
- Current state:
  - issue [#233](https://github.com/bensonlee5/tab-foundry/issues/233) tracked
    this post-performance architecture lane and closes on the medium-only
    closeout recorded here
  - sweep `tf_rd_024_classification_knob_sweep_v1` inherited the kept
    compile-first runtime-policy experiment
    `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
    rather than the stale pre-compile TF-RD-022 surface
  - the inherited TF-RD-022 runtime surface stayed fixed for every TF-RD-024
    run: `mixed_precision=bf16`, `trace_activations=false`,
    `activation_checkpointing=true`, `compile_model=true`,
    `compile_backend=eager`, and `compile_dynamic=true`
  - the fresh compile-eager-dynamic anchor completed on the closed TF-RD-010
    benchmark contract at medium `final_log_loss=0.6820309591` and large
    `final_log_loss=0.9298541427`
  - the completed seven-row medium screen found exactly one clear keep-worthy
    improvement: row `02` (`sandwich_heads=2`) at `final_log_loss=0.6762878243`
  - the other seven-row medium results were all worse than the anchor:
    `sandwich_latents=12` `0.7050649599`, `sandwich_ff_expansion=1`
    `0.7113842731`, `sandwich_summary_tokens_per_axis=1` `0.6940640479`,
    `sandwich_self_attention_per_cross=1` `0.7173978046`,
    `head_hidden_dim=64` `0.6912802875`, and `head_hidden_dim=128`
    `0.6834016822`
  - one explicit two-row medium-only follow-up then tested `sandwich_heads=1`
    and `sandwich_pre_row_attention_layers=2` as independent single-change
    probes against the same compile anchor
  - `sandwich_heads=1` won the follow-up at `final_log_loss=0.6603575333`,
    beating both the fresh compile anchor and the earlier `sandwich_heads=2`
    winner
  - `sandwich_pre_row_attention_layers=2` improved over the anchor at
    `0.6780725432` but stayed worse than `sandwich_heads=2` and well behind
    `sandwich_heads=1`
  - large validation was intentionally skipped by decision for the TF-RD-024
    closeout; this lane now relies on medium-only evidence to choose the
    pre-scaling family
  - completed sweep `tf_rd_025_sandwich_rational_activation_screen_v1` remains
    sidecar evidence only; it did not alter the TF-RD-024 closeout or the
    scaling handoff
- Closeout decision:
  - carry `sandwich_heads=1` into TF-RD-009 as the pre-scaling family winner
  - keep `sandwich_pre_row_attention_layers=1`
  - keep the remaining non-scaling knobs at the fresh compile-eager-dynamic
    anchor values unless TF-RD-009 produces new evidence that reopens them
- Exit criteria:
  - satisfied: the repo now has one explicit keep/defer decision on the bounded
    post-performance sandwich knob set under the inherited TF-RD-022 runtime
    policy
  - satisfied: TF-RD-009 can now freeze the remaining non-scaling architecture
    surface and proceed on the inherited benchmark and runtime contract

### TF-RD-026: Broad-ML Grid Sandwich Performance Campaign

- Status: `completed`
- Milestone: `Completed`
- Goal: test broad transformer/optimization-inspired changes against the
  promoted `grid_sandwich` anchor without biasing toward tabular-specific model
  families.
- Current state:
  - row `10`, `delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1`, is the
    promoted sweep anchor and current `cls_workstation_grid_sandwich` surface
  - row `10` cycles two distinct SwiGLU grid mixer layers four times for eight
    total grid-core applications and uses the same medium v6 corpus, Muon
    optimizer, and 5000-step budget as the control replay
  - row `10` final metrics: log loss `0.4181767299`, Brier `0.2551413012`,
    ROC AUC `0.8132547851`, and `2,205,754` parameters
  - row `11` is close comparison evidence at final log loss `0.4183364953`,
    but it uses four unique grid layers and more parameters
  - row `09` was stopped early and is not planned as a canonical completed row
- Anchor status:
  - the sweep anchor and `reference/system_delta_sweeps/index.yaml` now point
    at row `10`
  - `cls_workstation_grid_sandwich` resolves to the promoted row `10`
    architecture; old row configs remain only for reproducibility of the sweep
- Exit criteria:
  - satisfied for medium-rung promotion: row `10` beats the previous grid
    target `0.4221534937` without material Brier or ROC-AUC regression
  - remaining guardrail: validate on a larger or harder rung before treating
    the row `10` shape as a scaling-law replacement rather than the current
    medium anchor

### TF-RD-027: Expedited Grid FFN And Stability Sweeps

- Status: `completed`
- Milestone: `Completed`
- Tracking issue: [#289](https://github.com/bensonlee5/tab-foundry/issues/289)
- Goal: run short 2500-step screens from the TF-RD-026 row `10`
  `grid_sandwich` anchor without reopening width or unique-layer recurrence.
- Config-only sweep:
  - `tf_rd_027_grid_ffn_wd_config_v1` completed the three ready FFN rows:
    GELU 4:1 (`grid_ffn_mode=gelu`, `sandwich_ff_expansion=4`) reached final
    log loss `0.4254928909`, and SwiGLU 8:3 (`grid_ffn_mode=swiglu`,
    `sandwich_ff_expansion=4`) reached `0.4261833356`, and GEGLU 8:3
    (`grid_ffn_mode=geglu`, `sandwich_ff_expansion=4`) reached `0.4268446367`
  - all three 2500-step rows lost to the existing TF-RD-026 row `10` anchor final
    log loss `0.4181767299`; per the winner rule, keep the current anchor FFN
  - the existing TF-RD-026 row `10` anchor remains the comparator; no fresh
    anchor replay is planned
  - row `3` completed on the current anchor FFN surface with
    `optimizer.weight_decay=0.1` and reached final log loss `0.4263516327`;
    keep the carried `0.01` weight decay for the stability probes
- Implementation-backed follow-up:
  - `tf_rd_027_grid_stability_impl_v1` completed on the carried TF-RD-026 row
    `10` anchor config with the original `optimizer.weight_decay=0.01`
  - classification z-loss (`training.classification_z_loss_coeff=1e-4`) reached
    final log loss `0.4243303110`
  - tanh logit softcap (`model.classification_logit_softcap=30.0`) reached
    final log loss `0.4251839620`
  - QK-norm (`model.attention_qk_norm=true`) reached final log loss
    `0.4741464959`
- Exit decision:
  - all TF-RD-027 rows lost to the TF-RD-026 row `10` anchor final log loss
    `0.4181767299` by `final_log_loss_at_matched_regime_budget`
  - keep the TF-RD-026 row `10` anchor unchanged, with the original FFN surface
    and `optimizer.weight_decay=0.01`
  - close TF-RD-027 without stacking z-loss, softcap, or QK-norm

### TF-RD-028: Grid MoE Balanced Capacity Sweep

- Status: `completed`
- Milestone: `Completed`
- Tracking issue: [#295](https://github.com/bensonlee5/tab-foundry/issues/295)
- Goal: test whether sparse core-only SwiGLU MoE FFNs can add usable Grid-Sandwich
  capacity while keeping active FFN compute near the TF-RD-026 row `10` anchor.
- Completed execution:
  - `tf_rd_028_grid_moe_balanced_capacity_v1` compares against the locked
    TF-RD-026 row `10` anchor final log loss `0.4181767299`
  - row `01` completed the 100-step benchmark-capable `E=4`, `top_k=1` smoke
    gate with W&B online, finite MoE aux metrics, checkpoint export, benchmark
    registration, and final log loss `0.7274165025`
  - `E=2`, `E=4`, and `E=8` top-1 rows completed at the 5000-step budget with
    final log losses `0.4228972081`, `0.4221998345`, and `0.4219883094`
  - the original `E=4`, `top_k=2` row failed from single-GPU CUDA OOM on the
    L40S VM and is treated as resource-confounded
  - the remediated `E=4`, `top_k=2`, `task_batch_size=8`,
    `grad_accum_steps=8` row completed with final log loss `0.4234291888`
  - W&B route-health telemetry stayed finite and non-collapsed across completed
    MoE rows; the remediated top-2 row finished with load-balance `1.0115`,
    router z-loss `0.1266`, entropy `1.3059`, and expert fractions
    `0.2358` to `0.2619`
- Exit decision:
  - no TF-RD-028 row beat the TF-RD-026 row `10` anchor by
    `final_log_loss_at_matched_regime_budget`
  - best MoE candidate was `E=8`, `top_k=1` at final log loss `0.4219883094`,
    still `+0.0038115794` worse than the anchor
  - top-2/min-2 routing is technically healthy after microbatch remediation but
    costs substantially more wall time and underperforms both the anchor and the
    cheaper top-1 MoE rows
  - close TF-RD-028 without promoting a MoE row; if MoE is revisited, do not
    spend another rung on `E=8` top-1, and instead go directly to a top-2/min-2
    follow-up, ideally with either normalized top-k routing or an always-on
    shared expert rather than extending this completed sweep

### TF-RD-029: Grid MoE Normalized Top-2 Follow-Up

- Status: `completed`
- Milestone: `Next`
- Tracking issue: [#296](https://github.com/bensonlee5/tab-foundry/issues/296)
- Goal: test whether token-centric top-2 routing works better when the selected
  router probabilities are renormalized per token before expert outputs are
  combined, without spending another candidate rung on `E=8` top-1.
- Completed sweep:
  - `tf_rd_029_grid_moe_top2_normalized_v1` compares against the locked
    TF-RD-026 row `10` anchor final log loss `0.4181767299`
  - row `01`, `delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1`, is a
    100-step benchmark-capable smoke on `E=4`, `top_k=2`,
    `model.grid_moe_normalize_top_k=true`, `task_batch_size=8`, and
    `grad_accum_steps=8`
  - row `02`, `delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1`, is the
    5000-step candidate with the same `E=4`, normalized top-2, and remediated
    microbatch settings
  - both rows pin the exact TF-RD-026 row `10` corpus artifact
    `tf_rd_010_dagzoo_medium_control_curated_v6/tf_rd_010_dagzoo_medium_control_curated_v6__b54f7409b6d5`
- Comparison set:
  - TF-RD-026 row `10` anchor: final log loss `0.4181767299`
  - TF-RD-028 `E=4`, `top_k=1`: final log loss `0.4221998345`
  - TF-RD-028 `E=8`, `top_k=1`: final log loss `0.4219883094`
  - TF-RD-028 remediated raw `E=4`, `top_k=2`: final log loss `0.4234291888`
- Completed outcomes:
  - smoke row completed with W&B online, checkpoint export, benchmark artifacts,
    and final log loss `0.7118454598`
  - row `02` completed with W&B online run `sqoffpeo`, final log loss
    `0.4218944597`, delta `+0.0037177297` versus the anchor, final Brier
    `0.2575044437`, final ROC AUC `0.8111413310`, and train elapsed seconds
    `9812.1306`
  - route health was finite and non-collapsed at closeout: load-balance loss
    `1.0079`, router z-loss `0.1010`, router entropy `1.2750`, expert fraction
    range `0.2375` to `0.2618`, and expert fraction std `0.0093`
  - normalized top-2 became the best MoE row so far, narrowly beating TF-RD-028
    `E=8` top-1 by log loss, but it still underperformed the TF-RD-026 row `10`
    dense anchor and took roughly `+5738.8s` more training time than the anchor
- Exit decision:
  - close TF-RD-029 without promotion
  - continue MoE only through the separately scoped TF-RD-030 shared-expert and
    auxiliary-schedule hypothesis; otherwise keep TF-RD-026 row `10` as the
    active anchor

### TF-RD-030: Grid MoE Shared-Expert Follow-Up

- Status: `in_progress`
- Milestone: `Next`
- Tracking issue: [#297](https://github.com/bensonlee5/tab-foundry/issues/297)
- Goal: test whether preserving an always-on dense SwiGLU path inside the
  grid-core FFN while adding token-centric routed experts makes MoE easier to
  optimize than forcing all FFN capacity through sparse routing.
- Planned sweep:
  - `tf_rd_030_grid_moe_shared_expert_v1` compares against the locked
    TF-RD-026 row `10` anchor final log loss `0.4181767299`
  - row `01`, `delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1`, is a
    100-step benchmark-capable smoke on `E=4`, `top_k=1`,
    `model.grid_moe_shared_expert=true`, `task_batch_size=8`, and
    `grad_accum_steps=8`
  - row `02`, `delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1`, is the
    full shared-expert top-1 candidate with the same remediated microbatch
    settings
  - row `03`,
    `delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1`, combines the
    shared expert with normalized token-centric top-2 routing
  - row `04`,
    `delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1`, keeps the
    normalized top-2 path without a shared expert and decays the load-balance
    coefficient from `1e-2` to `1e-3` after the early routing-settlement window
  - row `05`,
    `delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1`, is an
    optional router-temperature smoothing probe if the shared top-1 row is
    healthy but still noisy
  - all rows pin the exact TF-RD-026 row `10` corpus artifact
    `tf_rd_010_dagzoo_medium_control_curated_v6/tf_rd_010_dagzoo_medium_control_curated_v6__b54f7409b6d5`
- Execution gate:
  - do not start the full rows until the shared-expert smoke row completes on
    Vast with W&B online, finite MoE losses, route-health telemetry,
    checkpoint export, and benchmark artifacts
  - prioritize rows `02` and `03`; run row `04` if TF-RD-029 or row `03`
    suggests top-2 is healthy but over-regularized, and run row `05` only if
    the shared top-1 path shows accuracy oscillation worth smoothing
- Decision rule:
  - promote a MoE variant only if it beats the TF-RD-026 row `10` anchor on
    final log loss under the same benchmark contract without route collapse or
    unacceptable wall-time cost
  - otherwise close the grid-core MoE thread as negative evidence and keep
    TF-RD-026 row `10` as the active anchor

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

- Status: `research`
- Milestone: `Later`
- Goal: keep steering-derived dagzoo corpus-front expansion available as a
  synthetic-data sidecar without blocking the main TF-RD-022 → TF-RD-024 →
  TF-RD-009 path on the closed TF-RD-010 benchmark contract
- Current state:
  - issue [#165](https://github.com/bensonlee5/tab-foundry/issues/165) is the
    successor synthetic-data epic after the first sandwich many-class
    benchmark contract is established
  - TF-RD-020 already records the historical staged-control harder-front
    winners, while TF-RD-010 now records the fixed first sandwich benchmark
    contract that later lanes already inherit directly
  - completed dagzoo issue
    [#246](https://github.com/bensonlee5/dagzoo/issues/246) remains historical
    steering context, while dagzoo epics
    [#249](https://github.com/bensonlee5/dagzoo/issues/249) and
    [#247](https://github.com/bensonlee5/dagzoo/issues/247) now define the
    upstream surface expansion that TF-RD-021 will inherit
  - this lane no longer blocks TF-RD-022 runtime work, the new TF-RD-024
    bounded architecture sweep, or TF-RD-009 scaling-law preparation
- Required work:
  - if this sidecar resumes, reuse the closed TF-RD-010 benchmark contract,
    then wait for dagzoo RD-002 and RD-005 under issues
    [#249](https://github.com/bensonlee5/dagzoo/issues/249) and
    [#247](https://github.com/bensonlee5/dagzoo/issues/247) to expand the
    admissible candidate surface and metadata contract
  - freeze one evaluation-ready post-RD-002/RD-005 dagzoo candidate set under
    a dedicated child of issue
    [#165](https://github.com/bensonlee5/tab-foundry/issues/165), including
    named presets, regime identifiers, and the compact metadata needed for
    matched-regime-budget comparisons
  - run one bounded carry-forward sweep under issue
    [#167](https://github.com/bensonlee5/tab-foundry/issues/167) on that
    frozen candidate set: one incumbent control row on the carried sandwich
    dagzoo slice plus a small set of named post-RD-002/RD-005 corpus rows
  - hold architecture, many-class plus missingness regime definition, and
    benchmark contract fixed across every row
  - interpret `final_log_loss_at_matched_regime_budget` first, with
    calibration, runtime, clipped-step fraction, and stability telemetry as
    guardrails
  - keep exactly one expanded dagzoo carry-forward surface only if it clearly
    beats the incumbent control; otherwise keep the original carried slice
  - keep this epic synthetic-data-only rather than absorbing imbalance,
    runtime-kernel, or scaling-law conclusions
- Exit criteria:
  - the repo has one explicit keep/defer decision on whether any frozen
    post-RD-002/RD-005 corpus front replaces the carried sandwich dagzoo slice
  - the relationship between TF-RD-021 and TF-RD-010, TF-RD-017, TF-RD-022,
    TF-RD-024, and TF-RD-009 is explicit and non-overlapping

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
    synthetic training fronts feeding `tab-realdata-hub` validation manifests;
    the corrected medium rerun is now complete and the large-rung replay is
    now recorded as a completed local benchmark-only pass, with the original
    `medium_v4` control explicitly retained over the worse `medium_v5`
    sorted-order replay
- Exit criteria:
  - satisfied: the repo has an explicit keep/defer decision that the sandwich
    architecture surface is adequate enough to move into benchmark-defined
    multiclass evaluation
  - satisfied: later work now reuses one coherent carried architecture family
    instead of reopening general simplification or staged-family divergence

### TF-RD-009: Scaling-Law Design And Measurement On The Classification-First Sandwich Target

- Status: `in_progress`
- Milestone: `Next`
- Active issue chain: umbrella issue
  [#51](https://github.com/bensonlee5/tab-foundry/issues/51), completed
  design-note child [#229](https://github.com/bensonlee5/tab-foundry/issues/229),
  completed sweep-program design child
  [#140](https://github.com/bensonlee5/tab-foundry/issues/140), active fixed-budget
  family epic [#253](https://github.com/bensonlee5/tab-foundry/issues/253),
  historical schedulefree width-transfer child
  [#254](https://github.com/bensonlee5/tab-foundry/issues/254), completed
  historical schedulefree joint width-depth child
  [#255](https://github.com/bensonlee5/tab-foundry/issues/255), completed
  historical schedulefree Kaplan-exact Phase-2 fit/report child
  [#256](https://github.com/bensonlee5/tab-foundry/issues/256), follow-on
  historical large-rung validation / hardware-freeze child
  [#257](https://github.com/bensonlee5/tab-foundry/issues/257),
  completed Muon Phase-1 child
  [#275](https://github.com/bensonlee5/tab-foundry/issues/275),
  completed corrected Muon Phase-2 child
  [#274](https://github.com/bensonlee5/tab-foundry/issues/274),
  active Muon cleanup child
  [#283](https://github.com/bensonlee5/tab-foundry/issues/283),
  active compact Muon training-dynamics selector child
  [#284](https://github.com/bensonlee5/tab-foundry/issues/284),
  post-fit frontier and robustness umbrella
  [#258](https://github.com/bensonlee5/tab-foundry/issues/258),
  later Muon upper-family reopen child
  [#269](https://github.com/bensonlee5/tab-foundry/issues/269),
  compute-frontier child
  [#259](https://github.com/bensonlee5/tab-foundry/issues/259), and
  curriculum/repetition slice child
  [#260](https://github.com/bensonlee5/tab-foundry/issues/260)
- Goal: fit the first classification scaling laws on the simplified sandwich
  family only after the repo has the closed TF-RD-010 benchmark contract, one
  TF-RD-022 runtime policy, one TF-RD-024 bounded architecture read, and a
  literature-grounded law-design note
- Current state:
  - as of April 8, 2026, TF-RD-024 closed via
    [#233](https://github.com/bensonlee5/tab-foundry/issues/233), so TF-RD-009
    remained the active `Next` lane on the main roadmap path
  - PR [#252](https://github.com/bensonlee5/tab-foundry/pull/252) merged the
    literature-deepened TF-RD-009 design note and closed
    [#229](https://github.com/bensonlee5/tab-foundry/issues/229)
  - PR [#261](https://github.com/bensonlee5/tab-foundry/pull/261) merged the
    sweep-program design tree and closed
    [#140](https://github.com/bensonlee5/tab-foundry/issues/140)
  - [#254](https://github.com/bensonlee5/tab-foundry/issues/254) replayed the
    carried `sandwich_heads=1` row because the historical TF-RD-024 follow-up
    result was queue-only and not benchmark-registry-backed on `main`
  - the replay established the formal TF-RD-009 anchor at `d_icl=60` with
    `final_log_loss=0.6620`
  - the executed medium width family was `{48, 60(anchor), 96, 128}` under the
    closed TF-RD-010 multiclass contract, the kept TF-RD-022
    compile-eager-dynamic runtime bundle, and matched regime budget
  - `d_icl=48` underperformed the anchor at `final_log_loss=0.6939`
  - `d_icl=96` improved the matched-regime-budget objective to
    `final_log_loss=0.6331`, improved Brier score and ROC AUC, and kept run
    health at `ok`
  - `d_icl=128` achieved the best raw objective result at
    `final_log_loss=0.6225`, but it also produced a health `warn`,
    `max_grad_norm=54.6871`, and sharply worse legacy BPC/BPF diagnostics
  - width-only family conclusion: keep width-only as a live empirical baseline,
    but carry `d_icl=96` into [#255](https://github.com/bensonlee5/tab-foundry/issues/255)
    as the explicit joint width-depth handoff because it is the cleanest
    improved row
  - `tf_rd_009_width_depth_medium_v1` is now complete on the corrected dense
    diagonal `72x1 -> 96x2 -> 112x3 -> 128x4 -> 152x5 -> 176x6` under the same
    closed TF-RD-010 contract and matched regime budget
  - `72x1` improved over the formal `60x2` anchor to `final_log_loss=0.6376`
    and kept the lower diagonal viable without changing the runtime policy
  - `112x3` improved the matched-budget objective to `final_log_loss=0.6046`
    and `final_roc_auc=0.6966`, establishing the first clear mixed-depth gain
    above the carried `96x2` baseline
  - `128x4` regressed to `final_log_loss=0.6348`, so the completed diagonal is
    not monotone in depth and should be fit on measured rows rather than on the
    queue-construction planning axis
  - `152x5` is the current fixed-budget winner at `final_log_loss=0.5740` and
    `final_roc_auc=0.7351`
  - the first registry-backed [#257](https://github.com/bensonlee5/tab-foundry/issues/257)
    rerun exposed an artifact-resolution bug: telemetry listed checkpoints
    through step 2500, but the reusable artifact retained numbered
    `step_*.pt` files only through step 600 plus `latest.pt`, so the old
    `benchmark_checkpoint_selection=all` path stopped on the last preserved
    numbered snapshot
  - `tf_rd_009_large_validation_152x5_v1` is now complete under
    [#257](https://github.com/bensonlee5/tab-foundry/issues/257): corrected
    rerun `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2`
    filtered telemetry-only missing late numbered checkpoints, appended the
    retained terminal `latest.pt` checkpoint at `global_step=2500`, completed
    `25/25` checkpoint comparisons, and finished at
    `final_log_loss=0.7436636568`, `final_brier_score=0.4288940`, and
    `final_roc_auc=0.7650940` on `openml_classification_large_v1`
  - the #257 gate is explicit: keep/defer on whether `152x5` beats the
    TF-RD-010 large clean-control anchor `0.8974410961` at
    `final_log_loss_at_matched_regime_budget`; only a pass should freeze the
    first hardware baseline entry, and the large row itself does not join the
    medium constraint-model evidence
  - the corrected large-rung result is a keep: `delta_final_log_loss=-0.1538`
    and `delta_final_roc_auc=+0.1327` versus the carried anchor, so
    `src/tab_foundry/bench/hardware_architecture_baselines_v1.json` now
    freezes `tf_rd_009_rtx8000_44gb_classification_medium_v1` from medium
    evidence rows only with preferred `152x5`, formal anchor `60x2`, and
    baseline `96x2`; the large row remains a gate rather than a fitted
    constraint-model point
  - `176x6` completed cleanly at `final_log_loss=0.5816` and
    `final_roc_auc=0.7238`; keep it as upper-family and near-ceiling evidence,
    and treat the old "stop at `176x6`" guidance as historical closeout for
    the first fixed-budget family rather than a live global rule
  - observed training VRAM reserved for the top rows was `16.59 GiB` at
    `152x5` and `17.84 GiB` at `176x6`, materially below the old pre-run
    width-evidence memory bridge; after the corrected [#257](https://github.com/bensonlee5/tab-foundry/issues/257)
    pass, the frozen baseline now records the medium-evidence constraint model
    `P_local(d, L) ≈ 18638.80 + 77.94 * d^2 + 47.93 * L * d^2`,
    `reserved_vram_gb ≈ 8.69 + 9.271e-07 * params`, and
    `train_wall_seconds ≈ 8298.45 + 2.275e-04 * params` instead of treating
    the old bridge as freeze-time evidence
  - TF-RD-021 remains sidecar corpus context under
    [#165](https://github.com/bensonlee5/tab-foundry/issues/165) rather than a
    blocker for this lane
  - as of April 12, 2026, the current Phase-2 result is a complete
    validation-backed fit over 44 points: 24 `ns_core` rows from
    `tf_rd_009_ns_medium_v1` and 20 `batch_critical` rows from
    `tf_rd_009_batch_critical_medium_v1`, rooted under
    `outputs/research_scaling/tf_rd_009_phase2`
  - as of April 13, 2026 PT, that Phase-2 fit is historical/superseded for
    scaling-law and `Cmin` interpretation: the higher-budget `NS` and
    `batch_critical` rows cycle the same 143,976-task train manifest, so the
    corrected rerun uses `tf_rd_010_dagzoo_medium_control_curated_v6` and new
    artifact ids `tf_rd_009_ns_one_epoch_medium_v1`,
    `tf_rd_009_batch_critical_one_epoch_medium_v1`, and
    `tf_rd_009_phase2_one_epoch_v1`
  - the validation-backed `L(N,S)` surface is the useful primary signal:
    `alpha_n=0.0302565`, `alpha_s=0.331430`, `Nc=258222760.6`, `Sc=608.501`,
    `log_space_r2=0.820915`, and `rmse=0.033284`
  - the C axis has been audited and corrected for reused 2,500-step rows:
    NS orders `07`, `11`, `15`, `19`, `23`, plus batch-critical order `11`;
    the full `L(C)` diagnostic is `alpha_c=0.521774`,
    `Cc=5.456582059841496e11`, `log_space_r2=0.237521`, and `rmse=0.036255`
  - the completed batch-critical envelope is weak and should be carried as a
    caution: `Bcrit(L)` uses only two envelope points with `alpha_b=0.00459242`
    and `log_space_r2=-0.0649503`; the derived `L(Cmin)` fit reports
    `alpha_cmin=0.123823`, `log_space_r2=0.915117`, and `rmse=0.014289`, but
    depends on that weak `Bcrit(L)` relation
  - `tab-foundry research scaling audit` now adds the Phase-2 fit-audit layer:
    validation-vs-benchmark target comparisons, leave-one-geometry and
    leave-one-step residual checks, bootstrap intervals, diagnostic
    broken-power-law univariate checks, and an iso-loss `Bcrit(L)` readiness
    gate before treating any derived `Cmin` relation as compute-optimal
  - as of April 13, 2026, [#269](https://github.com/bensonlee5/tab-foundry/issues/269)
    reopened the historical fixed-budget upper family under the corrected
    post-`#257` hardware model and recorded a D-optimal continuation in the
    historical selector artifact; treat that continuation as preserved context
    only for the schedulefree branch, because the active Muon reboot reruns the
    upper-family selector only after Muon Phase 2 lands
  - `tf_rd_009_ns_upper_extension_medium_v1` stays intentionally empty until
    the reopened gate rows return benchmark-backed health=`ok`; this child is
    law-information-first, so healthy upper rows may expand even if they do
    not beat `152x5` on the carried matched-budget benchmark objective
  - the interrupted upper-extension gate launch is also historical diagnostic
    evidence under the one-epoch correction; corrected upper-extension ids
    `tf_rd_009_width_depth_upper_extension_one_epoch_medium_v1`,
    `tf_rd_009_ns_upper_extension_one_epoch_medium_v1`, and
    `tf_rd_009_phase2_upper_extension_one_epoch_v1` remain scaffolded until
    `tf_rd_009_phase2_one_epoch_v1` lands and the D-optimal row selection is
    rerun on corrected validation `L(N,S)`
  - as of April 14, 2026 PT, the schedulefree/v5 branch plus the corrected
    schedulefree one-epoch rerun under [#254](https://github.com/bensonlee5/tab-foundry/issues/254),
    [#255](https://github.com/bensonlee5/tab-foundry/issues/255),
    [#256](https://github.com/bensonlee5/tab-foundry/issues/256), and
    [#257](https://github.com/bensonlee5/tab-foundry/issues/257) is preserved
    as historical TF-RD-009 context only; keep [#256](https://github.com/bensonlee5/tab-foundry/issues/256)
    closed and unchanged
  - the active reboot is now the fresh Muon family under completed Muon
    Phase-1 child [#275](https://github.com/bensonlee5/tab-foundry/issues/275)
    then
    [#274](https://github.com/bensonlee5/tab-foundry/issues/274), frozen to
    `tf_rd_010_dagzoo_medium_control_curated_v6`, the post-PR
    [#271](https://github.com/bensonlee5/tab-foundry/pull/271) packed/runtime
    stack on `main`, and the explicit Muon experiment alias
    `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
  - the fresh Muon width screen `tf_rd_009_muon_width_screen_medium_v1` has now
    landed on the fixed v6 contract at the full 2500-step budget:
    `60x2=0.4172`, `48x2=0.4147`, `96x2=0.4128`, and `128x2=0.3951` final
    matched-regime-budget log loss; keep `60x2` as the formal external Muon
    anchor, but carry `128x2` forward as the current in-family baseline for the
    diagonal derivation
  - as of April 15, 2026 PT, `tf_rd_009_muon_width_depth_medium_v1` is complete
    on the rederived fresh-Muon Phase-1 family `72x1`, `112x3`, `144x4`,
    `192x5`, and `264x6`, ordered with the carried width-screen baseline as
    `{72x1, 112x3, 128x2, 144x4, 192x5, 264x6}` by measured parameter count;
    matched-budget final log loss landed at `0.4135`, `0.4137`, `0.4116`,
    `0.4146`, and `0.4009` respectively, so `264x6` is the current benchmark-
    backed Muon Phase-1 winner
  - April 15, 2026 A6000 W&B system-memory evidence recorded peak allocated
    memory `4.28 GB` at `72x1`, `5.48 GB` at `112x3`, `8.11 GB` at `144x4`, and
    the winning `264x6` row still fit comfortably with `peak_vram_reserved`
    `17.20 GB`; keep that as headroom evidence, but do not reopen a larger-
    model Muon Phase-1 extension before Phase 2
  - `src/tab_foundry/bench/hardware_architecture_baselines_v1.json` now keeps
    the Muon placeholder
    `tf_rd_009_rtx8000_44gb_classification_medium_muon_v1` in `planned` state
    while carrying `264x6` as the current preferred candidate; the freeze still
    waits for a later large-rung Muon validation on the candidate architecture
  - [#274](https://github.com/bensonlee5/tab-foundry/issues/274) is now
    complete and merged via [#280](https://github.com/bensonlee5/tab-foundry/pull/280)
    under corrected sparse multiclass replay against
    `openml_classification_medium_v1`; the earlier provisional binary-backed
    interpretation is superseded. The corrected NS family prefers
    `144x4 @ 5000 = 0.4914031270`, while the corrected batch-critical family
    prefers `264x6 @ grad_accum=16 @ 5000 = 0.4646411887`, which is the
    current best corrected Muon Phase-2 row overall. All `40/40` rows are now
    validation-backed, the primary `L(N,S)` validation fit lands at
    `log_space_r2=0.9123` with `rmse=0.0172`, and the audit explicitly keeps
    `Bcrit` and derived `Cmin` diagnostic-only with `ready_for_cmin=false`
  - the corrected Phase-2 result changes the next defended move: batch-side
    gains now dominate geometry-only gains on the active Muon surface, so the
    next lane is training-dynamics-first rather than immediate upper-family
    reopen or compute-frontier work
  - the active Muon lineage is now the canonical TF-RD-009 path:
    `tf_rd_009_muon_width_screen_medium_v1` ->
    `tf_rd_009_muon_width_depth_medium_v1` ->
    `tf_rd_009_muon_ns_one_epoch_medium_v1` ->
    `tf_rd_009_muon_batch_critical_one_epoch_medium_v1` ->
    `tf_rd_009_muon_phase2_one_epoch_v1`; the historical schedulefree family
    under [#254](https://github.com/bensonlee5/tab-foundry/issues/254),
    [#255](https://github.com/bensonlee5/tab-foundry/issues/255),
    [#256](https://github.com/bensonlee5/tab-foundry/issues/256), and
    [#257](https://github.com/bensonlee5/tab-foundry/issues/257) remains
    preserved context only
  - the completed `#284` execution surface is the strict shared-anchor LMO
    transfer sweep:
    `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`
  - the earlier screen-based transfer sweeps
    `tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1` and
    `tf_rd_009_muon_training_dynamics_transfer_medium_v1` remain preserved
    superseded context only
  - the earlier heuristic endpoint selector
    `tf_rd_009_muon_training_dynamics_endpoint_medium_v1` remains tracked
    superseded context only; it is no longer the canonical training-dynamics
    decision surface
- Closeout state:
  - keep [#253](https://github.com/bensonlee5/tab-foundry/issues/253) as the
    authoritative umbrella for TF-RD-009, but treat the historical
    schedulefree family under [#254](https://github.com/bensonlee5/tab-foundry/issues/254),
    [#255](https://github.com/bensonlee5/tab-foundry/issues/255),
    [#256](https://github.com/bensonlee5/tab-foundry/issues/256), and
    [#257](https://github.com/bensonlee5/tab-foundry/issues/257) as completed
    context only
  - treat [#275](https://github.com/bensonlee5/tab-foundry/issues/275) as the
    completed fresh-Muon Phase-1 closeout: keep `60x2` as the formal external
    anchor, keep `128x2` as the carried in-family baseline, and carry the now-
    benchmark-backed `264x6` winner into later Muon planning without reopening
    a larger-model Phase-1 extension first
  - keep [#274](https://github.com/bensonlee5/tab-foundry/issues/274) and
    merged PR [#280](https://github.com/bensonlee5/tab-foundry/pull/280) as the
    corrected Muon Phase-2 closeout: keep `L(N,S)` on validation loss as the
    primary directional law, keep benchmark loss as external ranking evidence,
    and keep `Bcrit` / `Cmin` diagnostic-only with `ready_for_cmin=false`
  - [#283](https://github.com/bensonlee5/tab-foundry/issues/283) is now
    satisfied on this branch: `research sweep list-sweeps` renders the lineage
    tree, `research sweep inspect` exposes the active benchmark/corpus/carried
    baseline/shared-anchor winner surface, `research scaling inspect` exposes
    linked sweeps plus winner and fit/audit readiness, and the default-anchor
    contract guards now fail fast on benchmark, baseline, corpus, anchor, and
    replay-completeness drift
  - [#284](https://github.com/bensonlee5/tab-foundry/issues/284) is complete
    on the fixed `144x4` sandwich surface (`d_icl=144`,
    `sandwich_layers=4`, `sandwich_latents=24`, `sandwich_heads=1`,
    `sandwich_summary_tokens_per_axis=3`, `sandwich_ff_expansion=2`,
    `sandwich_self_attention_per_cross=4`,
    `sandwich_pre_row_attention_layers=1`,
    `sandwich_pre_column_attention_layers=1`, and
    `sandwich_pre_column_inducing_tokens=16`) against the corrected
    `openml_classification_medium_v1` benchmark and
    `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - the completed strict shared-anchor LMO sweep kept Regime `B` over Regime
    `D` by the tracked `T2`-then-`T1` rule:
    `B@T1=0.5239001271`, `B@T2=0.5143437440`,
    `D@T1=0.5268794041`, `D@T2=0.5169818590`
  - the best overall row in the completed study remains the imported carried
    low-batch baseline `T2` row `3` at `0.4914031270`; the carried high-batch
    `T2` row `6` landed at `0.5135392585`
  - final transfer-study verdict: the kept law-derived regime `B` did **not**
    beat carried high-batch at `T2`
    (`0.5143437440` vs `0.5135392585`, delta `+0.0008044855` log loss), so
    close `#284` without opening any later Phase-2B rerun issue from this lane
  - the April 20-21, 2026 routed/grid sidecar on the carried `144x4` medium
    surface (`head_hidden_dim=96`) is complete: `routed_control=0.5120092736`
    and `routed_rebalance=0.5661516574` are negative evidence for the first
    routed-residual/evidence-bank design, while `grid_pilot=0.4221534937`
    established the grid-preserving family; TF-RD-026 row `10` has since
    promoted the current repo architecture anchor through
    `cls_workstation_grid_sandwich` at final log loss `0.4181767299`
  - only after the completed cleanup child and negative faithful transfer study
    closeout should
    [#269](https://github.com/bensonlee5/tab-foundry/issues/269) be revisited
    for any Muon upper-family move; treat any later larger-model probe as
    post-selector work instead of as a blocker on the current active reboot path
  - only after some later separate study produces a better-conditioned
    multi-geometry batch surface should
    [#259](https://github.com/bensonlee5/tab-foundry/issues/259) be revisited
    for the compute-frontier / Chinchilla-style branch; keep
    [#260](https://github.com/bensonlee5/tab-foundry/issues/260) separate as a
    repetition or curriculum slice
  - maintain hardware baselines statefully in
    `src/tab_foundry/bench/hardware_architecture_baselines_v1.json`: the
    historical frozen schedulefree entry remains
    `tf_rd_009_rtx8000_44gb_classification_medium_v1`, while the new
    `tf_rd_009_rtx8000_44gb_classification_medium_muon_v1` entry stays
    `planned` even though it now carries the real Muon width-screen evidence and
    active Phase-1 sweep id; if a later Muon medium winner emerges, require a
    fresh one-row large-rung validation before replacing the frozen preferred
    baseline
  - keep the closed TF-RD-010 benchmark contract, the kept post-#271 packed
    runtime surface, and the retained compact non-scaling sandwich knobs fixed
    across the fresh Muon family; compare by matched regime budget with
    `final_log_loss_at_matched_regime_budget` as the ranking objective and
    calibration, stability, and runtime as explicit guardrails
- Exit criteria:
  - the repo can fit width-depth classification laws on the simplified sandwich
    architecture under the closed TF-RD-010 classification benchmark contract
    and one inherited runtime policy
  - scaling artifacts compare runs by matched regime budget with final log loss
    as the primary objective, and stability, calibration, and runtime as
    guardrails
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
  active grid path now runs: grid-anchor promotion, first
  benchmark-defined many-class plus missingness gate, bounded kernel/runtime
  tuning, bounded post-performance architecture sweep, then scaling-law work.
- The current large-anchor hybrid line is diagnostic evidence, not the intended
  architecture destination.
