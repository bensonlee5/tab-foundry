# TF-RD-010: Benchmark-Defined Multiclass Evolution On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-010](../../docs/development/roadmap.md#tf-rd-010-benchmark-defined-multiclass-evolution-on-the-classification-first-sandwich-target).

- Status: `partial`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md), feeds
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md),
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md), and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md), and does so through a
  benchmark program rather than a separate architecture family

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is primarily `EquiTabPFN` plus broader tabular
  foundation-model references that keep label conditioning modular
- Dedicated many-class benchmark literature remains thin inside the repo; the
  next sources to curate are multiclass calibration, missingness-aware
  evaluation, class-imbalance reporting, and many-class efficiency references

## Repo-Local Evidence

- issue [#52](https://github.com/bensonlee5/tab-foundry/issues/52) is the
  historical umbrella for this lane, issue
  [#99](https://github.com/bensonlee5/tab-foundry/issues/99) is the historical
  first execution issue, and issue
  [#202](https://github.com/bensonlee5/tab-foundry/issues/202) is the active
  trusted-rerun umbrella
- historical child issues
  [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
  [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
  [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
  [#200](https://github.com/bensonlee5/tab-foundry/issues/200) define the
  TF-RD-010 corpora and freeze the missing baselines
- successor issues [#205](https://github.com/bensonlee5/tab-foundry/issues/205)
  and [#203](https://github.com/bensonlee5/tab-foundry/issues/203) now own the
  trusted medium and large reruns, and issue
  [#204](https://github.com/bensonlee5/tab-foundry/issues/204) is the required
  sandwich refactor follow-up that lands before those reruns
- `tab-realdata-hub` issue
  [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) is the
  canonical upstream dependency for medium and large classification validation
  bundles and materialized manifests
- the reset sweep contracts now live in
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v1/`
  and
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v1/`
- the evolved sandwich benchmark config fixes:
  - `feature_type_conditioning=film`
  - `sandwich_summary_tokens_per_axis=3`
  - `many_class_base=10`
  - `training.loss_surface=cell_bpc`
  - direct multiclass head
- `dagzoo` remains the owner of the synthetic training fronts used by these
  sweeps
- `tab-foundry` benchmark execution already expects materialized manifest
  parquet for validation surfaces, which makes the hub-owned manifest contract
  the right long-term boundary
- medium and large validation manifests now live under the local
  benchmark-manifest output root, with the legacy local output ids
  `nanotabpfn_openml_classification_medium_v1` and
  `nanotabpfn_openml_classification_large_v1`
- those manifests are materialized from
  `tab-realdata-hub/src/tab_realdata_hub/bench/openml_classification_medium_v1.json`
  and
  `tab-realdata-hub/src/tab_realdata_hub/bench/openml_classification_large_v1.json`,
  whose checked-in bundle policy is `min_classes=2`, `max_classes=10`, and
  `max_missing_pct=20.0`
- `tab-foundry` froze
  the legacy baseline ids `cls_benchmark_linear_multiclass_medium_v1` and
  `cls_benchmark_linear_multiclass_large_v1` against those manifests before
  execution

## Current Interpretation

- This lane should be benchmark-first, not anchor-first
- Prior TF-RD-021B evidence is historical context only; the active target is an
  evolved sandwich classification surface
- The benchmark program should make the repo-to-repo linkage explicit:
  - `dagzoo` defines synthetic training fronts with balanced explicit coverage:
    row totals `128/256/512/1024`, feature counts `6/10/14/20`, and class
    counts covering every integer `2..10`, with every synthetic dataset capped
    at `<=1024` total rows
  - `tab-realdata-hub` defines medium and large real-data validation bundles
    plus materialized manifests, with `min_classes=2`, `max_classes=10`, and
    `max_missing_pct=20.0`
  - `tab-foundry` consumes those manifests and ranks rows by
    `final_bpc_at_matched_regime_budget`
- Trusted reruns now use one synthetic corpus pass only:
  `prior_dump_batch_size=64`, budgeted over corpus manifest records/tasks, with
  the concrete runtime step count derived from the corpus task count instead of
  a fixed 400-step contract
- BPC is the normalized log-loss view for the first expanded classification regime;
  raw log loss, calibration, runtime, and stability remain supporting guardrails
- The benchmark contract remains valid, but the previously recorded medium and
  large executions are no longer trusted as canonical evidence after later
  training and sandwich correctness fixes
- Those old 400-step outcomes remain historical context only:
  all medium and large rows deferred, every row failed the short-run stability
  guardrail, and MCAR gave the best BPC deltas on both rungs (`-2.5701` medium
  and `-62725.0640` large) without clearing the promotion guardrails
- Missingness should be addressed in both places:
  - synthetic training fronts via control, MCAR, MAR, and MNAR corpora
  - validation via the medium and large hub bundles, both of which now permit
    missing-valued tasks under the upstream bundle policy
- Class imbalance should be made explicit in benchmark coverage and reporting,
  but a dedicated imbalance ladder remains TF-RD-017 follow-on work

## Closed Evidence Gaps

- `tab-realdata-hub` now owns the medium and large classification bundle
  materialization flow, and `tab-foundry` consumes the resulting manifest
  parquet directly
- the legacy medium and large TF-RD-010 control baselines are frozen in the
  canonical registry
- the first medium and large benchmark packages were executed historically, but
  that evidence has now been invalidated and reset out of the canonical sweep state

## Open Evidence Gaps

- the trusted TF-RD-010 medium rerun under
  [#205](https://github.com/bensonlee5/tab-foundry/issues/205) still needs to
  re-establish canonical medium-rung evidence on the refactored sandwich
  surface and the single-epoch synthetic contract
- the trusted TF-RD-010 large rerun under
  [#203](https://github.com/bensonlee5/tab-foundry/issues/203) still needs to
  re-establish canonical large-rung evidence on the refactored sandwich
  surface and the single-epoch synthetic contract
- the sandwich refactor follow-up under
  [#204](https://github.com/bensonlee5/tab-foundry/issues/204) lands before any
  new TF-RD-010 rerun is recorded as canonical evidence

## Exit Signals

- the repo has one explicit medium-plus-large classification benchmark program
  on
  the evolved sandwich family
- medium and large validation manifests are owned upstream in
  `tab-realdata-hub` and referenced directly by `tab-foundry`
- later steering, imbalance, runtime, and scaling lanes inherit a fixed
  `dagzoo -> tab-realdata-hub -> tab-foundry` contract rather than reopening
  regime selection
- trusted medium and large reruns replace the invalidated historical executions
  before later lanes treat TF-RD-010 execution as canonical again
