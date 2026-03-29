# TF-RD-010: Benchmark-Defined Multiclass Evolution On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-010](../../docs/development/roadmap.md#tf-rd-010-benchmark-defined-multiclass-evolution-on-the-classification-first-sandwich-target).

- Status: `completed`
- Milestone: `Completed`
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
  umbrella for this lane, and issue
  [#99](https://github.com/bensonlee5/tab-foundry/issues/99) is the first
  execution issue; both now serve as historical handoff context for the
  completed package
- completed child issues
  [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
  [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
  [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
  [#200](https://github.com/bensonlee5/tab-foundry/issues/200) define the
  multiclass corpora, freeze the missing baselines, and execute the medium and
  large validation packages
- `tab-realdata-hub` issue
  [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) is the
  canonical upstream dependency for medium and large multiclass validation
  bundles and materialized manifests
- the completed sweep contracts now live in
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
- medium and large validation manifests now exist at
  `data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet`
  and
  `data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet`
- `tab-foundry` froze
  `cls_benchmark_linear_multiclass_medium_v1` and
  `cls_benchmark_linear_multiclass_large_v1` against those manifests before
  execution

## Current Interpretation

- This lane should be benchmark-first, not anchor-first
- Prior TF-RD-021B evidence is historical context only; the active target is an
  evolved sandwich classification surface
- The benchmark program should make the repo-to-repo linkage explicit:
  - `dagzoo` defines synthetic training fronts
  - `tab-realdata-hub` defines medium and large real-data validation bundles
    plus materialized manifests
  - `tab-foundry` consumes those manifests and ranks rows by
    `final_bpc_at_matched_regime_budget`
- BPC is the normalized log-loss view for the first expanded multiclass regime;
  raw log loss, calibration, runtime, and stability remain supporting guardrails
- The completed short-run package established the benchmark contract without
  promoting a new carried front: all medium and large rows deferred, and every
  row failed the short-run stability guardrail
- MCAR gave the best BPC deltas on both rungs (`-2.5701` medium and
  `-62725.0640` large), but it still degraded final ROC AUC and did not clear
  the guardrails needed for promotion
- Missingness should be addressed in both places:
  - synthetic training fronts via control, MCAR, MAR, and MNAR corpora
  - validation via the large allow-missing benchmark rung
- Class imbalance should be made explicit in benchmark coverage and reporting,
  but a dedicated imbalance ladder remains TF-RD-017 follow-on work

## Closed Evidence Gaps

- `tab-realdata-hub` now owns the medium and large multiclass bundle
  materialization flow, and `tab-foundry` consumes the resulting manifest
  parquet directly
- the multiclass medium and large control baselines are frozen in the canonical
  registry
- the first medium and large benchmark packages have executed, with all rows
  deferred under the short-run guardrails

## Exit Signals

- the repo has one explicit medium-plus-large multiclass benchmark program on
  the evolved sandwich family
- medium and large validation manifests are owned upstream in
  `tab-realdata-hub` and referenced directly by `tab-foundry`
- later steering, imbalance, runtime, and scaling lanes inherit a fixed
  `dagzoo -> tab-realdata-hub -> tab-foundry` contract rather than reopening
  regime selection
