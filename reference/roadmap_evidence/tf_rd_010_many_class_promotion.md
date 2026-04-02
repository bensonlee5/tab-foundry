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
  historical umbrella for this lane, issue
  [#99](https://github.com/bensonlee5/tab-foundry/issues/99) is the historical
  first execution issue, and issue
  [#202](https://github.com/bensonlee5/tab-foundry/issues/202) is the completed
  trusted-rerun umbrella
- historical child issues
  [#197](https://github.com/bensonlee5/tab-foundry/issues/197),
  [#198](https://github.com/bensonlee5/tab-foundry/issues/198),
  [#199](https://github.com/bensonlee5/tab-foundry/issues/199), and
  [#200](https://github.com/bensonlee5/tab-foundry/issues/200) define the
  TF-RD-010 corpora and freeze the missing baselines
- successor issue [#205](https://github.com/bensonlee5/tab-foundry/issues/205)
  records the completed trusted medium rerun package, issue
  [#203](https://github.com/bensonlee5/tab-foundry/issues/203) records the
  completed trusted large-rung replay, and issue
  [#204](https://github.com/bensonlee5/tab-foundry/issues/204) is the completed
  sandwich refactor follow-up that landed before those reruns
- the factorization-first adequacy gate is tracked in
  `reference/roadmap_evidence/tf_rd_010_synthetic_adequacy_gate.md`
- `tab-realdata-hub` issue
  [#1](https://github.com/bensonlee5/tab-realdata-hub/issues/1) is the
  canonical upstream dependency for medium and large classification validation
  bundles and materialized manifests
- the completed March 30, 2026 historical medium rerun lives in
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v1/`
  and the preserved reset-contract large reference lives in
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v1/`
- the expanded one-epoch successor lineage now lives in
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v2/`,
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v3/`,
  `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/`,
  and `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v2/`
- the evolved sandwich benchmark config fixes:
  - `feature_type_conditioning=film`
  - `sandwich_summary_tokens_per_axis=3`
  - `many_class_base=10`
  - `training.loss_surface=classification`
  - direct multiclass head
- `dagzoo` remains the owner of the synthetic training fronts used by these
  sweeps
- `tab-foundry` benchmark execution already expects materialized manifest
  parquet for validation surfaces, which makes the hub-owned manifest contract
  the right long-term boundary
- medium and large validation manifests now live under the local
  benchmark-manifest output root, with the legacy local output ids
  `openml_classification_medium_v1` and
  `openml_classification_large_v1`
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
    `final_log_loss_at_matched_regime_budget`, interpreted explicitly as
    label-target log loss per test cell
- The active successor sweeps now use one expanded synthetic corpus pass only:
  `prior_dump_batch_size=64`, budgeted over `159984` corpus manifest
  records/tasks per front, which resolves to `2500` optimizer steps instead of
  the historical fixed 400-step or reset-contract 3-step budgets
- active classification-evolution work now optimizes natural-log CE on label
  targets and ranks by matched-budget final log loss per test cell; the older
  `cell_bpc` / BPC lane remains historical context only
- The trusted rerun contract is now closed on the corrected sandwich/training
  surface:
  - `medium_v4` row 1 remains the carried medium control at
    `final_log_loss=0.6811727401`
  - the completed `medium_v5` sorted-order replay is negative control-order
    evidence at `0.6849303354`, so TF-RD-010 keeps the original `medium_v4`
    control rather than switching carry-forward anchors
  - the completed `large_v2` replay preserves the same ordering on the harder
    rung, with control `0.8974410961` ahead of MCAR `0.9155278224`, MAR
    `0.9418792099`, and MNAR `0.9411754209`
- Those older 400-step outcomes remain historical context only:
  all medium and large rows deferred, every row failed the short-run stability
  guardrail, and none of that evidence should be read strongly against the now-closed
  rerun package
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
  that evidence was invalidated and reset out of the canonical sweep state before
  the trusted rerun package landed
- the March 30, 2026 medium rerun is now preserved as historical evidence in
  `tf_rd_010_classification_evolution_medium_v1`, while the large reset
  contract is preserved only as a superseded reference in
  `tf_rd_010_classification_evolution_large_v1`
- `medium_v4`, `medium_v5`, and `large_v2` now provide the canonical rerun
  evidence for the first benchmark-defined many-class plus missingness gate
- the first `medium_v4` CE control-row CPU pilot
  (`sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v3`)
  was intentionally stopped at step `1324` after checkpoint benchmarking
  showed the current best sampled medium-manifest log loss at
  `step_001200.pt` (`0.9988591380293615`), beating both the frozen medium
  control baseline (`1.1045793217810176`) and the preserved `medium_v3`
  control row's best log loss (`1.086222860423438`), but that pilot is now
  historical operational context only because `dagzoo` factorization changed
- the preserved `medium_v3` no-clipping package is historical overfit evidence
  only: rows 1-3 reached very early best benchmark steps and then drifted
  badly, and row 4 was intentionally stopped rather than extended

## Open Evidence Gaps

- no TF-RD-010 rerun gap remains inside this lane
- follow-on missingness work moves to
  [TF-RD-014](tf_rd_014_missingness_robustness.md) only if later robustness
  work wants to go beyond the explicit no-missingness-promotion result from the
  first medium/large package
- follow-on imbalance work remains
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md)
- carry-forward surface expansion, runtime policy, and scaling all move to
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md), and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md)

## Exit Signals

- the repo has one explicit medium-plus-large classification benchmark program
  on
  the evolved sandwich family
- medium and large validation manifests are owned upstream in
  `tab-realdata-hub` and referenced directly by `tab-foundry`
- later steering, imbalance, runtime, and scaling lanes inherit a fixed
  `dagzoo -> tab-realdata-hub -> tab-foundry` contract rather than reopening
  regime selection
- trusted medium and large reruns now replace the invalidated historical
  executions, and later lanes inherit the original `medium_v4` control as the
  carried benchmark slice with no missingness promotion from TF-RD-010
