# System Delta Matrix

- Sweep id: `tf_rd_010_classification_evolution_medium_v1`
- Sweep status: `draft`
- Anchor run id: `none`
- Locked validation bundle: `nanotabpfn_openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`

## Purpose

- Close TF-RD-016 by treating the next step as a benchmark-definition and modest head-evolution package rather than a replay of the historical four-token anchor.
- Keep the sandwich backbone family intact while evaluating the evolved classification contract:
  - `feature_type_conditioning=film`
  - `sandwich_summary_tokens_per_axis=3`
  - `many_class_base=10`
  - direct multiclass head
- Keep `dagzoo` as the synthetic training owner and `tab-realdata-hub` as the real-data validation owner.

## Validation Contract

- Medium validation surface: hub-materialized manifest for `nanotabpfn_openml_classification_medium_v1`
- Validation owner: `tab-realdata-hub` issue `bensonlee5/tab-realdata-hub#1`
- Benchmark intent: clean no-missing multiclass rung
- Ranking metric: `final_bpc_at_matched_regime_budget`
- Supporting guardrails: raw log loss, calibration, runtime, stability, feature-count coverage, class-count coverage, and minority-class reporting

## Rows

| Order | Delta | Training Front | Draft Purpose |
| --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | `tf_rd_010_dagzoo_medium_control_v1` | Clean multiclass control |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | `tf_rd_010_missingness_mcar_v1` | Random missingness comparison |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | `tf_rd_010_missingness_mar_v1` | Structured observed-feature missingness |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | `tf_rd_010_missingness_mnar_v1` | Strongest self-masking missingness |

## Linkages

- `dagzoo` defines the synthetic training fronts.
- `tab-realdata-hub` defines the medium and large multiclass validation bundles and materialized manifests.
- `tab-foundry` consumes those manifests for BPC-ranked benchmark evaluation.
- Class imbalance is addressed through explicit benchmark coverage and reporting, while any stronger skew ladder remains TF-RD-017 follow-on work.
