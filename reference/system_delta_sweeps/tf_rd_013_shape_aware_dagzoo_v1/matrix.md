# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_013_shape_aware_dagzoo_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_013_data_source_contract_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4215`, final Brier score `0.2644`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final training time `2550.1s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| model anchor | TabICLv2 is the primary row-first architectural reference, but it does not define this exact repo-local promoted-anchor contract. | The settled promoted row-first benchmark anchor `row_cls + qass + no tfcol`. | TF-RD-013 changes only the training-data comparison surface, not the promoted model surface. |
| training data surface | TabICLv2 motivates synthetic pretraining at scale but does not define this repo-local manifest contract. | Current manifest-backed prior-training corpus with data surface label `anchor_manifest_default`. | The broader TF-RD-013 follow-up keeps the current corpus as baseline while testing one explicit multi-invocation dagzoo alternative plus the curated real-data comparator lane. |
| dagzoo provenance contract | Not applicable. | No dagzoo provenance is attached to the current-corpus anchor surface. | The broader dagzoo candidate row should keep one top-level `dagzoo_provenance` payload and make each generate invocation explicit inside `dagzoo_provenance.invocations`. |
| benchmark and control context | TabICLv2 is the architectural reference, while nanoTabPFN remains the current benchmark/control bundle family used by this repo. | Benchmark bundle `openml_binary_large` remains the benchmark-facing evaluation surface. | TF-RD-013 should keep benchmark/control context stable while it reads the training-data surface change. |
| training recipe | TabICLv2 informs the row-first staged recipe direction, but there is no repo-local shared prior-dump training-surface contract to copy literally. | Registered anchor training surface label `prior_linear_warmup_decay`. | The issue 127 follow-up should not mix optimizer or schedule changes into the data-source decision. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_dagzoo_shape_aware_multi_invocation` | provenance | yes | completed | none | Point training at the multi-invocation, shape-aware dagzoo manifest with explicit per-invocation provenance. | Use this completed issue 127 follow-up to close issue 127, keep dagzoo deferred for issue 96, and record in issue 107 that the current corpus remains the representative post-008 training-data surface. |
| 2 | `delta_data_manifest_curated_realdata_comparator` | source | yes | completed | none | Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations. | Keep this comparator evidence-only while closing issue 127 and recording the TF-RD-013 defer decision in issues 96 and 107. |

## Detailed Rows

### 1. `delta_data_manifest_root_dagzoo_shape_aware_multi_invocation`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the multi-invocation, shape-aware dagzoo manifest with explicit per-invocation provenance.
- Rationale: Replace the first single-invocation dagzoo candidate with an explicit shape-aware config ladder so TF-RD-013 can test a broader synthetic-data contract before touching filtering policy.
- Hypothesis: A broader dagzoo surface that mixes smaller, anchor-scale, and larger-shape generated corpora may better approximate the intended long-term synthetic-data lane and produce a more decisive promoted-anchor read than the first default-config-only comparison.
- Upstream delta: Not applicable; this is a repo-local synthetic-data generation axis.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but replace the current corpus with a TF-RD-013 multi-invocation dagzoo manifest assembled from three explicit config-backed generate runs and one merged manifest.
- Expected effect: Broader synthetic training coverage across small, anchor-scale, and large-shape dagzoo regimes while keeping provenance reviewable.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_multi_invocation`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0074`); context (grad `0.0457`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against the current-corpus anchor before reading any benchmark outcome.
  - Name which config-backed dagzoo regimes are present in the merged surface before deciding whether the broader synthetic lane is representative.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the anchor and curated comparator
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Confounders:
  - Direct nanoTabPFN helper comparison remains partially confounded: the missing-value helper run failed (`helper_failed_on_missing_bundle`), and the historical failure mode was tied to `Fitness_Club` non-finite probabilities.
- Notes:
  - Reference-only support bundle: `reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/support/materialization_summary.json` and `reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/support/manifest_characteristics_summary.json`.
  - Issue 124 remains the later filtering-policy question rather than part of this broader shape-aware follow-up.
  - Historical execution `sd_tf_rd_013_shape_aware_dagzoo_v1_01_delta_data_manifest_root_dagzoo_shape_aware_multi_invocation_v1` was invalid because manifest-backed training ran through the prior-dump backend.
  - The completed rerun kept dagzoo deferred: the broader multi-invocation surface still underperformed the anchor on final large-bundle log loss and Brier, so issue 127 does not change the representative-data decision.
  - Training now consumes the full tracked manifest rows; the broader feature-shape mix is preserved without runtime row subsampling.
  - Supersedes historical queue run `sd_tf_rd_013_shape_aware_dagzoo_v1_01_delta_data_manifest_root_dagzoo_shape_aware_multi_invocation_v1`; that invalid run id is preserved in queue notes only.
  - Canonical rerun registered as `sd_tf_rd_013_shape_aware_dagzoo_v1_01_delta_data_manifest_root_dagzoo_shape_aware_multi_invocation_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_shape_aware_dagzoo_v1/delta_data_manifest_root_dagzoo_shape_aware_multi_invocation/result_card.md`
- Registered run: `sd_tf_rd_013_shape_aware_dagzoo_v1_01_delta_data_manifest_root_dagzoo_shape_aware_multi_invocation_v2` with final log loss `0.4658`, delta final log loss `+0.0443`, final Brier score `0.2979`, delta final Brier score `+0.0335`, best ROC AUC `0.5055`, final ROC AUC `0.5111`, final-minus-best `+0.0056`, delta final ROC AUC `-0.1591`, delta drift `+0.0056`, delta final training time `-2222.5s`

### 2. `delta_data_manifest_curated_realdata_comparator`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations.
- Rationale: Rerun the curated real-data comparator inside the issue 127 follow-up sweep so the broader dagzoo result is interpreted against a self-contained current-vs-dagzoo-vs-curated comparison package.
- Hypothesis: The curated comparator should stay OpenML-first and evidence-only, but rerunning it inside the follow-up sweep keeps the broader dagzoo read comparable without reopening any data-loader boundary.
- Upstream delta: Not applicable; this is a repo-local comparator contract layered on top of the benchmark-native OpenML baseline.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but rerun the curated real-data comparator manifest family as the explicit evidence-only lane beside the broader shape-aware dagzoo candidate.
- Expected effect: A real-data comparator lane that is explicit enough to compare against current-corpus and dagzoo candidates without reopening loader boundaries.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_curated_realdata_comparator`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `1.7357`); context (grad `0.4873`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Keep the OpenML baseline canonical and cite approved external augmentations only after issue 114 license approval rows exist.
  - Interpret this row as comparator-surface contract work, not as a new ingestion pathway.
- Adequacy knobs to dimension explicitly:
  - Approved ledger rows for every dataset referenced by the comparator manifest family.
  - OpenML baseline versus approved external augmentation coverage notes.
  - Manifest lineage and regime-coverage notes attached before any benchmark read is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Confounders:
  - Direct nanoTabPFN helper comparison remains partially confounded: the missing-value helper run failed (`helper_failed_on_missing_bundle`), and the historical failure mode was tied to `Fitness_Club` non-finite probabilities.
- Notes:
  - This row defines comparator policy only; it does not authorize any dataset outside the review ledger.
  - The issue 127 follow-up reruns the same OpenML-first comparator lane for same-sweep evidence consistency.
  - The completed same-sweep comparator remained evidence-only and materially worse than the anchor on final large-bundle log loss and Brier.
  - Canonical rerun registered as `sd_tf_rd_013_shape_aware_dagzoo_v1_02_delta_data_manifest_curated_realdata_comparator_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_shape_aware_dagzoo_v1/delta_data_manifest_curated_realdata_comparator/result_card.md`
- Registered run: `sd_tf_rd_013_shape_aware_dagzoo_v1_02_delta_data_manifest_curated_realdata_comparator_v1` with final log loss `2.0849`, delta final log loss `+1.6634`, final Brier score `0.4740`, delta final Brier score `+0.2096`, best ROC AUC `0.5718`, final ROC AUC `0.6002`, final-minus-best `+0.0284`, delta final ROC AUC `-0.0700`, delta drift `+0.0284`, delta final training time `-2221.7s`
