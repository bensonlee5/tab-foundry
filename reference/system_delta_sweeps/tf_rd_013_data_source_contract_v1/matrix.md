# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_013_data_source_contract_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_013_data_source_contract_v1`
- Sweep status: `draft`
- Parent sweep id: `qass_tfcol_large_missing_validation_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4215`, final Brier score `0.2644`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final training time `2550.1s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| model anchor | Upstream nanoTabPFN has no repo-local row-first promoted-anchor contract. | The settled promoted row-first benchmark anchor `row_cls + qass + no tfcol`. | TF-RD-013 changes only the training-data comparison surface, not the promoted model surface. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Current manifest-backed prior-training corpus with data surface label `anchor_manifest_default`. | The current corpus is the baseline comparator; TF-RD-013 rows define explicit dagzoo and curated real-data alternatives against it. |
| dagzoo provenance contract | Not applicable. | No dagzoo provenance is attached to the current-corpus anchor surface. | Dagzoo candidate rows must carry one explicit `dagzoo_provenance` payload with fixed keys before the sweep becomes runnable. |
| real-data comparator policy | Not applicable. | Benchmark bundle `nanotabpfn_openml_binary_large` remains the benchmark-facing evaluation surface. | Curated real-data ladders are comparator evidence and must keep one OpenML baseline plus approved manifest-backed augmentations rather than introducing a new loader path. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | TF-RD-013 contract work should not mix optimizer or schedule changes into the data-source decision. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_filter_policy_accepted_only` | provenance | yes | blocked_on_artifacts | none | Rebuild the training manifest with accepted-only dagzoo filtering while holding model and preprocessing at the anchor. | Materialize the accepted-only manifest and fill the dagzoo command lineage under issue 120 before scheduling any comparison run. |
| 2 | `delta_data_manifest_root_curated_dagzoo` | provenance | yes | blocked_on_artifacts | none | Point training at a curated dagzoo manifest root with explicit dagzoo command provenance. | Materialize the curated dagzoo root lineage and support artifacts under issue 120, then wire the promoted-anchor comparison row under issue 121. |
| 3 | `delta_data_manifest_curated_realdata_comparator` | source | yes | blocked_on_artifacts | none | Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations. | Materialize the approved OpenML-baseline comparator manifest and cite any approved manifest-backed augmentations before scheduling this row. |

## Detailed Rows

### 1. `delta_data_filter_policy_accepted_only`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Rebuild the training manifest with accepted-only dagzoo filtering while holding model and preprocessing at the anchor.
- Rationale: Define the first canonical dagzoo candidate surface by tightening filtering while keeping the promoted row-first anchor otherwise fixed.
- Hypothesis: An accepted-only dagzoo candidate may improve signal quality, but its value is only interpretable once manifest lineage and corpus-shape deltas are explicit.
- Upstream delta: Not applicable; this is a repo-local manifest/provenance decision.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but replace the current corpus with a TF-RD-013 accepted-only dagzoo candidate manifest that carries the canonical dagzoo provenance payload.
- Expected effect: Cleaner data quality with a possible diversity-versus-quality tradeoff.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_accepted_only`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/tf_rd_013/dagzoo_accepted_only/manifest.parquet', 'filter_policy': 'accepted_only', 'dagzoo_provenance': {'corpus_variant': 'dagzoo_accepted_only', 'comparator_role': 'promoted_anchor_candidate', 'commands': [], 'config_refs': ['configs/default.yaml'], 'curated_root_lineage': [], 'materialization_issue': 120}}`
- Parameter adequacy plan:
  - Compare manifest characteristics against the current-corpus anchor before reading any benchmark outcome.
  - Treat the empty `commands` list as a contract placeholder only; issue 120 is responsible for filling the actual command lineage.
- Adequacy knobs to dimension explicitly:
  - Must report dataset-count and feature/class distribution shifts versus the anchor manifest.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row is contract-only until the accepted-only manifest and lineage are materialized.
- Notes:
  - This row defines the canonical top-level dagzoo provenance keys expected on TF-RD-013 promoted-anchor comparison surfaces.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_data_source_contract_v1/delta_data_filter_policy_accepted_only/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_curated_dagzoo`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at a curated dagzoo manifest root with explicit dagzoo command provenance.
- Rationale: Define the promoted-anchor curated dagzoo root as a second canonical dagzoo candidate surface rather than leaving it as an old binary-md precursor row.
- Hypothesis: A curated dagzoo root may better match intended post-008 use than the current corpus, but it must be read through explicit lineage from the accepted-only candidate rather than as an opaque manifest swap.
- Upstream delta: Not applicable; this is a repo-local dataset-generation axis.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but point training at the curated dagzoo root contract while preserving the same canonical dagzoo provenance payload shape.
- Expected effect: Potentially cleaner training data with a materially different corpus composition.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_curated_dagzoo_root`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/tf_rd_013/curated_dagzoo/manifest.parquet', 'dagzoo_provenance': {'corpus_variant': 'dagzoo_curated_root', 'comparator_role': 'promoted_anchor_candidate', 'commands': [], 'config_refs': ['configs/default.yaml'], 'curated_root_lineage': ['dagzoo_accepted_only', 'curated_root_selection'], 'materialization_issue': 120}}`
- Parameter adequacy plan:
  - Compare this row first against the accepted-only dagzoo candidate, then against the current-corpus anchor.
  - Treat `curated_root_lineage` as required contract metadata, not as optional commentary.
- Adequacy knobs to dimension explicitly:
  - dagzoo generate/filter commands
  - curated root lineage
  - dataset count and split distribution deltas
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row remains a contract placeholder until issue 120 materializes the curated root manifest and lineage.
- Notes:
  - This row supersedes the old `binary_md_v1` `delta_data_manifest_root_curated_dagzoo` precursor as the TF-RD-013 dagzoo-root contract surface.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_data_source_contract_v1/delta_data_manifest_root_curated_dagzoo/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_curated_realdata_comparator`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations.
- Rationale: Make the curated real-data comparator contract explicit so TF-RD-013 reads dagzoo against a defined comparator lane instead of a vague future workstream.
- Hypothesis: The first comparator should stay anchored on one canonical OpenML baseline and only add approved manifest-backed external augmentations when they cover regimes OpenML misses.
- Upstream delta: Not applicable; this is a repo-local comparator contract layered on top of the benchmark-native OpenML baseline.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but define the curated real-data comparator manifest family as a separate contract surface from both the current corpus and dagzoo candidates.
- Expected effect: A real-data comparator lane that is explicit enough to compare against current-corpus and dagzoo candidates without reopening loader boundaries.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_curated_realdata_comparator`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/manifest.parquet'}`
- Parameter adequacy plan:
  - Keep the OpenML baseline canonical and cite approved external augmentations only after issue 114 license approval rows exist.
  - Interpret this row as comparator-surface contract work, not as a new ingestion pathway.
- Adequacy knobs to dimension explicitly:
  - Approved ledger rows for every dataset referenced by the comparator manifest family.
  - OpenML baseline versus approved external augmentation coverage notes.
  - Manifest lineage and regime-coverage notes attached before any benchmark read is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - The comparator surface depends on later curation and license backfill work under issues 97, 106, and 114.
- Notes:
  - This row defines comparator policy only; it does not authorize any dataset outside the review ledger.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_data_source_contract_v1/delta_data_manifest_curated_realdata_comparator/result_card.md`
- Benchmark metrics: pending
