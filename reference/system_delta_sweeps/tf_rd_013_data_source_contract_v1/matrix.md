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

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| model anchor | TabICLv2 is the primary row-first architectural reference, but it does not define this exact repo-local promoted-anchor contract. | The settled promoted row-first benchmark anchor `row_cls + qass + no tfcol`. | TF-RD-013 changes only the training-data comparison surface, not the promoted model surface. |
| training data surface | TabICLv2 motivates synthetic pretraining at scale but does not define this repo-local manifest contract. | Current manifest-backed prior-training corpus with data surface label `anchor_manifest_default`. | The current corpus is the baseline comparator; TF-RD-013 starts with one explicit unfiltered dagzoo alternative plus a later curated real-data comparator lane. |
| dagzoo provenance contract | Not applicable. | No dagzoo provenance is attached to the current-corpus anchor surface. | Dagzoo candidate rows must carry one explicit `dagzoo_provenance` payload with fixed keys before the sweep becomes runnable. |
| benchmark and control context | TabICLv2 is the architectural reference, while nanoTabPFN remains the current benchmark/control bundle family used by this repo. | Benchmark bundle `nanotabpfn_openml_binary_large` remains the benchmark-facing evaluation surface. | TF-RD-013 should keep benchmark/control context stable while it reads the training-data surface change. |
| training recipe | TabICLv2 informs the row-first staged recipe direction, but there is no repo-local shared prior-dump training-surface contract to copy literally. | Training surface label `prior_linear_warmup_decay`. | TF-RD-013 contract work should not mix optimizer or schedule changes into the data-source decision. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_dagzoo_generated_source` | provenance | yes | ready | none | Point training at the unfiltered dagzoo generated-source manifest with explicit dagzoo generate provenance. | Run the promoted-anchor current-versus-unfiltered-dagzoo comparison and record whether the raw generated-source corpus is strong enough to justify later filtering follow-up under issue 124. |
| 2 | `delta_data_manifest_curated_realdata_comparator` | source | yes | blocked_on_artifacts | none | Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations. | Materialize the approved OpenML-baseline comparator manifest and cite any approved manifest-backed augmentations before scheduling this comparator against the current corpus and the unfiltered dagzoo row. |

## Detailed Rows

### 1. `delta_data_manifest_root_dagzoo_generated_source`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the unfiltered dagzoo generated-source manifest with explicit dagzoo generate provenance.
- Rationale: Define the first canonical dagzoo candidate surface as the raw generated-source corpus so TF-RD-013 can read current-versus-dagzoo without waiting on any later filtering policy.
- Hypothesis: The unfiltered dagzoo generated-source corpus may already be competitive enough for the promoted anchor, and the first comparison should establish that before the repo decides whether any later filtering is warranted.
- Upstream delta: Not applicable; this is a repo-local synthetic-data generation axis.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but replace the current corpus with a TF-RD-013 unfiltered dagzoo generated-source manifest that carries the canonical dagzoo provenance payload.
- Expected effect: Higher-throughput synthetic training data with no post-generation filtering and explicit provenance.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_generated_source`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet', 'dagzoo_provenance': {'corpus_variant': 'dagzoo_generated_source', 'comparator_role': 'promoted_anchor_candidate', 'commands': ['cd "$DAGZOO_ROOT" && uv run dagzoo generate --config configs/default.yaml --handoff-root "$TAB_FOUNDRY_ROOT/outputs/staged_ladder_support/tf_rd_013/generated_source" --num-datasets 200 --seed 1 --device cpu --hardware-policy none', 'cd "$TAB_FOUNDRY_ROOT" && ./.venv/bin/tab-foundry data build-manifest --data-root outputs/staged_ladder_support/tf_rd_013/generated_source/generated --out-manifest outputs/staged_ladder_support/tf_rd_013/generated_source/manifest.parquet'], 'config_refs': ['configs/default.yaml'], 'curated_root_lineage': [], 'materialization_issue': 120}}`
- Parameter adequacy plan:
  - Compare manifest characteristics against the current-corpus anchor before reading any benchmark outcome.
  - Treat `filter_status_counts.not_run` as a deliberate property of the initial unfiltered sweep, not as a hidden data-contract bug.
- Adequacy knobs to dimension explicitly:
  - dagzoo generate command lineage
  - dataset count and split distribution deltas
  - filter-status distribution versus the anchor manifest
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row intentionally measures the raw generated-source corpus before any later filtering-policy decision.
  - Any win or weakness here may reflect the unfiltered corpus directly rather than the value of a future filtering pass.
- Notes:
  - This row defines the canonical top-level `dagzoo_provenance` keys expected on TF-RD-013 promoted-anchor comparison surfaces.
  - Reference-only support bundle: `reference/system_delta_sweeps/tf_rd_013_data_source_contract_v1/support/materialization_summary.json` and `reference/system_delta_sweeps/tf_rd_013_data_source_contract_v1/support/manifest_characteristics_summary.json`.
  - Issue 124 is subsequent policy work on whether any filtered dagzoo variants should exist after the initial unfiltered read.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_data_source_contract_v1/delta_data_manifest_root_dagzoo_generated_source/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_curated_realdata_comparator`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Define the curated real-data comparator manifest contract as one OpenML baseline plus any approved manifest-backed augmentations.
- Rationale: Make the curated real-data comparator contract explicit so TF-RD-013 reads unfiltered dagzoo against a defined comparator lane instead of a vague future workstream.
- Hypothesis: The first comparator should stay anchored on one canonical OpenML baseline and only add approved manifest-backed external augmentations when they cover regimes OpenML misses.
- Upstream delta: Not applicable; this is a repo-local comparator contract layered on top of the benchmark-native OpenML baseline.
- Anchor delta: Keep the promoted anchor model, preprocessing, and training recipe fixed, but define the curated real-data comparator manifest family as a separate contract surface from both the current corpus and the unfiltered dagzoo candidate.
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
