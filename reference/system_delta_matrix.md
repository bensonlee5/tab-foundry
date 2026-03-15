# System Delta Matrix

This file is rendered from `reference/system_delta_queue.yaml` plus the canonical benchmark registry.

## Locked Surface

- Anchor run id: `01_nano_exact_md_prior_parity_fix_binary_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7615`, final ROC AUC `0.7599`, final training time `355.6s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Same nano feature encoder path. | Functional parity should be close; remaining differences come from the staged wrapper and metadata surface, not the active encoder math. |
| target conditioning | Mean-padded linear target encoder. | Same mean-padded linear target conditioner. | The anchor intentionally preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm block. | This is the strongest structural anchor to upstream parity. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Later row-pool experiments should be interpreted as explicit departures from upstream, not minor tuning. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Grouped-token changes must be isolated from later row/context changes. |
| context encoder | None on the direct binary path. | None on the direct binary path. | Any context encoder row changes both compute graph depth and target-conditioning semantics and therefore needs its own adequacy commentary. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Repo-local manifest-driven prior-training surface at data/manifests/default.parquet. | Data-source changes are first-class sweep rows, not background assumptions. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Runtime support-set preprocessing with mean imputation and train-only label remap/filter semantics. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy entanglement | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_label_token` | target_conditioning | yes | ready | label_token | Replace mean-padded linear target conditioning with train-label embeddings plus a learned test token. | Train the first anchor-only label-token run on the locked medium binary surface. |
| 2 | `delta_shared_feature_norm` | feature_encoding | yes | ready | shared_norm | Replace the internal nano feature path with the shared linear feature encoder while keeping scalar-per-feature tokenization. | Run the isolated shared-feature-encoder comparison. |
| 3 | `delta_prenorm_block` | table_block | yes | ready | prenorm_block | Switch the cell transformer from the nano post-norm block to the staged pre-norm block, still without test self-attention. | Run the pre-norm-only ablation. |
| 4 | `delta_test_self_attention` | table_block | yes | ready | test_self | Allow test-token self-attention inside the pre-norm block without adding any test-to-test cross-attention. | Queue after the pre-norm-only run so the masking effect has a local comparator. |
| 5 | `delta_shifted_grouped_tokenizer` | tokenization | yes | blocked_on_surface_semantics | grouped_tokens | Replace scalar-per-feature tokenization with the shifted grouped tokenizer while keeping downstream row/context structure anchored. | Unblock only after the comparison surface makes tokenizer changes effective, or replace this row with a genuinely isolatable tokenization experiment. |
| 6 | `delta_row_cls_pool` | row_pool | yes | ready | row_cls_pool | Replace target-column pooling with CLS-based row pooling while leaving tokenizer, column encoder, and context encoder otherwise anchored. | Run the isolated row-pool toggle and require interpretation before any verdict. |
| 7 | `delta_column_set_encoder` | column_encoding | yes | ready | column_set | Add the TFCol/ISAB-style set encoder over columns while keeping row pooling and context encoding otherwise fixed. | Run the isolated column-set encoder delta after the row-pool row is understood. |
| 8 | `delta_qass_context` | context_encoder | yes | ready | qass_context | Add the QASS-based sequence context encoder over row embeddings. | Run after simpler model deltas so the interpretation has cleaner context. |
| 9 | `delta_data_filter_policy_accepted_only` | provenance | yes | blocked_on_artifacts | none | Rebuild the training manifest with accepted-only dagzoo filtering while holding model and preprocessing at the anchor. | Materialize the accepted-only manifest and attach its provenance before scheduling training. |
| 10 | `delta_data_manifest_root_curated_dagzoo` | provenance | yes | blocked_on_artifacts | none | Point training at a curated dagzoo manifest root with explicit dagzoo command provenance. | Capture the dagzoo generation/filter lineage first. |
| 11 | `delta_data_manifest_source_binary_iris` | source | yes | ready | none | Swap the training manifest to the small binary iris support surface while keeping the model and preprocessing anchored. | Run only after the matrix explicitly records the anchor-versus-iris manifest characteristics. |
| 12 | `delta_preproc_impute_missing_off` | missing_values | yes | ready | none | Disable runtime mean imputation while keeping the fitted label-remap path intact. | Run once the matrix records the anchor manifest's missingness expectations. |
| 13 | `delta_preproc_all_nan_fill_nonzero` | missing_values | yes | ready | none | Keep imputation on but change the all-NaN fallback fill value from 0.0 to 1.0. | Run after the no-imputation row so missing-value sensitivity has context. |
| 14 | `delta_preproc_label_mapping_surface` | label_mapping | yes | blocked_on_policy_impl | none | Compare the current train-only remap/filter label policy against an alternate surfaced label-mapping policy. | Add a second supported label policy before activating this row. |

## Detailed Rows

### 1. `delta_label_token`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `label_token`
- Description: Replace mean-padded linear target conditioning with train-label embeddings plus a learned test token.
- Rationale: This is the smallest staged departure from upstream nanoTabPFN and should be measured before shared normalization or block-style changes.
- Hypothesis: Discrete label tokens may sharpen small-binary support-set conditioning, but any loss could reflect embedding initialization or calibration drift rather than the idea itself.
- Upstream delta: Upstream nanoTabPFN does not use label tokens on the binary path.
- Anchor delta: Changes only target_conditioner from mean_padded_linear to label_token; feature encoder, tokenizer, table block, row pooling, data, and preprocessing remain anchored.
- Expected effect: Better label separation on some binary tasks with possible increased variance on tiny support sets.
- Effective labels: model=`delta_label_token`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'target_conditioner': 'label_token'}`
- Parameter adequacy plan:
  - Confirm the isolated toggle before changing optimizer settings.
  - If weak, inspect whether the issue is decision-boundary sharpness or late-curve drift.
- Adequacy knobs to dimension explicitly:
  - many_class_base should stay at 2 on the locked binary surface.
  - Negative evidence should be checked for calibration drift before rejection.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_label_token/result_card.md`
- Benchmark metrics: pending

### 2. `delta_shared_feature_norm`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `shared_norm`
- Description: Replace the internal nano feature path with the shared linear feature encoder while keeping scalar-per-feature tokenization.
- Rationale: This isolates shared normalization and shared feature weights without changing the table block or row readout.
- Hypothesis: Shared feature encoding may help the staged family generalize across varied feature identities, but it also moves normalization responsibility outside the nano-exact path.
- Upstream delta: Upstream nanoTabPFN keeps its own internal feature normalization/encoding path.
- Anchor delta: Changes only feature_encoder from nano to shared; tokenizer, target conditioning, table block, row pooling, data, and preprocessing remain fixed.
- Expected effect: Better transfer across heterogeneous columns, with risk that the shared path simply mismatches the compact binary anchor scale.
- Effective labels: model=`delta_shared_feature_norm`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'feature_encoder': 'shared'}`
- Parameter adequacy plan:
  - Keep optimizer and row/context structure fixed for the first pass.
  - Only treat a weak result as structural evidence after checking whether normalization mismatch is the dominant failure mode.
- Adequacy knobs to dimension explicitly:
  - input_normalization remains train_zscore_clip on the anchor unless the queue row explicitly changes preprocessing.
  - If performance drops, note whether the harm appears early or only after longer training.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_shared_feature_norm/result_card.md`
- Benchmark metrics: pending

### 3. `delta_prenorm_block`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `prenorm_block`
- Description: Switch the cell transformer from the nano post-norm block to the staged pre-norm block, still without test self-attention.
- Rationale: This isolates normalization/block ordering before introducing any new attention edges or tokenization changes.
- Hypothesis: Pre-norm may stabilize optimization at compact scale, but a loss may simply mean the head/optimizer neighborhood was tuned around post-norm behavior.
- Upstream delta: Upstream nanoTabPFN uses the post-norm block only.
- Anchor delta: Changes only table_block_style from nano_postnorm to prenorm with allow_test_self_attention=false.
- Expected effect: Potentially smoother optimization and lower drift, with the risk that post-norm remains better matched to the benchmark wrapper.
- Effective labels: model=`delta_prenorm_block`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'table_block_style': 'prenorm', 'allow_test_self_attention': False}`
- Parameter adequacy plan:
  - Compare best ROC AUC, final ROC AUC, and drift jointly.
  - Defer rather than reject if only the late portion of training worsens without a clear hyperparameter adequacy check.
- Adequacy knobs to dimension explicitly:
  - The relevant knobs are still tficl_n_heads, tficl_n_layers, and head_hidden_dim.
  - A weak result is ambiguous unless late-curve stability clearly worsens despite similar best-step performance.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_prenorm_block/result_card.md`
- Benchmark metrics: pending

### 4. `delta_test_self_attention`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `test_self`
- Description: Allow test-token self-attention inside the pre-norm block without adding any test-to-test cross-attention.
- Rationale: This isolates the masking change from tokenizer, row-pool, and context-encoder changes.
- Hypothesis: Keeping each test token's own state through the block may help the direct head, but any failure could mean the surrounding block settings are still mismatched.
- Upstream delta: Upstream nanoTabPFN blocks test-token self attention in the direct path.
- Anchor delta: Starting from the explicit pre-norm block, change only allow_test_self_attention from false to true.
- Expected effect: Better preservation of per-test-token state with possible over-conditioning on individual test rows.
- Effective labels: model=`delta_test_self_attention`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'table_block_style': 'prenorm', 'allow_test_self_attention': True}`
- Parameter adequacy plan:
  - Compare directly against both the anchor and the pre-norm-only row in the written interpretation.
- Adequacy knobs to dimension explicitly:
  - Interpretation depends on the paired pre-norm-only result; a weak result may just indicate the block style itself remains unsettled.
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row is easiest to interpret after delta_prenorm_block has data.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_test_self_attention/result_card.md`
- Benchmark metrics: pending

### 5. `delta_shifted_grouped_tokenizer`

- Dimension family: `model`
- Status: `blocked_on_surface_semantics`
- Binary applicable: `True`
- Legacy cumulative entanglement: `grouped_tokens`
- Description: Replace scalar-per-feature tokenization with the shifted grouped tokenizer while keeping downstream row/context structure anchored.
- Rationale: The anchor keeps the nano feature encoder, which bypasses tokenizer outputs entirely, so this row is not isolatable until the surface changes away from the nano path.
- Hypothesis: Local grouped views may help feature interaction modeling, but this row cannot produce that signal while the nano encoder remains active.
- Upstream delta: Upstream nanoTabPFN keeps one scalar token per feature.
- Anchor delta: Would change only tokenizer from scalar_per_feature to shifted_grouped, but that override is currently ineffective under the anchor's nano feature encoder.
- Expected effect: None until the tokenizer becomes part of the effective computation graph on this surface.
- Effective labels: model=`delta_shifted_grouped_tokenizer`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'tokenizer': 'shifted_grouped'}`
- Parameter adequacy plan:
  - Do not run while tokenizer changes are bypassed by the active feature encoder.
- Adequacy knobs to dimension explicitly:
  - Revisit only after the comparison surface switches to a non-nano feature encoder.
- Interpretation status: `blocked`
- Decision: `None`
- Confounders:
  - The nano feature encoder ignores tokenizer outputs, so this row would execute the anchor path unchanged.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_shifted_grouped_tokenizer/result_card.md`
- Benchmark metrics: pending

### 6. `delta_row_cls_pool`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `row_cls_pool`
- Description: Replace target-column pooling with CLS-based row pooling while leaving tokenizer, column encoder, and context encoder otherwise anchored.
- Rationale: This is the row you called out directly; the structure change is large enough that its introduced capacity knobs must be reported and interpreted, not hidden.
- Hypothesis: CLS pooling may extract better row summaries, but a weak result is not evidence against the mechanism unless tfrow_n_heads, tfrow_n_layers, and tfrow_cls_tokens are adequate.
- Upstream delta: Upstream nanoTabPFN does not use row-level CLS pooling.
- Anchor delta: Changes only row_pool from target_column to row_cls.
- Expected effect: Richer row aggregation with substantial added capacity and masking sensitivity.
- Effective labels: model=`delta_row_cls_pool`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'row_pool': 'row_cls'}`
- Parameter adequacy plan:
  - Initial run isolates the structural toggle only.
  - If neutral-to-bad, perform a bounded adequacy sweep over tfrow_n_heads, tfrow_n_layers, and tfrow_cls_tokens before any reject decision.
  - Document whether the row pool appears under-capacity, over-capacity, or simply incompatible with the anchor tokenizer.
- Adequacy knobs to dimension explicitly:
  - tfrow_n_heads
  - tfrow_n_layers
  - tfrow_cls_tokens
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Row CLS pooling adds both a new aggregation rule and new capacity knobs.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_row_cls_pool/result_card.md`
- Benchmark metrics: pending

### 7. `delta_column_set_encoder`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `column_set`
- Description: Add the TFCol/ISAB-style set encoder over columns while keeping row pooling and context encoding otherwise fixed.
- Rationale: Column-set encoding should be evaluated separately from row CLS pooling and QASS context so its effect is attributable.
- Hypothesis: Explicit column-set reasoning may help heterogeneous feature identity, but harm may indicate that tfcol_n_heads / tfcol_n_layers / tfcol_n_inducing are poorly chosen rather than that the idea is bad.
- Upstream delta: Upstream nanoTabPFN does not include a dedicated column-set encoder.
- Anchor delta: Changes only column_encoder from none to tfcol.
- Expected effect: Better feature-set aggregation with extra capacity and inducing-point sensitivity.
- Effective labels: model=`delta_column_set_encoder`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'column_encoder': 'tfcol'}`
- Parameter adequacy plan:
  - If the first run underperforms, interpret whether the issue looks like under-capacity, inducing-point mismatch, or interaction with the unchanged row pool.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_column_set_encoder/result_card.md`
- Benchmark metrics: pending

### 8. `delta_qass_context`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `qass_context`
- Description: Add the QASS-based sequence context encoder over row embeddings.
- Rationale: Context encoding changes both depth and label-conditioned message passing, so it belongs late in the binary queue after simpler toggles.
- Hypothesis: QASS may help longer-range support/test interactions, but weak results may simply indicate the compact benchmark surface does not justify the added context depth.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder.
- Anchor delta: Changes only context_encoder from none to qass.
- Expected effect: Richer row-sequence conditioning with risk of unnecessary depth on the compact binary surface.
- Effective labels: model=`delta_qass_context`, data=`anchor_manifest_default`, preprocessing=`runtime_default`
- Model overrides: `{'context_encoder': 'qass'}`
- Parameter adequacy plan:
  - Distinguish context-value from pure added-depth effects in the result card.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_qass_context/result_card.md`
- Benchmark metrics: pending

### 9. `delta_data_filter_policy_accepted_only`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Rebuild the training manifest with accepted-only dagzoo filtering while holding model and preprocessing at the anchor.
- Rationale: Filter policy is a data-surface decision, not a silent background tweak.
- Hypothesis: Accepted-only data may improve robustness if the current anchor manifest still contains marginal datasets, but a loss may simply reflect reduced diversity.
- Upstream delta: Not applicable; this is a repo-local manifest/provenance decision.
- Anchor delta: Changes only the data filter policy and the manifest artifact it yields.
- Expected effect: Cleaner data quality with a possible diversity-versus-quality tradeoff.
- Effective labels: model=`anchor_model`, data=`accepted_only_manifest`, preprocessing=`runtime_default`
- Data overrides: `{'filter_policy': 'accepted_only'}`
- Parameter adequacy plan:
  - Rebuild the manifest and compare manifest characteristics before training.
  - If weaker, discuss whether coverage loss or provenance tightening is the likely cause.
- Adequacy knobs to dimension explicitly:
  - Must report dataset-count and feature/class distribution shifts versus the anchor manifest.
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Requires a manifest rebuild to become runnable.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_data_filter_policy_accepted_only/result_card.md`
- Benchmark metrics: pending

### 10. `delta_data_manifest_root_curated_dagzoo`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Point training at a curated dagzoo manifest root with explicit dagzoo command provenance.
- Rationale: The queue must be able to compare not only model structure but also the exact dagzoo call set that defined the training corpus.
- Hypothesis: A more curated dagzoo root may improve signal quality, but any change must be interpreted through manifest lineage and dataset-characteristic shifts.
- Upstream delta: Not applicable; this is a repo-local dataset-generation axis.
- Anchor delta: Changes only the manifest root and its documented dagzoo provenance.
- Expected effect: Potentially cleaner training data with a materially different corpus composition.
- Effective labels: model=`anchor_model`, data=`curated_dagzoo_root`, preprocessing=`runtime_default`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/curated_dagzoo/manifest.parquet', 'dagzoo_provenance': {'commands': [], 'config_refs': [], 'curated_root_lineage': []}}`
- Parameter adequacy plan:
  - Do not run until the dagzoo provenance payload is filled in and the manifest characteristics are attached.
- Adequacy knobs to dimension explicitly:
  - dagzoo generate/filter commands
  - curated root lineage
  - dataset count and split distribution deltas
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Requires a materialized curated dagzoo manifest and provenance references.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_data_manifest_root_curated_dagzoo/result_card.md`
- Benchmark metrics: pending

### 11. `delta_data_manifest_source_binary_iris`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Swap the training manifest to the small binary iris support surface while keeping the model and preprocessing anchored.
- Rationale: This gives the queue one immediately runnable data-source ablation using an existing repo-tracked manifest family.
- Hypothesis: A very clean but tiny training surface may sharpen some local behavior but is likely to underperform because it collapses diversity; that should be interpreted as a coverage tradeoff, not a verdict on the model.
- Upstream delta: Not applicable; upstream nanoTabPFN does not share this repo-local prior-training manifest contract.
- Anchor delta: Changes only the manifest path / source surface.
- Expected effect: Faster, cleaner, but much lower-diversity training data.
- Effective labels: model=`anchor_model`, data=`binary_iris_manifest`, preprocessing=`runtime_default`
- Data overrides: `{'source': 'manifest', 'manifest_path': 'outputs/staged_ladder_support/binary_iris_manifest/manifest.parquet'}`
- Parameter adequacy plan:
  - Treat any weakness primarily as evidence about training-surface coverage unless metrics collapse unambiguously.
- Adequacy knobs to dimension explicitly:
  - Compare manifest row/feature/class distributions to the anchor before training.
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row changes dataset diversity far more than model structure.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_data_manifest_source_binary_iris/result_card.md`
- Benchmark metrics: pending

### 12. `delta_preproc_impute_missing_off`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Disable runtime mean imputation while keeping the fitted label-remap path intact.
- Rationale: Missing-value handling is a first-class preprocessing dimension and should be measured without mixing in data or model changes.
- Hypothesis: If missingness is rare on the anchor surface this may be neutral; if not, harm may reflect simple information loss rather than a meaningful architectural weakness.
- Upstream delta: Upstream notebook preprocessing imputes as part of its benchmark helper path.
- Anchor delta: Changes only preprocessing.impute_missing from true to false.
- Expected effect: Cleaner attribution around imputation usefulness, but possibly brittle feature tensors on some datasets.
- Effective labels: model=`anchor_model`, data=`anchor_manifest_default`, preprocessing=`runtime_no_impute`
- Preprocessing overrides: `{'impute_missing': False}`
- Parameter adequacy plan:
  - If the result is weak, note whether the benchmark tasks actually contain meaningful missingness on the support side.
- Adequacy knobs to dimension explicitly:
  - Interpret against manifest missingness prevalence where available.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_preproc_impute_missing_off/result_card.md`
- Benchmark metrics: pending

### 13. `delta_preproc_all_nan_fill_nonzero`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Keep imputation on but change the all-NaN fallback fill value from 0.0 to 1.0.
- Rationale: This isolates a preprocessing policy knob that is usually hidden inside fitted-state defaults.
- Hypothesis: The row is mainly interpretive; a weak result may simply mean all-NaN columns are rare on the current surface.
- Upstream delta: Upstream helper logic does not expose this repo-local fill-value contract as a first-class surface.
- Anchor delta: Changes only preprocessing.all_nan_fill from 0.0 to 1.0.
- Expected effect: Usually small, but important for understanding whether hidden fill defaults matter on this corpus.
- Effective labels: model=`anchor_model`, data=`anchor_manifest_default`, preprocessing=`runtime_all_nan_fill_one`
- Preprocessing overrides: `{'all_nan_fill': 1.0}`
- Parameter adequacy plan:
  - Interpret near-zero movement as expected if the surface rarely exercises this branch.
- Adequacy knobs to dimension explicitly:
  - Report whether all-NaN columns are actually present in the compared manifests.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_preproc_all_nan_fill_nonzero/result_card.md`
- Benchmark metrics: pending

### 14. `delta_preproc_label_mapping_surface`

- Dimension family: `preprocessing`
- Status: `blocked_on_policy_impl`
- Binary applicable: `True`
- Legacy cumulative entanglement: `none`
- Description: Compare the current train-only remap/filter label policy against an alternate surfaced label-mapping policy.
- Rationale: Label mapping is now explicit in the surface record, but there is not yet a second supported runtime policy to compare.
- Hypothesis: This row is blocked until a legitimate alternate policy exists; until then it serves as a reminder that label policy is part of the system surface, not a hidden constant.
- Upstream delta: Upstream notebook code does not expose this repo-local label-policy contract as a benchmarkable axis.
- Anchor delta: Would change only preprocessing.label_mapping and/or preprocessing.unseen_test_label_policy.
- Expected effect: Unknown until an alternate policy is implemented.
- Effective labels: model=`anchor_model`, data=`anchor_manifest_default`, preprocessing=`runtime_label_policy_alt`
- Preprocessing overrides: `{'label_mapping': 'train_only_remap', 'unseen_test_label_policy': 'filter'}`
- Parameter adequacy plan:
  - Do not run until a real alternative policy exists.
- Adequacy knobs to dimension explicitly:
  - Must record how unseen test labels are handled and whether the effective task definition changes.
- Interpretation status: `blocked`
- Decision: `None`
- Confounders:
  - No alternate runtime label policy has been implemented yet.
- Follow-up run ids: `[]`
- Result card path: `/Users/bensonlee/dev/tab-foundry/outputs/staged_ladder/research/delta_preproc_label_mapping_surface/result_card.md`
- Benchmark metrics: pending
