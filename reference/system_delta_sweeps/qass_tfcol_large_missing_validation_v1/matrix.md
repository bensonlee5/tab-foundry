# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/qass_tfcol_large_missing_validation_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `qass_tfcol_large_missing_validation_v1`
- Sweep status: `completed`
- Parent sweep id: `qass_tfcol_large_no_missing_validation_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_no_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4227`, final Brier score `0.2712`, best ROC AUC `0.8567`, final ROC AUC `0.8567`, final training time `2399.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block with test-self attention enabled. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Shifted grouped tokenizer. | Tokenizer changes reshape the effective table sequence and need their own adequacy commentary. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Row-CLS pooling path. | Row-pool changes alter how the table summary is extracted and should be isolated from context changes. |
| context encoder | None on the upstream direct path. | QASS context encoder. | QASS changes both compute graph depth and label-context semantics and needs explicit adequacy notes. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_large` (12 tasks, missing values permitted) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_qass_no_column_v3` | context_encoder | yes | completed | qass_context | Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface. | Treat this row as the settled post-008 default row-first classification anchor for future work on the missing-permitting large bundle and later post-008 fronts. |
| 2 | `delta_qass_context_tfcol_heads4_v1` | column_encoding | yes | completed | qass_context | Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty. | Retain this row as a documented calibration-oriented alternative, but do not treat it as the default post-008 line. |

## Detailed Rows

### 1. `delta_qass_no_column_v3`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface.
- Rationale: Establish the no-TFCol QASS control on the larger missing-permitting bundle so the TFCol candidate could be judged against a same-bundle row-first baseline during TF-RD-008 closure.
- Hypothesis: The larger missing-permitting bundle should keep the no-TFCol QASS line as a competitive default baseline even if the TFCol line still improves some calibration-facing metrics.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder and no QASS stage.
- Anchor delta: Re-run the completed `delta_qass_no_column_v3` architecture on the larger missing-permitting bundle to create the same-bundle control for the heads4 validation row.
- Expected effect: Isolate whether QASS itself helps on the row-embedding surface without TFCol confounding the result.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0266`); context (grad `0.0475`)
- Model overrides: `{'module_overrides': {'column_encoder': 'none'}, 'stage': 'qass_context'}`
- Parameter adequacy plan:
  - This row serves only as the same-bundle control for row 2 on the TF-RD-008 missing-data validator and should be interpreted in that paired context.
  - Do not interpret this row by itself as a new attribution result; the decision comes from row 2 versus row 1 on this bundle.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - The missing-bundle closure sweep finished with a mixed result, so this simpler no-TFCol row becomes the default by the explicit simplicity and lower-runtime tie-breaker.
  - Final metrics for the canonical run were log loss `0.42151056`, Brier `0.26437641`, and ROC AUC `0.67022423`.
  - This row stays materially smaller than the retained TFCol variant (`868,706` parameters versus `1,295,618`), which is part of the default-selection rationale.
  - `nanotabpfn_error.kind=helper_failed_on_missing_bundle` remains recorded for the external helper, but it does not block closure because the tab-foundry comparison package completed on the same benchmark surface for both rows.
  - Canonical rerun registered as `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_large_missing_validation_v1/delta_qass_no_column_v3/result_card.md`
- Registered run: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1` with final log loss `0.4215`, delta final log loss `-0.0012`, final Brier score `0.2644`, delta final Brier score `-0.0069`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final-minus-best `+0.0000`, delta final ROC AUC `-0.1865`, delta drift `+0.0000`, delta final training time `+150.8s`

### 2. `delta_qass_context_tfcol_heads4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty.
- Rationale: Validate whether the selected calibration-oriented `qass + tfcol_heads4` line remains default-worthy once missing-valued tasks are allowed on the larger bundle.
- Hypothesis: If the no-missing `heads4` win generalizes rather than collapsing on missing-valued tasks, row 2 should beat row 1 on both final log loss and final Brier while keeping the final ROC AUC gap within `-0.005`.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from row 1 on the larger missing-permitting bundle, add TFCol back in at the selected `heads4` setting and compare the result only against the same-bundle no-TFCol QASS control.
- Expected effect: Preserve the calibration gain of the QASS+TFCol line while reducing TFCol attention cost enough to recover ROC or runtime if the default head count is unnecessary.
- Effective labels: model=`delta_qass_context_tfcol_heads4_v1`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0144`); row (grad `0.0236`); context (grad `0.0590`)
- Model overrides: `{'stage': 'qass_context', 'tfcol_n_heads': 4}`
- Parameter adequacy plan:
  - Read this row only against row 1 on the same large missing-permitting bundle using `vs_parent`.
  - Promote `row_cls + qass + tfcol_heads4` only if row 2 beats row 1 on final log loss and final Brier, and the final ROC AUC delta is no worse than `-0.005`.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Row 2 improved final Brier and final ROC AUC on the missing-permitting bundle, but it missed the planned promotion rule because final log loss was slightly worse than row 1 by about `+0.0000045`.
  - Final metrics for the canonical run were log loss `0.42151508`, Brier `0.26432957`, and ROC AUC `0.67529660`; this remains useful retained evidence for calibration-oriented follow-up work.
  - The best checkpoint looked stronger than the final checkpoint, but TF-RD-008 settles on the planned final-metric rule rather than reopening another adequacy branch inside this issue.
  - This row remains materially larger than the default no-TFCol line (`1,295,618` parameters versus `868,706`), so it is retained as a non-default variant.
  - `nanotabpfn_error.kind=helper_failed_on_missing_bundle` remains recorded for the external helper, but it does not block closure because the tab-foundry comparison package completed on the same benchmark surface for both rows.
  - Canonical rerun registered as `sd_qass_tfcol_large_missing_validation_v1_02_delta_qass_context_tfcol_heads4_v1_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_large_missing_validation_v1/delta_qass_context_tfcol_heads4_v1/result_card.md`
- Registered run: `sd_qass_tfcol_large_missing_validation_v1_02_delta_qass_context_tfcol_heads4_v1_v1` with final log loss `0.4215`, delta final log loss `-0.0012`, final Brier score `0.2643`, delta final Brier score `-0.0069`, best ROC AUC `0.6768`, final ROC AUC `0.6753`, final-minus-best `-0.0015`, delta final ROC AUC `-0.1814`, delta drift `-0.0015`, delta final training time `+1423.0s`
