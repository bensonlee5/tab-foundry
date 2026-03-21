# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/qass_tfcol_adequacy_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `qass_tfcol_adequacy_v1`
- Sweep status: `draft`
- Parent sweep id: `row_embedding_attribution_v3`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_embedding_attribution_v3_03_delta_qass_context_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3925`, final Brier score `0.2554`, best ROC AUC `0.7528`, final ROC AUC `0.7516`, final training time `267.9s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block with test-self attention enabled. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Shifted grouped tokenizer. | Tokenizer changes reshape the effective table sequence and need their own adequacy commentary. |
| column encoder | None on the upstream direct path. | Transformer column-set encoder (`tfcol`). | Column-set encoding changes how feature interactions are aggregated before row reasoning. |
| row readout | Target-column readout from the final cell tensor. | Row-CLS pooling path. | Row-pool changes alter how the table summary is extracted and should be isolated from context changes. |
| context encoder | None on the upstream direct path. | QASS context encoder. | QASS changes both compute graph depth and label-context semantics and needs explicit adequacy notes. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_qass_context_tfcol_inducing64_v1` | column_encoding | yes | completed | qass_context | Use the public `qass_context` stage, but reduce TFCol inducing points to 64 so the calibration-winning row-first surface tests whether the current set size is overbuilt. | Run first and compare directly against the completed `delta_qass_context_v3` anchor to see whether a smaller inducing set preserves calibration while recovering ROC or runtime. |
| 2 | `delta_qass_context_tfcol_layers1_v1` | column_encoding | yes | completed | qass_context | Use the public `qass_context` stage, but reduce TFCol depth to one layer so the calibration-winning row-first surface tests whether excess column-encoder depth is driving the ROC penalty. | Run second and compare directly against the completed `delta_qass_context_v3` anchor to see whether shallower TFCol depth preserves calibration while improving ROC or runtime. |
| 3 | `delta_qass_context_tfcol_heads4_v1` | column_encoding | yes | completed | qass_context | Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty. | Run third and compare directly against the completed `delta_qass_context_v3` anchor to see whether fewer TFCol heads preserve calibration while improving ROC or runtime. |

## Detailed Rows

### 1. `delta_qass_context_tfcol_inducing64_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but reduce TFCol inducing points to 64 so the calibration-winning row-first surface tests whether the current set size is overbuilt.
- Rationale: Evaluate whether the calibration-winning `qass_context` line is carrying more TFCol inducing capacity than this no-missing medium bundle actually needs.
- Hypothesis: If the default TFCol inducing set is overbuilt, halving it to 64 should preserve the log-loss/Brier gain of `delta_qass_context_v3` while recovering ROC or training time.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from the completed `delta_qass_context_v3` anchor, reduce only `tfcol_n_inducing` from 128 to 64 and keep the rest of the QASS row-first surface fixed.
- Expected effect: Preserve the calibration gain of the QASS+TFCol line while reducing TFCol capacity enough to recover ROC or runtime if the default inducing set is oversized.
- Effective labels: model=`delta_qass_context_tfcol_inducing64_v1`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0132`); row (grad `0.0237`); context (grad `0.0430`)
- Model overrides: `{'stage': 'qass_context', 'tfcol_n_inducing': 64}`
- Parameter adequacy plan:
  - Compare directly against the completed `delta_qass_context_v3` anchor and keep the calibration-first objective fixed.
  - Promote only if the row keeps the calibration win over `delta_qass_no_column_v3` and improves either ROC or training time versus the anchor.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Treat this as a pure inducing-capacity adequacy row; do not reopen the broader TFCol-versus-no-TFCol attribution question on this bundle.
  - If this row wins, validate it next against `delta_qass_no_column_v3` on `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json`.
  - Execution must use the same benchmark bundle, control baseline, and reuse signature as the v3 anchor so only TFCol inducing capacity changes.
  - Canonical rerun registered as `sd_qass_tfcol_adequacy_v1_01_delta_qass_context_tfcol_inducing64_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_adequacy_v1/delta_qass_context_tfcol_inducing64_v1/result_card.md`
- Registered run: `sd_qass_tfcol_adequacy_v1_01_delta_qass_context_tfcol_inducing64_v1_v1` with final log loss `0.4371`, delta final log loss `+0.0446`, final Brier score `0.2858`, delta final Brier score `+0.0304`, best ROC AUC `0.7438`, final ROC AUC `0.7454`, final-minus-best `+0.0017`, delta final ROC AUC `-0.0062`, delta drift `+0.0029`, delta final training time `-18.0s`

### 2. `delta_qass_context_tfcol_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but reduce TFCol depth to one layer so the calibration-winning row-first surface tests whether excess column-encoder depth is driving the ROC penalty.
- Rationale: Evaluate whether the calibration-winning `qass_context` line is paying unnecessary ROC or runtime cost for a TFCol stack that is too deep for this medium no-missing bundle.
- Hypothesis: If the default TFCol depth is the source of the ROC penalty, reducing the column encoder to one layer should preserve most of the calibration gain while improving ROC or training time.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from the completed `delta_qass_context_v3` anchor, reduce only `tfcol_n_layers` from 3 to 1 and keep the rest of the QASS row-first surface fixed.
- Expected effect: Preserve the calibration gain of the QASS+TFCol line while reducing depth enough to recover ROC or runtime if the default column stack is too deep for this bundle.
- Effective labels: model=`delta_qass_context_tfcol_layers1_v1`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0120`); row (grad `0.0248`); context (grad `0.0535`)
- Model overrides: `{'stage': 'qass_context', 'tfcol_n_layers': 1}`
- Parameter adequacy plan:
  - Compare directly against the completed `delta_qass_context_v3` anchor and keep the calibration-first objective fixed.
  - Promote only if the row keeps the calibration win over `delta_qass_no_column_v3` and improves either ROC or training time versus the anchor.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Treat this as a pure depth adequacy row; do not reinterpret plain context or QASS itself from this result.
  - If this row wins, validate it next against `delta_qass_no_column_v3` on `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json`.
  - Execution must use the same benchmark bundle, control baseline, and reuse signature as the v3 anchor so only TFCol depth changes.
  - Canonical rerun registered as `sd_qass_tfcol_adequacy_v1_02_delta_qass_context_tfcol_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_adequacy_v1/delta_qass_context_tfcol_layers1_v1/result_card.md`
- Registered run: `sd_qass_tfcol_adequacy_v1_02_delta_qass_context_tfcol_layers1_v1_v1` with final log loss `0.4024`, delta final log loss `+0.0099`, final Brier score `0.2624`, delta final Brier score `+0.0070`, best ROC AUC `0.7584`, final ROC AUC `0.7463`, final-minus-best `-0.0120`, delta final ROC AUC `-0.0053`, delta drift `-0.0108`, delta final training time `-61.5s`

### 3. `delta_qass_context_tfcol_heads4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty.
- Rationale: Evaluate whether the calibration-winning `qass_context` line can keep its log-loss/Brier gain with a lighter TFCol attention budget on the medium no-missing bundle.
- Hypothesis: If the default TFCol head count is unnecessary, reducing it to four heads should preserve the calibration gain while improving ROC or training time.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from the completed `delta_qass_context_v3` anchor, reduce only `tfcol_n_heads` from 8 to 4 and keep the rest of the QASS row-first surface fixed.
- Expected effect: Preserve the calibration gain of the QASS+TFCol line while reducing TFCol attention cost enough to recover ROC or runtime if the default head count is unnecessary.
- Effective labels: model=`delta_qass_context_tfcol_heads4_v1`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0207`); row (grad `0.0244`); context (grad `0.0507`)
- Model overrides: `{'stage': 'qass_context', 'tfcol_n_heads': 4}`
- Parameter adequacy plan:
  - Compare directly against the completed `delta_qass_context_v3` anchor and keep the calibration-first objective fixed.
  - Promote only if the row keeps the calibration win over `delta_qass_no_column_v3` and improves either ROC or training time versus the anchor.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Treat this as a pure attention-budget adequacy row; do not reopen the TFCol-presence decision on this bundle from this result alone.
  - If this row wins, validate it next against `delta_qass_no_column_v3` on `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json`.
  - Execution must use the same benchmark bundle, control baseline, and reuse signature as the v3 anchor so only TFCol head count changes.
  - Canonical rerun registered as `sd_qass_tfcol_adequacy_v1_03_delta_qass_context_tfcol_heads4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_adequacy_v1/delta_qass_context_tfcol_heads4_v1/result_card.md`
- Registered run: `sd_qass_tfcol_adequacy_v1_03_delta_qass_context_tfcol_heads4_v1_v1` with final log loss `0.3963`, delta final log loss `+0.0038`, final Brier score `0.2595`, delta final Brier score `+0.0041`, best ROC AUC `0.7526`, final ROC AUC `0.7538`, final-minus-best `+0.0012`, delta final ROC AUC `+0.0022`, delta drift `+0.0024`, delta final training time `-26.5s`
