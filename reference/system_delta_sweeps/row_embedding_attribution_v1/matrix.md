# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/row_embedding_attribution_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `row_embedding_attribution_v1`
- Sweep status: `draft`
- Parent sweep id: `tokenization_migration_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4002`, final Brier score `0.2618`, best ROC AUC `0.7413`, final ROC AUC `0.7413`, final training time `421.9s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block with test-self attention enabled. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Shifted grouped tokenizer. | Tokenizer changes reshape the effective table sequence and need their own adequacy commentary. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_row_cls_pool` | row_pool | yes | completed | row_cls_pool | Replace target-column pooling with CLS-based row pooling while leaving tokenizer, column encoder, and context encoder otherwise anchored. | Run the isolated row-pool toggle and require interpretation before any verdict. |
| 2 | `delta_plain_context_encoder` | context_encoder | yes | completed | row_cls_pool | Add the plain sequence context encoder over row embeddings while leaving tokenizer, row pooling, and column encoding otherwise fixed. | Run only after the isolated row-pool row so context gains are not conflated with readout changes. |

## Detailed Rows

### 1. `delta_row_cls_pool`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `row_cls_pool`
- Description: Replace target-column pooling with CLS-based row pooling while leaving tokenizer, column encoder, and context encoder otherwise anchored.
- Rationale: Determine whether row embeddings help when introduced on the canonical grouped-token replay selected at the end of TF-RD-004.
- Hypothesis: If row embeddings are viable on the intended migration surface, the isolated row-pool row should outperform the grouped-token control strongly enough to keep the row-first line open.
- Upstream delta: Upstream nanoTabPFN does not use row-level CLS pooling.
- Anchor delta: Keep the benchmark-facing grouped-token replay fixed and replace only target-column pooling with row-CLS pooling while leaving context encoding disabled.
- Expected effect: Richer row aggregation with substantial added capacity and masking sensitivity.
- Effective labels: model=`delta_row_cls_pool`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0000`); row (grad `0.5904`, act delta `-0.0101`)
- Model overrides: `{'module_overrides': {'row_pool': 'row_cls'}}`
- Parameter adequacy plan:
  - Initial run isolates the structural toggle only.
  - If neutral-to-bad, perform a bounded adequacy sweep over tfrow_n_heads, tfrow_n_layers, and tfrow_cls_tokens before any reject decision.
  - Document whether the row pool appears under-capacity, over-capacity, or simply incompatible with the anchor tokenizer.
- Adequacy knobs to dimension explicitly:
  - tfrow_n_heads
  - tfrow_n_layers
  - tfrow_cls_tokens
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - The grouped-token replay anchor serves as the control surface for the three-surface TF-RD-005 comparison and should not be duplicated as an executable row.
  - Canonical rerun registered as `sd_row_embedding_attribution_v1_01_delta_row_cls_pool_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v1/delta_row_cls_pool/result_card.md`
- Registered run: `sd_row_embedding_attribution_v1_01_delta_row_cls_pool_v1` with final log loss `0.6815`, delta final log loss `+0.2814`, final Brier score `0.4884`, delta final Brier score `+0.2266`, best ROC AUC `0.5489`, final ROC AUC `0.5489`, final-minus-best `+0.0000`, delta final ROC AUC `-0.1924`, delta drift `+0.0000`, delta final training time `-11.5s`

### 2. `delta_plain_context_encoder`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `row_cls_pool`
- Description: Add the plain sequence context encoder over row embeddings while leaving tokenizer, row pooling, and column encoding otherwise fixed.
- Rationale: Add plain row-level context only after row embeddings are established so TF-RD-005 can separate readout value from context value.
- Hypothesis: If plain context earns its added depth, it should improve on the isolated row-pool row without needing QASS or column-set changes.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder.
- Anchor delta: Starting from the resolved `delta_row_cls_pool` row, add only the plain context encoder so the final surface matches public `row_cls_pool` behavior without hiding the intermediate row-pool comparison.
- Expected effect: Better row-sequence conditioning than the row-pool-only surface, with materially less added complexity than QASS.
- Effective labels: model=`delta_plain_context_encoder`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0000`); row (grad `0.0000`, act delta `-0.0001`); context (grad `0.3129`, act delta `+0.0002`)
- Model overrides: `{'module_overrides': {'context_encoder': 'plain'}}`
- Parameter adequacy plan:
  - Compare directly against both the grouped-token anchor and the isolated row-pool row.
  - Distinguish plain-context value from row-pool-only effects before introducing column-set or QASS changes.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row should be interpreted directly against both the grouped-token anchor and the resolved row-pool-only row.
  - Canonical rerun registered as `sd_row_embedding_attribution_v1_02_delta_plain_context_encoder_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v1/delta_plain_context_encoder/result_card.md`
- Registered run: `sd_row_embedding_attribution_v1_02_delta_plain_context_encoder_v1` with final log loss `0.6440`, delta final log loss `+0.2439`, final Brier score `0.4517`, delta final Brier score `+0.1899`, best ROC AUC `0.4938`, final ROC AUC `0.4938`, final-minus-best `+0.0000`, delta final ROC AUC `-0.2475`, delta drift `+0.0000`, delta final training time `-318.6s`
