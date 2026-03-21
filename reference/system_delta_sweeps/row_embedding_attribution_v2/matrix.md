# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/row_embedding_attribution_v2/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `row_embedding_attribution_v2`
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
| 1 | `delta_row_embeddings_no_context_v2` | row_pool | yes | completed | row_cls_pool | Use the public `row_cls_pool` stage on the grouped-token replay surface, but disable plain context so row embeddings are measured without added context depth. | Run first as the stage-native row-embedding control, then compare against the paired plain-context row. |
| 2 | `delta_row_embeddings_plain_context_v2` | context_encoder | yes | completed | row_cls_pool | Use the public `row_cls_pool` stage unchanged so plain row-level context is measured against the no-context row-embedding control. | Run second and interpret it directly against the no-context row-embedding control before moving to column-set. |
| 3 | `delta_column_set_plain_context_v2` | column_encoding | yes | completed | column_set | Use the public `column_set` stage unchanged so TF-RD-006 is measured on top of the established plain-context row-embedding surface. | Run only after the plain-context row so TF-RD-006 stays attributable before QASS is introduced. |
| 4 | `delta_qass_context_v2` | context_encoder | yes | completed | qass_context | Use the public `qass_context` stage unchanged so QASS is measured against the matched plain-context `column_set` row. | Run last and interpret only against the matched plain-context column-set row. |

## Detailed Rows

### 1. `delta_row_embeddings_no_context_v2`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `row_cls_pool`
- Description: Use the public `row_cls_pool` stage on the grouped-token replay surface, but disable plain context so row embeddings are measured without added context depth.
- Rationale: Establish the stage-native row-embedding control on top of the benchmark-facing grouped-token replay before any plain-context, column-set, or QASS depth is added.
- Hypothesis: If row-level aggregation is already useful on the grouped-token replay surface, the no-context `row_cls_pool` control should outperform the grouped-token anchor strongly enough to keep TF-RD-005 open even before plain context is enabled.
- Upstream delta: Upstream nanoTabPFN direct binary path has no row-level CLS pooling or separate context encoder.
- Anchor delta: Starting from the grouped-token replay anchor, switch to the public `row_cls_pool` stage but override `context_encoder=none` so row embeddings are measured without added context depth.
- Expected effect: Stage-native row embeddings without plain-context help, isolating whether row-level aggregation alone improves the grouped-token replay surface.
- Effective labels: model=`delta_row_embeddings_no_context_v2`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0503`)
- Model overrides: `{'module_overrides': {'context_encoder': 'none'}, 'stage': 'row_cls_pool'}`
- Parameter adequacy plan:
  - Compare directly against the grouped-token replay anchor before enabling plain row-level context.
  - If neutral-to-bad, perform a bounded adequacy sweep over tfrow_n_heads, tfrow_n_layers, and tfrow_cls_tokens before any reject decision.
  - Treat this as the no-context control for the paired plain-context row.
- Adequacy knobs to dimension explicitly:
  - tfrow_n_heads
  - tfrow_n_layers
  - tfrow_cls_tokens
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Execution must use `--device auto`, `--nanotabpfn-root ~/dev/nanoTabPFN`, and `--prior-dump ~/dev/nanoTabPFN/300k_150x5_2.h5` so the anchor nanoTabPFN curve is reused instead of refreshed.
  - Canonical rerun registered as `sd_row_embedding_attribution_v2_01_delta_row_embeddings_no_context_v2_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v2/delta_row_embeddings_no_context_v2/result_card.md`
- Registered run: `sd_row_embedding_attribution_v2_01_delta_row_embeddings_no_context_v2_v1` with final log loss `0.3991`, delta final log loss `-0.0010`, final Brier score `0.2627`, delta final Brier score `+0.0009`, best ROC AUC `0.7575`, final ROC AUC `0.7580`, final-minus-best `+0.0006`, delta final ROC AUC `+0.0167`, delta drift `+0.0006`, delta final training time `-272.3s`

### 2. `delta_row_embeddings_plain_context_v2`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `row_cls_pool`
- Description: Use the public `row_cls_pool` stage unchanged so plain row-level context is measured against the no-context row-embedding control.
- Rationale: Measure the public `row_cls_pool` stage as the matched plain-context companion to the no-context row-embedding control so TF-RD-005 can separate row embeddings from row-level context value.
- Hypothesis: If plain row-level context is useful beyond row embeddings alone, the stage-native `row_cls_pool` row should beat row 1 without needing TFCol or QASS.
- Upstream delta: Upstream nanoTabPFN direct binary path has no row-level CLS pooling or separate context encoder.
- Anchor delta: Starting from row 1, enable the public plain context path by removing the no-context override while keeping the rest of the `row_cls_pool` stage fixed.
- Expected effect: The public row-embedding stage may justify its added plain-context depth if row-level context is genuinely useful on the grouped-token replay surface.
- Effective labels: model=`delta_row_embeddings_plain_context_v2`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0294`); context (grad `0.0385`)
- Model overrides: `{'stage': 'row_cls_pool'}`
- Parameter adequacy plan:
  - Compare directly against both the grouped-token replay anchor and the no-context row-embedding row.
  - Keep column-set and QASS disabled so plain-context value is separated from later row-first structure.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Execution must preserve the same full-budget replay payload as row 1 so only plain context changes.
  - Canonical rerun registered as `sd_row_embedding_attribution_v2_02_delta_row_embeddings_plain_context_v2_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v2/delta_row_embeddings_plain_context_v2/result_card.md`
- Registered run: `sd_row_embedding_attribution_v2_02_delta_row_embeddings_plain_context_v2_v1` with final log loss `0.4002`, delta final log loss `+0.0001`, final Brier score `0.2638`, delta final Brier score `+0.0020`, best ROC AUC `0.7518`, final ROC AUC `0.7518`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0105`, delta drift `+0.0000`, delta final training time `-248.2s`

### 3. `delta_column_set_plain_context_v2`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `column_set`
- Description: Use the public `column_set` stage unchanged so TF-RD-006 is measured on top of the established plain-context row-embedding surface.
- Rationale: Evaluate TF-RD-006 only after the row-embedding and plain-context surface is established, so column-set evidence is not conflated with earlier row-first transitions.
- Hypothesis: If explicit column-set reasoning belongs in the promoted row-first line, the public `column_set` stage should outperform row 2 on the same 2500-step replay contract.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder.
- Anchor delta: Starting from row 2, add only the public `column_set` stage so TFCol is measured on the established plain-context row-embedding surface.
- Expected effect: Better feature-set aggregation on the stage-native row-first surface, with inducing-point and capacity risk kept explicit.
- Effective labels: model=`delta_column_set_plain_context_v2`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0126`); row (grad `0.0275`); context (grad `0.0435`)
- Model overrides: `{'stage': 'column_set'}`
- Parameter adequacy plan:
  - Compare directly against the plain-context row-embedding row before reading any QASS result.
  - If neutral-to-bad, interpret whether the issue looks like under-capacity, inducing-point mismatch, or interaction with the unchanged plain context path.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row is the matched non-QASS, plain-context control that TF-RD-007 should compare against after it completes.
  - Canonical rerun registered as `sd_row_embedding_attribution_v2_03_delta_column_set_plain_context_v2_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v2/delta_column_set_plain_context_v2/result_card.md`
- Registered run: `sd_row_embedding_attribution_v2_03_delta_column_set_plain_context_v2_v1` with final log loss `0.4271`, delta final log loss `+0.0269`, final Brier score `0.2803`, delta final Brier score `+0.0185`, best ROC AUC `0.7417`, final ROC AUC `0.7396`, final-minus-best `-0.0021`, delta final ROC AUC `-0.0017`, delta drift `-0.0021`, delta final training time `-152.8s`

### 4. `delta_qass_context_v2`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage unchanged so QASS is measured against the matched plain-context `column_set` row.
- Rationale: Evaluate QASS last, and only against the matched plain-context `column_set` row, so TF-RD-007 answers whether QASS earns its added cost beyond the non-QASS row-first baseline.
- Hypothesis: If QASS is genuinely useful on this row-first line, the public `qass_context` stage should beat row 3 on the same 2500-step replay contract rather than just adding depth without value.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder and no QASS stage.
- Anchor delta: Starting from row 3, switch only from plain context to the public `qass_context` stage so QASS is compared against a matched plain-context `column_set` control.
- Expected effect: Richer row-sequence conditioning than the matched plain-context column-set row, with any gain or loss attributable to QASS rather than to row embeddings or TFCol.
- Effective labels: model=`delta_qass_context_v2`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0135`); row (grad `0.0257`); context (grad `0.0456`)
- Model overrides: `{'stage': 'qass_context'}`
- Parameter adequacy plan:
  - Compare directly against the matched plain-context `column_set` row.
  - Treat the plain-context `column_set` row as the non-QASS added-depth control for TF-RD-007.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Do not execute this row if the nanoTabPFN reuse preflight fails; fix the signature mismatch first instead of allowing a fresh helper run.
  - Canonical rerun registered as `sd_row_embedding_attribution_v2_04_delta_qass_context_v2_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v2/delta_qass_context_v2/result_card.md`
- Registered run: `sd_row_embedding_attribution_v2_04_delta_qass_context_v2_v1` with final log loss `0.3925`, delta final log loss `-0.0077`, final Brier score `0.2554`, delta final Brier score `-0.0064`, best ROC AUC `0.7528`, final ROC AUC `0.7516`, final-minus-best `-0.0012`, delta final ROC AUC `+0.0103`, delta drift `-0.0012`, delta final training time `-143.9s`
