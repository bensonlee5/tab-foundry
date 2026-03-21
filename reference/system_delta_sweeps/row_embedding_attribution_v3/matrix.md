# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/row_embedding_attribution_v3/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `row_embedding_attribution_v3`
- Sweep status: `draft`
- Parent sweep id: `row_embedding_attribution_v2`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_embedding_attribution_v2_01_delta_row_embeddings_no_context_v2_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3991`, final Brier score `0.2627`, best ROC AUC `0.7575`, final ROC AUC `0.7580`, final training time `149.6s`

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
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_qass_no_column_v3` | context_encoder | yes | completed | qass_context | Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface. | Run first and compare directly against the v2 no-context row-embedding anchor before attributing any TFCol effect. |
| 2 | `delta_column_set_no_context_v3` | column_encoding | yes | completed | column_set | Use the public `column_set` stage, but disable plain context so TFCol is measured directly on top of the winning no-context row-embedding surface. | Run second and read it as TFCol-only evidence on the no-context row-embedding base. |
| 3 | `delta_qass_context_v3` | column_encoding | yes | completed | qass_context | Use the public `qass_context` stage unchanged so TFCol is measured as the incremental addition to the QASS-only row-first surface. | Run last and interpret only against the QASS-only row to decide whether TFCol belongs in the promoted row-first line. |

## Detailed Rows

### 1. `delta_qass_no_column_v3`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface.
- Rationale: Evaluate whether QASS itself belongs on the promoted row-first line after v2 showed the no-context row-embedding surface was the clean positive result.
- Hypothesis: If QASS is the actual source of the row-first improvement, `qass_context` with the column encoder disabled should beat the v2 no-context row-embedding anchor directly.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder and no QASS stage.
- Anchor delta: Starting from the v2 no-context row-embedding anchor, enable only the QASS context path while keeping the column encoder off so QASS is isolated from TFCol.
- Expected effect: Isolate whether QASS itself helps on the row-embedding surface without TFCol confounding the result.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0261`); context (grad `0.0478`)
- Model overrides: `{'module_overrides': {'column_encoder': 'none'}, 'stage': 'qass_context'}`
- Parameter adequacy plan:
  - Compare directly against the v2 no-context row-embedding anchor before reading any TFCol result.
  - Treat this as the QASS-only corner of the row-first follow-up factorization.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Execution must use the same benchmark bundle, control baseline, and reuse signature as v2 so QASS is the only modeled change.
  - Canonical rerun registered as `sd_row_embedding_attribution_v3_01_delta_qass_no_column_v3_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v3/delta_qass_no_column_v3/result_card.md`
- Registered run: `sd_row_embedding_attribution_v3_01_delta_qass_no_column_v3_v1` with final log loss `0.4011`, delta final log loss `+0.0019`, final Brier score `0.2625`, delta final Brier score `-0.0002`, best ROC AUC `0.7588`, final ROC AUC `0.7588`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0008`, delta drift `-0.0006`, delta final training time `+25.4s`

### 2. `delta_column_set_no_context_v3`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `column_set`
- Description: Use the public `column_set` stage, but disable plain context so TFCol is measured directly on top of the winning no-context row-embedding surface.
- Rationale: Evaluate whether TFCol itself belongs on the promoted row-first line without reintroducing the plain-context branch that v2 already ruled out.
- Hypothesis: If TFCol alone is useful on the row-embedding base, `column_set` with context disabled should beat the v2 no-context row-embedding anchor directly.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder.
- Anchor delta: Starting from the v2 no-context row-embedding anchor, enable only the column-set encoder while keeping context disabled so TFCol is isolated from both plain context and QASS.
- Expected effect: Isolate whether TFCol itself helps on the row-embedding surface without plain or QASS context confounding the result.
- Effective labels: model=`delta_column_set_no_context_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0309`); row (grad `0.0498`)
- Model overrides: `{'module_overrides': {'context_encoder': 'none'}, 'stage': 'column_set'}`
- Parameter adequacy plan:
  - Compare directly against the v2 no-context row-embedding anchor before reading any QASS-plus-TFCol result.
  - Treat this as the TFCol-only corner of the row-first follow-up factorization.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Do not interpret this row as plain-context evidence; the context path stays disabled so only TFCol changes.
  - Canonical rerun registered as `sd_row_embedding_attribution_v3_02_delta_column_set_no_context_v3_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v3/delta_column_set_no_context_v3/result_card.md`
- Registered run: `sd_row_embedding_attribution_v3_02_delta_column_set_no_context_v3_v1` with final log loss `0.4208`, delta final log loss `+0.0216`, final Brier score `0.2707`, delta final Brier score `+0.0080`, best ROC AUC `0.7467`, final ROC AUC `0.7414`, final-minus-best `-0.0053`, delta final ROC AUC `-0.0166`, delta drift `-0.0058`, delta final training time `+90.1s`

### 3. `delta_qass_context_v3`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage unchanged so TFCol is measured as the incremental addition to the QASS-only row-first surface.
- Rationale: Complete the missing TFCol × QASS factorization by measuring whether TFCol adds value after QASS is already present on the winning row-embedding base.
- Hypothesis: If TFCol is only useful once QASS is already present, the public `qass_context` stage should beat row 1; otherwise the QASS-only row should remain the promoted architecture line.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from row 1, remove the `column_encoder=none` override while keeping QASS fixed so TFCol is the only incremental change.
- Expected effect: Determine whether TFCol adds value once QASS is already present on top of the row-embedding base.
- Effective labels: model=`delta_qass_context_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0135`); row (grad `0.0257`); context (grad `0.0456`)
- Model overrides: `{'stage': 'qass_context'}`
- Parameter adequacy plan:
  - Compare directly against the matched QASS-only row so the incremental TFCol effect stays isolated.
  - Do not interpret this row as plain-context evidence; v2 already resolved that branch.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - If this row is flat or worse than row 1, promote `row_cls + qass + no tfcol` and demote default TFCol from the row-first path.
  - Canonical rerun registered as `sd_row_embedding_attribution_v3_03_delta_qass_context_v3_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_embedding_attribution_v3/delta_qass_context_v3/result_card.md`
- Registered run: `sd_row_embedding_attribution_v3_03_delta_qass_context_v3_v1` with final log loss `0.3925`, delta final log loss `-0.0066`, final Brier score `0.2554`, delta final Brier score `-0.0073`, best ROC AUC `0.7528`, final ROC AUC `0.7516`, final-minus-best `-0.0012`, delta final ROC AUC `-0.0064`, delta drift `-0.0018`, delta final training time `+118.3s`
