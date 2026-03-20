# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/qass_tfcol_large_no_missing_validation_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `qass_tfcol_large_no_missing_validation_v1`
- Sweep status: `draft`
- Parent sweep id: `qass_tfcol_adequacy_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_embedding_attribution_v3_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4011`, final Brier score `0.2625`, best ROC AUC `0.7588`, final ROC AUC `0.7588`, final training time `175.1s`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_large_no_missing` (64 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_qass_no_column_v3` | context_encoder | yes | ready | qass_context | Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface. | Run first and treat it as the same-bundle no-TFCol control before reading the heads4 validation row. |
| 2 | `delta_qass_context_tfcol_heads4_v1` | column_encoding | yes | ready | qass_context | Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty. | Run second and interpret only against row 1 on the same large no-missing bundle. |

## Detailed Rows

### 1. `delta_qass_no_column_v3`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but disable the column-set encoder so QASS is measured directly on top of the winning no-context row-embedding surface.
- Rationale: Establish the no-TFCol QASS control on the larger no-missing bundle so the selected calibration-first TFCol row can be judged against a same-bundle row-first baseline.
- Hypothesis: The larger no-missing bundle should keep the no-TFCol QASS line as a strong ROC-oriented baseline even if the calibration-first TFCol line remains preferable for log loss and Brier.
- Upstream delta: Upstream nanoTabPFN direct binary path has no separate context encoder and no QASS stage.
- Anchor delta: Re-run the completed `delta_qass_no_column_v3` architecture on the larger no-missing bundle to create the same-bundle control for the heads4 validation row.
- Expected effect: Isolate whether QASS itself helps on the row-embedding surface without TFCol confounding the result.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'column_encoder': 'none'}, 'stage': 'qass_context'}`
- Parameter adequacy plan:
  - Use this row only as the same-bundle control for row 2 on the large no-missing validator.
  - Do not interpret this row by itself as a new attribution result; the decision comes from row 2 versus row 1 on this bundle.
- Adequacy knobs to dimension explicitly:
  - tficl_n_heads
  - tficl_n_layers
  - tficl_ff_expansion
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row exists to establish the large-bundle control; do not reopen medium-bundle attribution from it.
  - Execution must use the same control baseline, replay payload, and reuse signature as the matching heads4 row so bundle shift is the only external change.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_large_no_missing_validation_v1/delta_qass_no_column_v3/result_card.md`
- Benchmark metrics: pending

### 2. `delta_qass_context_tfcol_heads4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `qass_context`
- Description: Use the public `qass_context` stage, but reduce TFCol attention heads to four so the calibration-winning row-first surface tests whether a lighter attention budget can keep calibration while softening the ROC penalty.
- Rationale: Validate whether the selected calibration-first `qass + tfcol_heads4` line still wins once task count and feature width expand on the larger no-missing bundle.
- Hypothesis: If the medium-bundle `heads4` win is real rather than overfit to the small validator, row 2 should beat row 1 on both final log loss and final Brier while keeping the final ROC AUC gap within `-0.005`.
- Upstream delta: Upstream nanoTabPFN direct binary path has no dedicated column-set encoder and no separate context encoder.
- Anchor delta: Starting from row 1 on the larger no-missing bundle, add TFCol back in at the selected `heads4` setting and compare the result only against the same-bundle no-TFCol QASS control.
- Expected effect: Preserve the calibration gain of the QASS+TFCol line while reducing TFCol attention cost enough to recover ROC or runtime if the default head count is unnecessary.
- Effective labels: model=`delta_qass_context_tfcol_heads4_v1`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'stage': 'qass_context', 'tfcol_n_heads': 4}`
- Parameter adequacy plan:
  - Read this row only against row 1 on the same large no-missing bundle using `vs_parent`.
  - Promote `row_cls + qass + tfcol_heads4` only if row 2 beats row 1 on final log loss and final Brier, and the final ROC AUC delta is no worse than `-0.005`.
- Adequacy knobs to dimension explicitly:
  - tfcol_n_heads
  - tfcol_n_layers
  - tfcol_n_inducing
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Do not reopen `inducing64` or `layers1` unless this row is materially ambiguous on the large-bundle validator.
  - If this row passes, the next follow-up is a single missing-data generalization check on `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`.
  - Execution must use the same control baseline, replay payload, and reuse signature as row 1 so the large-bundle heads4 comparison stays isolated.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/qass_tfcol_large_no_missing_validation_v1/delta_qass_context_tfcol_heads4_v1/result_card.md`
- Benchmark metrics: pending
