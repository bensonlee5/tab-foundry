# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/input_norm_none_followup/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `input_norm_none_followup`
- Sweep status: `completed`
- Parent sweep id: `input_norm_followup`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3972`, final Brier score `0.1307`, best ROC AUC `0.7634`, final ROC AUC `0.7634`, final training time `257.5s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Same nano feature encoder path with internal benchmark normalization. | Feature encoding remains close to upstream parity; later deltas should be attributed elsewhere. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Same mean-padded linear target conditioner. | The anchor preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block without test-self attention. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Tokenization remains aligned with upstream parity. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Row-CLS pooling path. | Row-pool changes alter how the table summary is extracted and should be isolated from context changes. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head remains on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_input_norm_none_batch64_sqrt` | input_normalization | yes | completed | none | Remove explicit input normalization on the batch64 sqrt-scaled bridge winner and keep the larger-batch surface fixed. | Keep `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2` as the preferred batch64 surface; do not reopen `input_normalization=none` on this bundle without new evidence. |

## Detailed Rows

### 1. `dpnb_input_norm_none_batch64_sqrt`

- Dimension family: `preprocessing`
- Status: `completed`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Remove explicit input normalization on the batch64 sqrt-scaled bridge winner and keep the larger-batch surface fixed.
- Rationale: Test the user-preferred simpler preprocessing choice on the current best tested systems surface rather than reopening the weaker batch32 rows.
- Hypothesis: If batch64 with sqrt LR scaling was the real driver of the gain, explicit input normalization may be removable without losing the benchmark lift.
- Upstream delta: Upstream nanoTabPFN keeps internal z-score-plus-clip handling on the exact path; this repo-local surface can disable explicit pre-encoder normalization entirely.
- Anchor delta: Changes only `model.input_normalization` from `train_zscore_clip` to `none` while keeping the row-7 batch64 sqrt-scaled bridge surface fixed.
- Expected effect: Tests whether batch size was the real driver of the gain and whether the batch64 winner can stay competitive without any explicit normalization.
- Effective labels: model=`dpnb_input_norm_none_batch64_sqrt`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Keep the batch64 sqrt-scaled training surface identical to the clipped batch64 anchor.
  - Compare final ROC AUC and drift directly against the batch64 clipped winner before simplifying the default preprocessing surface.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical CUDA rerun registered as `sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v1`.
  - Canonical CUDA comparison recorded against the batch64 clipped anchor; interpret this row before simplifying the preprocessing default.
  - Removing explicit normalization lost 0.0048 final ROC AUC versus the clipped batch64 anchor, so the simpler `none` surface is not justified on this bundle.
  - Supersedes historical queue run `sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_none_followup/dpnb_input_norm_none_batch64_sqrt/result_card.md`
- Registered run: `sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v2` with final log loss `0.4052`, delta final log loss `+0.0080`, final Brier score `0.1332`, delta final Brier score `+0.0025`, best ROC AUC `0.7597`, final ROC AUC `0.7586`, final-minus-best `-0.0011`, delta final ROC AUC `-0.0048`, delta drift `-0.0011`, delta final training time `+0.7s`
