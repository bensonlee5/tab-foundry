# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v3/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v3`
- Sweep status: `active`
- Parent sweep id: `binary_md_v2`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `01_shared_norm_post_ln_binary_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7670`, final ROC AUC `0.7643`, final training time `505.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Same mean-padded linear target conditioner. | The anchor preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm cell transformer block. | This keeps the strongest structural tie to upstream nanoTabPFN. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Tokenization remains aligned with upstream parity. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head remains on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_constant_lr`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_global_rmsnorm` | normalization | yes | ready | none | Keep the anchor structure fixed but switch the staged/global LayerNorm family to RMSNorm, including the row-pool norm override for completeness. | Train on the locked medium binary surface and benchmark against the v3 anchor. |

## Detailed Rows

### 1. `delta_global_rmsnorm`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Keep the anchor structure fixed but switch the staged/global LayerNorm family to RMSNorm, including the row-pool norm override for completeness.
- Rationale: Test bias removal (baked into QASS and shared encoder layers) combined with global RMSNorm on the promoted shared+LayerNorm anchor.
- Hypothesis: RMSNorm may improve optimization stability, and bias removal provides mild regularization by eliminating redundant parameters behind pre-norm layers.
- Upstream delta: Upstream nanoTabPFN uses LayerNorm in its exact transformer blocks.
- Anchor delta: Changes the global norm family from LayerNorm to RMSNorm (norm_type and tfrow_norm) while keeping the shared feature encoder, post-encoder LayerNorm, nano-postnorm block style, and all other anchor surfaces fixed.
- Expected effect: Potentially smoother optimization or lower gradient spikes, with the risk that RMSNorm only shifts calibration or late-curve behavior.
- Effective labels: model=`delta_global_rmsnorm`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Model overrides: `{'module_overrides': {'feature_encoder': 'shared', 'post_encoder_norm': 'layernorm'}, 'norm_type': 'rmsnorm', 'tfrow_norm': 'rmsnorm'}`
- Parameter adequacy plan:
  - Compare best/final ROC AUC and training stability against the v3 anchor.
  - Distinguish early-step stabilization from true end-of-run quality improvements.
  - Note the bias-removal confound — this run tests both changes simultaneously.
- Adequacy knobs to dimension explicitly:
  - model.norm_type
  - model.tfrow_norm
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Bias removal is baked into the code; cannot isolate RMSNorm from bias removal.
- Notes:
  - The v1 RMSNorm run (sd_binary_md_v1_10_delta_global_rmsnorm_v1) scored 0.7601 best ROC AUC on the nano-only anchor.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_global_rmsnorm/result_card.md`
- Benchmark metrics: pending
