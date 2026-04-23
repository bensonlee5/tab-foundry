# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_027_grid_stability_impl_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_027_grid_stability_impl_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_027_grid_stability_impl_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_027_grid_ffn_wd_config_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_027_grid_stability_impl_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `191b8254cb14647bab5e9e050b7e6c1a8b6fa8b23f114343b3a95a33d8ed23a9`

## Locked Surface

- Anchor run id: `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_workstation_grid_sandwich`
- Training config profile: `cls_workstation_grid_sandwich`
- Surface role: `classification_grid_broad_ml_followup`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4182`, final Brier score `0.2551`, best ROC AUC `0.8135`, final ROC AUC `0.8133`, final training time `4073.3s`

## Anchor Comparison

Upstream reference: `Broad-ML grid sandwich architecture follow-ons` from `Hyper-Connections, Differential Transformer, SwiGLU gated FFNs, and recurrent refinement`.

| Dimension | Upstream Broad-ML grid sandwich architecture follow-ons | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Staged feature encoder `unknown` from the benchmark registry surface. | Feature encoder changes alter the per-cell representation and should be interpreted explicitly. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Target conditioner `unknown` from the staged surface. | Target-conditioning changes should be interpreted separately from encoder or context changes. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Cell transformer block `unknown` from the staged surface. | Cell-block changes affect the core table computation and should be isolated carefully. |
| tokenizer | One scalar token per feature. | Tokenizer `unknown` from the staged surface. | Tokenizer changes alter the token sequence presented to the transformer stack. |
| column encoder | None on the upstream direct path. | Column encoder `unknown` from the staged surface. | Column-encoder changes should be read separately from row pooling or context changes. |
| row readout | Target-column readout from the final cell tensor. | Row pool `unknown` from the staged surface. | Row-pool changes alter the readout contract and require their own interpretation. |
| context encoder | None on the upstream direct path. | Context encoder `unknown` from the staged surface. | Context-encoder changes alter how training rows condition test rows. |
| prediction head | Direct binary logits head. | Prediction head `unknown` from the staged surface. | Head changes alter the task contract and output semantics. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest `data/manifests/bench/openml_classification_medium_v1/manifest.parquet` sourced from `openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_027_grid_classification_z_loss_1e4_v1` | grid_sandwich_ffn_stability | no | completed | none | On the config-only winner, add canonical classification z-loss with coefficient 1e-4 for a 2500-step isolated stability row. | Completed; classification z-loss lost to the locked TF-RD-026 row 10 anchor, so do not carry it. |
| 2 | `delta_tf_rd_027_grid_logit_softcap_30_v1` | grid_sandwich_ffn_stability | no | completed | none | On the config-only winner, apply a tanh classification-logit softcap of 30.0 for a 2500-step isolated stability row. | Completed; logit softcap lost to the locked TF-RD-026 row 10 anchor, so do not carry it. |
| 3 | `delta_tf_rd_027_grid_qk_norm_v1` | grid_sandwich_ffn_stability | no | completed | none | On the config-only winner, enable QK-norm across grid-sandwich attention sites for a 2500-step isolated stability row. | Completed; QK-norm lost to the locked TF-RD-026 row 10 anchor, so do not carry it. |

## Detailed Rows

### 1. `delta_tf_rd_027_grid_classification_z_loss_1e4_v1`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: On the config-only winner, add canonical classification z-loss with coefficient 1e-4 for a 2500-step isolated stability row.
- Rationale: Contextualize `delta_tf_rd_027_grid_classification_z_loss_1e4_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_stability_impl_v1`.
- Hypothesis: none
- Upstream delta: Implementation-backed TF-RD-027 softmax-stability mechanism.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_classification_z_loss_1e4_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Penalizes large classification log normalizers via coeff * mean(logsumexp(logits)^2) without changing benchmark evaluation semantics.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `9fad736e62c02d257d00d599119b871b06d82ebfa4b0de310e0aa2dcc2f56812`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Stage-local stability: row (grad `0.0537`)
- Training overrides: `{'runtime': {'max_steps': 2500, 'activation_checkpointing': False, 'checkpoint_every': None}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Use the current TF-RD-026 row 10 anchor config because the FFN rows and the weight-decay 0.1 follow-up all lost by matched-budget final log loss.
  - Compare directly against the config-only winner by matched-budget final log loss and instability telemetry.
- Adequacy knobs to dimension explicitly:
  - run only after the config-only FFN/weight-decay winner is selected
  - isolated z-loss change only; no logit softcap or QK-norm in the same row
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Config-only gate completed; carry the original TF-RD-026 row 10 anchor config because `optimizer.weight_decay=0.1` lost to the anchor (`0.4263516327` vs. `0.4181767299` final log loss).
  - Canonical rerun registered as `sd_tf_rd_027_grid_stability_impl_v1_01_delta_tf_rd_027_grid_classification_z_loss_1e4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Final log loss `0.4243303110` lost to the locked anchor `0.4181767299` by `+0.0061535811`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_stability_impl_v1/delta_tf_rd_027_grid_classification_z_loss_1e4_v1/result_card.md`
- Registered run: `sd_tf_rd_027_grid_stability_impl_v1_01_delta_tf_rd_027_grid_classification_z_loss_1e4_v1_v1` with final log loss `0.4243`, delta final log loss `+0.0062`, final Brier score `0.2584`, delta final brier score `+0.0033`, final ROC AUC `0.8097`, delta final roc auc `-0.0035`, best ROC AUC `0.8097`, final-minus-best `+0.0000`

### 2. `delta_tf_rd_027_grid_logit_softcap_30_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: On the config-only winner, apply a tanh classification-logit softcap of 30.0 for a 2500-step isolated stability row.
- Rationale: Contextualize `delta_tf_rd_027_grid_logit_softcap_30_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_stability_impl_v1`.
- Hypothesis: none
- Upstream delta: Implementation-backed TF-RD-027 softmax-stability mechanism.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_logit_softcap_30_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Bounds classification logits with cap * tanh(logits / cap) before the classification loss and benchmark logits leave the model.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `691c4bba7b97bcba6a229f759c501aebaa1231f3af5852cd201eddcf5c8b9d6c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Stage-local stability: row (grad `0.0538`)
- Model overrides: `{'classification_logit_softcap': 30.0}`
- Parameter adequacy plan:
  - Use the current TF-RD-026 row 10 anchor config because the FFN rows and the weight-decay 0.1 follow-up all lost by matched-budget final log loss.
  - Verify the exported model spec carries the softcap field before interpreting benchmark metrics.
- Adequacy knobs to dimension explicitly:
  - run only after the config-only FFN/weight-decay winner is selected
  - isolated logit-softcap change only; no z-loss or QK-norm in the same row
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Config-only gate completed; carry the original TF-RD-026 row 10 anchor config because `optimizer.weight_decay=0.1` lost to the anchor (`0.4263516327` vs. `0.4181767299` final log loss).
  - Canonical rerun registered as `sd_tf_rd_027_grid_stability_impl_v1_02_delta_tf_rd_027_grid_logit_softcap_30_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Final log loss `0.4251839620` lost to the locked anchor `0.4181767299` by `+0.0070072320`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_stability_impl_v1/delta_tf_rd_027_grid_logit_softcap_30_v1/result_card.md`
- Registered run: `sd_tf_rd_027_grid_stability_impl_v1_02_delta_tf_rd_027_grid_logit_softcap_30_v1_v1` with final log loss `0.4252`, delta final log loss `+0.0070`, final Brier score `0.2591`, delta final brier score `+0.0040`, final ROC AUC `0.8113`, delta final roc auc `-0.0020`, best ROC AUC `0.8113`, final-minus-best `+0.0000`

### 3. `delta_tf_rd_027_grid_qk_norm_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: On the config-only winner, enable QK-norm across grid-sandwich attention sites for a 2500-step isolated stability row.
- Rationale: Contextualize `delta_tf_rd_027_grid_qk_norm_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_stability_impl_v1`.
- Hypothesis: none
- Upstream delta: Implementation-backed TF-RD-027 attention-stability mechanism.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_qk_norm_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Normalizes query and key vectors and applies a learnable per-head scale initialized to sqrt(head_dim) across grid sandwich attention.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `74d7961b47e03037e66dd58e8d4e8eb0f1dc16494562834b7e133f92c9e192f8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Stage-local stability: row (grad `0.0856`)
- Model overrides: `{'attention_qk_norm': True}`
- Parameter adequacy plan:
  - Use the current TF-RD-026 row 10 anchor config because the FFN rows and the weight-decay 0.1 follow-up all lost by matched-budget final log loss.
  - Verify packed self-attention and cross-attention paths before interpreting benchmark metrics.
- Adequacy knobs to dimension explicitly:
  - run only after the config-only FFN/weight-decay winner is selected
  - isolated QK-norm change only; no z-loss or logit softcap in the same row
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Config-only gate completed; carry the original TF-RD-026 row 10 anchor config because `optimizer.weight_decay=0.1` lost to the anchor (`0.4263516327` vs. `0.4181767299` final log loss).
  - Canonical rerun registered as `sd_tf_rd_027_grid_stability_impl_v1_03_delta_tf_rd_027_grid_qk_norm_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Final log loss `0.4741464959` lost to the locked anchor `0.4181767299` by `+0.0559697660`; no TF-RD-027 stability row beat the anchor.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_stability_impl_v1/delta_tf_rd_027_grid_qk_norm_v1/result_card.md`
- Registered run: `sd_tf_rd_027_grid_stability_impl_v1_03_delta_tf_rd_027_grid_qk_norm_v1_v1` with final log loss `0.4741`, delta final log loss `+0.0560`, final Brier score `0.2908`, delta final brier score `+0.0357`, final ROC AUC `0.7860`, delta final roc auc `-0.0272`, best ROC AUC `0.7860`, final-minus-best `+0.0000`
