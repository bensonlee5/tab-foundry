# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_027_grid_ffn_wd_config_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_027_grid_ffn_wd_config_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_027_grid_ffn_wd_config_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_026_grid_sandwich_broad_ml_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_027_grid_ffn_wd_config_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `0613e113a159be4247e4c62a3d31fa7db8ca52551ae51efca402267f6496764e`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest `/Users/bensonlee/dev/tab-foundry/data/manifests/bench/openml_classification_medium_v1/manifest.parquet` sourced from `openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_027_grid_ffn_gelu4_v1` | grid_sandwich_ffn_stability | no | ready | none | Test the TF-RD-026 row 10 recurrent grid anchor with GELU FFNs at a 4:1 feedforward multiplier for a 2500-step expedited screen. | Execute as the first TF-RD-027 FFN multiplier screen row. |
| 2 | `delta_tf_rd_027_grid_ffn_swiglu8_3_v1` | grid_sandwich_ffn_stability | no | ready | none | Test the TF-RD-026 row 10 recurrent grid anchor with SwiGLU at the 8:3 parameter-matched feedforward multiplier for a 2500-step expedited screen. | Execute as the second TF-RD-027 FFN multiplier screen row. |
| 3 | `delta_tf_rd_027_grid_weight_decay_0_1_v1` | grid_sandwich_ffn_stability | no | blocked | none | On the TF-RD-027 FFN winner only, test Muon optimizer weight decay 0.1 for a 2500-step expedited follow-up. | Unblock only after the FFN winner has been selected and copied into this row's effective model surface. |

## Detailed Rows

### 1. `delta_tf_rd_027_grid_ffn_gelu4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 recurrent grid anchor with GELU FFNs at a 4:1 feedforward multiplier for a 2500-step expedited screen.
- Rationale: Contextualize `delta_tf_rd_027_grid_ffn_gelu4_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_ffn_wd_config_v1`.
- Hypothesis: none
- Upstream delta: Config-only FFN multiplier screen from TF-RD-027.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_ffn_gelu4_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Establishes whether the current SwiGLU recurrent anchor is still better when GELU receives the requested 4:1 feedforward budget.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6b4fbd3928344005770bd40ffdb1e2783b6ccfce7536c477cbfaa8af6d8b81c2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'grid_ffn_mode': 'gelu', 'sandwich_ff_expansion': 4}`
- Parameter adequacy plan:
  - Compare against the existing TF-RD-026 row 10 anchor and the TF-RD-027 SwiGLU 8:3 row by final_log_loss_at_matched_regime_budget.
  - If both 2500-step FFN rows lose to the existing anchor, keep the current anchor FFN.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - d_icl=144, grid_recurrence_steps=8, and grid_recurrence_unique_layers=2 inherited from the anchor
  - Muon optimizer, medium v6 corpus, OpenML medium benchmark, and W&B logging held fixed
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_ffn_wd_config_v1/delta_tf_rd_027_grid_ffn_gelu4_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_027_grid_ffn_swiglu8_3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 recurrent grid anchor with SwiGLU at the 8:3 parameter-matched feedforward multiplier for a 2500-step expedited screen.
- Rationale: Contextualize `delta_tf_rd_027_grid_ffn_swiglu8_3_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_ffn_wd_config_v1`.
- Hypothesis: none
- Upstream delta: Config-only FFN multiplier screen from TF-RD-027.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_ffn_swiglu8_3_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Tests whether the current SwiGLU anchor improves when the feedforward expansion uses sandwich_ff_expansion=4, which maps to the 8:3 SwiGLU hidden-size rule.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d4073f9469f1b8dae18cb00f051bc7296b28e465e913cf5693339689a37facc8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'grid_ffn_mode': 'swiglu', 'sandwich_ff_expansion': 4}`
- Parameter adequacy plan:
  - Compare against the existing TF-RD-026 row 10 anchor and the TF-RD-027 GELU 4:1 row by final_log_loss_at_matched_regime_budget.
  - If both 2500-step FFN rows lose to the existing anchor, keep the current anchor FFN.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - d_icl=144, grid_recurrence_steps=8, and grid_recurrence_unique_layers=2 inherited from the anchor
  - Muon optimizer, medium v6 corpus, OpenML medium benchmark, and W&B logging held fixed
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_ffn_wd_config_v1/delta_tf_rd_027_grid_ffn_swiglu8_3_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_027_grid_weight_decay_0_1_v1`

- Dimension family: `training`
- Status: `blocked`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: On the TF-RD-027 FFN winner only, test Muon optimizer weight decay 0.1 for a 2500-step expedited follow-up.
- Rationale: Contextualize `delta_tf_rd_027_grid_weight_decay_0_1_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_027_grid_ffn_wd_config_v1`.
- Hypothesis: none
- Upstream delta: Config-only optimizer regularization follow-up from TF-RD-027.
- Anchor delta: Delta description pending for `delta_tf_rd_027_grid_weight_decay_0_1_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Checks whether stronger weight decay improves the chosen FFN surface without changing width, recurrence, corpus, or runtime policy.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `99dfebb1b79fd5e1286301314405c32f256baefadd81b7c4e69a45ec4635ad04`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'weight_decay': 0.1}, 'runtime': {'max_steps': 2500, 'activation_checkpointing': False, 'checkpoint_every': None}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Update this row's anchor/model surface to the FFN winner before execution.
  - Carry the better of weight decay 0.01 and 0.1 into the implementation-backed stability sweep.
- Adequacy knobs to dimension explicitly:
  - run only after the FFN winner is selected
  - compare optimizer.weight_decay=0.1 against the carried 0.01 setting on the same FFN winner surface
  - train for exactly 2500 prior-dump steps
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_027_grid_ffn_wd_config_v1/delta_tf_rd_027_grid_weight_decay_0_1_v1/result_card.md`
- Benchmark metrics: pending
