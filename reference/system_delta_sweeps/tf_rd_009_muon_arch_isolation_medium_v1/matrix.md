# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_arch_isolation_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_arch_isolation_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_arch_isolation_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_width_screen_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_arch_isolation_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `f51ad6e4aa803bef93402488a6b2d4128fe79fb5102ec05a511dd145a5b72682`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_muon_cls_sandwich_last_full_refresh_v1` | classification_scaling_law | no | ready | none | Add a final-stage full-cell refresh to the current TF-RD-009 Muon `128x2` sandwich leader while keeping the closed medium benchmark contract fixed. | Benchmark the final-stage full-cell refresh in isolation against the carried `128x2` Muon leader; only combine it later if it wins cleanly. |
| 2 | `delta_tf_rd_009_muon_cls_sandwich_split_column_summary_v1` | classification_scaling_law | no | ready | none | Split the column summary stream into train-conditioned and test-conditioned branches on top of the current TF-RD-009 Muon `128x2` sandwich leader. | Benchmark split role-conditioned column summaries in isolation against the carried `128x2` Muon leader; only combine them later if they win cleanly. |
| 3 | `delta_tf_rd_009_muon_cls_sandwich_feature_encoder_mlp2_v1` | classification_scaling_law | no | ready | none | Replace the linear per-cell encoder with a shared two-layer MLP on top of the current TF-RD-009 Muon `128x2` sandwich leader. | Benchmark the shared `mlp2` feature encoder in isolation against the carried `128x2` Muon leader; only combine it later if it wins cleanly. |
| 4 | `delta_tf_rd_009_muon_cls_sandwich_class_memory_v1` | classification_scaling_law | no | ready | none | Add train-only class-memory slots ahead of the direct multiclass head on top of the current TF-RD-009 Muon `128x2` sandwich leader. | Benchmark train-only class memory in isolation against the carried `128x2` Muon leader; only combine it later if it wins cleanly. |

## Detailed Rows

### 1. `delta_tf_rd_009_muon_cls_sandwich_last_full_refresh_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Add a final-stage full-cell refresh to the current TF-RD-009 Muon `128x2` sandwich leader while keeping the closed medium benchmark contract fixed.
- Rationale: Execute the final-stage full-cell refresh idea in isolation on top of the current Muon `128x2` leader while keeping the measured runtime bundle and v6 corpus contract fixed.
- Hypothesis: If the current sandwich loses detail because only stage 0 gets a high-bandwidth full-cell read, reintroducing that read at the final Perceiver stage should improve benchmark quality without reopening width or optimizer tuning.
- Upstream delta: This is an architecture-isolation follow-up on the current Muon width winner `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
- Anchor delta: Isolated `128x2` Muon architecture row with only `sandwich_last_stage_full_cell_refresh=true` relative to the carried leader.
- Expected effect: If the current `128x2` sandwich is bottlenecked by only reading the full-cell stream once at stage 0, a final full-cell refresh should recover missed detailed evidence without reopening width, depth, or optimizer tuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6ecb9081dabba15a125ca482f8a55ca1b26fae0463e1e49171dc2482b3ee09a6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'sandwich_last_stage_full_cell_refresh': True, 'sandwich_column_summary_mode': 'shared_unconditioned', 'sandwich_feature_encoder_kind': 'linear', 'sandwich_use_class_memory': False, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the current Muon `128x2` width winner and the full v6 runtime bundle fixed; this row changes only whether the final Perceiver stage rereads the full-cell stream.
  - Compare directly against the carried `128x2` Muon leader at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Do not combine this delta with any other architecture change unless it first lands as a clean standalone winner.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - fixed Muon optimizer, runtime, compile, and one-epoch contract bundle from the measured `128x2` width winner
  - final-stage full-cell refresh only; width, depth, summaries-per-axis, and head geometry remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_arch_isolation_medium_v1/delta_tf_rd_009_muon_cls_sandwich_last_full_refresh_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_muon_cls_sandwich_split_column_summary_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Split the column summary stream into train-conditioned and test-conditioned branches on top of the current TF-RD-009 Muon `128x2` sandwich leader.
- Rationale: Execute the split train/test column-summary routing idea in isolation on top of the current Muon `128x2` leader while keeping the measured runtime bundle and v6 corpus contract fixed.
- Hypothesis: If the current summary-only refinement stages are limited by generic column summaries, splitting them into train-conditioned and test-conditioned branches should preserve more task structure without reopening width or optimizer tuning.
- Upstream delta: This is an architecture-isolation follow-up on the current Muon width winner `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
- Anchor delta: Isolated `128x2` Muon architecture row with only `sandwich_column_summary_mode=split_role_conditioned` relative to the carried leader.
- Expected effect: If the current summary-only refinement stages are under-informed by generic column summaries, separate train/test role-conditioned column summaries should preserve more structure without reopening the main runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `9e7e5e98d54b6e0c0ad4a8975b953e943607f5c10569849c2a61ec8e9e6363be`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'sandwich_last_stage_full_cell_refresh': False, 'sandwich_column_summary_mode': 'split_role_conditioned', 'sandwich_feature_encoder_kind': 'linear', 'sandwich_use_class_memory': False, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the current Muon `128x2` width winner and the full v6 runtime bundle fixed; this row changes only how column summaries are built and routed.
  - Compare directly against the carried `128x2` Muon leader at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Do not combine this delta with any other architecture change unless it first lands as a clean standalone winner.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - fixed Muon optimizer, runtime, compile, and one-epoch contract bundle from the measured `128x2` width winner
  - column summary routing only; width, depth, latent count, and direct head remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_arch_isolation_medium_v1/delta_tf_rd_009_muon_cls_sandwich_split_column_summary_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_muon_cls_sandwich_feature_encoder_mlp2_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replace the linear per-cell encoder with a shared two-layer MLP on top of the current TF-RD-009 Muon `128x2` sandwich leader.
- Rationale: Execute the stronger shared `mlp2` feature-encoder idea in isolation on top of the current Muon `128x2` leader while keeping the measured runtime bundle and v6 corpus contract fixed.
- Hypothesis: If the current sandwich is compensating for a shallow per-cell encoder with excess global attention work, replacing the shared linear encoder with a small MLP should improve token quality without reopening width or optimizer tuning.
- Upstream delta: This is an architecture-isolation follow-up on the current Muon width winner `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
- Anchor delta: Isolated `128x2` Muon architecture row with only `sandwich_feature_encoder_kind=mlp2` relative to the carried leader.
- Expected effect: If the current sandwich is spending too much global capacity compensating for a shallow local cell encoder, a shared `mlp2` feature encoder should improve token quality without reopening the broader Muon runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3f69400b6928e0065e7e43f187d1ff8f48f379e872af94bda2d5387f9721d6ce`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'sandwich_last_stage_full_cell_refresh': False, 'sandwich_column_summary_mode': 'shared_unconditioned', 'sandwich_feature_encoder_kind': 'mlp2', 'sandwich_use_class_memory': False, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the current Muon `128x2` width winner and the full v6 runtime bundle fixed; this row changes only the shared per-cell encoder.
  - Compare directly against the carried `128x2` Muon leader at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Do not combine this delta with any other architecture change unless it first lands as a clean standalone winner.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - fixed Muon optimizer, runtime, compile, and one-epoch contract bundle from the measured `128x2` width winner
  - feature encoder only; width, depth, latent count, and summary routing remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_arch_isolation_medium_v1/delta_tf_rd_009_muon_cls_sandwich_feature_encoder_mlp2_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_muon_cls_sandwich_class_memory_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Add train-only class-memory slots ahead of the direct multiclass head on top of the current TF-RD-009 Muon `128x2` sandwich leader.
- Rationale: Execute the explicit train-only class-memory idea in isolation on top of the current Muon `128x2` leader while keeping the measured runtime bundle and v6 corpus contract fixed.
- Hypothesis: If the current direct head is under-structured for small-class classification, augmenting pooled test rows with train-only class-memory slots should improve class evidence routing without reopening width or optimizer tuning.
- Upstream delta: This is an architecture-isolation follow-up on the current Muon width winner `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
- Anchor delta: Isolated `128x2` Muon architecture row with only `sandwich_use_class_memory=true` relative to the carried leader.
- Expected effect: If the current direct head is under-structured for small-class classification, explicit class-memory slots built only from train rows should provide a cleaner class-evidence path without reopening the fixed Muon runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `50225a63c23fed5cd5e633a48b806db7eb405b8b4b6ef098392b437e0d28e290`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'sandwich_last_stage_full_cell_refresh': False, 'sandwich_column_summary_mode': 'shared_unconditioned', 'sandwich_feature_encoder_kind': 'linear', 'sandwich_use_class_memory': True, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the current Muon `128x2` width winner and the full v6 runtime bundle fixed; this row changes only whether pooled test rows can read from train-only class-memory slots.
  - Compare directly against the carried `128x2` Muon leader at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Do not combine this delta with any other architecture change unless it first lands as a clean standalone winner.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - fixed Muon optimizer, runtime, compile, and one-epoch contract bundle from the measured `128x2` width winner
  - class-memory augmentation only; width, depth, tokenization, and summary routing remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_arch_isolation_medium_v1/delta_tf_rd_009_muon_cls_sandwich_class_memory_v1/result_card.md`
- Benchmark metrics: pending
