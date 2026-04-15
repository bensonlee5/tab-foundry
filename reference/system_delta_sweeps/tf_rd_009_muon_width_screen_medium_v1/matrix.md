# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_width_screen_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_width_screen_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_width_screen_medium_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_024_classification_heads_prerow_followup_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_width_screen_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `6aff188ecf0f026f9f201b59e7693cc7f77669353bfafb84df649964e837b636`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1729`, final BPF `2.1729`, final log loss `0.4172`, final Brier score `0.2748`, best ROC AUC `0.7235`, final ROC AUC `0.7506`, final training time `1890.0s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_024_followup_cls_sandwich_heads1_v1` | architecture_followup | no | completed | none | Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface. | Keep `60x2` as the formal external Muon anchor, but do not carry it forward as the in-family diagonal baseline now that `128x2` has won the measured width screen. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl48_v1` | classification_scaling_law | no | completed | none | Reduce TF-RD-009 classification sandwich width from `d_icl=60` to `48` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Keep `48x2` as lower-width evidence only; use it to constrain the Muon diagonal derivation, but do not carry it forward as the baseline geometry. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Keep `96x2` as mid-width evidence for the fresh Muon diagonal, but do not carry it forward over the stronger `128x2` result. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl128_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `128` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Carry `128x2` into `tf_rd_009_muon_width_depth_medium_v1` as the current in-family Muon baseline for the diagonal derivation. |

## Detailed Rows

### 1. `delta_tf_rd_024_followup_cls_sandwich_heads1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface.
- Rationale: Establish the fresh Muon `60x2` external anchor candidate at the full 2500-step budget on the v6 no-repeat contract.
- Hypothesis: 
- Upstream delta: Extends the TF-RD-024 attention-head family one bracket lower after `sandwich_heads=2` won the initial medium screen.
- Anchor delta: Fresh Muon width-screen row `60x2`; treat this as the formal external anchor candidate for the rebooted family.
- Expected effect: If the medium gain from the lower-head bracket reflects excess head factorization rather than a narrow `2`-head sweet spot, `sandwich_heads=1` may remain competitive or improve further on the closed medium contract.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2467cd80ac43b33bba690c39f7647013f04db7921194e86dda92a6279ea29b03`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Establish the formal external Muon anchor at `60x2` before any fresh width-depth diagonal is derived.
  - Do not inherit carried-baseline status from historical schedulefree TF-RD-009; choose the in-family baseline only from measured Muon results.
  - Use the benchmark-backed `60x2` result as the reference row for the Muon width screen and the later planned hardware-freeze replacement.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime policy surface
  - attention partitioning only; no width, depth, batching, or optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon anchor replay of the carried `sandwich_heads=1` surface on the corrected v6 medium contract.
  - This row replaces no historical TF-RD-009 id in place; it starts a new Muon family.
  - Canonical rerun registered as `sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - This row remains the formal external Muon anchor even though `128x2` emerged as the carried in-family width winner.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_screen_medium_v1/delta_tf_rd_024_followup_cls_sandwich_heads1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1` with final log loss `0.4172`, delta final log loss `+0.0000`, final Brier score `0.2748`, delta final brier score `+0.0000`, final ROC AUC `0.7506`, delta final roc auc `+0.0000`, final BPC (legacy feature-cell diagnostic) `2.1729`, delta final bpc (legacy feature-cell diagnostic) `+0.0000`, final BPF (legacy feature-cell diagnostic) `2.1729`, delta final bpf (legacy feature-cell diagnostic) `+0.0000`, best ROC AUC `0.7235`, delta final training time `+0.0s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl48_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reduce TF-RD-009 classification sandwich width from `d_icl=60` to `48` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute the fresh Muon lower-width bracket at `48x2` under the fixed packed-runtime plus v6 no-repeat contract.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the first live scaling axis after the TF-RD-024 heads1 replay establishes the benchmark-registry-backed anchor.
- Anchor delta: Fresh Muon width-screen row `48x2`; interpret this row only inside the bounded family once the `60x2` anchor is benchmark-backed.
- Expected effect: If the carried heads1 classification surface is width-overprovisioned on the closed medium contract, dropping to `d_icl=48` should preserve most benchmark quality at the matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2fe49d94269f4bb92aab7f2345f2e52d12d04876ca2b0d2cd9b1489f6ad5130d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 48, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the post-#271 packed Muon runtime surface fixed and vary width only inside the bounded family `{48x2, 60x2, 96x2, 128x2}`.
  - Compare directly against the measured Muon `60x2` anchor candidate at matched regime budget after order 1 lands.
  - Use this row only as lower-width evidence for the fresh Muon family; do not inherit any decision from the historical schedulefree branch.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon TF-RD-009 width-screen row on `tf_rd_010_dagzoo_medium_control_curated_v6`.
  - Historical schedulefree TF-RD-009 artifacts remain preserved and must not be mixed into this family.
  - Canonical rerun registered as `sd_tf_rd_009_muon_width_screen_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl48_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_width_screen_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1` with final log loss `0.4147`, delta final log loss `-0.0026`, final Brier score `0.2713`, delta final brier score `-0.0035`, final ROC AUC `0.7608`, delta final roc auc `+0.0102`, final BPC (legacy feature-cell diagnostic) `2.7336`, delta final bpc (legacy feature-cell diagnostic) `+0.5607`, final BPF (legacy feature-cell diagnostic) `2.7336`, delta final bpf (legacy feature-cell diagnostic) `+0.5607`, best ROC AUC `0.6588`, delta final training time `-26.8s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute the fresh Muon upper-width bracket at `96x2` under the fixed packed-runtime plus v6 no-repeat contract.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Fresh Muon width-screen row `96x2`; interpret this row only inside the bounded Muon width family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b282813d27fa9d26e717f468ce0fe9fdef0e73537306d700c5658528159b6e70`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the post-#271 packed Muon runtime surface fixed and vary width only inside the bounded family `{48x2, 60x2, 96x2, 128x2}`.
  - Compare directly against the measured Muon `60x2` anchor candidate at matched regime budget after order 1 lands.
  - Allow this row to become the carried in-family baseline only if the measured Muon result is healthy and clearly better than the rest of the Muon width family.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon TF-RD-009 width-screen row on `tf_rd_010_dagzoo_medium_control_curated_v6`.
  - Historical schedulefree TF-RD-009 evidence must not be used to pre-assign baseline status to this row.
  - Canonical rerun registered as `sd_tf_rd_009_muon_width_screen_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_width_screen_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.4128`, delta final log loss `-0.0044`, final Brier score `0.2710`, delta final brier score `-0.0037`, final ROC AUC `0.7573`, delta final roc auc `+0.0067`, final BPC (legacy feature-cell diagnostic) `2.2405`, delta final bpc (legacy feature-cell diagnostic) `+0.0676`, final BPF (legacy feature-cell diagnostic) `2.2405`, delta final bpf (legacy feature-cell diagnostic) `+0.0676`, best ROC AUC `0.7035`, delta final training time `-37.6s`

### 4. `delta_tf_rd_009_cls_sandwich_dicl128_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `128` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute the fresh Muon upper-width bracket at `128x2` under the fixed packed-runtime plus v6 no-repeat contract.
- Hypothesis: 
- Upstream delta: This is the larger-width bracket for the first TF-RD-009 width-transfer family once the carried heads1 anchor is replayed and registered.
- Anchor delta: Fresh Muon width-screen row `128x2`; interpret this row only inside the bounded Muon width family.
- Expected effect: If width-only improvement remains monotone beyond `d_icl=96`, expanding to `128` should either continue the gain or show that width-only transfer is flattening before joint width-depth scaling begins.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `21499a9506ea43ef2c15417c36825517dca0010aedd37c3eb55492d42a2d76e3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep the post-#271 packed Muon runtime surface fixed and vary width only inside the bounded family `{48x2, 60x2, 96x2, 128x2}`.
  - Compare directly against the measured Muon `60x2` anchor candidate at matched regime budget after order 1 lands.
  - Carry this row forward only if it remains healthy and materially informative for the fresh Muon diagonal derivation.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Fresh Muon TF-RD-009 width-screen row on `tf_rd_010_dagzoo_medium_control_curated_v6`.
  - Historical schedulefree TF-RD-009 evidence must not be used to pre-assign baseline status to this row.
  - Canonical rerun registered as `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - This is the measured in-family width winner and the current carried baseline for the fresh Muon diagonal derivation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1` with final log loss `0.3951`, delta final log loss `-0.0222`, final Brier score `0.2585`, delta final brier score `-0.0163`, final ROC AUC `0.7583`, delta final roc auc `+0.0077`, final BPC (legacy feature-cell diagnostic) `2.1383`, delta final bpc (legacy feature-cell diagnostic) `-0.0346`, final BPF (legacy feature-cell diagnostic) `2.1383`, delta final bpf (legacy feature-cell diagnostic) `-0.0346`, best ROC AUC `0.7563`, delta final training time `-64.6s`
