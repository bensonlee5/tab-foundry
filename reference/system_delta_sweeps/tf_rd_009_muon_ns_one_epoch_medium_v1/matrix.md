# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_ns_one_epoch_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `e49737bff76cab1b5cbfcf8b503222eb83477f3efe9abc957f1d649f82538b0f`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law_phase2_ns`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `8686.8s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 1 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 2 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 3 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 4 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 5 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 6 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 7 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 8 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 9 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 10 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 11 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 12 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | ready | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 13 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | ready | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 14 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | ready | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 15 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | ready | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 16 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 17 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 18 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 19 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 20 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6bc5e58b3d6eb722123dec779c13510e38cd499f5bee16abf73ecd124fdf8b63`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `72x1` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=625; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6b02e5b4989d4bdda91148987d85c262917d44f16be42ae0b185e8b55da97bf8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `72x1` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=1250; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `11a9899045fd44740fde0f7f7265eb9a975e0e66524421326da56910864cc285`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `72x1` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=2500; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6038bd24731cf3b938929c4d0cb6970b900fbc0b841c7adef8893c744d7a9b6d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `72x1` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=5000; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `776b7d36255e50f6b979789114e3375a11478e66e407cbc6fe41e3eb9792ebaa`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `112x3` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=625; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `eb4d171f76f24850bdc80ff8178bc0d0ee45f5cc909f3b0b5af6de08a6297053`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `112x3` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=1250; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `85814ddf829eacaa8e9c2669efea325c087993e60bdbfd609e9f40154d65a615`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `112x3` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=2500; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ab1135181c4f4eac7893709ddb315d4abf82c82221451c90a3329d2dc95bc49c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry `112x3` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=5000; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4479cd814203f6aabc243fa81db084555096ca87dd9a352d0cbe409d12b2c087`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Parameter adequacy plan:
  - Keep geometry `144x4` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=625; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aa4e2daa3dd663eee1b46c867006bc05815f6fcbd5d28297551e392ba4f3f9aa`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Parameter adequacy plan:
  - Keep geometry `144x4` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=1250; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e30f9244fd24579ca5ba343e6f70aee431b39702c986760815536c21f2d96d12`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Parameter adequacy plan:
  - Keep geometry `144x4` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=2500; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ebb772f009e6be9247db30b1a4f02d577b988b020515501b5b90295152457d39`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Parameter adequacy plan:
  - Keep geometry `144x4` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=5000; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 13. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a21ad32f0fdfa4f6c42aa8e93878b4a3d7f8f4e9edf88585b3e8930dc8968d15`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 192, 'sandwich_layers': 5}`
- Parameter adequacy plan:
  - Keep geometry `192x5` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=625; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 14. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `11155918a8dd050d56e25ea7b9325bd4384189931f236eb332b3d6adcb789af9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 192, 'sandwich_layers': 5}`
- Parameter adequacy plan:
  - Keep geometry `192x5` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=1250; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 15. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a677177127d418cf5bcc004ada3e268d95d8047b69f20227e79277ae7d283488`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 192, 'sandwich_layers': 5}`
- Parameter adequacy plan:
  - Keep geometry `192x5` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=2500; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 16. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e3a31112009f0875eb0a94062a5bf9bf2dc3089cd55c1e8aa48d90386cfaad43`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 192, 'sandwich_layers': 5}`
- Parameter adequacy plan:
  - Keep geometry `192x5` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=5000; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 17. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cc37c06521b8b512d95fa67e25eab46e4838223a3af997656f490ca5ca853e5b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Keep geometry `264x6` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=625; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 18. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c5c181dd9254b4b1e9112840df2e4805eba9a00d19fbcb6d8c46e13fdabc56c1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Keep geometry `264x6` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=1250; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 19. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65da367784e87491aa576cccb3457df140819cc49c8244e749915e549d9891b6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Keep geometry `264x6` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=2500; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 20. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ac26faef6594ce195426ff6277ce364956e066ee3636e32eec35863a66423a43`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Keep geometry `264x6` on the completed fresh Muon Phase-1 family `72x1/112x3/144x4/192x5/264x6`.
  - Use this row only for the fresh Muon `L(N,S)` matrix at max_steps=5000; do not reinterpret it as additional Phase-1 evidence.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only geometry or step budget within the Muon family.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Historical schedulefree TF-RD-009 Phase-2 rows remain context only and must not enter the fresh Muon fit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending
