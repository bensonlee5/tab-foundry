# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_ns_one_epoch_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_ns_one_epoch_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `fc1c9b8a031395b8814bb6a9a27bf423238e2fbc5f6f96abac62fa0eb6b8cc22`

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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 1 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 2 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 3 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 4 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 5 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 6 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 7 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 8 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 9 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 10 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 11 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 12 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | completed | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 13 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | completed | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 14 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | completed | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 15 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | completed | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Execute order 16 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 17 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 18 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 19 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 20 as fresh Muon one-epoch row in `tf_rd_009_muon_ns_one_epoch_medium_v1`; admit it only into Muon Phase-2 fits. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6f0e1fd8e93f08d8ba2cd94ade329a20f0b63109b9309bf0bbe6bf701639c2ca`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.6831`, delta final log loss `+0.2880`, final Brier score `0.4239`, delta final brier score `+0.1654`, final ROC AUC `0.6064`, delta final roc auc `-0.1519`, final BPC (legacy feature-cell diagnostic) `2.1335`, delta final bpc (legacy feature-cell diagnostic) `-0.0048`, final BPF (legacy feature-cell diagnostic) `2.1335`, delta final bpf (legacy feature-cell diagnostic) `-0.0048`, best ROC AUC `0.6064`, delta final training time `-8087.3s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ef956c010dc46a225d5169ceade1a2373ccabdcefbcb0fe4d7e4646fb6f3f2b5`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.5966`, delta final log loss `+0.2015`, final Brier score `0.3683`, delta final brier score `+0.1098`, final ROC AUC `0.7022`, delta final roc auc `-0.0561`, final BPC (legacy feature-cell diagnostic) `2.1974`, delta final bpc (legacy feature-cell diagnostic) `+0.0591`, final BPF (legacy feature-cell diagnostic) `2.1974`, delta final bpf (legacy feature-cell diagnostic) `+0.0591`, best ROC AUC `0.7022`, delta final training time `-7632.7s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ecd9e950c96d7c7f76e800787de8a969d27544175e2d6536c0d14102a0a54b94`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.5161`, delta final log loss `+0.1210`, final Brier score `0.3142`, delta final brier score `+0.0557`, final ROC AUC `0.7692`, delta final roc auc `+0.0109`, final BPC (legacy feature-cell diagnostic) `2.3315`, delta final bpc (legacy feature-cell diagnostic) `+0.1932`, final BPF (legacy feature-cell diagnostic) `2.3314`, delta final bpf (legacy feature-cell diagnostic) `+0.1931`, best ROC AUC `0.7692`, delta final training time `-6694.5s`

### 4. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute fresh Muon one-epoch NS row `72x1` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two Muon scaling row `72x1` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1b3c47af7fe1ef60083943ab669223b22fd279a9f3729b76736a17eb89d5f80e`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.5068`, delta final log loss `+0.1117`, final Brier score `0.3087`, delta final brier score `+0.0502`, final ROC AUC `0.7744`, delta final roc auc `+0.0161`, final BPC (legacy feature-cell diagnostic) `11.3314`, delta final bpc (legacy feature-cell diagnostic) `+9.1931`, final BPF (legacy feature-cell diagnostic) `11.3248`, delta final bpf (legacy feature-cell diagnostic) `+9.1866`, best ROC AUC `0.7744`, delta final training time `-4866.5s`

### 5. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e82c0cb2754df1da9cf8829730301ac219a1b3fb6b2f771fb7e25581b4a83f2a`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.6496`, delta final log loss `+0.2546`, final Brier score `0.4039`, delta final brier score `+0.1454`, final ROC AUC `0.6583`, delta final roc auc `-0.1000`, final BPC (legacy feature-cell diagnostic) `2.4610`, delta final bpc (legacy feature-cell diagnostic) `+0.3227`, final BPF (legacy feature-cell diagnostic) `2.4610`, delta final bpf (legacy feature-cell diagnostic) `+0.3227`, best ROC AUC `0.6583`, delta final training time `-8115.8s`

### 6. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5292eb8cd69733b7c777eba801c1569b88f9ca11991dc0318556398da8b034e4`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.5353`, delta final log loss `+0.1403`, final Brier score `0.3267`, delta final brier score `+0.0682`, final ROC AUC `0.7602`, delta final roc auc `+0.0019`, final BPC (legacy feature-cell diagnostic) `2.4705`, delta final bpc (legacy feature-cell diagnostic) `+0.3322`, final BPF (legacy feature-cell diagnostic) `2.4705`, delta final bpf (legacy feature-cell diagnostic) `+0.3322`, best ROC AUC `0.7602`, delta final training time `-7650.3s`

### 7. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d8ad7100f0869a335eb4b1a82905808ad8195112876ed9cfd23f41c9e82062f0`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.5098`, delta final log loss `+0.1147`, final Brier score `0.3103`, delta final brier score `+0.0517`, final ROC AUC `0.7722`, delta final roc auc `+0.0139`, final BPC (legacy feature-cell diagnostic) `2.4540`, delta final bpc (legacy feature-cell diagnostic) `+0.3157`, final BPF (legacy feature-cell diagnostic) `2.4540`, delta final bpf (legacy feature-cell diagnostic) `+0.3157`, best ROC AUC `0.7722`, delta final training time `-6720.1s`

### 8. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute fresh Muon one-epoch NS row `112x3` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two Muon scaling row `112x3` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `48c3fdb47b51f9dcf8deba82be16e4479a1c1b002f7535340d31fde52ca0d2a2`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.4986`, delta final log loss `+0.1036`, final Brier score `0.3031`, delta final brier score `+0.0446`, final ROC AUC `0.7804`, delta final roc auc `+0.0221`, final BPC (legacy feature-cell diagnostic) `2.5782`, delta final bpc (legacy feature-cell diagnostic) `+0.4400`, final BPF (legacy feature-cell diagnostic) `2.5782`, delta final bpf (legacy feature-cell diagnostic) `+0.4399`, best ROC AUC `0.7804`, delta final training time `-4896.3s`

### 9. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4fc459d869ad4846c6cfae41b5c03ff051b4d47a924f1126afbec3d8ee4a7c5a`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1` with final log loss `0.6461`, delta final log loss `+0.2511`, final Brier score `0.4026`, delta final brier score `+0.1441`, final ROC AUC `0.6658`, delta final roc auc `-0.0925`, final BPC (legacy feature-cell diagnostic) `2.1886`, delta final bpc (legacy feature-cell diagnostic) `+0.0504`, final BPF (legacy feature-cell diagnostic) `2.1886`, delta final bpf (legacy feature-cell diagnostic) `+0.0504`, best ROC AUC `0.6658`, delta final training time `-8123.2s`

### 10. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a6ff9687d4654cb123272f6d73d5955904440fc8ef1dda28a4df3cce8131f1c2`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1` with final log loss `0.5268`, delta final log loss `+0.1317`, final Brier score `0.3211`, delta final brier score `+0.0626`, final ROC AUC `0.7658`, delta final roc auc `+0.0075`, final BPC (legacy feature-cell diagnostic) `2.2353`, delta final bpc (legacy feature-cell diagnostic) `+0.0971`, final BPF (legacy feature-cell diagnostic) `2.2353`, delta final bpf (legacy feature-cell diagnostic) `+0.0970`, best ROC AUC `0.7658`, delta final training time `-7660.7s`

### 11. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a4c7d3c6553c38115717e1178045a9b795c21c50d5104cbea7c9e02b3125eb6f`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1` with final log loss `0.5079`, delta final log loss `+0.1128`, final Brier score `0.3095`, delta final brier score `+0.0510`, final ROC AUC `0.7719`, delta final roc auc `+0.0136`, final BPC (legacy feature-cell diagnostic) `2.3965`, delta final bpc (legacy feature-cell diagnostic) `+0.2583`, final BPF (legacy feature-cell diagnostic) `2.3965`, delta final bpf (legacy feature-cell diagnostic) `+0.2582`, best ROC AUC `0.7719`, delta final training time `-6724.9s`

### 12. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `144x4` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Phase-two Muon scaling row `144x4` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7ced1d99f22a54bfcb426e1cfb57b5f40bfc925b674839077baadc1b8d838e40`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1` with final log loss `0.4914`, delta final log loss `+0.0963`, final Brier score `0.2989`, delta final brier score `+0.0404`, final ROC AUC `0.7866`, delta final roc auc `+0.0283`, final BPC (legacy feature-cell diagnostic) `3.0328`, delta final bpc (legacy feature-cell diagnostic) `+0.8946`, final BPF (legacy feature-cell diagnostic) `3.0327`, delta final bpf (legacy feature-cell diagnostic) `+0.8944`, best ROC AUC `0.7866`, delta final training time `-4906.3s`

### 13. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4e8ef4060e6e381e4c0190714219f1c0840a8daa296ccf925c7d9589f8ccff4d`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1` with final log loss `0.5710`, delta final log loss `+0.1759`, final Brier score `0.3496`, delta final brier score `+0.0911`, final ROC AUC `0.7438`, delta final roc auc `-0.0145`, final BPC (legacy feature-cell diagnostic) `2.1439`, delta final bpc (legacy feature-cell diagnostic) `+0.0056`, final BPF (legacy feature-cell diagnostic) `2.1439`, delta final bpf (legacy feature-cell diagnostic) `+0.0056`, best ROC AUC `0.7438`, delta final training time `-8124.8s`

### 14. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e1a088c900e2dd3107c63b6634b5287947ac96814bb28c201483c027f56c3cae`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1` with final log loss `0.5159`, delta final log loss `+0.1208`, final Brier score `0.3136`, delta final brier score `+0.0551`, final ROC AUC `0.7760`, delta final roc auc `+0.0177`, final BPC (legacy feature-cell diagnostic) `2.1753`, delta final bpc (legacy feature-cell diagnostic) `+0.0370`, final BPF (legacy feature-cell diagnostic) `2.1753`, delta final bpf (legacy feature-cell diagnostic) `+0.0370`, best ROC AUC `0.7760`, delta final training time `-7646.0s`

### 15. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `361d639f57f338ee430d40cfa5eeb3a263e1679918bb03b071b322a43224c933`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1` with final log loss `0.5080`, delta final log loss `+0.1129`, final Brier score `0.3091`, delta final brier score `+0.0506`, final ROC AUC `0.7724`, delta final roc auc `+0.0141`, final BPC (legacy feature-cell diagnostic) `2.2106`, delta final bpc (legacy feature-cell diagnostic) `+0.0723`, final BPF (legacy feature-cell diagnostic) `2.2106`, delta final bpf (legacy feature-cell diagnostic) `+0.0723`, best ROC AUC `0.7724`, delta final training time `-6699.7s`

### 16. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Execute fresh Muon one-epoch NS row `192x5` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Phase-two Muon scaling row `192x5` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `938e958e142b760df3b909f4af57f44383d54f3dd6babb27da1f8a3c5c39f647`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1` with final log loss `0.4936`, delta final log loss `+0.0985`, final Brier score `0.3003`, delta final brier score `+0.0418`, final ROC AUC `0.7808`, delta final roc auc `+0.0225`, final BPC (legacy feature-cell diagnostic) `2.3067`, delta final bpc (legacy feature-cell diagnostic) `+0.1685`, final BPF (legacy feature-cell diagnostic) `2.3067`, delta final bpf (legacy feature-cell diagnostic) `+0.1685`, best ROC AUC `0.7808`, delta final training time `-4883.5s`

### 17. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aa5b409db83c82febe75d601e0f90d2906bd787c9ea7952f78ed4280b2055774`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5417`, delta final log loss `+0.1466`, final Brier score `0.3294`, delta final brier score `+0.0709`, final ROC AUC `0.7585`, delta final roc auc `+0.0002`, final BPC (legacy feature-cell diagnostic) `2.3881`, delta final bpc (legacy feature-cell diagnostic) `+0.2498`, final BPF (legacy feature-cell diagnostic) `2.3881`, delta final bpf (legacy feature-cell diagnostic) `+0.2498`, best ROC AUC `0.7585`, delta final training time `-8067.5s`

### 18. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3e22982f1f83795b1ad1ec4ef934c3d99f444b657c4493e78b03f10009d574c`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5125`, delta final log loss `+0.1174`, final Brier score `0.3124`, delta final brier score `+0.0539`, final ROC AUC `0.7722`, delta final roc auc `+0.0139`, final BPC (legacy feature-cell diagnostic) `2.3005`, delta final bpc (legacy feature-cell diagnostic) `+0.1622`, final BPF (legacy feature-cell diagnostic) `2.3005`, delta final bpf (legacy feature-cell diagnostic) `+0.1622`, best ROC AUC `0.7722`, delta final training time `-7556.3s`

### 19. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3bb44390f194c0ebe7c79bbd152d7cb7569533370182cf968fa9f17929312a2`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on `openml_classification_medium_v1` using `best_and_final` checkpoints.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5067`, delta final log loss `+0.1117`, final Brier score `0.3077`, delta final brier score `+0.0492`, final ROC AUC `0.7744`, delta final roc auc `+0.0161`, final BPC (legacy feature-cell diagnostic) `2.2683`, delta final bpc (legacy feature-cell diagnostic) `+0.1300`, final BPF (legacy feature-cell diagnostic) `2.2683`, delta final bpf (legacy feature-cell diagnostic) `+0.1300`, best ROC AUC `0.7744`, delta final training time `-6517.9s`

### 20. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch NS row `264x6` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon scaling row `264x6` remains benchmarked against the locked `128x2` width-screen anchor while changing only the declared geometry or step budget for the fresh Muon study family.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d0954348936f014ca6fb8b6093f79c14336c2a53f5fce4baf1423acd8517a408`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch scaling row for phase-one geometry `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4951`, delta final log loss `+0.1000`, final Brier score `0.3015`, delta final brier score `+0.0430`, final ROC AUC `0.7820`, delta final roc auc `+0.0237`, final BPC (legacy feature-cell diagnostic) `2.4878`, delta final bpc (legacy feature-cell diagnostic) `+0.3495`, final BPF (legacy feature-cell diagnostic) `2.4878`, delta final bpf (legacy feature-cell diagnostic) `+0.3495`, best ROC AUC `0.7820`, delta final training time `-4261.3s`
