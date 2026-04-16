# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_ns_one_epoch_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `e603175853889ac6cad04c23dca20924155c88483e945bed6c267fbcb74256bf`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law_phase2_batch`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `8686.8s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 1 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 2 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 3 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 4 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 5 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 6 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 7 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 8 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 9 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 10 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 11 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 12 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 13 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 14 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 15 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 16 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 17 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 18 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 19 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 20 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e49d94e8c87d92913f93c37e9ac8932b4e8aced310ab2336cc1b750080effdd4`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=625 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=10000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `49a7360f0dc6b2afebe09dc8c516a52c7cab7a7e44372de87d8a59b4671c127e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=1250 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=20000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4bf5f6d7cb01c65dc9ff8c0b4f942989be9be6f497ebefa58fa99e0d88bcba79`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=2500 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f097db5d0db1361d92016956dd90cfdc446a63785f17f47ee7ba1121cfc8fd25`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=5000 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `929c9693d05e13bfe692cd03beeda18f4e91e6da0ff7ea79ecd77936a61bc31e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=625 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=20000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5afdae331f17e22e6cfa62922c170ffea663b6bdb173269e1d46f2ce5a274a2b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=1250 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e31fc926374c14488f2d23e8bac25376f14e6b392fc0850fa79384b13878d574`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=2500 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d5b35fcb9da99d4cea5ef077f87bdfa4289012278b9102f0275702d2e459e98c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=5000 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cc37c06521b8b512d95fa67e25eab46e4838223a3af997656f490ca5ca853e5b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=625 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c5c181dd9254b4b1e9112840df2e4805eba9a00d19fbcb6d8c46e13fdabc56c1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=1250 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65da367784e87491aa576cccb3457df140819cc49c8244e749915e549d9891b6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=2500 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ac26faef6594ce195426ff6277ce364956e066ee3636e32eec35863a66423a43`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=5000 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 13. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b0cca8ef1b287468376c2baeeeb0e75717174e5726d4b5892d17d554d8329b0f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=625 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 14. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8b7e21434249c20c10db850cf06b84832a73c0d1306905012f46589385c6be51`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=1250 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 15. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `912d7043d44a85a9da10095b917acbca2cd275e5261e0905fccb106b1d3926f9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=2500 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 16. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6d7fab558e42c45d8ea7d074b2d0a683cd4448f3d47cc95946fa12cc0f9075cb`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=5000 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=640000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 17. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `9a07e3bb7c70d272078524bd131b86752bbf9e214cd93d56f978c09039325d05`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=625 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 18. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1ec15952d93ea83dbdbe32d45beef1e544e73ce86e90fa8f6f7c0ee53adf177c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=1250 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 19. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b3088896e8578ea550d96f3bcd12b6e03255fbfededd6d4bc08143eb54bdcc9d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=2500 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=640000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 20. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c1f96d623ab69a7df23951e466460c59f987696509b8ce5da1eecbbfbeb72c3b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Hold geometry `264x6` at the completed fresh Muon Phase-1 winner while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=5000 for the fresh Muon batch ladder; keep `Bcrit` and derived `Cmin` diagnostic-only in later reporting.
  - Keep the post-#271 packed Muon runtime surface and the v6 one-epoch contract fixed; this row changes only effective batch or step budget.
  - Corrected one-epoch contract: required_train_tasks=1280000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
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
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending
