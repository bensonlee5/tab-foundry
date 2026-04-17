# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_ns_one_epoch_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `870fc98bf670f253d2ded21cb42c804e4faabeb639f240f327fd2ae5c2e34315`

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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 1 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 2 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 3 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 4 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 5 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 6 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 7 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 8 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 9 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 10 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 11 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 12 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 13 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 14 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 15 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 16 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 17 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 18 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 19 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | completed | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Execute order 20 as fresh Muon batch-critical row in `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`; treat `Bcrit` and `Cmin` as diagnostic-only. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a37e5e96079a006a8f0527e1dc7cfce9922ecaebb03aa0ccbe0f922aae0b5b3c`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.6293`, delta final log loss `+0.2343`, final Brier score `0.3860`, delta final brier score `+0.1275`, final ROC AUC `0.6770`, delta final roc auc `-0.0813`, final BPC (legacy feature-cell diagnostic) `2.4060`, delta final bpc (legacy feature-cell diagnostic) `+0.2677`, final BPF (legacy feature-cell diagnostic) `2.4060`, delta final bpf (legacy feature-cell diagnostic) `+0.2677`, best ROC AUC `0.6770`, delta final training time `-8238.0s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ac4441d692818e19c8b588668709c507caf701aed195505ae70eae052e98a12d`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5291`, delta final log loss `+0.1340`, final Brier score `0.3212`, delta final brier score `+0.0627`, final ROC AUC `0.7710`, delta final roc auc `+0.0127`, final BPC (legacy feature-cell diagnostic) `2.3123`, delta final bpc (legacy feature-cell diagnostic) `+0.1740`, final BPF (legacy feature-cell diagnostic) `2.3123`, delta final bpf (legacy feature-cell diagnostic) `+0.1740`, best ROC AUC `0.7710`, delta final training time `-8068.0s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `428ae7f9a12fe555119aa4079cfcf576a6e8feafc954b4926136b49c0d0fa49b`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5180`, delta final log loss `+0.1230`, final Brier score `0.3153`, delta final brier score `+0.0568`, final ROC AUC `0.7713`, delta final roc auc `+0.0130`, final BPC (legacy feature-cell diagnostic) `2.3946`, delta final bpc (legacy feature-cell diagnostic) `+0.2564`, final BPF (legacy feature-cell diagnostic) `2.3946`, delta final bpf (legacy feature-cell diagnostic) `+0.2563`, best ROC AUC `0.7713`, delta final training time `-7591.4s`

### 4. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=1 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `88fa2c3ca6050c988ad90c65ed82431feec04b0e780fcabddac4f60598378c4d`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=1.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5150`, delta final log loss `+0.1199`, final Brier score `0.3123`, delta final brier score `+0.0538`, final ROC AUC `0.7693`, delta final roc auc `+0.0110`, final BPC (legacy feature-cell diagnostic) `2.3104`, delta final bpc (legacy feature-cell diagnostic) `+0.1722`, final BPF (legacy feature-cell diagnostic) `2.3104`, delta final bpf (legacy feature-cell diagnostic) `+0.1722`, best ROC AUC `0.7693`, delta final training time `-6593.5s`

### 5. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4e495487d3ae4b1979ecc5d1b2c9b67a39912d9ee62c87fc5a335a4da4387a2f`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5523`, delta final log loss `+0.1572`, final Brier score `0.3344`, delta final brier score `+0.0759`, final ROC AUC `0.7566`, delta final roc auc `-0.0017`, final BPC (legacy feature-cell diagnostic) `2.2902`, delta final bpc (legacy feature-cell diagnostic) `+0.1519`, final BPF (legacy feature-cell diagnostic) `2.2901`, delta final bpf (legacy feature-cell diagnostic) `+0.1519`, best ROC AUC `0.7566`, delta final training time `-8146.6s`

### 6. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `40966da86cc6640b773384af5a1ed2c292b48133c86280a90bc73bac39405436`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5228`, delta final log loss `+0.1278`, final Brier score `0.3200`, delta final brier score `+0.0615`, final ROC AUC `0.7668`, delta final roc auc `+0.0085`, final BPC (legacy feature-cell diagnostic) `2.3264`, delta final bpc (legacy feature-cell diagnostic) `+0.1881`, final BPF (legacy feature-cell diagnostic) `2.3264`, delta final bpf (legacy feature-cell diagnostic) `+0.1881`, best ROC AUC `0.7668`, delta final training time `-7772.0s`

### 7. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5d99ce575e770c3c8550669bdd4ceea710a5d8fed891352ddd9de55bbb6676c6`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5021`, delta final log loss `+0.1070`, final Brier score `0.3057`, delta final brier score `+0.0472`, final ROC AUC `0.7783`, delta final roc auc `+0.0200`, final BPC (legacy feature-cell diagnostic) `2.3916`, delta final bpc (legacy feature-cell diagnostic) `+0.2533`, final BPF (legacy feature-cell diagnostic) `2.3915`, delta final bpf (legacy feature-cell diagnostic) `+0.2532`, best ROC AUC `0.7783`, delta final training time `-6940.4s`

### 8. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=2 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `64b2c30ee764cc9d1f67ea1de2a393d06d6cbc4d934f33c9f9b6c80d83e5bcdb`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=2.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5061`, delta final log loss `+0.1110`, final Brier score `0.3081`, delta final brier score `+0.0496`, final ROC AUC `0.7755`, delta final roc auc `+0.0172`, final BPC (legacy feature-cell diagnostic) `2.4026`, delta final bpc (legacy feature-cell diagnostic) `+0.2644`, final BPF (legacy feature-cell diagnostic) `2.4027`, delta final bpf (legacy feature-cell diagnostic) `+0.2644`, best ROC AUC `0.7755`, delta final training time `-5358.2s`

### 9. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aa5b409db83c82febe75d601e0f90d2906bd787c9ea7952f78ed4280b2055774`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5401`, delta final log loss `+0.1451`, final Brier score `0.3278`, delta final brier score `+0.0693`, final ROC AUC `0.7627`, delta final roc auc `+0.0044`, final BPC (legacy feature-cell diagnostic) `2.3787`, delta final bpc (legacy feature-cell diagnostic) `+0.2404`, final BPF (legacy feature-cell diagnostic) `2.3787`, delta final bpf (legacy feature-cell diagnostic) `+0.2404`, best ROC AUC `0.7627`, delta final training time `-7855.0s`

### 10. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3e22982f1f83795b1ad1ec4ef934c3d99f444b657c4493e78b03f10009d574c`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5143`, delta final log loss `+0.1192`, final Brier score `0.3135`, delta final brier score `+0.0550`, final ROC AUC `0.7758`, delta final roc auc `+0.0176`, final BPC (legacy feature-cell diagnostic) `2.2480`, delta final bpc (legacy feature-cell diagnostic) `+0.1098`, final BPF (legacy feature-cell diagnostic) `2.2481`, delta final bpf (legacy feature-cell diagnostic) `+0.1098`, best ROC AUC `0.7758`, delta final training time `-7160.1s`

### 11. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3bb44390f194c0ebe7c79bbd152d7cb7569533370182cf968fa9f17929312a2`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5046`, delta final log loss `+0.1095`, final Brier score `0.3066`, delta final brier score `+0.0481`, final ROC AUC `0.7789`, delta final roc auc `+0.0206`, final BPC (legacy feature-cell diagnostic) `2.6439`, delta final bpc (legacy feature-cell diagnostic) `+0.5056`, final BPF (legacy feature-cell diagnostic) `2.6439`, delta final bpf (legacy feature-cell diagnostic) `+0.5056`, best ROC AUC `0.7789`, delta final training time `-5724.0s`

### 12. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d0954348936f014ca6fb8b6093f79c14336c2a53f5fce4baf1423acd8517a408`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=4.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4962`, delta final log loss `+0.1011`, final Brier score `0.3020`, delta final brier score `+0.0435`, final ROC AUC `0.7784`, delta final roc auc `+0.0201`, final BPC (legacy feature-cell diagnostic) `2.5484`, delta final bpc (legacy feature-cell diagnostic) `+0.4101`, final BPF (legacy feature-cell diagnostic) `2.5483`, delta final bpf (legacy feature-cell diagnostic) `+0.4100`, best ROC AUC `0.7784`, delta final training time `-2890.5s`

### 13. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `721bbdb60643da636db1d37b9aa3c1e619e5c8e4625af4671467f85fbf474826`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5320`, delta final log loss `+0.1369`, final Brier score `0.3229`, delta final brier score `+0.0644`, final ROC AUC `0.7644`, delta final roc auc `+0.0061`, final BPC (legacy feature-cell diagnostic) `2.3565`, delta final bpc (legacy feature-cell diagnostic) `+0.2182`, final BPF (legacy feature-cell diagnostic) `2.3565`, delta final bpf (legacy feature-cell diagnostic) `+0.2182`, best ROC AUC `0.7644`, delta final training time `-7226.2s`

### 14. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7488f6a4c5f4592aa2edeb449448ede10a7c2d4844fd9bd5fba45f02761dd098`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5059`, delta final log loss `+0.1109`, final Brier score `0.3079`, delta final brier score `+0.0494`, final ROC AUC `0.7779`, delta final roc auc `+0.0196`, final BPC (legacy feature-cell diagnostic) `2.2874`, delta final bpc (legacy feature-cell diagnostic) `+0.1491`, final BPF (legacy feature-cell diagnostic) `2.2874`, delta final bpf (legacy feature-cell diagnostic) `+0.1491`, best ROC AUC `0.7779`, delta final training time `-5912.2s`

### 15. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1e6a5547eadba4aa0f134e65f86fb821c5eede915952ec524754811482f87f10`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4948`, delta final log loss `+0.0997`, final Brier score `0.3013`, delta final brier score `+0.0428`, final ROC AUC `0.7814`, delta final roc auc `+0.0231`, final BPC (legacy feature-cell diagnostic) `2.4554`, delta final bpc (legacy feature-cell diagnostic) `+0.3172`, final BPF (legacy feature-cell diagnostic) `2.4554`, delta final bpf (legacy feature-cell diagnostic) `+0.3172`, best ROC AUC `0.7814`, delta final training time `-3267.8s`

### 16. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=8 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5b9c993b32c52b1f2966a4a0a8c9f5048ccc70359cd5986b223e07d59a3931fd`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=8.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Execution attempt `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` failed: [enforce fail at inline_container.cc:672] . unexpected pos 27950848 vs 27950768
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4853`, delta final log loss `+0.0903`, final Brier score `0.2951`, delta final brier score `+0.0366`, final ROC AUC `0.7860`, delta final roc auc `+0.0277`, final BPC (legacy feature-cell diagnostic) `2.6024`, delta final bpc (legacy feature-cell diagnostic) `+0.4642`, final BPF (legacy feature-cell diagnostic) `2.6024`, delta final bpf (legacy feature-cell diagnostic) `+0.4641`, best ROC AUC `0.7860`, delta final training time `+1561.3s`

### 17. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=625 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cbfc8ee13ffdbd4b7165b5588ed5f90da71000becbcaf4be323273a51c03c957`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=625 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5210`, delta final log loss `+0.1259`, final Brier score `0.3178`, delta final brier score `+0.0593`, final ROC AUC `0.7714`, delta final roc auc `+0.0131`, final BPC (legacy feature-cell diagnostic) `2.2054`, delta final bpc (legacy feature-cell diagnostic) `+0.0671`, final BPF (legacy feature-cell diagnostic) `2.2054`, delta final bpf (legacy feature-cell diagnostic) `+0.0671`, best ROC AUC `0.7714`, delta final training time `-6003.8s`

### 18. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=1250 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `67e15975badf1ba2539d0d0c61e95605710a3ab7f1534ff064388ae2783f566d`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=1250 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.5024`, delta final log loss `+0.1073`, final Brier score `0.3060`, delta final brier score `+0.0475`, final ROC AUC `0.7773`, delta final roc auc `+0.0190`, final BPC (legacy feature-cell diagnostic) `2.2896`, delta final bpc (legacy feature-cell diagnostic) `+0.1513`, final BPF (legacy feature-cell diagnostic) `2.2896`, delta final bpf (legacy feature-cell diagnostic) `+0.1513`, best ROC AUC `0.7773`, delta final training time `-3468.7s`

### 19. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=2500 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `07954ee775856a8500a6406eeb2e0578a28491ef0bfe0b918dcbcfa3effa1a1f`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=2500 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4803`, delta final log loss `+0.0852`, final Brier score `0.2917`, delta final brier score `+0.0332`, final ROC AUC `0.7881`, delta final roc auc `+0.0298`, final BPC (legacy feature-cell diagnostic) `2.2720`, delta final bpc (legacy feature-cell diagnostic) `+0.1337`, final BPF (legacy feature-cell diagnostic) `2.2720`, delta final bpf (legacy feature-cell diagnostic) `+0.1337`, best ROC AUC `0.7881`, delta final training time `+1727.8s`

### 20. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Execute fresh Muon one-epoch batch-critical row `264x6` at max_steps=5000 and grad_accum_steps=16 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Phase-two Muon batch-critical row `264x6` keeps the `264x6` family fixed while the sweep remains benchmarked against the locked `128x2` width-screen anchor.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `29b12512e861faff1a4d9904f8a2cebc115794c70d16e82368317f3dfce9237f`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Fresh Muon one-epoch batch-critical row for phase-one winner `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` at max_steps=5000 and grad_accum_steps=16.
  - This Muon rerun stays on `tf_rd_010_dagzoo_medium_control_curated_v6` and the post-#271 packed/runtime stack.
  - Do not mix this row with historical schedulefree TF-RD-009 Phase-2 evidence.
  - Execution attempt `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` failed: [Errno 28] No space left on device
  - Canonical rerun registered as `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Corrected sparse multiclass replay recorded on openml_classification_medium_v1 using best_and_final checkpoints.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_batch_critical_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_muon_batch_critical_one_epoch_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1` with final log loss `0.4646`, delta final log loss `+0.0696`, final Brier score `0.2832`, delta final brier score `+0.0247`, final ROC AUC `0.7946`, delta final roc auc `+0.0363`, final BPC (legacy feature-cell diagnostic) `2.4329`, delta final bpc (legacy feature-cell diagnostic) `+0.2946`, final BPF (legacy feature-cell diagnostic) `2.4329`, delta final bpf (legacy feature-cell diagnostic) `+0.2946`, best ROC AUC `0.7946`, delta final training time `+12911.4s`
