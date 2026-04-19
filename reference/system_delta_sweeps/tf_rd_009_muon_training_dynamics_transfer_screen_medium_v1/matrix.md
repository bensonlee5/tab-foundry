# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `4e0f5eb1a50b7ef600b55d93c12d9cfeb813add417a9064b385064cf54d7ec3e`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_training_dynamics_transfer_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `8686.8s`

## Anchor Comparison

Upstream reference: `Deriving Hyperparameter Scaling Laws via Modern Optimization Theory` from `https://arxiv.org/abs/2603.15958`.

| Dimension | Upstream Deriving Hyperparameter Scaling Laws via Modern Optimization Theory | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Screen regime B on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 21 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 22 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 23 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 24 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 25 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 26 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 27 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 28 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 29 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |
| 30 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Screen regime D on T0 and keep only the winning candidate for benchmark-backed transfer validation. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.001, momentum=0.9, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.001_m0.9_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.001, momentum=0.95, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.001_m0.95_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.001, momentum=0.975, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.001_m0.975_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.002, momentum=0.9, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.002_m0.9_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.002, momentum=0.95, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.002_m0.95_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime B at 144x4 on T0.
- Hypothesis: Regime B candidate with lr_max=0.002, momentum=0.975, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the screened T0 anchor should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'B', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_b_lr0.002_m0.975_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.9, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.9_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.95, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.95_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.975, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.975_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.9, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.9_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.95, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.95_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.975, B_eff=64 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b1b16140b1c4030d2da1e02fa2bb39adca58cda45df052302e78c0db9961893f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.975_beff64', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 13. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.9, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0652cd7a1a2f4f4378b0bc4201fd46927d3237fff532f57e865f87d945e715a1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.9_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 14. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.95, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0652cd7a1a2f4f4378b0bc4201fd46927d3237fff532f57e865f87d945e715a1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.95_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 15. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.975, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0652cd7a1a2f4f4378b0bc4201fd46927d3237fff532f57e865f87d945e715a1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.975_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 16. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.9, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `633593b78c242bf9b3d024d12f3f9e74515c92df6c4610eb748f401437e0ce50`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.9_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 17. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.95, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `633593b78c242bf9b3d024d12f3f9e74515c92df6c4610eb748f401437e0ce50`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.95_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 18. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.975, B_eff=80 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `633593b78c242bf9b3d024d12f3f9e74515c92df6c4610eb748f401437e0ce50`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 500, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.975_beff80', 'target_effective_batch': 80, 'realized_effective_batch': 80, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 19. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.9, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `df8dc783cbed79e092a8cb91b10990e689e1bb0f79e5e24184b66da7ef85d9c8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.9_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 20. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.95, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `df8dc783cbed79e092a8cb91b10990e689e1bb0f79e5e24184b66da7ef85d9c8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.95_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 21. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.975, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `df8dc783cbed79e092a8cb91b10990e689e1bb0f79e5e24184b66da7ef85d9c8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.975_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 22. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.9, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `66ff51cd4381a71a8b2e32bab1785f7d94c1361830e0bdb47353ec65466265dd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.9_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 23. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.95, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `66ff51cd4381a71a8b2e32bab1785f7d94c1361830e0bdb47353ec65466265dd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.95_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 24. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.975, B_eff=96 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `66ff51cd4381a71a8b2e32bab1785f7d94c1361830e0bdb47353ec65466265dd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 417}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 417}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 417, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.975_beff96', 'target_effective_batch': 96, 'realized_effective_batch': 96, 'target_effective_budget': 40000, 'realized_effective_budget': 40032, 'budget_drift': 0.0007999999999999119, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 25. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.9, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1770a916bb5a44aa876d02f68fefe83658abdd2f262db49c43d7611a0fb12d56`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.9_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 26. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.95, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1770a916bb5a44aa876d02f68fefe83658abdd2f262db49c43d7611a0fb12d56`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.95_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 27. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.001, momentum=0.975, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1770a916bb5a44aa876d02f68fefe83658abdd2f262db49c43d7611a0fb12d56`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.001_m0.975_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 28. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.9, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `09eba2e92f3e1b5494dd6795f09424de49e9daead5b5c778f16708a87cd1efb7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.9_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 29. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.95, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `09eba2e92f3e1b5494dd6795f09424de49e9daead5b5c778f16708a87cd1efb7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.95_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 30. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Paper-derived Muon transfer screen row for regime D at 144x4 on T0.
- Hypothesis: Regime D candidate with lr_max=0.002, momentum=0.975, B_eff=128 may be the best T0 anchor for faithful transfer.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Hold 144x4 fixed and screen the paper-derived T0 anchor before any extrapolation.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the screened T0 anchor should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `09eba2e92f3e1b5494dd6795f09424de49e9daead5b5c778f16708a87cd1efb7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 313}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 8, 'max_steps': 313}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 313, 'lr_max': 0.002, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'screen', 'regime_label': 'D', 'formula_label': 'paper T0 screen anchor', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'regime_d_lr0.002_m0.975_beff128', 'target_effective_batch': 128, 'realized_effective_batch': 128, 'target_effective_budget': 40000, 'realized_effective_budget': 40064, 'budget_drift': 0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use train-only screen metrics to choose exactly one T0 anchor per paper regime before any benchmark or larger-budget transfer.
  - Rank candidates by lower upper_block_final_window_mean, then upper_block_post_warmup_mean_slope, then clipped_step_fraction, then final_train_loss_ema.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Faithful paper-derived transfer screen under issue #284; hold geometry fixed at 144x4.
  - This row is train-only and exists solely to choose the T0 anchor for its regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending
