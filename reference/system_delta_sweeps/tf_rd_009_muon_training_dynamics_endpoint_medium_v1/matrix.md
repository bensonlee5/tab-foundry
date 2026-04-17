# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_training_dynamics_endpoint_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `de30d58f0cbf8ffb299d5f2670f2152987dfc7017dc6cd16289ab2b63d891b6e`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_training_dynamics_selector`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `8686.8s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_lowbatch_v1` | classification_scaling_law | no | ready | none | Re-run the carried Muon `128x2` in-family baseline at the fixed 5000-step endpoint with the low-batch `B_eff=64` contract (`task_batch_size=16`, `grad_accum_steps=4`) as the selector floor. | Benchmark this low-batch endpoint row and rank it on the corrected medium loss/time Pareto frontier before any larger-family work. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run the carried Muon `128x2` geometry at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward contract (`grad_accum_steps=16`) and the original Muon LR/beta surface unchanged. | Benchmark this high-batch carried reference and compare it against the low-batch row on the corrected medium loss/time Pareto frontier. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_linear_lr_batch_v1` | classification_scaling_law | no | ready | none | Test the literature-backed linear LR/batch prescription on `128x2` at `B_eff=256`, scaling `lr_max` to `4e-3` and `min_lr` to `4e-6` while keeping `betas=(0.9, 0.95)` and `weight_decay=0.01`. | Benchmark this linear LR/batch row and keep it only if it improves the corrected medium loss/time Pareto frontier. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_momentum_timescale_v1` | classification_scaling_law | no | ready | none | Test the momentum-timescale Muon selector prescription on `128x2` at `B_eff=256`, keeping `lr_max=4e-3`, `min_lr=4e-6`, `beta2=0.95`, and raising `beta1` to `0.975`. | Benchmark this momentum-timescale row and keep it only if it improves the corrected medium loss/time Pareto frontier versus the simpler high-batch rows. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_lowbatch_v1` | classification_scaling_law | no | ready | none | Re-run the corrected Muon NS winner geometry `144x4` at the fixed 5000-step endpoint with the carried low-batch `B_eff=64` contract. | Benchmark this low-batch endpoint row and rank it on the corrected medium loss/time Pareto frontier before any larger-family work. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface. | Benchmark this high-batch carried reference and compare it against the low-batch row on the corrected medium loss/time Pareto frontier. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_linear_lr_batch_v1` | classification_scaling_law | no | ready | none | Test the linear LR/batch Muon selector prescription on `144x4` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `betas=(0.9, 0.95)`, and `weight_decay=0.01`. | Benchmark this linear LR/batch row and keep it only if it improves the corrected medium loss/time Pareto frontier. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_momentum_timescale_v1` | classification_scaling_law | no | ready | none | Test the momentum-timescale Muon selector prescription on `144x4` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `beta1=0.975`, `beta2=0.95`, and `weight_decay=0.01`. | Benchmark this momentum-timescale row and keep it only if it improves the corrected medium loss/time Pareto frontier versus the simpler high-batch rows. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_lowbatch_v1` | classification_scaling_law | no | ready | none | Re-run the corrected overall Muon Phase-2 winner geometry `264x6` at the fixed 5000-step endpoint with the low-batch `B_eff=64` carry-forward contract. | Benchmark this low-batch endpoint row and rank it on the corrected medium loss/time Pareto frontier before any larger-family work. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run `264x6` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface. | Benchmark this high-batch carried reference and compare it against the low-batch row on the corrected medium loss/time Pareto frontier. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_linear_lr_batch_v1` | classification_scaling_law | no | ready | none | Test the linear LR/batch Muon selector prescription on `264x6` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `betas=(0.9, 0.95)`, and `weight_decay=0.01`. | Benchmark this linear LR/batch row and keep it only if it improves the corrected medium loss/time Pareto frontier. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_momentum_timescale_v1` | classification_scaling_law | no | ready | none | Test the momentum-timescale Muon selector prescription on `264x6` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `beta1=0.975`, `beta2=0.95`, and `weight_decay=0.01`. | Benchmark this momentum-timescale row and keep it only if it improves the corrected medium loss/time Pareto frontier versus the simpler high-batch rows. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_lowbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run the carried Muon `128x2` in-family baseline at the fixed 5000-step endpoint with the low-batch `B_eff=64` contract (`task_batch_size=16`, `grad_accum_steps=4`) as the selector floor.
- Rationale: Compact Muon training-dynamics selector row for `128x2` using the carried low-batch baseline at a fixed 5000-step horizon.
- Hypothesis: Re-running the carried low-batch Muon surface at the fixed 5000-step endpoint gives the compact time-efficient selector baseline.
- Upstream delta: This is the compact endpoint selector baseline that follows corrected Muon Phase 2 and the cleanup-first navigation lane; it keeps the corrected multiclass benchmark contract and the `v6` training corpus fixed while measuring the carried low-batch dynamics point directly.
- Anchor delta: Keep the carried Muon optimizer/runtime surface and `B_eff=64` as the selector baseline.
- Expected effect: If the carried low-batch Muon surface is still on the quality/time frontier after the corrected Phase-2 closeout, this row should remain the compact selector floor that later higher-batch prescriptions must beat.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `63f4e190b563f9ecc902d6baa68d10a621f95685f210fff34ad80e76e3a29f94`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon runtime and optimizer surface with `weight_decay=0.01`
  - no repeats; selector ranks rows on the corrected quality/time Pareto frontier
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `128x2` with carried low-batch baseline.
  - Selector baseline row: keep the carried Muon optimizer surface and `B_eff=64` (`task_batch_size=16`, `grad_accum_steps=4`).
  - This row exists to quantify the quality/time floor before any higher-batch or literature-backed prescription is admitted.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_lowbatch_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run the carried Muon `128x2` geometry at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward contract (`grad_accum_steps=16`) and the original Muon LR/beta surface unchanged.
- Rationale: Compact Muon training-dynamics selector row for `128x2` using the carried high-batch empirical reference at a fixed 5000-step horizon.
- Hypothesis: Holding the carried Muon optimizer surface fixed while raising `B_eff` to 256 should test whether the Phase-2 batch-side gain transfers across the compact geometry set.
- Upstream delta: Corrected Muon Phase 2 showed that batch-side movement dominated geometry-only movement; this row tests whether the measured high-batch gain transfers to the compact `128x2` selector geometry without opening the optimizer prescription yet.
- Anchor delta: Raise effective batch to `B_eff=256` without changing the carried Muon LR or beta surface.
- Expected effect: If the Phase-2 batch-side gain transfers cleanly, this row should beat or match the low-batch `128x2` floor at acceptable extra training time and justify keeping high-batch dynamics in the next law-conditioning lane.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `10a9234a87808903bf9a2ccaabe0073e5dc4af95b5c9857f2b9d90a1b04cc29e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 16, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon LR, beta, and weight-decay surface
  - high-batch empirical reference only; no broader optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `128x2` with carried high-batch empirical reference.
  - Empirical carry-forward row: keep the measured Muon LR/beta surface fixed and only raise `B_eff` to 256.
  - This row tests whether the corrected Phase-2 batch-side gain transfers cleanly across the compact geometry set.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_linear_lr_batch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the literature-backed linear LR/batch prescription on `128x2` at `B_eff=256`, scaling `lr_max` to `4e-3` and `min_lr` to `4e-6` while keeping `betas=(0.9, 0.95)` and `weight_decay=0.01`.
- Rationale: Compact Muon training-dynamics selector row for `128x2` using the linear LR/batch prescription at a fixed 5000-step horizon.
- Hypothesis: Under fixed corpus and fixed weight decay, a first-pass `eta ∝ B` prescription may outperform the carried high-batch reference without reopening geometry.
- Upstream delta: This row encodes the first defended optimizer-timescale variant for the compact selector after corrected Muon Phase 2, testing the simplest `eta proportional to batch` prescription under the fixed corpus, benchmark, and horizon contract.
- Anchor delta: Keep `B_eff=256` and test the literature-backed linear LR/batch prescription `eta ∝ B`.
- Expected effect: If the corrected Muon surface follows the linear LR/batch invariant closely enough, this row should improve quality at the high-batch endpoint without requiring a separate geometry change.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `06b20c3067b9dac0deb69aaec88af3cfd1f6e44736f5971b398a40c0cb91b0ea`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 16, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with linear LR scaling
  - `weight_decay=0.01` stays frozen while batch and LR move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `128x2` with linear LR/batch prescription.
  - Literature-backed selector row: test the first defended invariant `eta ∝ B` under fixed corpus and fixed weight decay.
  - Do not reinterpret this row as an optimizer scaling law claim; it is a compact selector candidate for the next Muon contract.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_linear_lr_batch_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_momentum_timescale_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the momentum-timescale Muon selector prescription on `128x2` at `B_eff=256`, keeping `lr_max=4e-3`, `min_lr=4e-6`, `beta2=0.95`, and raising `beta1` to `0.975`.
- Rationale: Compact Muon training-dynamics selector row for `128x2` using the momentum-timescale prescription at a fixed 5000-step horizon.
- Hypothesis: If the batch-side Phase-2 gain reflects a timescale effect, keeping `B(1-beta1)` approximately constant at high batch may outperform the plain linear LR/batch row.
- Upstream delta: Corrected Muon Phase 2 justified training-dynamics-first work; this row is the second defended invariant-backed probe, testing whether a larger effective momentum timescale transfers better than simple linear LR scaling on the compact selector geometry.
- Anchor delta: Keep `B_eff=256`, raise LR with batch, and test a momentum-timescale variant with `beta1=0.975`.
- Expected effect: If momentum-timescale coupling matters on the corrected Muon surface, this row should outperform the linear LR/batch prescription at comparable wall time on `128x2`.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `06b20c3067b9dac0deb69aaec88af3cfd1f6e44736f5971b398a40c0cb91b0ea`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.975, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 16, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with `beta1=0.975`
  - `weight_decay=0.01` stays frozen while LR and momentum move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `128x2` with momentum-timescale prescription.
  - Literature-backed selector row: test a timescale-style invariant by pairing high batch with `beta1=0.975` while keeping `beta2=0.95`.
  - This row is still selector evidence only; do not promote it into a larger-family law claim without the later Phase-2B rerun.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_momentum_timescale_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_lowbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run the corrected Muon NS winner geometry `144x4` at the fixed 5000-step endpoint with the carried low-batch `B_eff=64` contract.
- Rationale: Compact Muon training-dynamics selector row for `144x4` using the carried low-batch baseline at a fixed 5000-step horizon.
- Hypothesis: Re-running the carried low-batch Muon surface at the fixed 5000-step endpoint gives the compact time-efficient selector baseline.
- Upstream delta: Corrected Muon Phase 2 showed that `144x4` is the best NS geometry while `264x6` only wins after batch tuning; this row carries the NS winner into the compact training-dynamics selector without changing the geometry.
- Anchor delta: Keep the carried Muon optimizer/runtime surface and `B_eff=64` as the selector baseline.
- Expected effect: If `144x4` remains the best compact geometry under the selector contract, this low-batch row should establish the geometry’s time-efficient frontier point before any higher-batch or literature-backed prescription is kept.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ad68d3446dbc6cbc160bd2cb424a3e7849cb1f44f713092cfbc276b2f844799d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon runtime and optimizer surface with `weight_decay=0.01`
  - no repeats; selector ranks rows on the corrected quality/time Pareto frontier
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `144x4` with carried low-batch baseline.
  - Selector baseline row: keep the carried Muon optimizer surface and `B_eff=64` (`task_batch_size=16`, `grad_accum_steps=4`).
  - This row exists to quantify the quality/time floor before any higher-batch or literature-backed prescription is admitted.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_lowbatch_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface.
- Rationale: Compact Muon training-dynamics selector row for `144x4` using the carried high-batch empirical reference at a fixed 5000-step horizon.
- Hypothesis: Holding the carried Muon optimizer surface fixed while raising `B_eff` to 256 should test whether the Phase-2 batch-side gain transfers across the compact geometry set.
- Upstream delta: This is the direct test of whether the corrected Phase-2 batch-side gain also improves the `144x4` NS winner when geometry is held fixed.
- Anchor delta: Raise effective batch to `B_eff=256` without changing the carried Muon LR or beta surface.
- Expected effect: If batch-side movement generalizes across the compact frontier, this row should improve `144x4` materially enough to challenge the carried low-batch point on the corrected quality/time frontier.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d788d8328141bedffd258acd3393ef7087e3c906ed36df06bd0f618fd56f7baa`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon LR, beta, and weight-decay surface
  - empirical high-batch reference only; no broader optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `144x4` with carried high-batch empirical reference.
  - Empirical carry-forward row: keep the measured Muon LR/beta surface fixed and only raise `B_eff` to 256.
  - This row tests whether the corrected Phase-2 batch-side gain transfers cleanly across the compact geometry set.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_linear_lr_batch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the linear LR/batch Muon selector prescription on `144x4` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `betas=(0.9, 0.95)`, and `weight_decay=0.01`.
- Rationale: Compact Muon training-dynamics selector row for `144x4` using the linear LR/batch prescription at a fixed 5000-step horizon.
- Hypothesis: Under fixed corpus and fixed weight decay, a first-pass `eta ∝ B` prescription may outperform the carried high-batch reference without reopening geometry.
- Upstream delta: This row applies the simplest defended high-batch optimizer invariant to the corrected NS-winner geometry so the selector can decide whether `144x4` wants a new dynamics contract before any later Phase-2B rerun.
- Anchor delta: Keep `B_eff=256` and test the literature-backed linear LR/batch prescription `eta ∝ B`.
- Expected effect: If `eta proportional to batch` is a better organizing invariant than pure carry-forward at `144x4`, this row should improve corrected benchmark quality enough to stay on the Pareto frontier.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `31d415ca121d3bd94a067bd33fb7d11a278f3492933bd46bd6fa5872d22049af`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with linear LR scaling
  - `weight_decay=0.01` stays frozen while batch and LR move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `144x4` with linear LR/batch prescription.
  - Literature-backed selector row: test the first defended invariant `eta ∝ B` under fixed corpus and fixed weight decay.
  - Do not reinterpret this row as an optimizer scaling law claim; it is a compact selector candidate for the next Muon contract.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_linear_lr_batch_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_momentum_timescale_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the momentum-timescale Muon selector prescription on `144x4` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `beta1=0.975`, `beta2=0.95`, and `weight_decay=0.01`.
- Rationale: Compact Muon training-dynamics selector row for `144x4` using the momentum-timescale prescription at a fixed 5000-step horizon.
- Hypothesis: If the batch-side Phase-2 gain reflects a timescale effect, keeping `B(1-beta1)` approximately constant at high batch may outperform the plain linear LR/batch row.
- Upstream delta: This row is the second invariant-backed training-dynamics probe for the corrected NS-winner geometry, testing whether a longer effective momentum timescale gives a cleaner high-batch transfer than the linear LR/batch rule.
- Anchor delta: Keep `B_eff=256`, raise LR with batch, and test a momentum-timescale variant with `beta1=0.975`.
- Expected effect: If momentum-timescale coupling is the better invariant on `144x4`, this row should outperform the linear LR/batch `144x4` prescription at comparable high-batch wall time.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `31d415ca121d3bd94a067bd33fb7d11a278f3492933bd46bd6fa5872d22049af`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.975, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with `beta1=0.975`
  - `weight_decay=0.01` stays frozen while LR and momentum move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `144x4` with momentum-timescale prescription.
  - Literature-backed selector row: test a timescale-style invariant by pairing high batch with `beta1=0.975` while keeping `beta2=0.95`.
  - This row is still selector evidence only; do not promote it into a larger-family law claim without the later Phase-2B rerun.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_momentum_timescale_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_lowbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run the corrected overall Muon Phase-2 winner geometry `264x6` at the fixed 5000-step endpoint with the low-batch `B_eff=64` carry-forward contract.
- Rationale: Compact Muon training-dynamics selector row for `264x6` using the carried low-batch baseline at a fixed 5000-step horizon.
- Hypothesis: Re-running the carried low-batch Muon surface at the fixed 5000-step endpoint gives the compact time-efficient selector baseline.
- Upstream delta: Corrected Muon Phase 2 showed that `264x6` only wins once effective batch is pushed up; this row measures the larger geometry’s low-batch floor inside the compact selector so the later high-batch gain can be interpreted cleanly.
- Anchor delta: Keep the carried Muon optimizer/runtime surface and `B_eff=64` as the selector baseline.
- Expected effect: If `264x6` is genuinely under-optimized at low batch, this row should underperform the high-batch `264x6` rows and justify training-dynamics-first sequencing over immediate larger-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f827b6780fc6bf418ddc836c458638901244403df61fcd6c23148eba212584f0`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon runtime and optimizer surface with `weight_decay=0.01`
  - no repeats; selector ranks rows on the corrected quality/time Pareto frontier
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `264x6` with carried low-batch baseline.
  - Selector baseline row: keep the carried Muon optimizer surface and `B_eff=64` (`task_batch_size=16`, `grad_accum_steps=4`).
  - This row exists to quantify the quality/time floor before any higher-batch or literature-backed prescription is admitted.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_lowbatch_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run `264x6` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface.
- Rationale: Compact Muon training-dynamics selector row for `264x6` using the carried high-batch empirical reference at a fixed 5000-step horizon.
- Hypothesis: Holding the carried Muon optimizer surface fixed while raising `B_eff` to 256 should test whether the Phase-2 batch-side gain transfers across the compact geometry set.
- Upstream delta: This row carries forward the corrected Phase-2 result that made `264x6` the best overall row, but now inside the compact selector where quality is judged jointly with training time.
- Anchor delta: Raise effective batch to `B_eff=256` without changing the carried Muon LR or beta surface.
- Expected effect: If the corrected Phase-2 high-batch gain is robust enough to survive the selector replay, this row should stay on the corrected quality/time Pareto frontier and justify keeping `264x6` in later Phase-2B planning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2600b55293a704b039f05de7486db9377acc90231d0ad52433d0b5a231bab53f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - carried Muon LR, beta, and weight-decay surface
  - empirical high-batch reference only; no broader optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `264x6` with carried high-batch empirical reference.
  - Empirical carry-forward row: keep the measured Muon LR/beta surface fixed and only raise `B_eff` to 256.
  - This row tests whether the corrected Phase-2 batch-side gain transfers cleanly across the compact geometry set.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_linear_lr_batch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the linear LR/batch Muon selector prescription on `264x6` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `betas=(0.9, 0.95)`, and `weight_decay=0.01`.
- Rationale: Compact Muon training-dynamics selector row for `264x6` using the linear LR/batch prescription at a fixed 5000-step horizon.
- Hypothesis: Under fixed corpus and fixed weight decay, a first-pass `eta ∝ B` prescription may outperform the carried high-batch reference without reopening geometry.
- Upstream delta: This row asks whether the larger corrected winner geometry wants the simple `eta proportional to batch` prescription strongly enough to beat the empirical carry-forward contract once quality and time are judged together.
- Anchor delta: Keep `B_eff=256` and test the literature-backed linear LR/batch prescription `eta ∝ B`.
- Expected effect: If linear LR scaling is the better invariant for the larger geometry, this row should improve corrected benchmark quality enough to remain Pareto-admissible despite the geometry’s higher training cost.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d0291fe363bc08a66ae43f4aebfa1ce01f75a221b3f3b1c6438c9aae4c5aff31`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with linear LR scaling
  - `weight_decay=0.01` stays frozen while batch and LR move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `264x6` with linear LR/batch prescription.
  - Literature-backed selector row: test the first defended invariant `eta ∝ B` under fixed corpus and fixed weight decay.
  - Do not reinterpret this row as an optimizer scaling law claim; it is a compact selector candidate for the next Muon contract.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_linear_lr_batch_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_momentum_timescale_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the momentum-timescale Muon selector prescription on `264x6` at `B_eff=256` with `lr_max=4e-3`, `min_lr=4e-6`, `beta1=0.975`, `beta2=0.95`, and `weight_decay=0.01`.
- Rationale: Compact Muon training-dynamics selector row for `264x6` using the momentum-timescale prescription at a fixed 5000-step horizon.
- Hypothesis: If the batch-side Phase-2 gain reflects a timescale effect, keeping `B(1-beta1)` approximately constant at high batch may outperform the plain linear LR/batch row.
- Upstream delta: This row is the second invariant-backed high-batch probe for the larger corrected winner geometry, testing whether a longer momentum timescale produces a better kept dynamics contract than simple linear LR scaling.
- Anchor delta: Keep `B_eff=256`, raise LR with batch, and test a momentum-timescale variant with `beta1=0.975`.
- Expected effect: If momentum-timescale coupling is the better organizing invariant for the larger geometry, this row should beat the linear LR/batch `264x6` prescription on corrected benchmark quality at comparable wall time.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d0291fe363bc08a66ae43f4aebfa1ce01f75a221b3f3b1c6438c9aae4c5aff31`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.975, 0.95], 'min_lr': 4e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
- Adequacy knobs to dimension explicitly:
  - corrected `openml_classification_medium_v1` benchmark contract
  - fixed `tf_rd_010_dagzoo_medium_control_curated_v6` corpus
  - fixed `5000`-step one-epoch endpoint
  - high-batch `B_eff=256` endpoint with `beta1=0.975`
  - `weight_decay=0.01` stays frozen while LR and momentum move together
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Selector geometry `264x6` with momentum-timescale prescription.
  - Literature-backed selector row: test a timescale-style invariant by pairing high batch with `beta1=0.975` while keeping `beta2=0.95`.
  - This row is still selector evidence only; do not promote it into a larger-family law claim without the later Phase-2B rerun.
  - Keep the corrected `openml_classification_medium_v1` benchmark and `tf_rd_010_dagzoo_medium_control_curated_v6` corpus fixed across the selector lane.
  - This selector is downstream of cleanup issue `#283` and execution issue `#284`; historical schedulefree TF-RD-009 remains preserved context only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_endpoint_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_momentum_timescale_v1/result_card.md`
- Benchmark metrics: pending
