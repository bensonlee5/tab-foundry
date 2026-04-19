# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_batch_critical_one_epoch_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `93c50ed09ab4c89cb673c50762262e5b366354859f8aa06adcf06de34e53982d`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_training_dynamics_transfer`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `8686.8s`

## Anchor Comparison

Upstream reference: `Deriving Hyperparameter Scaling Laws via Modern Optimization Theory` from `https://arxiv.org/abs/2603.15958`.

| Dimension | Upstream Deriving Hyperparameter Scaling Laws via Modern Optimization Theory | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Benchmark this imported low-batch baseline into the strict shared-anchor LMO transfer study without retraining. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Benchmark this imported low-batch baseline into the strict shared-anchor LMO transfer study without retraining. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Benchmark this imported low-batch baseline into the strict shared-anchor LMO transfer study without retraining. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface. | Benchmark this carried high-batch baseline and compare it against the strict shared-anchor LMO transfer regimes on corrected benchmark log loss. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface. | Benchmark this carried high-batch baseline and compare it against the strict shared-anchor LMO transfer regimes on corrected benchmark log loss. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1` | classification_scaling_law | no | ready | none | Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface. | Benchmark this carried high-batch baseline and compare it against the strict shared-anchor LMO transfer regimes on corrected benchmark log loss. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Execute the strict shared-anchor LMO transfer row and compare it against the carried high-batch baseline at the matched budget. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget. | Execute the strict shared-anchor LMO transfer row and compare it against the carried high-batch baseline at the matched budget. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Execute the strict shared-anchor LMO transfer row and compare it against the carried high-batch baseline at the matched budget. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1` | classification_scaling_law | no | ready | none | Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget. | Execute the strict shared-anchor LMO transfer row and compare it against the carried high-batch baseline at the matched budget. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Import the corrected 144x4 low-batch Muon baseline from the completed NS sweep without retraining.
- Hypothesis: The carried low-batch 144x4 baseline anchors the faithful transfer study and should remain a valid corrected benchmark reference at the matching budget rung.
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Reuse the completed NS low-batch train artifact and benchmark it into the transfer study without retraining.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6da2f3eeb7d91544df4285a2b2060bda44950c8e0b2a4f2f82552abb2b8f6567`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train`
- Reuse training surface fingerprint: `4fc459d869ad4846c6cfae41b5c03ff051b4d47a924f1126afbec3d8ee4a7c5a`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_lowbatch', 'formula_label': 'carried low-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'carry_lowbatch', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 40000, 'realized_effective_budget': 40000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Imported baseline provenance: `{'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 9, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}`
- Parameter adequacy plan:
  - Do not retrain the carried low-batch baseline; benchmark only from the preserved NS train artifact.
  - Keep the corrected medium anchor benchmark fixed so the low-batch baseline remains directly comparable to the transfer regimes.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Import low-batch baseline provenance from tf_rd_009_muon_ns_one_epoch_medium_v1 order 09 (sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1).
  - This baseline remains corrected-medium multiclass only and should not drift to any binary benchmark surface.
  - This imported low-batch row is also the shared T0 anchor for the strict LMO transfer regimes.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics:
  - Best log loss: `0.6461` (step 625)
  - Final log loss: `0.6461`
  - Final Brier score: `0.4026`
  - Final ROC AUC: `0.6658`
  - Drift (final − best): `0.0000`
  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows.
  - Final BPC (legacy feature-cell diagnostic): `2.1886`
  - Final BPF (legacy feature-cell diagnostic): `2.1886`
  - max_grad_norm: `3.595`

### 2. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Import the corrected 144x4 low-batch Muon baseline from the completed NS sweep without retraining.
- Hypothesis: The carried low-batch 144x4 baseline anchors the faithful transfer study and should remain a valid corrected benchmark reference at the matching budget rung.
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Reuse the completed NS low-batch train artifact and benchmark it into the transfer study without retraining.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c4930ff2cd45ed5cbe10bc853eb181e43201c6a89c7ee86ae7b608df225e9ccd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train`
- Reuse training surface fingerprint: `a4c7d3c6553c38115717e1178045a9b795c21c50d5104cbea7c9e02b3125eb6f`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_lowbatch', 'formula_label': 'carried low-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'carry_lowbatch', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 160000, 'realized_effective_budget': 160000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Imported baseline provenance: `{'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 11, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}`
- Parameter adequacy plan:
  - Do not retrain the carried low-batch baseline; benchmark only from the preserved NS train artifact.
  - Keep the corrected medium anchor benchmark fixed so the low-batch baseline remains directly comparable to the transfer regimes.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Import low-batch baseline provenance from tf_rd_009_muon_ns_one_epoch_medium_v1 order 11 (sd_tf_rd_009_muon_ns_one_epoch_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1).
  - This baseline remains corrected-medium multiclass only and should not drift to any binary benchmark surface.
  - This imported low-batch row is also the shared T0 anchor for the strict LMO transfer regimes.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics:
  - Best log loss: `0.5079` (step 2500)
  - Final log loss: `0.5079`
  - Final Brier score: `0.3095`
  - Final ROC AUC: `0.7719`
  - Drift (final − best): `0.0000`
  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows.
  - Final BPC (legacy feature-cell diagnostic): `2.3965`
  - Final BPF (legacy feature-cell diagnostic): `2.3965`
  - max_grad_norm: `4.271`

### 3. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Import the corrected 144x4 low-batch Muon baseline from the completed NS sweep without retraining.
- Hypothesis: The carried low-batch 144x4 baseline anchors the faithful transfer study and should remain a valid corrected benchmark reference at the matching budget rung.
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Reuse the completed NS low-batch train artifact and benchmark it into the transfer study without retraining.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ad68d3446dbc6cbc160bd2cb424a3e7849cb1f44f713092cfbc276b2f844799d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train`
- Reuse training surface fingerprint: `7ced1d99f22a54bfcb426e1cfb57b5f40bfc925b674839077baadc1b8d838e40`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_lowbatch', 'formula_label': 'carried low-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'carry_lowbatch', 'target_effective_batch': 64, 'realized_effective_batch': 64, 'target_effective_budget': 320000, 'realized_effective_budget': 320000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Imported baseline provenance: `{'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 12, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}`
- Parameter adequacy plan:
  - Do not retrain the carried low-batch baseline; benchmark only from the preserved NS train artifact.
  - Keep the corrected medium anchor benchmark fixed so the low-batch baseline remains directly comparable to the transfer regimes.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Import low-batch baseline provenance from tf_rd_009_muon_ns_one_epoch_medium_v1 order 12 (sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1).
  - This baseline remains corrected-medium multiclass only and should not drift to any binary benchmark surface.
  - This imported low-batch row is also the shared T0 anchor for the strict LMO transfer regimes.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics:
  - Best log loss: `0.4914` (step 5000)
  - Final log loss: `0.4914`
  - Final Brier score: `0.2989`
  - Final ROC AUC: `0.7866`
  - Drift (final − best): `0.0000`
  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows.
  - Final BPC (legacy feature-cell diagnostic): `3.0328`
  - Final BPF (legacy feature-cell diagnostic): `3.0327`
  - max_grad_norm: `2.936`

### 4. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface.
- Rationale: Carry the empirical 144x4 Muon high-batch baseline into the faithful transfer study at the requested budget rung.
- Hypothesis: A pure carry-forward high-batch baseline distinguishes the value of budget-faithful transfer from a simple fixed high-batch contract.
- Upstream delta: This is the direct test of whether the corrected Phase-2 batch-side gain also improves the `144x4` NS winner when geometry is held fixed.
- Anchor delta: Keep the carried Muon high-batch optimizer surface and adjust only the step budget needed for the requested rung.
- Expected effect: If batch-side movement generalizes across the compact frontier, this row should improve `144x4` materially enough to challenge the carried low-batch point on the corrected quality/time frontier.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `07d38633c5b9e05798095e8fd8a85279f9eb442a88381f66864d11a25d4ab0e9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 156}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 156}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 156, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_highbatch', 'formula_label': 'carried high-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T0', 'candidate_label': 'carry_highbatch', 'target_effective_batch': 256, 'realized_effective_batch': 256, 'target_effective_budget': 40000, 'realized_effective_budget': 39936, 'budget_drift': -0.0016000000000000458, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use the carried high-batch Muon contract as a baseline against the paper-derived transfer regimes.
  - Do not let runtime or wall time override corrected benchmark log loss when deciding whether this baseline still merits a later Phase-2B rerun.
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
  - The earlier heuristic selector remains preserved context only; this row is the budget-faithful carried high-batch baseline for the transfer study.
  - This row remains a carried high-batch baseline; it is not the source of the shared anchor transfer law.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface.
- Rationale: Carry the empirical 144x4 Muon high-batch baseline into the faithful transfer study at the requested budget rung.
- Hypothesis: A pure carry-forward high-batch baseline distinguishes the value of budget-faithful transfer from a simple fixed high-batch contract.
- Upstream delta: This is the direct test of whether the corrected Phase-2 batch-side gain also improves the `144x4` NS winner when geometry is held fixed.
- Anchor delta: Keep the carried Muon high-batch optimizer surface and adjust only the step budget needed for the requested rung.
- Expected effect: If batch-side movement generalizes across the compact frontier, this row should improve `144x4` materially enough to challenge the carried low-batch point on the corrected quality/time frontier.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8f07b12c8809233dd626a6e00b4fea37644d2bd014798857d85d2443dac4440f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 625}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 625, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_highbatch', 'formula_label': 'carried high-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'carry_highbatch', 'target_effective_batch': 256, 'realized_effective_batch': 256, 'target_effective_budget': 160000, 'realized_effective_budget': 160000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use the carried high-batch Muon contract as a baseline against the paper-derived transfer regimes.
  - Do not let runtime or wall time override corrected benchmark log loss when deciding whether this baseline still merits a later Phase-2B rerun.
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
  - The earlier heuristic selector remains preserved context only; this row is the budget-faithful carried high-batch baseline for the transfer study.
  - This row remains a carried high-batch baseline; it is not the source of the shared anchor transfer law.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Re-run `144x4` at the fixed 5000-step endpoint with the empirical high-batch `B_eff=256` carry-forward Muon contract and unchanged LR/beta surface.
- Rationale: Carry the empirical 144x4 Muon high-batch baseline into the faithful transfer study at the requested budget rung.
- Hypothesis: A pure carry-forward high-batch baseline distinguishes the value of budget-faithful transfer from a simple fixed high-batch contract.
- Upstream delta: This is the direct test of whether the corrected Phase-2 batch-side gain also improves the `144x4` NS winner when geometry is held fixed.
- Anchor delta: Keep the carried Muon high-batch optimizer surface and adjust only the step budget needed for the requested rung.
- Expected effect: If batch-side movement generalizes across the compact frontier, this row should improve `144x4` materially enough to challenge the carried low-batch point on the corrected quality/time frontier.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `05faa2cd610b3208f72916bef78cb0beb436f3a289b4d483370cad05e09b4988`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.95}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 16, 'max_steps': 1250}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 1250, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Transfer context: `{'phase': 'baseline', 'regime_label': 'carry_highbatch', 'formula_label': 'carried high-batch baseline', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'carry_highbatch', 'target_effective_batch': 256, 'realized_effective_batch': 256, 'target_effective_budget': 320000, 'realized_effective_budget': 320000, 'budget_drift': 0.0, 'batch_drift': 0.0}`
- Parameter adequacy plan:
  - Use the carried high-batch Muon contract as a baseline against the paper-derived transfer regimes.
  - Do not let runtime or wall time override corrected benchmark log loss when deciding whether this baseline still merits a later Phase-2B rerun.
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
  - The earlier heuristic selector remains preserved context only; this row is the budget-faithful carried high-batch baseline for the transfer study.
  - This row remains a carried high-batch baseline; it is not the source of the shared anchor transfer law.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Apply faithful paper-derived transfer regime B at 144x4 on the T1 rung.
- Hypothesis: Regime B should extrapolate from its winning T0 anchor to T1 without local retuning.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Lock the winning T0 anchor for regime B and apply pure paper-derived transfer to T1.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the strict shared T0 anchor transfer should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `52a7a31f13b3d6c11dd9239ded32f13a215da2b8f44d9bb3efec3002ba0d7c12`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 3.5355339059327384e-07, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 2500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.0003535533905932738, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Dynamic training overrides: `{'transfer_schedule': {'kind': 'shared_anchor_transfer', 'anchor_order': 1, 'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_label': 'carry_lowbatch_shared_anchor_t0', 'regime_label': 'B', 'base_effective_budget': 40000, 'target_effective_budget': 160000, 'fixed_effective_batch': 64, 'min_lr_ratio': 0.001, 'max_budget_drift': 0.02, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor'}}`
- Transfer context: `{'phase': 'validation', 'regime_label': 'B', 'formula_label': 'paper regime B shared-anchor transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'shared_anchor_regime_b', 'target_effective_budget': 160000}`
- Transfer resolution: `{'phase': 'validation', 'regime_label': 'B', 'formula_label': 'Theorem 2 fixed-batch transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'shared_anchor_regime_b', 'target_effective_budget': 160000, 'base_effective_budget': 40000, 'realized_effective_budget': 160000, 'base_effective_batch': 64, 'target_effective_batch': 64.0, 'realized_effective_batch': 64, 'base_lr_max': 0.001, 'target_lr_max': 0.0003535533905932738, 'base_momentum': 0.95, 'target_momentum': 0.975, 'base_alpha': 0.050000000000000044, 'target_alpha': 0.025000000000000022, 'grad_accum_steps': 4, 'max_steps': 2500, 'min_lr': 3.5355339059327384e-07, 'budget_drift': 0.0, 'batch_drift': 0.0, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor', 'shared_anchor_provenance': {'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_order': 1, 'anchor_delta_id': 'delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1', 'anchor_run_dir': 'outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train', 'anchor_imported_baseline_provenance': {'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 9, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}, 'anchor_candidate_label': 'carry_lowbatch_shared_anchor_t0'}}`
- Parameter adequacy plan:
  - Use the carried low-batch 144x4 Muon T0 artifact as the single shared anchor for LMO transfer.
  - Apply the paper-derived law directly at the requested budget without any regime-specific T0 search or local retuning.
  - Choose the kept regime by corrected benchmark log loss at T2, using T1 only as a tie-break and stability check.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Strict shared-anchor LMO transfer under issue #284; derive this row from the carried low-batch T0 anchor with no empirical T0 search.
  - Use corrected openml_classification_medium_v1 only and keep the T2-first, T1 tie-break regime winner rule.
  - Resolved transfer training overrides `transfer_schedule` from shared anchor row `1` in `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime B, which keeps effective batch fixed while transferring Muon momentum and lr with budget.
- Rationale: Apply faithful paper-derived transfer regime B at 144x4 on the T2 rung.
- Hypothesis: Regime B should extrapolate from its winning T0 anchor to T2 without local retuning.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Lock the winning T0 anchor for regime B and apply pure paper-derived transfer to T2.
- Expected effect: If the paper fixed-batch transfer law is the better inductive bias, the strict shared T0 anchor transfer should extrapolate cleanly to T1/T2 without local retuning.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `90f46b16e7a68ad05bac9b9476347cadc60d270c4cbc8bce67b7e1284939802a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2.1022410381342863e-07, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9823223304703363}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 4, 'max_steps': 5000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.00021022410381342862, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Dynamic training overrides: `{'transfer_schedule': {'kind': 'shared_anchor_transfer', 'anchor_order': 1, 'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_label': 'carry_lowbatch_shared_anchor_t0', 'regime_label': 'B', 'base_effective_budget': 40000, 'target_effective_budget': 320000, 'fixed_effective_batch': 64, 'min_lr_ratio': 0.001, 'max_budget_drift': 0.02, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor'}}`
- Transfer context: `{'phase': 'validation', 'regime_label': 'B', 'formula_label': 'paper regime B shared-anchor transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'shared_anchor_regime_b', 'target_effective_budget': 320000}`
- Transfer resolution: `{'phase': 'validation', 'regime_label': 'B', 'formula_label': 'Theorem 2 fixed-batch transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'shared_anchor_regime_b', 'target_effective_budget': 320000, 'base_effective_budget': 40000, 'realized_effective_budget': 320000, 'base_effective_batch': 64, 'target_effective_batch': 64.0, 'realized_effective_batch': 64, 'base_lr_max': 0.001, 'target_lr_max': 0.00021022410381342862, 'base_momentum': 0.95, 'target_momentum': 0.9823223304703363, 'base_alpha': 0.050000000000000044, 'target_alpha': 0.017677669529663705, 'grad_accum_steps': 4, 'max_steps': 5000, 'min_lr': 2.1022410381342863e-07, 'budget_drift': 0.0, 'batch_drift': 0.0, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor', 'shared_anchor_provenance': {'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_order': 1, 'anchor_delta_id': 'delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1', 'anchor_run_dir': 'outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train', 'anchor_imported_baseline_provenance': {'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 9, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}, 'anchor_candidate_label': 'carry_lowbatch_shared_anchor_t0'}}`
- Parameter adequacy plan:
  - Use the carried low-batch 144x4 Muon T0 artifact as the single shared anchor for LMO transfer.
  - Apply the paper-derived law directly at the requested budget without any regime-specific T0 search or local retuning.
  - Choose the kept regime by corrected benchmark log loss at T2, using T1 only as a tie-break and stability check.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Strict shared-anchor LMO transfer under issue #284; derive this row from the carried low-batch T0 anchor with no empirical T0 search.
  - Use corrected openml_classification_medium_v1 only and keep the T2-first, T1 tie-break regime winner rule.
  - Resolved transfer training overrides `transfer_schedule` from shared anchor row `1` in `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Apply faithful paper-derived transfer regime D at 144x4 on the T1 rung.
- Hypothesis: Regime D should extrapolate from its winning T0 anchor to T1 without local retuning.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Lock the winning T0 anchor for regime D and apply pure paper-derived transfer to T1.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the strict shared T0 anchor transfer should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `baefd0b2c2c956be40d031eda08bf5f16b7aba46eb935b0b929195aee58be3c7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 5, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2000}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 4.4544935907016965e-07, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.9685019737526281}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 5, 'max_steps': 2000}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2000, 'lr_max': 0.00044544935907016964, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Dynamic training overrides: `{'transfer_schedule': {'kind': 'shared_anchor_transfer', 'anchor_order': 1, 'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_label': 'carry_lowbatch_shared_anchor_t0', 'regime_label': 'D', 'base_effective_budget': 40000, 'target_effective_budget': 160000, 'fixed_effective_batch': None, 'min_lr_ratio': 0.001, 'max_budget_drift': 0.02, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor'}}`
- Transfer context: `{'phase': 'validation', 'regime_label': 'D', 'formula_label': 'paper regime D shared-anchor transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'shared_anchor_regime_d', 'target_effective_budget': 160000}`
- Transfer resolution: `{'phase': 'validation', 'regime_label': 'D', 'formula_label': 'Theorem 3 joint-transfer proxy', 'base_budget_label': 'T0', 'target_budget_label': 'T1', 'candidate_label': 'shared_anchor_regime_d', 'target_effective_budget': 160000, 'base_effective_budget': 40000, 'realized_effective_budget': 160000, 'base_effective_batch': 64, 'target_effective_batch': 80.63494719327188, 'realized_effective_batch': 80, 'base_lr_max': 0.001, 'target_lr_max': 0.00044544935907016964, 'base_momentum': 0.95, 'target_momentum': 0.9685019737526281, 'base_alpha': 0.050000000000000044, 'target_alpha': 0.03149802624737186, 'grad_accum_steps': 5, 'max_steps': 2000, 'min_lr': 4.4544935907016965e-07, 'budget_drift': 0.0, 'batch_drift': -0.007874342519875399, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor', 'shared_anchor_provenance': {'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_order': 1, 'anchor_delta_id': 'delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1', 'anchor_run_dir': 'outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train', 'anchor_imported_baseline_provenance': {'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 9, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}, 'anchor_candidate_label': 'carry_lowbatch_shared_anchor_t0'}}`
- Parameter adequacy plan:
  - Use the carried low-batch 144x4 Muon T0 artifact as the single shared anchor for LMO transfer.
  - Apply the paper-derived law directly at the requested budget without any regime-specific T0 search or local retuning.
  - Choose the kept regime by corrected benchmark log loss at T2, using T1 only as a tie-break and stability check.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Strict shared-anchor LMO transfer under issue #284; derive this row from the carried low-batch T0 anchor with no empirical T0 search.
  - Use corrected openml_classification_medium_v1 only and keep the T2-first, T1 tie-break regime winner rule.
  - Resolved transfer training overrides `transfer_schedule` from shared anchor row `1` in `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Hold 144x4 fixed and instantiate paper Regime D, which jointly transfers effective batch, Muon momentum, and lr with budget.
- Rationale: Apply faithful paper-derived transfer regime D at 144x4 on the T2 rung.
- Hypothesis: Regime D should extrapolate from its winning T0 anchor to T2 without local retuning.
- Upstream delta: Faithful TF-RD-009 Muon transfer study derived from https://arxiv.org/abs/2603.15958.
- Anchor delta: Lock the winning T0 anchor for regime D and apply pure paper-derived transfer to T2.
- Expected effect: If the joint batch+momentum+lr transfer law is the better inductive bias, the strict shared T0 anchor transfer should beat both carried baselines on corrected benchmark log loss.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `91b360b6bd1ad5ef9e56441aa71787fb618e33a5b3cf59c7dfcb3e31091f1c7e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 6, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 3333}`
- Training overrides: `{'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 2.9730177875068024e-07, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True, 'momentum': 0.975}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'non_blocking_device_transfer': True, 'compile_model': True, 'compile_backend': 'eager', 'compile_dynamic': True, 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'signature_family_run_length': 4, 'grad_accum_steps': 6, 'max_steps': 3333}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 3333, 'lr_max': 0.00029730177875068024, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Dynamic training overrides: `{'transfer_schedule': {'kind': 'shared_anchor_transfer', 'anchor_order': 1, 'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_label': 'carry_lowbatch_shared_anchor_t0', 'regime_label': 'D', 'base_effective_budget': 40000, 'target_effective_budget': 320000, 'fixed_effective_batch': None, 'min_lr_ratio': 0.001, 'max_budget_drift': 0.02, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor'}}`
- Transfer context: `{'phase': 'validation', 'regime_label': 'D', 'formula_label': 'paper regime D shared-anchor transfer', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'shared_anchor_regime_d', 'target_effective_budget': 320000}`
- Transfer resolution: `{'phase': 'validation', 'regime_label': 'D', 'formula_label': 'Theorem 3 joint-transfer proxy', 'base_budget_label': 'T0', 'target_budget_label': 'T2', 'candidate_label': 'shared_anchor_regime_d', 'target_effective_budget': 320000, 'base_effective_budget': 40000, 'realized_effective_budget': 319968, 'base_effective_batch': 64, 'target_effective_batch': 90.50966799187808, 'realized_effective_batch': 96, 'base_lr_max': 0.001, 'target_lr_max': 0.00029730177875068024, 'base_momentum': 0.95, 'target_momentum': 0.975, 'base_alpha': 0.050000000000000044, 'target_alpha': 0.025000000000000022, 'grad_accum_steps': 6, 'max_steps': 3333, 'min_lr': 2.9730177875068024e-07, 'budget_drift': -9.999999999998899e-05, 'batch_drift': 0.060660171779821415, 'resolved_from_order': 1, 'resolved_from_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'resolved_candidate_label': 'carry_lowbatch_shared_anchor_t0', 'resolution_reason': 'shared_anchor', 'shared_anchor_provenance': {'anchor_sweep_id': 'tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1', 'anchor_order': 1, 'anchor_delta_id': 'delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1', 'anchor_run_dir': 'outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1/train', 'anchor_imported_baseline_provenance': {'source_sweep_id': 'tf_rd_009_muon_ns_one_epoch_medium_v1', 'source_order': 9, 'source_run_id': 'sd_tf_rd_009_muon_ns_one_epoch_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1', 'source_kind': 'carry_lowbatch'}, 'anchor_candidate_label': 'carry_lowbatch_shared_anchor_t0'}}`
- Parameter adequacy plan:
  - Use the carried low-batch 144x4 Muon T0 artifact as the single shared anchor for LMO transfer.
  - Apply the paper-derived law directly at the requested budget without any regime-specific T0 search or local retuning.
  - Choose the kept regime by corrected benchmark log loss at T2, using T1 only as a tie-break and stability check.
- Adequacy knobs to dimension explicitly:
  - paper-derived budget transfer law
  - corrected openml_classification_medium_v1 anchor benchmark
  - 144x4 geometry held fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Strict shared-anchor LMO transfer under issue #284; derive this row from the carried low-batch T0 anchor with no empirical T0 search.
  - Use corrected openml_classification_medium_v1 only and keep the T2-first, T1 tie-break regime winner rule.
  - Resolved transfer training overrides `transfer_schedule` from shared anchor row `1` in `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_d_v1/result_card.md`
- Benchmark metrics: pending
