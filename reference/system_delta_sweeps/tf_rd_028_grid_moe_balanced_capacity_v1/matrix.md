# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_028_grid_moe_balanced_capacity_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_028_grid_moe_balanced_capacity_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_028_grid_moe_balanced_capacity_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_027_grid_stability_impl_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_028_grid_moe_balanced_capacity_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `c4634dc14e6e81833acea1194ce3273dd9e35f314faecde10eebe85f0822c791`

## Locked Surface

- Anchor run id: `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_workstation_grid_sandwich`
- Training config profile: `cls_workstation_grid_sandwich`
- Surface role: `classification_grid_moe_followup`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4182`, final Brier score `0.2551`, best ROC AUC `0.8135`, final ROC AUC `0.8133`, final training time `4073.3s`

## Anchor Comparison

Upstream reference: `Grid-Sandwich MoE balanced-capacity follow-up` from `Switch-style sparse SwiGLU experts for the promoted Grid-Sandwich core FFNs`.

| Dimension | Upstream Grid-Sandwich MoE balanced-capacity follow-up | Locked anchor | Interpretation |
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
| 1 | `delta_tf_rd_028_grid_moe_e4_top1_smoke_v1` | grid_sandwich_moe_balanced_capacity | no | ready | none | Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-1 routing. | Execute first on the Vast VM and unblock the full MoE ladder only after smoke training, benchmark artifacts, and W&B logging are clean. |
| 2 | `delta_tf_rd_028_grid_moe_e2_top1_v1` | grid_sandwich_moe_balanced_capacity | no | ready | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using two experts and top-1 routing. | Execute as the first full-budget TF-RD-028 top-1 MoE candidate after the smoke row passes. |
| 3 | `delta_tf_rd_028_grid_moe_e4_top1_v1` | grid_sandwich_moe_balanced_capacity | no | ready | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-1 routing. | Execute as the main TF-RD-028 top-1 MoE candidate after the smoke row passes. |
| 4 | `delta_tf_rd_028_grid_moe_e8_top1_v1` | grid_sandwich_moe_balanced_capacity | no | ready | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using eight experts and top-1 routing. | Execute after E=2 and E=4 top-1 rows show finite losses and usable route-health metrics. |
| 5 | `delta_tf_rd_028_grid_moe_e4_top2_v1` | grid_sandwich_moe_balanced_capacity | no | ready | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-2 routing. | Execute only after the top-1 MoE rows pass route-health inspection and quality is not explained by a routing bug. |

## Detailed Rows

### 1. `delta_tf_rd_028_grid_moe_e4_top1_smoke_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-1 routing.
- Rationale: Contextualize `delta_tf_rd_028_grid_moe_e4_top1_smoke_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_028_grid_moe_balanced_capacity_v1`.
- Hypothesis: none
- Upstream delta: Diagnostic execution gate for the TF-RD-028 Grid-Sandwich MoE capacity ladder.
- Anchor delta: Delta description pending for `delta_tf_rd_028_grid_moe_e4_top1_smoke_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should prove that MoE config resolution, prior training, differentiable auxiliary losses, route-health metrics, checkpoint export, benchmark registration, and W&B logging all work before spending a full candidate budget.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e22f10d0f5085e2c0a26cdb6ec8c748a048128884330921193921c9210493fb2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 100, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01}`
- Parameter adequacy plan:
  - Treat this row as diagnostic only; do not compare it as a quality candidate against the 5000-step anchor.
  - Full TF-RD-028 candidate execution is blocked on finite training losses, finite MoE auxiliary metrics, completed benchmark artifacts, and W&B logging from this smoke row.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - Switch-style top-1 routing with no token dropping
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - train for exactly 100 prior-dump smoke steps before benchmark plumbing
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_028_grid_moe_balanced_capacity_v1/delta_tf_rd_028_grid_moe_e4_top1_smoke_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_028_grid_moe_e2_top1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using two experts and top-1 routing.
- Rationale: Contextualize `delta_tf_rd_028_grid_moe_e2_top1_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_028_grid_moe_balanced_capacity_v1`.
- Hypothesis: none
- Upstream delta: First full-budget TF-RD-028 sparse expert count rung.
- Anchor delta: Delta description pending for `delta_tf_rd_028_grid_moe_e2_top1_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Adds a low-risk amount of sparse expert capacity while keeping active FFN compute near the dense anchor.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8315967364764e0ac3174595c61566c3cc03dbba9c348ec7346d9be19115f6dd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 2, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01}`
- Parameter adequacy plan:
  - Execute after the diagnostic smoke row passes.
  - Compare directly against the locked TF-RD-026 row 10 anchor by final_log_loss_at_matched_regime_budget and route-health telemetry.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with two sparse experts
  - Switch-style top-1 routing with no token dropping
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_028_grid_moe_balanced_capacity_v1/delta_tf_rd_028_grid_moe_e2_top1_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_028_grid_moe_e4_top1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-1 routing.
- Rationale: Contextualize `delta_tf_rd_028_grid_moe_e4_top1_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_028_grid_moe_balanced_capacity_v1`.
- Hypothesis: none
- Upstream delta: Main recommended TF-RD-028 sparse capacity row.
- Anchor delta: Delta description pending for `delta_tf_rd_028_grid_moe_e4_top1_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Adds the planned first meaningful sparse parameter increase while keeping active FFN compute near the dense anchor.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `33e88ecbe3cc98f9b6f23e9bc44e13cbf22a36e0fae9ca4d6fce6e2108dd3989`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01}`
- Parameter adequacy plan:
  - Execute after the diagnostic smoke row passes.
  - Compare directly against the locked TF-RD-026 row 10 anchor by final_log_loss_at_matched_regime_budget and route-health telemetry.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - Switch-style top-1 routing with no token dropping
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_028_grid_moe_balanced_capacity_v1/delta_tf_rd_028_grid_moe_e4_top1_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_028_grid_moe_e8_top1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using eight experts and top-1 routing.
- Rationale: Contextualize `delta_tf_rd_028_grid_moe_e8_top1_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_028_grid_moe_balanced_capacity_v1`.
- Hypothesis: none
- Upstream delta: Higher sparse expert count rung for the TF-RD-028 MoE ladder.
- Anchor delta: Delta description pending for `delta_tf_rd_028_grid_moe_e8_top1_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Adds substantially more total expert parameters while keeping active FFN compute near the dense anchor, exposing whether sparse capacity continues to scale before expert-parallel memory work.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `02ce9c5ece3b193e4a3a0adfcbcd877cffca57acb790cabc4745e3139be8626b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 8, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01}`
- Parameter adequacy plan:
  - Execute after lower expert-count top-1 rows pass basic route-health checks.
  - Compare directly against the locked TF-RD-026 row 10 anchor by final_log_loss_at_matched_regime_budget and route-health telemetry.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with eight sparse experts
  - Switch-style top-1 routing with no token dropping
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_028_grid_moe_balanced_capacity_v1/delta_tf_rd_028_grid_moe_e8_top1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_028_grid_moe_e4_top2_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts and top-2 routing.
- Rationale: Contextualize `delta_tf_rd_028_grid_moe_e4_top2_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_028_grid_moe_balanced_capacity_v1`.
- Hypothesis: none
- Upstream delta: Conditional TF-RD-028 top-k follow-up after top-1 route health is known.
- Anchor delta: Delta description pending for `delta_tf_rd_028_grid_moe_e4_top2_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Increases active expert mixing and router gradient density relative to top-1, trading extra active FFN compute for a possible quality or route-health improvement.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d5f11be2a9bcd4c836b219657dab7920cfa632f4c13520cee6ba4b484b0484e3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 2, 'grid_moe_router_init_std': 0.01}`
- Parameter adequacy plan:
  - Execute only after top-1 MoE rows have finite auxiliary metrics and no persistent route collapse.
  - Compare against the E=4 top-1 row and locked TF-RD-026 row 10 anchor by final_log_loss_at_matched_regime_budget, route-health telemetry, and the additional active-compute cost.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - Switch-style top-2 routing with no token dropping
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_028_grid_moe_balanced_capacity_v1/delta_tf_rd_028_grid_moe_e4_top2_v1/result_card.md`
- Benchmark metrics: pending
