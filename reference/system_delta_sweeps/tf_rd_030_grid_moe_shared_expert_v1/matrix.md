# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_030_grid_moe_shared_expert_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_030_grid_moe_shared_expert_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_030_grid_moe_shared_expert_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_029_grid_moe_top2_normalized_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_030_grid_moe_shared_expert_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `ae185f6fe8bcc29b01fc00f2d2f4345f668906c976b8054ec332c394050fef52`

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

Upstream reference: `Grid-Sandwich MoE shared-expert follow-up` from `Token-centric sparse SwiGLU experts with an optional always-on shared SwiGLU expert, router-temperature smoothing, and load-balancing schedule probes for the promoted Grid-Sandwich core FFNs`.

| Dimension | Upstream Grid-Sandwich MoE shared-expert follow-up | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Staged feature encoder `unknown` from the benchmark registry surface. | Feature encoder changes alter the per-cell representation and should be interpreted explicitly. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Target conditioner `unknown` from the staged surface. | Target-conditioning changes should be interpreted separately from encoder or context changes. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Cell transformer block `unknown` from the staged surface. | Cell-block changes affect the core table computation and should be isolated carefully. |
| tokenizer | One scalar token per feature. | Tokenizer `unknown` from the staged surface. | Tokenizer changes alter the token sequence presented to the transformer stack. |
| column encoder | None on the upstream direct path. | Column encoder `unknown` from the staged surface. | Column-encoder changes should be read separately from row pooling or context changes. |
| row readout | Target-column readout from the final cell tensor. | Row pool `unknown` from the staged surface. | Row-pool changes alter the readout contract and require their own interpretation. |
| context encoder | None on the upstream direct path. | Context encoder `unknown` from the staged surface. | Context-encoder changes alter how training rows condition test rows. |
| prediction head | Direct binary logits head. | Prediction head `unknown` from the staged surface. | Head changes alter the task contract and output semantics. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest `data/manifests/bench/openml_classification_medium_v1/manifest.parquet` sourced from `openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1` | grid_sandwich_moe_shared_expert | no | completed | none | Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with four sparse grid-core SwiGLU experts, top-1 routing, and one always-on shared SwiGLU expert. | Execute first on the Vast VM and unblock shared-expert full rows only after smoke training, benchmark artifacts, and W&B logging are clean. |
| 2 | `delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1` | grid_sandwich_moe_shared_expert | no | completed | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with four sparse grid-core SwiGLU experts, top-1 routing, and one always-on shared SwiGLU expert. | Execute after the shared-expert smoke row passes. |
| 3 | `delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1` | grid_sandwich_moe_shared_expert | no | completed | none | Test four sparse grid-core SwiGLU experts with normalized top-2 routing plus one always-on shared SwiGLU expert. | Execute after the shared top-1 candidate if smoke health remains clean. |
| 4 | `delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1` | grid_sandwich_moe_aux_schedule | no | completed | none | Retest normalized top-2 MoE with a decayed load-balancing coefficient so routing is stabilized early but less constrained late. | Execute after TF-RD-029 normalized top-2 completes and shared-expert smoke is clean. |
| 5 | `delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1` | grid_sandwich_moe_router_temperature | no | screened | none | Test shared-expert top-1 MoE with router-temperature smoothing by setting the router softmax temperature to 1.25. | Do not execute in TF-RD-030 closeout; completed rows showed healthy route balance, so router-temperature smoothing does not target the observed quality gap. |

## Detailed Rows

### 1. `delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with four sparse grid-core SwiGLU experts, top-1 routing, and one always-on shared SwiGLU expert.
- Rationale: Contextualize `delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_030_grid_moe_shared_expert_v1`.
- Hypothesis: none
- Upstream delta: Diagnostic execution gate for the TF-RD-030 shared-expert MoE follow-up.
- Anchor delta: Delta description pending for `delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should prove that the shared-expert MoE path resolves, trains with finite MoE auxiliary metrics, exports checkpoints, registers benchmark artifacts, and logs to W&B online before full-row execution.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cb0209d1a2eca537ddad9d23f405cb2904e5e7c1c4eb194c6a31d57f0d14fb34`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 100, 'signature_family_optimizer_step_block_length': 4}`
- Stage-local stability: row (grad `0.1170`)
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01, 'grid_moe_shared_expert': True, 'grid_moe_shared_expert_scale': 1.0}`
- Parameter adequacy plan:
  - Treat this row as diagnostic only; do not compare it as a quality candidate against the 5000-step anchor.
  - Full TF-RD-030 candidate execution is blocked on finite training losses, finite MoE auxiliary metrics, completed benchmark artifacts, and W&B logging from this smoke row.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs add an always-on shared expert plus four routed sparse experts
  - top-1 routed expert contribution remains router-probability weighted
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for exactly 100 prior-dump smoke steps before benchmark plumbing
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_030_grid_moe_shared_expert_v1_01_delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_030_grid_moe_shared_expert_v1/delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1/result_card.md`
- Registered run: `sd_tf_rd_030_grid_moe_shared_expert_v1_01_delta_tf_rd_030_grid_moe_e4_top1_shared_smoke_v1_v1` with final log loss `0.7138`, delta final log loss `+0.2956`, final Brier score `0.4441`, delta final brier score `+0.1890`, final ROC AUC `0.6394`, delta final roc auc `-0.1739`, best ROC AUC `0.6394`, delta final training time `-3718.2s`

### 2. `delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with four sparse grid-core SwiGLU experts, top-1 routing, and one always-on shared SwiGLU expert.
- Rationale: Contextualize `delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_030_grid_moe_shared_expert_v1`.
- Hypothesis: none
- Upstream delta: Main TF-RD-030 hypothesis that MoE should add a stable shared FFN path instead of forcing all FFN capacity through routing.
- Anchor delta: Delta description pending for `delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Tests whether a shared dense path plus one routed expert preserves anchor-like optimization while adding specialist capacity at lower active routed compute than top-2.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c180ae671c183dc4758ec019293ef0aa00d0538ef621c7d4e27a3297915ca8ab`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Stage-local stability: row (grad `0.0471`)
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01, 'grid_moe_shared_expert': True, 'grid_moe_shared_expert_scale': 1.0}`
- Parameter adequacy plan:
  - Execute immediately after the shared-expert smoke row passes.
  - Compare against TF-RD-026 row 10, TF-RD-028 top-1 MoE rows, and TF-RD-029 normalized top-2 by final log loss, route-health telemetry, and wall time.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs add an always-on shared expert plus four routed sparse experts
  - top-1 routed expert contribution remains router-probability weighted
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_030_grid_moe_shared_expert_v1_02_delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_030_grid_moe_shared_expert_v1/delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1/result_card.md`
- Registered run: `sd_tf_rd_030_grid_moe_shared_expert_v1_02_delta_tf_rd_030_grid_moe_e4_top1_shared_microbatch8_v1_v1` with final log loss `0.4210`, delta final log loss `+0.0028`, final Brier score `0.2564`, delta final brier score `+0.0012`, final ROC AUC `0.8121`, delta final roc auc `-0.0011`, best ROC AUC `0.8121`, delta final training time `+6176.3s`

### 3. `delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test four sparse grid-core SwiGLU experts with normalized top-2 routing plus one always-on shared SwiGLU expert.
- Rationale: Contextualize `delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_030_grid_moe_shared_expert_v1`.
- Hypothesis: none
- Upstream delta: Tests whether the TF-RD-029 normalized top-2 path benefits from a stable always-on shared expert.
- Anchor delta: Delta description pending for `delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should preserve token-centric two-expert routing while the shared path carries dense anchor-like capacity and reduces pressure on the router to learn the whole FFN transformation.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2bedf5195538ff11a72beea46e73d0aa40e4541c8a53150f02821d83d0bc39cf`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Stage-local stability: row (grad `0.0466`)
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 2, 'grid_moe_router_init_std': 0.01, 'grid_moe_normalize_top_k': True, 'grid_moe_shared_expert': True, 'grid_moe_shared_expert_scale': 1.0}`
- Parameter adequacy plan:
  - Execute after the shared top-1 row unless smoke or resource health argues otherwise.
  - Compare against TF-RD-029 normalized top-2 and TF-RD-030 shared top-1.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs add an always-on shared expert plus four routed sparse experts
  - selected top-2 router probabilities renormalized per token
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_030_grid_moe_shared_expert_v1_03_delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_030_grid_moe_shared_expert_v1/delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1/result_card.md`
- Registered run: `sd_tf_rd_030_grid_moe_shared_expert_v1_03_delta_tf_rd_030_grid_moe_e4_top2_shared_norm_microbatch8_v1_v1` with final log loss `0.4223`, delta final log loss `+0.0042`, final Brier score `0.2575`, delta final brier score `+0.0024`, final ROC AUC `0.8109`, delta final roc auc `-0.0023`, best ROC AUC `0.8109`, delta final training time `+7083.5s`

### 4. `delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Retest normalized top-2 MoE with a decayed load-balancing coefficient so routing is stabilized early but less constrained late.
- Rationale: Contextualize `delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_030_grid_moe_shared_expert_v1`.
- Hypothesis: none
- Upstream delta: Tests whether the constant TF-RD-029 load-balancing loss is over-regularizing once route health is already acceptable.
- Anchor delta: Delta description pending for `delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should retain finite, non-collapsed route health while reducing late-training pressure toward perfectly uniform expert usage.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a09397dffbf680e56f3d7cfc2b035ed29b4014de200387e052291b499c414f24`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Stage-local stability: row (grad `0.0468`)
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 2, 'grid_moe_router_init_std': 0.01, 'grid_moe_normalize_top_k': True}`
- Parameter adequacy plan:
  - Execute after TF-RD-029 normalized top-2 completes and route-health metrics remain finite.
  - Compare against TF-RD-029 normalized top-2 to isolate the auxiliary-loss schedule effect.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - selected top-2 router probabilities renormalized per token
  - load-balance coefficient decays from 1e-2 to 1e-3 with warmup/decay
  - router z-loss coefficient 1e-4 remains constant
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_030_grid_moe_shared_expert_v1_04_delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_030_grid_moe_shared_expert_v1/delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1/result_card.md`
- Registered run: `sd_tf_rd_030_grid_moe_shared_expert_v1_04_delta_tf_rd_030_grid_moe_e4_top2_norm_lb_decay_microbatch8_v1_v1` with final log loss `0.4229`, delta final log loss `+0.0047`, final Brier score `0.2579`, delta final brier score `+0.0027`, final ROC AUC `0.8117`, delta final roc auc `-0.0016`, best ROC AUC `0.8117`, delta final training time `+5988.2s`

### 5. `delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1`

- Dimension family: `model`
- Status: `screened`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test shared-expert top-1 MoE with router-temperature smoothing by setting the router softmax temperature to 1.25.
- Rationale: Contextualize `delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_030_grid_moe_shared_expert_v1`.
- Hypothesis: none
- Upstream delta: Tests whether smoother early routing helps shared-expert top-1 MoE avoid brittle expert assignment without paying top-2 compute.
- Anchor delta: Delta description pending for `delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should keep route entropy higher and expert fractions more even than the default-temperature shared top-1 candidate while preserving lower routed compute than top-2.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2649b46f669aa37d8d616d83e153f99168ff14921297ca7cba9673413c45a411`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 1, 'grid_moe_router_init_std': 0.01, 'grid_moe_shared_expert': True, 'grid_moe_shared_expert_scale': 1.0, 'grid_moe_router_temperature': 1.25}`
- Parameter adequacy plan:
  - Execute after the default-temperature shared top-1 row if route health or loss oscillation suggests smoothing is worth isolating.
  - Compare against default-temperature shared top-1 by final log loss, route entropy, expert fraction spread, and wall time.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs add an always-on shared expert plus four routed sparse experts
  - top-1 routing uses router softmax temperature 1.25
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Screened out after rows 02 through 04 completed with finite, non-collapsed route health but all remained worse than the TF-RD-026 row 10 dense anchor.
  - Row 02 shared top-1 was the best TF-RD-030 MoE quality result at final log loss 0.4210, still +0.0028 worse than the locked anchor.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_030_grid_moe_shared_expert_v1/delta_tf_rd_030_grid_moe_e4_top1_shared_temp125_microbatch8_v1/result_card.md`
- Benchmark metrics: pending
