# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_029_grid_moe_top2_normalized_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_029_grid_moe_top2_normalized_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_029_grid_moe_top2_normalized_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_028_grid_moe_balanced_capacity_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_029_grid_moe_top2_normalized_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `62967b0d53c8b07bf7757bb9de5c96ab361b29c8a9df242bdac495149b5b834e`

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

Upstream reference: `Grid-Sandwich MoE normalized top-2 follow-up` from `Token-centric top-2 sparse SwiGLU experts with per-token selected-probability normalization for the promoted Grid-Sandwich core FFNs`.

| Dimension | Upstream Grid-Sandwich MoE normalized top-2 follow-up | Locked anchor | Interpretation |
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
| 1 | `delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1` | grid_sandwich_moe_top2_normalized | no | ready | none | Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts, top-2 routing, and per-token top-k probability normalization. | Execute first on the Vast VM and unblock the full normalized top-2 MoE row only after smoke training, benchmark artifacts, and W&B logging are clean. |
| 2 | `delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1` | grid_sandwich_moe_top2_normalized | no | ready | none | Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts, normalized top-2 routing, and a smaller task microbatch. | Execute after the normalized top-2 smoke row passes. |

## Detailed Rows

### 1. `delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Run a short benchmark-capable smoke of the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts, top-2 routing, and per-token top-k probability normalization.
- Rationale: Contextualize `delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_029_grid_moe_top2_normalized_v1`.
- Hypothesis: none
- Upstream delta: Diagnostic execution gate for the TF-RD-029 top-2-first MoE follow-up.
- Anchor delta: Delta description pending for `delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Should prove that normalized top-2 routing resolves config, trains with finite MoE auxiliary metrics, exports checkpoints, registers benchmarks, and logs to W&B before spending a full candidate budget.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `44c5bd0ba112fe745b55600a5f6000eedb8458fb780955427f00d62852c646f2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 100, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 2, 'grid_moe_router_init_std': 0.01, 'grid_moe_normalize_top_k': True}`
- Parameter adequacy plan:
  - Treat this row as diagnostic only; do not compare it as a quality candidate against the 5000-step anchor.
  - Full TF-RD-029 candidate execution is blocked on finite training losses, finite MoE auxiliary metrics, completed benchmark artifacts, and W&B logging from this smoke row.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - Switch-style top-2 routing with no token dropping
  - selected top-2 router probabilities renormalized per token before expert outputs are combined
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for exactly 100 prior-dump smoke steps before benchmark plumbing
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_029_grid_moe_top2_normalized_v1/delta_tf_rd_029_grid_moe_e4_top2_norm_smoke_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Test the TF-RD-026 row 10 Grid-Sandwich anchor with core-only SwiGLU MoE FFNs using four experts, normalized top-2 routing, and a smaller task microbatch.
- Rationale: Contextualize `delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1` against anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2` for sweep `tf_rd_029_grid_moe_top2_normalized_v1`.
- Hypothesis: none
- Upstream delta: Top-2-first MoE follow-up after TF-RD-028 showed raw top-2 routing was healthy but underperformed and uncheckpointed top-2 exceeded single-GPU active memory.
- Anchor delta: Delta description pending for `delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1` against locked anchor `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
- Expected effect: Tests whether per-token normalization of selected top-2 router probabilities removes the output-scale shrinkage of raw top-2 weighting while keeping all tokens routed to two experts.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `173473290a098291bd96c39cf7d7742537f053cf888ee089bcf86b58e94ab8bd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_train_shuffle': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': None, 'val_batches': 0, 'max_steps': 5000, 'signature_family_optimizer_step_block_length': 4}`
- Model overrides: `{'grid_moe_scope': 'grid_core_ffn', 'grid_moe_num_experts': 4, 'grid_moe_top_k': 2, 'grid_moe_router_init_std': 0.01, 'grid_moe_normalize_top_k': True}`
- Parameter adequacy plan:
  - Execute immediately after the normalized top-2 smoke row passes; do not spend another E=8 top-1 rung in this lane.
  - Compare against TF-RD-028 raw top-2 microbatch8, TF-RD-028 E=4 top-1, and the locked TF-RD-026 row 10 anchor by final_log_loss_at_matched_regime_budget, route-health telemetry, and wall-time cost.
- Adequacy knobs to dimension explicitly:
  - current TF-RD-026 row 10 anchor shape held fixed
  - grid-core SwiGLU FFNs replaced with four sparse experts
  - Switch-style top-2 routing with no token dropping
  - selected top-2 router probabilities renormalized per token before expert outputs are combined
  - load-balance coefficient 1e-2 and router z-loss coefficient 1e-4
  - task batch size reduced to 8 with grad accumulation increased to 8
  - train for the anchor-matched 5000-step prior-dump budget
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_029_grid_moe_top2_normalized_v1/delta_tf_rd_029_grid_moe_e4_top2_norm_microbatch8_v1/result_card.md`
- Benchmark metrics: pending
