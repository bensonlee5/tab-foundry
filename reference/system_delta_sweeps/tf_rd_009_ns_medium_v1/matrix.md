# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_ns_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `95adb83bad76be6cb1753266f7254b7f8b245fba13c652b3c2e73b76c3073875`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law_phase2_ns`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.3481`, final BPF `2.3481`, final log loss `0.6331`, final Brier score `0.3914`, best ROC AUC `0.6109`, final ROC AUC `0.6716`, final training time `8529.8s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Staged feature encoder `unknown` from the benchmark registry surface. | Feature encoder changes alter the per-cell representation and should be interpreted explicitly. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Target conditioner `unknown` from the staged surface. | Target-conditioning changes should be interpreted separately from encoder or context changes. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Cell transformer block `unknown` from the staged surface. | Cell-block changes affect the core table computation and should be isolated carefully. |
| tokenizer | One scalar token per feature. | Tokenizer `unknown` from the staged surface. | Tokenizer changes alter the token sequence presented to the transformer stack. |
| column encoder | None on the upstream direct path. | Column encoder `unknown` from the staged surface. | Column-encoder changes should be read separately from row pooling or context changes. |
| row readout | Target-column readout from the final cell tensor. | Row pool `unknown` from the staged surface. | Row-pool changes alter the readout contract and require their own interpretation. |
| context encoder | None on the upstream direct path. | Context encoder `unknown` from the staged surface. | Context-encoder changes alter how training rows condition test rows. |
| prediction head | Direct binary logits head. | Prediction head `unknown` from the staged surface. | Head changes alter the task contract and output semantics. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest local id `openml_classification_medium_v1` sourced from `nanotabpfn_openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 1 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=625. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 2 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=1250. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 3 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=2500. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | completed | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 4 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=5000. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 5 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=625. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 6 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=1250. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 7 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=2500. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 8 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=5000. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 9 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=625. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 10 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=1250. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 11 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=2500. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | completed | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 12 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=5000. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 13 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=625. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 14 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=1250. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 15 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=2500. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | completed | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 16 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=5000. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | completed | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 17 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=625. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | completed | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 18 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=1250. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | completed | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 19 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=2500. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | completed | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 20 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=5000. |
| 21 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | completed | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 21 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=625. |
| 22 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | completed | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 22 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=1250. |
| 23 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | completed | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 23 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=2500. |
| 24 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | completed | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 24 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=5000. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `79156b2febdab225cea18d824599001b8bd424548da650c9adc7909d83834eb5`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.7257`, delta final log loss `+0.0926`, final Brier score `0.4541`, delta final brier score `+0.0626`, final ROC AUC `0.5058`, delta final roc auc `-0.1658`, final BPC (legacy feature-cell diagnostic) `2.7261`, delta final bpc (legacy feature-cell diagnostic) `+0.3780`, final BPF (legacy feature-cell diagnostic) `2.7261`, delta final bpf (legacy feature-cell diagnostic) `+0.3780`, best ROC AUC `0.5058`, delta final training time `-8165.1s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cd24e1dd6c6f8eb8bc56f8be9f26415428c49629be06b69968aff466eb89171b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.7112`, delta final log loss `+0.0780`, final Brier score `0.4428`, delta final brier score `+0.0514`, final ROC AUC `0.5726`, delta final roc auc `-0.0990`, final BPC (legacy feature-cell diagnostic) `2.7741`, delta final bpc (legacy feature-cell diagnostic) `+0.4260`, final BPF (legacy feature-cell diagnostic) `2.7741`, delta final bpf (legacy feature-cell diagnostic) `+0.4260`, best ROC AUC `0.5053`, delta final training time `-7833.7s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `59f2b28564e106c0cc977890d3201b521c052704ae7ab08404f7e2dafad4aadd`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.6380`, delta final log loss `+0.0048`, final Brier score `0.3975`, delta final brier score `+0.0061`, final ROC AUC `0.6569`, delta final roc auc `-0.0147`, final BPC (legacy feature-cell diagnostic) `3.5751`, delta final bpc (legacy feature-cell diagnostic) `+1.2270`, final BPF (legacy feature-cell diagnostic) `3.5751`, delta final bpf (legacy feature-cell diagnostic) `+1.2270`, best ROC AUC `0.6064`, delta final training time `-7344.0s`

### 4. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d0f29688a6d71756c2e012d054dbf1708467e1fb1d77644c0421930ab5fd0728`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1` with final log loss `0.6362`, delta final log loss `+0.0030`, final Brier score `0.3962`, delta final brier score `+0.0048`, final ROC AUC `0.6565`, delta final roc auc `-0.0151`, final BPC (legacy feature-cell diagnostic) `3.6933`, delta final bpc (legacy feature-cell diagnostic) `+1.3452`, final BPF (legacy feature-cell diagnostic) `3.6934`, delta final bpf (legacy feature-cell diagnostic) `+1.3453`, best ROC AUC `0.6101`, delta final training time `-6492.3s`

### 5. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `265318e6d554620e1681773432f61f1731cc8ab6f2cbe5055b37efe32b7d6a6e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7208`, delta final log loss `+0.0876`, final Brier score `0.4501`, delta final brier score `+0.0587`, final ROC AUC `0.5489`, delta final roc auc `-0.1227`, final BPC (legacy feature-cell diagnostic) `2.2862`, delta final bpc (legacy feature-cell diagnostic) `-0.0619`, final BPF (legacy feature-cell diagnostic) `2.2862`, delta final bpf (legacy feature-cell diagnostic) `-0.0619`, best ROC AUC `0.5275`, delta final training time `-8193.4s`

### 6. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0285f916ebe48a7d687992a5f842a3fb444dd2c4c46fcecf6077dd6218669583`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6775`, delta final log loss `+0.0443`, final Brier score `0.4191`, delta final brier score `+0.0277`, final ROC AUC `0.6279`, delta final roc auc `-0.0437`, final BPC (legacy feature-cell diagnostic) `2.3097`, delta final bpc (legacy feature-cell diagnostic) `-0.0385`, final BPF (legacy feature-cell diagnostic) `2.3097`, delta final bpf (legacy feature-cell diagnostic) `-0.0384`, best ROC AUC `0.5249`, delta final training time `-7864.5s`

### 7. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5e8e9c41a1aeac62ffb18143a9907b70506cc3e517dc6b14c7d126eeacedd6fb`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1/train`
- Reuse training surface fingerprint: `0724af85431f0e20692d70d56e2161bc033172bd2590ef3fb0ba6fc8aa23fe91`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6331`, delta final log loss `-0.0000`, final Brier score `0.3914`, delta final brier score `-0.0000`, final ROC AUC `0.6716`, delta final roc auc `-0.0000`, final BPC (legacy feature-cell diagnostic) `2.3481`, delta final bpc (legacy feature-cell diagnostic) `+0.0000`, final BPF (legacy feature-cell diagnostic) `2.3481`, delta final bpf (legacy feature-cell diagnostic) `-0.0000`, best ROC AUC `0.6109`, delta final training time `+0.0s`

### 8. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ff74c99dda8c48d165ca3895a3370a525993b22f9d969250c7c10a90e8efaa88`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6527`, delta final log loss `+0.0195`, final Brier score `0.4071`, delta final brier score `+0.0157`, final ROC AUC `0.6422`, delta final roc auc `-0.0294`, final BPC (legacy feature-cell diagnostic) `2.8418`, delta final bpc (legacy feature-cell diagnostic) `+0.4937`, final BPF (legacy feature-cell diagnostic) `2.8418`, delta final bpf (legacy feature-cell diagnostic) `+0.4937`, best ROC AUC `0.6094`, delta final training time `-6368.0s`

### 9. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d06a86c2af5c944efa7c23b7c1d73407563bf0171127f542271ce23ff688e1de`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.7240`, delta final log loss `+0.0909`, final Brier score `0.4531`, delta final brier score `+0.0616`, final ROC AUC `0.5292`, delta final roc auc `-0.1423`, final BPC (legacy feature-cell diagnostic) `2.2001`, delta final bpc (legacy feature-cell diagnostic) `-0.1481`, final BPF (legacy feature-cell diagnostic) `2.2000`, delta final bpf (legacy feature-cell diagnostic) `-0.1481`, best ROC AUC `0.5292`, delta final training time `-8171.1s`

### 10. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8f4e94d542c88e73abdc1afd580ca238d3830cdc409a3d58cea22aeb5ab925ac`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.6835`, delta final log loss `+0.0504`, final Brier score `0.4222`, delta final brier score `+0.0308`, final ROC AUC `0.6120`, delta final roc auc `-0.0596`, final BPC (legacy feature-cell diagnostic) `2.1970`, delta final bpc (legacy feature-cell diagnostic) `-0.1512`, final BPF (legacy feature-cell diagnostic) `2.1969`, delta final bpf (legacy feature-cell diagnostic) `-0.1512`, best ROC AUC `0.5523`, delta final training time `-7817.3s`

### 11. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e926390cb6f4e83bdeff0ca38637cd699ae9d0d20e1efe45e934da06bd28d267`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/sd_tf_rd_009_width_depth_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1/train`
- Reuse training surface fingerprint: `1a3c3253b50711655df3dee13106f0b051ff4db858e06b3e9ec786a51aec1583`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.6046`, delta final log loss `-0.0286`, final Brier score `0.3705`, delta final brier score `-0.0210`, final ROC AUC `0.6966`, delta final roc auc `+0.0250`, final BPC (legacy feature-cell diagnostic) `2.9666`, delta final bpc (legacy feature-cell diagnostic) `+0.6184`, final BPF (legacy feature-cell diagnostic) `2.9660`, delta final bpf (legacy feature-cell diagnostic) `+0.6179`, best ROC AUC `0.5176`, delta final training time `+582.8s`

### 12. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `66b765184dafba3868bb85efb6d6ea3e410ad5dce1201f35d2ed9e6e226e3c18`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1` with final log loss `0.6049`, delta final log loss `-0.0282`, final Brier score `0.3694`, delta final brier score `-0.0221`, final ROC AUC `0.7123`, delta final roc auc `+0.0407`, final BPC (legacy feature-cell diagnostic) `3.2329`, delta final bpc (legacy feature-cell diagnostic) `+0.8848`, final BPF (legacy feature-cell diagnostic) `3.2324`, delta final bpf (legacy feature-cell diagnostic) `+0.8843`, best ROC AUC `0.5143`, delta final training time `-6182.7s`

### 13. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e1b560ed1f43f608f1f4e965c9cb9628b49bbc5606982fe656cb129fd7d227ca`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1` with final log loss `0.7073`, delta final log loss `+0.0741`, final Brier score `0.4403`, delta final brier score `+0.0489`, final ROC AUC `0.5820`, delta final roc auc `-0.0896`, final BPC (legacy feature-cell diagnostic) `2.3103`, delta final bpc (legacy feature-cell diagnostic) `-0.0378`, final BPF (legacy feature-cell diagnostic) `2.3102`, delta final bpf (legacy feature-cell diagnostic) `-0.0379`, best ROC AUC `0.5820`, delta final training time `-8155.8s`

### 14. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `074ec53c7217f7f3843abe1f2459d0911c524a1c41aee21a346be23326ff7464`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1` with final log loss `0.6490`, delta final log loss `+0.0158`, final Brier score `0.4003`, delta final brier score `+0.0089`, final ROC AUC `0.6465`, delta final roc auc `-0.0251`, final BPC (legacy feature-cell diagnostic) `2.2889`, delta final bpc (legacy feature-cell diagnostic) `-0.0592`, final BPF (legacy feature-cell diagnostic) `2.2888`, delta final bpf (legacy feature-cell diagnostic) `-0.0593`, best ROC AUC `0.6176`, delta final training time `-7781.8s`

### 15. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4be6d5520b37bdb79537234d3cfe447dd224ce5bec668a209a3e79b1aec5a80e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/sd_tf_rd_009_width_depth_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1/train`
- Reuse training surface fingerprint: `5c472a66d458a9149ba21381a8af720511b2895f38e601879e8c3cc59836c787`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1` with final log loss `0.6348`, delta final log loss `+0.0017`, final Brier score `0.3933`, delta final brier score `+0.0019`, final ROC AUC `0.6457`, delta final roc auc `-0.0259`, final BPC (legacy feature-cell diagnostic) `2.7294`, delta final bpc (legacy feature-cell diagnostic) `+0.3812`, final BPF (legacy feature-cell diagnostic) `2.7291`, delta final bpf (legacy feature-cell diagnostic) `+0.3811`, best ROC AUC `0.6185`, delta final training time `+1038.1s`

### 16. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `bf2f0d8c5d60652fa5faa4caa450e1c497fb57c7fa0d1b30eceedd49353f1109`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1_v1` with final log loss `0.6683`, delta final log loss `+0.0351`, final Brier score `0.4156`, delta final brier score `+0.0242`, final ROC AUC `0.6295`, delta final roc auc `-0.0420`, final BPC (legacy feature-cell diagnostic) `7.9135`, delta final bpc (legacy feature-cell diagnostic) `+5.5654`, final BPF (legacy feature-cell diagnostic) `7.9119`, delta final bpf (legacy feature-cell diagnostic) `+5.5639`, best ROC AUC `0.6143`, delta final training time `-6064.4s`

### 17. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4fccef2d79cfd3ec76fb59fb660e13bca88532f767ddafabedffdd0268de4b82`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` with final log loss `0.7117`, delta final log loss `+0.0786`, final Brier score `0.4444`, delta final brier score `+0.0529`, final ROC AUC `0.5731`, delta final roc auc `-0.0985`, final BPC (legacy feature-cell diagnostic) `2.2902`, delta final bpc (legacy feature-cell diagnostic) `-0.0579`, final BPF (legacy feature-cell diagnostic) `2.2902`, delta final bpf (legacy feature-cell diagnostic) `-0.0579`, best ROC AUC `0.4905`, delta final training time `-8083.6s`

### 18. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `474351c2559999f2e62e74d9a7be39771831ad11483d497f4951211ecbd69fab`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` with final log loss `0.6172`, delta final log loss `-0.0159`, final Brier score `0.3789`, delta final brier score `-0.0126`, final ROC AUC `0.6966`, delta final roc auc `+0.0250`, final BPC (legacy feature-cell diagnostic) `2.7112`, delta final bpc (legacy feature-cell diagnostic) `+0.3631`, final BPF (legacy feature-cell diagnostic) `2.7111`, delta final bpf (legacy feature-cell diagnostic) `+0.3630`, best ROC AUC `0.5069`, delta final training time `-7636.7s`

### 19. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8fc45bcaafa29bcf657e50ddecbbe75de919507b5369594431be8b6b8aebada0`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1/train`
- Reuse training surface fingerprint: `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` with final log loss `0.7042`, delta final log loss `+0.0710`, final Brier score `0.4381`, delta final brier score `+0.0466`, final ROC AUC `0.5845`, delta final roc auc `-0.0871`, final BPC (legacy feature-cell diagnostic) `2.3356`, delta final bpc (legacy feature-cell diagnostic) `-0.0125`, final BPF (legacy feature-cell diagnostic) `2.3356`, delta final bpf (legacy feature-cell diagnostic) `-0.0125`, best ROC AUC `0.5137`, delta final training time `-5921.8s`

### 20. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7f5ec50e949723388af308f8a77c1ae5adc29ba5aaeeda94ea083785a225628e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` with final log loss `0.5774`, delta final log loss `-0.0557`, final Brier score `0.3520`, delta final brier score `-0.0394`, final ROC AUC `0.7419`, delta final roc auc `+0.0703`, final BPC (legacy feature-cell diagnostic) `4.6058`, delta final bpc (legacy feature-cell diagnostic) `+2.2577`, final BPF (legacy feature-cell diagnostic) `4.6061`, delta final bpf (legacy feature-cell diagnostic) `+2.2580`, best ROC AUC `0.6323`, delta final training time `-5509.4s`

### 21. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `426ee4797ff39bff8c9cc2986ce1366ce76d09bb6aee60ccd514b9e5a18d55cb`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_21_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_21_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1` with final log loss `0.6937`, delta final log loss `+0.0606`, final Brier score `0.4335`, delta final brier score `+0.0421`, final ROC AUC `0.6096`, delta final roc auc `-0.0620`, final BPC (legacy feature-cell diagnostic) `2.6977`, delta final bpc (legacy feature-cell diagnostic) `+0.3496`, final BPF (legacy feature-cell diagnostic) `2.6976`, delta final bpf (legacy feature-cell diagnostic) `+0.3496`, best ROC AUC `0.4955`, delta final training time `-8055.4s`

### 22. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `756889adc86f88b3dd7d10391b2e6b84373a8b952818de7984d0da630924e25a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_22_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_22_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1` with final log loss `0.6503`, delta final log loss `+0.0171`, final Brier score `0.4050`, delta final brier score `+0.0136`, final ROC AUC `0.6419`, delta final roc auc `-0.0297`, final BPC (legacy feature-cell diagnostic) `2.8881`, delta final bpc (legacy feature-cell diagnostic) `+0.5400`, final BPF (legacy feature-cell diagnostic) `2.8881`, delta final bpf (legacy feature-cell diagnostic) `+0.5400`, best ROC AUC `0.4977`, delta final training time `-7580.2s`

### 23. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5214b8e2a3b25cc4646434e054454e65248608d7cbfd21ae4de515ff7604183a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/sd_tf_rd_009_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1/train`
- Reuse training surface fingerprint: `b74f3dc0759fef1e2c75f5b45d99cdb94c46bfe3a529584a1cf25d01ac78536c`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_23_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_23_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1` with final log loss `0.7285`, delta final log loss `+0.0954`, final Brier score `0.4550`, delta final brier score `+0.0636`, final ROC AUC `0.5113`, delta final roc auc `-0.1603`, final BPC (legacy feature-cell diagnostic) `2.4525`, delta final bpc (legacy feature-cell diagnostic) `+0.1044`, final BPF (legacy feature-cell diagnostic) `2.4525`, delta final bpf (legacy feature-cell diagnostic) `+0.1044`, best ROC AUC `0.4991`, delta final training time `-7487.1s`

### 24. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ffe868324e53760ef2f8981e39fda91b46ec0b04cb667f3a84042adc33ac1a03`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_24_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_24_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1` with final log loss `0.6122`, delta final log loss `-0.0210`, final Brier score `0.3751`, delta final brier score `-0.0164`, final ROC AUC `0.7023`, delta final roc auc `+0.0307`, final BPC (legacy feature-cell diagnostic) `3.9204`, delta final bpc (legacy feature-cell diagnostic) `+1.5723`, final BPF (legacy feature-cell diagnostic) `3.9203`, delta final bpf (legacy feature-cell diagnostic) `+1.5722`, best ROC AUC `0.6315`, delta final training time `-5289.1s`
