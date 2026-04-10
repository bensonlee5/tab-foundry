# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_ns_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `588dbf478eaa429b5bea2e25359b9362970f05dd160f154a47085bd7f94fd2df`

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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 1 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=625. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 2 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=1250. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 3 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=2500. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 4 as the TF-RD-009 Phase 2 N x S row for geometry 72x1 at max_steps=5000. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 5 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=625. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 6 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=1250. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 7 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=2500. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 8 as the TF-RD-009 Phase 2 N x S row for geometry 96x2 at max_steps=5000. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 9 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=625. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 10 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=1250. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 11 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=2500. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 12 as the TF-RD-009 Phase 2 N x S row for geometry 112x3 at max_steps=5000. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 13 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=625. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 14 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=1250. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 15 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=2500. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 16 as the TF-RD-009 Phase 2 N x S row for geometry 128x4 at max_steps=5000. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 17 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=625. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 18 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=1250. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 19 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=2500. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 20 as the TF-RD-009 Phase 2 N x S row for geometry 152x5 at max_steps=5000. |
| 21 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 21 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=625. |
| 22 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 22 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=1250. |
| 23 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 23 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=2500. |
| 24 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 24 as the TF-RD-009 Phase 2 N x S row for geometry 176x6 at max_steps=5000. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4f5d113f9ccdfd6a6fdcc7b37e8c54e59b7bd4f4238c13594cf4687f5f1fad89`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5b0b09b14495df0a5e06ba27e3aaf7b22863d08f1f1e9187a692444bbb3de534`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5341a2ba42e1ee093f6f26393bcaf1f441d4064d5888e84098357e550e2e345d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute 72x1 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4fdb6a6eb13930d0d3587a245b55599ba7abfef70471717fd921de227af1388e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6c90c66128e6bf96308cb20bdf2a6e8652026e43309ce3618b3f465ca94569e1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d3f0c2d72b4a258722bbc81a032ea7b6b208cfee86664a8d58d9f90e64f6f3d0`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3a7e99b88ae55fc3546d8df3241f411768ce91bf05f474f2e7d8b8653b1dbe9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b603f709d7b760e1df005764c4d3fd522d1f2f2a059d3d2fecf649565cee722a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0afb1dc70beacbe71d0849860bef944115e2e76b7b34c95e79a79a4254eec728`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6816c3d4a6bca8f11b2b8b0631b437ff7a10beb7cee58e9554fe39caeecf2c20`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `21b1145f4e210135482236978601819e7b08b53ed009e3a63f9996766025c4b3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute 112x3 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d8193b1be2c8019962924421aa030911fb184a71959b4557e06bb82d959eb081`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 13. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2a50c870fe5080378825fb2e95e73bfa47046de9197ca3cfc709701df95da932`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 14. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c034905cc1cafc9f723855a2ebd1bf90dcc61615a17dfa6e111be79d6e5ca31f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 15. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d433732226811e802ad786a88f152a37fa4b3027bb94f09fbe5468a0f5b156cf`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 16. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute 128x4 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `15817f7390b6ea18f53ceb836d28e86dff311a49b9e3e5d1ec84d3deb8e9a492`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 17. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a0b6f05992f03676f9c393d90df1dd9c4d745287929be530115409abf41f8cac`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 18. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aba821f05e15a368eedaa660ac04d397dea411c375a73945e9904b5649e9b3e3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 19. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `846853cba0c44d719d5ace101c3f6d8c8ae35eee4b1ae8c3a6c5e1c4e95220e9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 20. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute 152x5 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `276d9aa7b9c9eb85a05f1d312fe507a624f6efe2cd4c845d5358c5e611dbf286`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 21. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=625 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c11b34910b308ad9a81bbb1ae1a88fe6e2d94d8f56067b18c21c1ba923b65410`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 22. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=1250 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f8668fbd1c451d4bcc5542386659e6567962823d3a99c62f17b3e9b8e45e863d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 23. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=2500 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cc53a2c49b417ab63a448bc18941816a42df63c08e60ea3bf443932d6f24a262`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 24. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute 176x6 at max_steps=5000 and grad_accum_steps=4 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `98a5ada8c29563144079b94b910117a1780d5bfc66cbce38333ae3de3878d853`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending
