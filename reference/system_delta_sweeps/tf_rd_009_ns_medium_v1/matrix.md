# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_ns_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_ns_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `1d81b869dc1d7b38ad0e2607c4c73acb470a93ee8a147c71c4ea13b54ebeef56`

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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65f969852d60efbe4c0340fa6f7eae828bf23f692f0a98880034ccd654dd1953`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d01c9f5bf95d173f1badbe8fbc6e89381f2e265b59b46038484ffe4cbb862b9f`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `811297672caa02afb3965067d1ed845e75598d3b7dfc5b1b462c1ec1cbfcceab`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d5bac018588409236ce9e9f0da779d0d2d77c7ffc4b487da3df06d08e0fd5b2d`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3303683ee867cad7208ceea358e1ab4be3d994fb790141f9cd2f281b54dfd98c`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b781542f3cce336edb5ad9bdb64c16ce0e8e7ccfaa5aa3ff0a6f8b9002d1c86d`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0724af85431f0e20692d70d56e2161bc033172bd2590ef3fb0ba6fc8aa23fe91`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e84051a7a1f546dc830e64266bdae7558d786733d11575fa2285c0473dfcafec`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1755531ca2f16861576e7051383badeb30de1194929a83eea15687c9cc5c4b15`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `63ce41a9f1f0014a90d78858faed460b87dbf36d01077d6c865a1ae760c71458`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1a3c3253b50711655df3dee13106f0b051ff4db858e06b3e9ec786a51aec1583`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `42b7a1020ff7c82924f6036def83ef7d4a36c48831a17d5bc004bb7343777311`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aaf8eeb0b001216a684666986c9593dc5e4d9d95b6a0a68ef70f3a7822773fb6`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b4369a2ae094b8c56396b5790671863a3ba6055ea7ea32972c20bf3c97c92c96`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5c472a66d458a9149ba21381a8af720511b2895f38e601879e8c3cc59836c787`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c2e1f60112403054e4dc1ed71480236fe31ed60ac13e2c959800c2ab204a4aaa`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2d4ed6a8dfc91bbb182ed2b911c2a553c84cc537e8f3148b24a859bd70962d2e`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1ccd4dc4d791ddefee6bada3cc13d513d44712daa6d77d068f6f44671f7ed75a`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e6265768a4e4cb1a40680eb3ef5529e81c2f00a946d36be63c2df7ccd7bb1305`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b0752100d7400c026715a3245129003063c36a02e77142bc2cb24ebe9bd20fbd`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `78a570a9bb8928daf741f1cd98303ee918d988ccea69da00c7779d64dbe497f5`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b74f3dc0759fef1e2c75f5b45d99cdb94c46bfe3a529584a1cf25d01ac78536c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3a1967e5adb4fb4271572942559f44488db4ec89e8b2b598bb5b4a4fef2a55df`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_ns_medium_v1_24_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Registered run: `sd_tf_rd_009_ns_medium_v1_24_delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1_v1` with final log loss `0.6122`, delta final log loss `-0.0210`, final Brier score `0.3751`, delta final brier score `-0.0164`, final ROC AUC `0.7023`, delta final roc auc `+0.0307`, final BPC (legacy feature-cell diagnostic) `3.9204`, delta final bpc (legacy feature-cell diagnostic) `+1.5723`, final BPF (legacy feature-cell diagnostic) `3.9203`, delta final bpf (legacy feature-cell diagnostic) `+1.5722`, best ROC AUC `0.6315`, delta final training time `-5289.1s`
