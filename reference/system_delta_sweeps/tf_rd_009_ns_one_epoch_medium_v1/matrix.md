# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_ns_one_epoch_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_ns_one_epoch_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_ns_one_epoch_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_ns_one_epoch_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `56121bfe42d71a2347397825f640489692f6257aa6ecd3bc0d3a85622ee0a75b`

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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 1 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 2 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 3 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute order 4 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 5 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 6 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 7 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | ready | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 8 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 9 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 10 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 11 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute order 12 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 13 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 14 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 15 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute order 16 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 17 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 18 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 19 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute order 20 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 21 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 21 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 22 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 22 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 23 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 23 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |
| 24 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute order 24 as corrected one-epoch row in `tf_rd_009_ns_one_epoch_medium_v1`; admit it only into corrected one-epoch Phase 2 fits. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d30304f18fb25fb27a4cbb6c2e6f3afb351496d06d9dadc475fadff325a72d1d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` order 1; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8283f9296905253a2c4da57882afa962bb0a022d88451154ea127591669276c7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` order 2; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `85fbe24baac2a01d5cb2a96cd778baf3ef17e769393946f07c60b8089d60eaee`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` order 3; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 72x1 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c2db2077d31eef9ed5e4c163de006f217619df9180aa9f8c95ec2fae6d4f856b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 72x1 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` order 4; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0aba9d35b388cb8408adfde56b2ba3a374ea84eb77815edc71fabbf9995546f2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl96_v1` order 5; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b0fef6399480f1df669cc4a3a823d3bc68974d276e6a21b7d9bb45c91d235b28`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl96_v1` order 6; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ec895e683420e957f0402669d68a61c070426f313b94a26570308348142e1529`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl96_v1` order 7; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `50d4794f333c9ae3a92750d3c4a330cb1d34467291f44589cd9dd5b3c780b069`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 96x2 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl96_v1` order 8; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `84ac6262a88bcf131f507cbb1bffac72de6bd1bb0a2b6c306bcabb7836ea0c8d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` order 9; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 10. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `01063f30e6d25604a9e77c222c72399803207f6dd70deb56ffa50f58706d6ca5`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` order 10; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 11. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2d04f14525545e5495b0526b0ab2d3db53946e86f56d23fa68bae07ef5dbebc1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` order 11; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 12. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Phase-two scaling row 112x3 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f64aa8b3669dd9060c07b759d46fff80698fcc0ba4a49d3ba342b145c6bfd342`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 112x3 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` order 12; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 13. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5cb52e3e1ebdbc86572480314e78e34024172c0b5594a9e3ae180350c73ea815`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` order 13; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 14. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `9b5d5789348b291f3bdcebeebbe6d70df9087f1e3e807902630ddb7af40c4084`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` order 14; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 15. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `2a3d8e8453a0aba8624a5238c3f4cd17a05431d2785db5100595d202acde6ec2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` order 15; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 16. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Phase-two scaling row 128x4 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `307c30ce5f3b5e7408e4b965fc3199f1839e93e4a42016ab6d10bf523a51ab9f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 128x4 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` order 16; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 17. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0839e9b092e77f60f3ce5e6b2256a25878cd84cf31bb92ed589d16c531904c9d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` order 17; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 18. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6c5b10a94ec85b51aff008b350039a856630d47634d403a2ac544ed68d2f4737`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` order 18; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 19. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1ad9fdf8163375b44dff7939c3a98ceafe07b6f2c7f37fa42cb9cab6e2664cfa`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` order 19; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 20. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Phase-two scaling row 152x5 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ac5a3bbff0af0a278d70555e361c9e5a95b636de32c36513b8c6935a7d6ec185`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 152x5 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` order 20; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 21. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute corrected one-epoch NS row at max_steps=625 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d3aa7222994c175699041d38366e32330ce04c191466d48fb7f2086051a6197a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=625; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=40000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` order 21; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 22. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute corrected one-epoch NS row at max_steps=1250 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0ccafe33cd433a3d8ccb3367264b6b284ca88fea45b44dc68f24fe3b48b71dba`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=1250; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=80000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` order 22; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 23. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute corrected one-epoch NS row at max_steps=2500 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8afddaafb90480fb05b6ad986eda521156b5119e572d1f0e058923dd5e65c73b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=2500; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=160000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` order 23; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending

### 24. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Execute corrected one-epoch NS row at max_steps=5000 and grad_accum_steps=4 on `tf_rd_010_dagzoo_medium_control_curated_v6`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Phase-two scaling row 176x6 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `9843ecf2f145c2c2f046af1ecfa464308399c35c6335418d764e8a4116046030`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Keep geometry 176x6 on the fixed diagonal {72x1, 96x2, 112x3, 128x4, 152x5, 176x6}.
  - Use this row only for the paper-faithful N x S matrix at max_steps=5000; do not reinterpret it as a phase-one queue row.
  - Final reported fits must use inspected canonical non-embedding params and measured compute after the run lands.
  - Corrected one-epoch contract: required_train_tasks=320000 must fit within the `tf_rd_010_dagzoo_medium_control_curated_v6` train split before execution.
  - Do not mix this corrected row with historical sub-epoch TF-RD-009 Phase 2 rows in scaling-law fits.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Corrected one-epoch rerun of historical `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` order 24; uses `tf_rd_010_dagzoo_medium_control_curated_v6` instead of `tf_rd_010_dagzoo_medium_control_curated_v5`.
  - Historical TF-RD-009 Phase 2 evidence remains preserved under the original sweep id and is superseded for Cmin interpretation.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_ns_one_epoch_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending
