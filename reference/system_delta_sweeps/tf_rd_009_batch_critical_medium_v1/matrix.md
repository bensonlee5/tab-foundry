# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_batch_critical_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_batch_critical_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_batch_critical_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_ns_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_batch_critical_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `6ebaf6bff503ad411bd698ecfe9dbf73846a51015076d6488aebce54a757bb58`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law_phase2_batch`
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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 1 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=1 and max_steps=625. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 2 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=1 and max_steps=1250. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 3 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=1 and max_steps=2500. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 4 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=1 and max_steps=5000. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 5 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=2 and max_steps=625. |
| 6 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 6 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=2 and max_steps=1250. |
| 7 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 7 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=2 and max_steps=2500. |
| 8 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 8 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=2 and max_steps=5000. |
| 9 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 9 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=4 and max_steps=625. |
| 10 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 10 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=4 and max_steps=1250. |
| 11 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 11 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=4 and max_steps=2500. |
| 12 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 12 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=4 and max_steps=5000. |
| 13 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 13 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=8 and max_steps=625. |
| 14 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 14 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=8 and max_steps=1250. |
| 15 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 15 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=8 and max_steps=2500. |
| 16 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 16 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=8 and max_steps=5000. |
| 17 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 17 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=16 and max_steps=625. |
| 18 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 18 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=16 and max_steps=1250. |
| 19 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 19 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=16 and max_steps=2500. |
| 20 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Execute order 20 as the TF-RD-009 Phase 2 batch-critical row for grad_accum_steps=16 and max_steps=5000. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=1 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b248049b1038c89e98f9a47a5e7854b82f2fe2a21dc6bb85fd25cfae2249eff2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=625 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7262`, delta final log loss `+0.0931`, final Brier score `0.4541`, delta final brier score `+0.0627`, final ROC AUC `0.5105`, delta final roc auc `-0.1611`, final BPC (legacy feature-cell diagnostic) `2.2869`, delta final bpc (legacy feature-cell diagnostic) `-0.0612`, final BPF (legacy feature-cell diagnostic) `2.2869`, delta final bpf (legacy feature-cell diagnostic) `-0.0612`, best ROC AUC `0.5034`, delta final training time `-8176.8s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=1 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f326cbf86e4e8f3668c4472fa9b11db57554737fc571860525f7d359a1a9bf98`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=1250 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7044`, delta final log loss `+0.0712`, final Brier score `0.4373`, delta final brier score `+0.0459`, final ROC AUC `0.5825`, delta final roc auc `-0.0891`, final BPC (legacy feature-cell diagnostic) `2.3097`, delta final bpc (legacy feature-cell diagnostic) `-0.0384`, final BPF (legacy feature-cell diagnostic) `2.3097`, delta final bpf (legacy feature-cell diagnostic) `-0.0384`, best ROC AUC `0.5048`, delta final training time `-7835.3s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=1 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e09c1374222cb839b1ec88696fd2a73a8ce9bd22c386ff3f08de20be473ebf5a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=2500 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6631`, delta final log loss `+0.0300`, final Brier score `0.4121`, delta final brier score `+0.0207`, final ROC AUC `0.6339`, delta final roc auc `-0.0376`, final BPC (legacy feature-cell diagnostic) `2.3462`, delta final bpc (legacy feature-cell diagnostic) `-0.0019`, final BPF (legacy feature-cell diagnostic) `2.3462`, delta final bpf (legacy feature-cell diagnostic) `-0.0018`, best ROC AUC `0.5232`, delta final training time `-7173.7s`

### 4. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=1 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6ce956d1fda9690bfdb30bd348667a84dad3844ed2128a9383b7e9db3f121fbb`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=1 and max_steps=5000 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6346`, delta final log loss `+0.0015`, final Brier score `0.3951`, delta final brier score `+0.0037`, final ROC AUC `0.6542`, delta final roc auc `-0.0174`, final BPC (legacy feature-cell diagnostic) `2.8638`, delta final bpc (legacy feature-cell diagnostic) `+0.5156`, final BPF (legacy feature-cell diagnostic) `2.8637`, delta final bpf (legacy feature-cell diagnostic) `+0.5156`, best ROC AUC `0.6082`, delta final training time `-5892.0s`

### 5. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=2 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8c0228d4fb1989de87b056f5795377869e8531c92cdbc5889c500545fa1097dc`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=625 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7255`, delta final log loss `+0.0924`, final Brier score `0.4539`, delta final brier score `+0.0624`, final ROC AUC `0.5119`, delta final roc auc `-0.1597`, final BPC (legacy feature-cell diagnostic) `2.2444`, delta final bpc (legacy feature-cell diagnostic) `-0.1037`, final BPF (legacy feature-cell diagnostic) `2.2444`, delta final bpf (legacy feature-cell diagnostic) `-0.1037`, best ROC AUC `0.5099`, delta final training time `-7872.3s`

### 6. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=2 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b592e5084f598b0f66570f6771d13003117db87c8410497aafabbd75e280799c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=1250 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_06_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6907`, delta final log loss `+0.0576`, final Brier score `0.4283`, delta final brier score `+0.0369`, final ROC AUC `0.6174`, delta final roc auc `-0.0542`, final BPC (legacy feature-cell diagnostic) `2.2348`, delta final bpc (legacy feature-cell diagnostic) `-0.1133`, final BPF (legacy feature-cell diagnostic) `2.2348`, delta final bpf (legacy feature-cell diagnostic) `-0.1133`, best ROC AUC `0.6174`, delta final training time `-7217.0s`

### 7. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=2 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `523f0400639a4425bbcb93299bbabc74922e2a5b40f98dabf0d8958861289618`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=2500 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_07_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6460`, delta final log loss `+0.0129`, final Brier score `0.4013`, delta final brier score `+0.0098`, final ROC AUC `0.6501`, delta final roc auc `-0.0215`, final BPC (legacy feature-cell diagnostic) `2.3607`, delta final bpc (legacy feature-cell diagnostic) `+0.0125`, final BPF (legacy feature-cell diagnostic) `2.3607`, delta final bpf (legacy feature-cell diagnostic) `+0.0126`, best ROC AUC `0.6266`, delta final training time `-5910.9s`

### 8. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=2 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7a1e8a857ce0afb68b9bd324a585d3501087cf3c15bc6c00de65dfb3dfe39c11`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 2, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=2 and max_steps=5000 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_08_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.5904`, delta final log loss `-0.0427`, final Brier score `0.3607`, delta final brier score `-0.0307`, final ROC AUC `0.7097`, delta final roc auc `+0.0381`, final BPC (legacy feature-cell diagnostic) `2.8843`, delta final bpc (legacy feature-cell diagnostic) `+0.5362`, final BPF (legacy feature-cell diagnostic) `2.8843`, delta final bpf (legacy feature-cell diagnostic) `+0.5362`, best ROC AUC `0.6089`, delta final training time `-3624.9s`

### 9. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

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
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=625 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_09_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7231`, delta final log loss `+0.0899`, final Brier score `0.4521`, delta final brier score `+0.0606`, final ROC AUC `0.5289`, delta final roc auc `-0.1426`, final BPC (legacy feature-cell diagnostic) `2.2916`, delta final bpc (legacy feature-cell diagnostic) `-0.0566`, final BPF (legacy feature-cell diagnostic) `2.2916`, delta final bpf (legacy feature-cell diagnostic) `-0.0565`, best ROC AUC `0.5018`, delta final training time `-7234.7s`

### 10. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

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
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=1250 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_10_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6725`, delta final log loss `+0.0394`, final Brier score `0.4153`, delta final brier score `+0.0239`, final ROC AUC `0.6303`, delta final roc auc `-0.0413`, final BPC (legacy feature-cell diagnostic) `2.3207`, delta final bpc (legacy feature-cell diagnostic) `-0.0274`, final BPF (legacy feature-cell diagnostic) `2.3207`, delta final bpf (legacy feature-cell diagnostic) `-0.0274`, best ROC AUC `0.6143`, delta final training time `-5970.0s`

### 11. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

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
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=2500 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_11_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6331`, delta final log loss `-0.0000`, final Brier score `0.3914`, delta final brier score `-0.0000`, final ROC AUC `0.6716`, delta final roc auc `-0.0000`, final BPC (legacy feature-cell diagnostic) `2.3481`, delta final bpc (legacy feature-cell diagnostic) `+0.0000`, final BPF (legacy feature-cell diagnostic) `2.3481`, delta final bpf (legacy feature-cell diagnostic) `-0.0000`, best ROC AUC `0.6109`, delta final training time `+0.0s`

### 12. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

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
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=4 and max_steps=5000 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6355`, delta final log loss `+0.0023`, final Brier score `0.3922`, delta final brier score `+0.0007`, final ROC AUC `0.6814`, delta final roc auc `+0.0098`, final BPC (legacy feature-cell diagnostic) `2.8915`, delta final bpc (legacy feature-cell diagnostic) `+0.5434`, final BPF (legacy feature-cell diagnostic) `2.8914`, delta final bpf (legacy feature-cell diagnostic) `+0.5433`, best ROC AUC `0.6087`, delta final training time `-451.0s`

### 13. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=8 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `817c0bd039cb61b4c68d46629ab77c4e3764393fe1a3b636806618ed54bbf169`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=625 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_13_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7113`, delta final log loss `+0.0781`, final Brier score `0.4424`, delta final brier score `+0.0510`, final ROC AUC `0.5685`, delta final roc auc `-0.1030`, final BPC (legacy feature-cell diagnostic) `2.2534`, delta final bpc (legacy feature-cell diagnostic) `-0.0947`, final BPF (legacy feature-cell diagnostic) `2.2534`, delta final bpf (legacy feature-cell diagnostic) `-0.0947`, best ROC AUC `0.5307`, delta final training time `-5920.8s`

### 14. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=8 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `36ca545661bc067e3d3afb25bce126b80fe65b519712f627e27ca977b508427d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=1250 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_14_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6729`, delta final log loss `+0.0397`, final Brier score `0.4166`, delta final brier score `+0.0251`, final ROC AUC `0.6367`, delta final roc auc `-0.0349`, final BPC (legacy feature-cell diagnostic) `2.2298`, delta final bpc (legacy feature-cell diagnostic) `-0.1183`, final BPF (legacy feature-cell diagnostic) `2.2298`, delta final bpf (legacy feature-cell diagnostic) `-0.1183`, best ROC AUC `0.6285`, delta final training time `-3759.3s`

### 15. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=8 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3b3261d073acaa581f22b6893072080ef1a450a5dadcab11488fbb739248297c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=2500 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_15_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6641`, delta final log loss `+0.0310`, final Brier score `0.4108`, delta final brier score `+0.0194`, final ROC AUC `0.6435`, delta final roc auc `-0.0281`, final BPC (legacy feature-cell diagnostic) `2.2217`, delta final bpc (legacy feature-cell diagnostic) `-0.1265`, final BPF (legacy feature-cell diagnostic) `2.2216`, delta final bpf (legacy feature-cell diagnostic) `-0.1264`, best ROC AUC `0.6252`, delta final training time `-6351.1s`

### 16. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=8 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b46020fea5a1227c2b3df7995437bdd60223989c0873c3de47b76ece68e42421`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 8, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=8 and max_steps=5000 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_16_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6665`, delta final log loss `+0.0333`, final Brier score `0.4122`, delta final brier score `+0.0208`, final ROC AUC `0.6436`, delta final roc auc `-0.0280`, final BPC (legacy feature-cell diagnostic) `2.2953`, delta final bpc (legacy feature-cell diagnostic) `-0.0528`, final BPF (legacy feature-cell diagnostic) `2.2953`, delta final bpf (legacy feature-cell diagnostic) `-0.0528`, best ROC AUC `0.6160`, delta final training time `-4685.4s`

### 17. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=625 and grad_accum_steps=16 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `be2b65f708cf4d486fd4c44a4350e6c429a5444c0312ecdbb7c12774fcc1304c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 625}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=625 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_17_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7054`, delta final log loss `+0.0723`, final Brier score `0.4381`, delta final brier score `+0.0467`, final ROC AUC `0.5821`, delta final roc auc `-0.0894`, final BPC (legacy feature-cell diagnostic) `2.2726`, delta final bpc (legacy feature-cell diagnostic) `-0.0755`, final BPF (legacy feature-cell diagnostic) `2.2726`, delta final bpf (legacy feature-cell diagnostic) `-0.0755`, best ROC AUC `0.5293`, delta final training time `-7251.6s`

### 18. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=1250 and grad_accum_steps=16 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c358c17186b9c9234eaec0240cdc3cdc1aff89ef25ad4f714438bf9f224562ff`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 1250}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=1250 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_18_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7009`, delta final log loss `+0.0678`, final Brier score `0.4347`, delta final brier score `+0.0433`, final ROC AUC `0.5881`, delta final roc auc `-0.0835`, final BPC (legacy feature-cell diagnostic) `2.2553`, delta final bpc (legacy feature-cell diagnostic) `-0.0928`, final BPF (legacy feature-cell diagnostic) `2.2553`, delta final bpf (legacy feature-cell diagnostic) `-0.0928`, best ROC AUC `0.5228`, delta final training time `-6358.0s`

### 19. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=2500 and grad_accum_steps=16 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `982779fab247077f4d1a5d783987fdf48cdbb97c9c1bfb2b1f680b35d1f601b7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=2500 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_19_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7068`, delta final log loss `+0.0736`, final Brier score `0.4389`, delta final brier score `+0.0474`, final ROC AUC `0.5790`, delta final roc auc `-0.0926`, final BPC (legacy feature-cell diagnostic) `2.2455`, delta final bpc (legacy feature-cell diagnostic) `-0.1027`, final BPF (legacy feature-cell diagnostic) `2.2454`, delta final bpf (legacy feature-cell diagnostic) `-0.1026`, best ROC AUC `0.5213`, delta final training time `-4599.5s`

### 20. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Execute 96x2 at max_steps=5000 and grad_accum_steps=16 for the TF-RD-009 Phase 2 scaling-law study.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Phase-two scaling row 96x2 reuses the locked 96x2 anchor surface and changes only the declared geometry or step budget for the study family.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `87703b4463f61564b1e077392bf3b4137bb0b825c8f0f093de2480db99bf4ce2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 16, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Hold geometry at 96x2 while varying only effective batch through grad_accum_steps.
  - Use grad_accum_steps=16 and max_steps=5000 for the Bcrit(L) ladder.
  - Final reported fits must use validation loss from completed runs plus inspected training-only compute accounting.
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
  - Canonical rerun registered as `sd_tf_rd_009_batch_critical_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_batch_critical_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_batch_critical_medium_v1_20_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.7123`, delta final log loss `+0.0792`, final Brier score `0.4422`, delta final brier score `+0.0508`, final ROC AUC `0.5696`, delta final roc auc `-0.1020`, final BPC (legacy feature-cell diagnostic) `2.2883`, delta final bpc (legacy feature-cell diagnostic) `-0.0599`, final BPF (legacy feature-cell diagnostic) `2.2882`, delta final bpf (legacy feature-cell diagnostic) `-0.0598`, best ROC AUC `0.5332`, delta final training time `-1082.2s`
