# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_width_depth_upper_extension_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_width_depth_upper_extension_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_width_depth_upper_extension_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_width_depth_upper_extension_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `8c10d8b2b5ce4ca4f19a28a23ff40e3e97a24df935d4ad73d7850b8872b2a75e`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law`
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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest `data/manifests/bench/openml_classification_medium_v1/manifest.parquet` sourced from `nanotabpfn_openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1` | classification_scaling_law | no | ready | none | Execute the first reopened TF-RD-009 upper-family gate row at `d_icl=192`, `sandwich_layers=7`, selected by the deterministic post-`#257` validation `L(N,S)` information-gain design pass. | Execute as order 1 in `tf_rd_009_width_depth_upper_extension_medium_v1`, then expand only after health=`ok`. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1` | classification_scaling_law | no | ready | none | Execute the second reopened TF-RD-009 upper-family gate row at `d_icl=208`, `sandwich_layers=8`, continuing the winning post-`#257` information-first continuation. | Execute as order 2 in `tf_rd_009_width_depth_upper_extension_medium_v1`, then expand only after health=`ok`. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1` | classification_scaling_law | no | ready | none | Execute the third reopened TF-RD-009 upper-family gate row at `d_icl=224`, `sandwich_layers=9`, continuing the winning post-`#257` information-first continuation. | Execute as order 3 in `tf_rd_009_width_depth_upper_extension_medium_v1`, then expand only after health=`ok`. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1` | classification_scaling_law | no | ready | none | Execute the reopened TF-RD-009 near-ceiling gate row at `d_icl=248`, `sandwich_layers=10`, selected as the largest member of the winning post-`#257` information-first continuation. | Execute as order 4 in `tf_rd_009_width_depth_upper_extension_medium_v1`, then expand only after health=`ok`. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first reopened TF-RD-009 upper-family gate row at `d_icl=192`, `sandwich_layers=7`, selected by the deterministic post-`#257` validation `L(N,S)` information-gain design pass.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_upper_extension_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 deliberately reopens the fixed-budget upper family after the corrected `#257` large-rung gate froze a lower-memory hardware model. This row belongs to the winning continuation `192x7 -> 208x8 -> 224x9 -> 248x10`, chosen by D-optimal information gain on the current validation `L(N,S)` fit rather than by direct model chasing.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the corrected headroom can support a wider reopened upper family, `192x7` should widen the validation `L(N,S)` design space at the carried matched-budget row while preserving the fixed-contract architecture surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `46d064ae3bf0dd14e6c9f5852f042f97b3c2e7d50fb84def704b04c18c86594b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 192, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 7, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only at the carried fixed-budget gate row before spending any NS-ladder budget on reopened upper geometries.
  - Promote to `tf_rd_009_ns_upper_extension_medium_v1` only if benchmark-backed and health=`ok`; keep health=`warn` as upper-family evidence only.
  - Use this row for validation `L(N,S)` information gain even if it is weaker than `152x5` on `final_log_loss_at_matched_regime_budget`, and do not freeze a new preferred baseline without a fresh large-rung validation gate.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - upper-family gate-first policy before any reopened NS ladder spend
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_upper_extension_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers7_upper_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second reopened TF-RD-009 upper-family gate row at `d_icl=208`, `sandwich_layers=8`, continuing the winning post-`#257` information-first continuation.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_upper_extension_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 reopens the upper family under the corrected medium-evidence parameter and VRAM fits, then spends compute only on rows selected by the deterministic validation `L(N,S)` design helper.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: `208x8` should probe whether a deeper upper row remains healthy enough to improve law information before the reopened branch reaches the near-40 GB regime.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4cabd07d4a7d0a33bdf95a38d4919791a3f417a31b81bd1850101ba4892c182a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 208, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 8, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only at the carried fixed-budget gate row before spending any NS-ladder budget on reopened upper geometries.
  - Promote to `tf_rd_009_ns_upper_extension_medium_v1` only if benchmark-backed and health=`ok`; keep health=`warn` as upper-family evidence only.
  - Use this row for validation `L(N,S)` information gain even if it is weaker than `152x5` on `final_log_loss_at_matched_regime_budget`, and do not freeze a new preferred baseline without a fresh large-rung validation gate.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - upper-family gate-first policy before any reopened NS ladder spend
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_upper_extension_medium_v1/delta_tf_rd_009_cls_sandwich_dicl208_layers8_upper_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the third reopened TF-RD-009 upper-family gate row at `d_icl=224`, `sandwich_layers=9`, continuing the winning post-`#257` information-first continuation.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_upper_extension_medium_v1`.
- Hypothesis: 
- Upstream delta: This row exists because the corrected frozen VRAM fit showed materially more headroom than the original `176x6` branch assumed, so TF-RD-009 now reopens the upper family by explicit validation-law design rather than by ad hoc capacity chasing.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: `224x9` should test whether the reopened upper family keeps adding validation `L(N,S)` curvature information near the corrected medium-rung ceiling.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `92e42be8b60262f4b42b770d3fda35c1321f3a997f63cee6dac9ffedcbf0f3ae`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 224, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 9, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only at the carried fixed-budget gate row before spending any NS-ladder budget on reopened upper geometries.
  - Promote to `tf_rd_009_ns_upper_extension_medium_v1` only if benchmark-backed and health=`ok`; keep health=`warn` as upper-family evidence only.
  - Use this row for validation `L(N,S)` information gain even if it is weaker than `152x5` on `final_log_loss_at_matched_regime_budget`, and do not freeze a new preferred baseline without a fresh large-rung validation gate.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - upper-family gate-first policy before any reopened NS ladder spend
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_upper_extension_medium_v1/delta_tf_rd_009_cls_sandwich_dicl224_layers9_upper_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the reopened TF-RD-009 near-ceiling gate row at `d_icl=248`, `sandwich_layers=10`, selected as the largest member of the winning post-`#257` information-first continuation.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_upper_extension_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 no longer stops at `176x6` by policy; it now intentionally reopens the upper family under the corrected frozen hardware model, but still gates every new geometry at the carried fixed-budget row before spending full NS-ladder budget.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: `248x10` should act as the reopened near-ceiling evidence row for the corrected `~40 GB` target while still participating in the same deterministic validation-law design set as the lower reopened rows.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d70c1476270f09f7b7c5c371f255a4d3bd0075848d99779e4fb139cf5e7e6d51`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 248, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 10, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only at the carried fixed-budget gate row before spending any NS-ladder budget on reopened upper geometries.
  - Promote to `tf_rd_009_ns_upper_extension_medium_v1` only if benchmark-backed and health=`ok`; keep health=`warn` as upper-family evidence only.
  - Use this row for validation `L(N,S)` information gain even if it is weaker than `152x5` on `final_log_loss_at_matched_regime_budget`, and do not freeze a new preferred baseline without a fresh large-rung validation gate.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - upper-family gate-first policy before any reopened NS ladder spend
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_upper_extension_medium_v1/delta_tf_rd_009_cls_sandwich_dicl248_layers10_upper_v1/result_card.md`
- Benchmark metrics: pending
