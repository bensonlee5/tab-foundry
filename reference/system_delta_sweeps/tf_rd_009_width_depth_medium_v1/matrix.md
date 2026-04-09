# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_width_depth_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_transfer_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `70ba02580c81e5644dd0d4a1e757264cac20b40870315985baf093d1579d6798`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest local id `openml_classification_medium_v1` sourced from `nanotabpfn_openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=88`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor-equivalent effective size under the repo-local bridge `S(d, L) = L * d^2`. | Execute as order 1 in `tf_rd_009_width_depth_medium_v1` against the carried `96x2` baseline. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl104_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=104`, `sandwich_layers=3`, derived by matching the effective size of the width-only upper evidence row `128x2` under the repo-local bridge `S(d, L) = L * d^2`. | Execute as order 2 in `tf_rd_009_width_depth_medium_v1` after the lower diagonal row is benchmark-backed. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=88`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor-equivalent effective size under the repo-local bridge `S(d, L) = L * d^2`.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then converts that into an exact repo-local row through the measured sandwich size bridge `P_local(d, L) ≈ 89 * L * d^2`.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the sandwich target can trade one layer for more width while preserving anchor-equivalent effective size, `88x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `97bec332be54c42ac85bd41275e3f270b9f8a889838c804692b495184ed40933`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 88, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after `d_icl=96`, `sandwich_layers=2` is established as the carried width-only baseline for `#255`.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Report relative to the formal `60x2` TF-RD-009 anchor, but interpret this row inside the diagonal family `{88x1, 96x2, 104x3}` rather than as a standalone keep/reject claim.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl104_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=104`, `sandwich_layers=3`, derived by matching the effective size of the width-only upper evidence row `128x2` under the repo-local bridge `S(d, L) = L * d^2`.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl104_layers3_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the first upper joint probe allowed by the literature-backed width-depth co-design rule without extrapolating to a non-derived `128x4` regime.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl104_layers3_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `104x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aa83bafa62db2d5a3a67fa7361fa50aa9ca6f58e5c51d4019dbceba96b28b36d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 104, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the lower diagonal `88x1` row so the first TF-RD-009 joint family remains an ordered diagonal rather than a grid.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - If benchmark-backed but health=`warn`, keep the row as evidence and do not silently replace it with a different upper geometry in the same branch.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl104_layers3_v1/result_card.md`
- Benchmark metrics: pending
