# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_width_depth_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_transfer_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_width_depth_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `a3281a4aab25bc048ac8ad4900594c94c81c9f5778a439592376be0f59c4fa8c`

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
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Execute as order 1 in `tf_rd_009_width_depth_medium_v1` against the carried `96x2` baseline. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Execute as order 2 in `tf_rd_009_width_depth_medium_v1` after the lower diagonal row is benchmark-backed. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface. | Execute as order 3 in `tf_rd_009_width_depth_medium_v1` after `112x3` is benchmark-backed. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | ready | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Execute as order 4 in `tf_rd_009_width_depth_medium_v1` after `128x4` is benchmark-backed. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` | classification_scaling_law | no | ready | none | Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family. | Execute as order 5 in `tf_rd_009_width_depth_medium_v1` after `152x5` is benchmark-backed; treat failure as ceiling evidence. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the carried `96x2` baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5341a2ba42e1ee093f6f26393bcaf1f441d4064d5888e84098357e550e2e345d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after `d_icl=96`, `sandwich_layers=2` is established as the carried width-only baseline for `#255`; this row is the empirical depth-aware parameter bridge's lower anchor-equivalent probe.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Report relative to the formal `60x2` TF-RD-009 anchor, but interpret this row inside the diagonal family `{72x1, 96x2, 112x3, 128x4, 152x5, 176x6}` rather than as a standalone keep/reject claim.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the fixed-budget law continues above the carried `96x2` baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `21b1145f4e210135482236978601819e7b08b53ed009e3a63f9996766025c4b3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the lower diagonal `72x1` row so the first TF-RD-009 joint family remains an ordered diagonal around the width-only `128x2` upper evidence target rather than a grid.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first post-seed upper TF-RD-009 joint width-depth row at `d_icl=128`, `sandwich_layers=4`, extending the diagonal family by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe on the frozen sandwich surface.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 keeps the literature-backed width-depth co-design shape, then extends the upper ladder by log-spacing predicted parameter scale between the `112x3` seed and the intended `176x6` ceiling probe under the empirical depth-aware bridge.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the joint law remains smooth beyond the first upper seed, `128x4` should continue the matched-budget trend without consuming an outsized share of the `rtx8000_44gb` VRAM budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d433732226811e802ad786a88f152a37fa4b3027bb94f09fbe5468a0f5b156cf`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after `112x3` is benchmark-backed so the broadened family preserves an ordered dense diagonal.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Treat this row as an intermediate reported-fit point once benchmark-backed, not a final keep/reject decision in isolation.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `846853cba0c44d719d5ace101c3f6d8c8ae35eee4b1ae8c3a6c5e1c4e95220e9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after `128x4` is benchmark-backed so the family remains a dense diagonal rather than a sparse extrapolation.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Keep any health=`warn` outcome as explicit ceiling evidence rather than silently replacing this row.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the intended TF-RD-009 ceiling probe at `d_icl=176`, `sandwich_layers=6`, chosen to land near the retained `rtx8000_44gb` surface's `32-33 GB` reserved-memory target while staying on the same dense diagonal family.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` against anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` for sweep `tf_rd_009_width_depth_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats this as an intentional hardware-ceiling probe derived from the empirical sandwich parameter bridge and the carried RTX 8000 VRAM fit rather than a paper-claimed closed-form exponent. The final reported law is still fit later on measured benchmark-registry `model_size.total_params`.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1` against locked anchor `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
- Expected effect: `176x6` should either extend the matched-budget law into the near-saturation regime or fail cleanly enough to mark the first medium-rung hardware ceiling on the carried runtime surface.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cc53a2c49b417ab63a448bc18941816a42df63c08e60ea3bf443932d6f24a262`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 176, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 6, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute last as the explicit `rtx8000_44gb` ceiling probe for the first broadened TF-RD-009 family; this rounded row is chosen to land near the empirical `32-33 GB` reserved-memory target rather than the older underfit `144x6` estimate.
  - Compare directly against the carried `96x2` baseline at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - If the row fails before benchmark registration, keep that failure as valid ceiling evidence and leave `#255` open rather than substituting a new upper row in the same branch.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1/result_card.md`
- Benchmark metrics: pending
