# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_021b_sandwich_feature_removal_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_021b_sandwich_feature_removal_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_021b_sandwich_feature_removal_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_021b_sandwich_width_capacity_sensitivity_v1`
- Complexity level: `binary_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_021b_sandwich_feature_removal_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `afcee716686a111ebb8863f67375080dda692a2dd7ac0a21635b2c0a28ff9ffd`

## Locked Surface

- Anchor run id: `tf_rd_021b_hybrid_full_cell_compact_prior_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_hybrid_prior`
- Training config profile: `cls_benchmark_sandwich_hybrid_prior`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4672`, final Brier score `0.3072`, best ROC AUC `0.7370`, final ROC AUC `0.7370`, final training time `470.5s`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `openml_binary_medium` (10 tasks) with data surface label `prior_dump`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_021b_sandwich_selfattn0_v1` | architecture_sensitivity | yes | completed | none | Remove latent self-attention refinement entirely between cross-attention reads while keeping the compact control otherwise fixed. | Completed in the four-row TF-RD-021B feature-removal screen; keep the compact hybrid anchor and do not promote this row. |
| 2 | `delta_tf_rd_021b_sandwich_ffexp1_v1` | architecture_sensitivity | yes | completed | none | Reduce the hybrid sandwich feed-forward expansion from 2x to 1x while keeping the compact control otherwise fixed. | Completed in the four-row TF-RD-021B feature-removal screen; keep the compact hybrid anchor and do not promote this row. |
| 3 | `delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1` | architecture_sensitivity | yes | completed | none | Remove latent self-attention refinement and collapse feed-forward expansion to 1x while keeping the compact control otherwise fixed. | Completed in the four-row TF-RD-021B feature-removal screen; keep the compact hybrid anchor and do not promote this row. |
| 4 | `delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1` | architecture_sensitivity | yes | completed | none | Remove latent self-attention refinement, collapse feed-forward expansion to 1x, and reduce summary-token multiplicity to 1 while keeping the compact control otherwise fixed. | Completed in the four-row TF-RD-021B feature-removal screen; keep the compact hybrid anchor and do not promote this row. |

## Detailed Rows

### 1. `delta_tf_rd_021b_sandwich_selfattn0_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove latent self-attention refinement entirely between cross-attention reads while keeping the compact control otherwise fixed.
- Rationale: Replace the earlier self-attention-depth shrink row with a true removal-first read on the locked compact hybrid control.
- Hypothesis: If stage-0 full-cell access and the final readout carry most of the gain, setting `sandwich_self_attention_per_cross=0` should be only weakly harmful.
- Upstream delta: The hybrid successor added repeated latent self-refinement between cross-attention segments; this row tests whether that feature can be removed outright.
- Anchor delta: Keep the compact hybrid control fixed and change only `sandwich_self_attention_per_cross` from `4` to `0`.
- Expected effect: If the structural gain comes from stage-0 full-cell access and dual readout more than from latent recycling, removing the self-attention stack may be only weakly harmful.
- Effective labels: model=`tabfoundry_sandwich`, data=`legacy_prior`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `904772e82e0ebe000e3709d7767e2eb4742446728131e1c39e539e5879cf7195`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 0, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Treat this as the removal-first replacement for the earlier self-attention-depth ablation.
  - If the drop is weak, prefer removing this feature entirely rather than retaining a reduced nonzero count.
- Adequacy knobs to dimension explicitly:
  - Keep stage count, latent count, and FF expansion fixed so this row isolates removal of the latent self-refinement feature.
  - Compare directly against the compact control before collapsing self-attention depth into later frozen defaults.
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_feature_removal_v1_01_delta_tf_rd_021b_sandwich_selfattn0_v1_v1`.
  - This row underperformed the locked sweep anchor on final log loss and final ROC AUC; keep the compact hybrid anchor and do not promote this row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_feature_removal_v1/delta_tf_rd_021b_sandwich_selfattn0_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_feature_removal_v1_01_delta_tf_rd_021b_sandwich_selfattn0_v1_v1` with final log loss `0.4865`, delta final log loss `+0.0193`, final Brier score `0.3220`, delta final brier score `+0.0148`, final ROC AUC `0.7043`, delta final roc auc `-0.0328`, best ROC AUC `0.7078`, delta final training time `-53.2s`

### 2. `delta_tf_rd_021b_sandwich_ffexp1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce the hybrid sandwich feed-forward expansion from 2x to 1x while keeping the compact control otherwise fixed.
- Rationale: Measure the smallest currently supported retained feed-forward setting before combining it with wider removal candidates.
- Hypothesis: If extra trunk MLP capacity is mostly surplus on the compact hybrid control, collapsing `sandwich_ff_expansion` from `2` to `1` should remain bounded.
- Upstream delta: FF expansion is one of the simplest capacity axes to simplify before introducing broader scaling laws.
- Anchor delta: Keep the compact hybrid control fixed and change only `sandwich_ff_expansion` from `2` to `1`.
- Expected effect: If most of the current gain comes from attention structure rather than trunk MLP capacity, shrinking FF expansion should only weakly degrade performance.
- Effective labels: model=`tabfoundry_sandwich`, data=`legacy_prior`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `11d5de64b295e49f83b5e4883247b1ca211e22f1172e317f7aa280811714c49f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 1, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Treat this as the minimal retained FF form for the removal-first follow-up.
  - Do not add `sandwich_ff_expansion=0` in this pass; freeze work can only choose between the anchor and the supported `=1` contraction.
- Adequacy knobs to dimension explicitly:
  - Keep latent count, stage count, and summary-token count fixed so this row isolates FF capacity.
  - Compare directly against the compact control before treating FF expansion as a live scaling axis.
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_feature_removal_v1_02_delta_tf_rd_021b_sandwich_ffexp1_v1_v1`.
  - This row underperformed the locked sweep anchor on final log loss and final ROC AUC; keep the compact hybrid anchor and do not promote this row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_feature_removal_v1/delta_tf_rd_021b_sandwich_ffexp1_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_feature_removal_v1_02_delta_tf_rd_021b_sandwich_ffexp1_v1_v1` with final log loss `0.4932`, delta final log loss `+0.0260`, final Brier score `0.3289`, delta final brier score `+0.0217`, final ROC AUC `0.7082`, delta final roc auc `-0.0288`, best ROC AUC `0.7084`, delta final training time `-8.7s`

### 3. `delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove latent self-attention refinement and collapse feed-forward expansion to 1x while keeping the compact control otherwise fixed.
- Rationale: Test the first compound removal package after the isolated self-attention removal row while keeping summary-token multiplicity fixed.
- Hypothesis: If the hybrid path is structurally adequate already, combining self-attention removal with `sandwich_ff_expansion=1` may preserve most of the fit while simplifying the trunk materially.
- Upstream delta: After the completed single-knob screen, this row tests whether the simplest acceptable parent can remove the self-refinement feature and retain only minimal trunk MLP expansion.
- Anchor delta: Keep the compact hybrid control fixed and change `sandwich_self_attention_per_cross` from `4` to `0` plus `sandwich_ff_expansion` from `2` to `1`.
- Expected effect: If the hybrid path is structurally adequate already, combining self-attention removal with 1x FF expansion may preserve most of the fit while simplifying the trunk materially.
- Effective labels: model=`tabfoundry_sandwich`, data=`legacy_prior`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5a2023d615a0689e95b4162e6ba033ddd3cf9945ba14b0e0646f77298cc49624`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 1, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 0, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Use this as the first compound removal candidate after the isolated removal row.
  - Prefer this row over the anchor only if the combined simplification remains bounded on final quality and stability.
- Adequacy knobs to dimension explicitly:
  - Keep summary-token multiplicity fixed so this row reads removal of latent refinement plus minimal FF capacity only.
  - Compare directly against the compact control before carrying this compound simplification into harder-surface work.
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_feature_removal_v1_03_delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1_v1`.
  - This row underperformed the locked sweep anchor on final log loss and final ROC AUC; keep the compact hybrid anchor and do not promote this row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_feature_removal_v1/delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_feature_removal_v1_03_delta_tf_rd_021b_sandwich_selfattn0_ffexp1_v1_v1` with final log loss `0.5194`, delta final log loss `+0.0522`, final Brier score `0.3492`, delta final brier score `+0.0420`, final ROC AUC `0.6828`, delta final roc auc `-0.0542`, best ROC AUC `0.6832`, delta final training time `-64.5s`

### 4. `delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove latent self-attention refinement, collapse feed-forward expansion to 1x, and reduce summary-token multiplicity to 1 while keeping the compact control otherwise fixed.
- Rationale: Test the smallest currently supported removal-first parent before any manual freeze decision is made.
- Hypothesis: If the hybrid architecture mainly needs the stage-0 full-cell path and the final cell-stream readout, this compound simplification may preserve bounded quality with the smallest remaining nonzero summary and FF settings.
- Upstream delta: This is the smallest removal-first parent expressible on the current sandwich surface without adding new zero-valued FF or summary-token semantics.
- Anchor delta: Keep the compact hybrid control fixed and change `sandwich_self_attention_per_cross` from `4` to `0`, `sandwich_ff_expansion` from `2` to `1`, and `sandwich_summary_tokens_per_axis` from `4` to `1`.
- Expected effect: If the hybrid architecture mainly needs the stage-0 full-cell path and the final cell-stream readout, this compound simplification may preserve bounded quality with the smallest remaining nonzero summary/FF settings.
- Effective labels: model=`tabfoundry_sandwich`, data=`legacy_prior`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `cf7ed650a7477adab8d4b2f2e38c9b5ebfbdaaec8bafd208b8fe906d98fe6764`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 1, 'sandwich_summary_tokens_per_axis': 1, 'sandwich_self_attention_per_cross': 0, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Use this as the smallest currently supported parent in the TF-RD-021B removal-first screen.
  - If this row is close enough to the anchor, the later freeze should prefer it over larger retained-feature variants.
- Adequacy knobs to dimension explicitly:
  - Treat this as the lower bound of the current expressible removal-first simplification package.
  - Compare directly against both the anchor and the simpler compound row before freezing any defaults.
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_feature_removal_v1_04_delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1_v1`.
  - This was the smallest simplification package and the fastest row in the sweep, but it still underperformed the locked sweep anchor; keep the compact hybrid anchor and do not promote this row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_feature_removal_v1/delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_feature_removal_v1_04_delta_tf_rd_021b_sandwich_selfattn0_ffexp1_summarytokens1_v1_v1` with final log loss `0.4947`, delta final log loss `+0.0275`, final Brier score `0.3269`, delta final brier score `+0.0197`, final ROC AUC `0.7053`, delta final roc auc `-0.0318`, best ROC AUC `0.7042`, delta final training time `-155.5s`
