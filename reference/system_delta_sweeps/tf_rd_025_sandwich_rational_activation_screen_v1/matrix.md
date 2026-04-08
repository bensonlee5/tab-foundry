# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_025_sandwich_rational_activation_screen_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_025_sandwich_rational_activation_screen_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_025_sandwich_rational_activation_screen_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_025_sandwich_rational_activation_screen_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `8243b280951c0002cb644f8c9c886c72638c3165b8881266924bd50e0923ef04`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- Surface role: `classification_activation_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1136`, final BPF `2.1136`, final log loss `0.6812`, final Brier score `0.4229`, best ROC AUC `0.6094`, final ROC AUC `0.6094`, final training time `7449.8s`

## Anchor Comparison

Upstream reference: `Rational-activation transformer theory` from `https://arxiv.org/abs/2602.12390`.

| Dimension | Upstream Rational-activation transformer theory | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| internal block normalization | The theory paper does not prescribe this repo-local sandwich block-norm split directly. | The carried TF-RD-010 medium anchor keeps sandwich internal block norms enabled through `sandwich_block_norm=layernorm`. | Row `01` isolates whether removing the internal sandwich block norms is a viable control surface before the rational swap is read. |
| activation family | The paper argues trainable rational activations can improve transformer optimization and expressivity. | The carried TF-RD-010 medium anchor uses GELU throughout the sandwich core FF blocks. | Row `02` reads the local version-A `5/4` rational activation only after the norm-free GELU control is defined. |
| corpus and validation contract | Not applicable. | Keep the manifest-backed dagzoo control corpus `tf_rd_010_dagzoo_medium_control_curated_v5` and the medium classification validation manifest fixed. | This sweep does not reopen corpus choice, missingness, or the real-data benchmark bundle. |
| runtime surface | The paper does not define this repo's prior-dump runtime policy. | Inherit the TF-RD-022 sandwich benchmark training surface, but force CPU execution with `mixed_precision="no"` for this screen. | This lane answers whether the activation change is promising enough to earn a later benchmark-rerun, not whether CPU is the final runtime policy. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_model_sandwich_block_norm_none_v1` | activation_followup | no | screened | none | Disable the internal sandwich block pre-norm modules while keeping the global `norm_type=layernorm` contract fixed on the TF-RD-010 multiclass benchmark surface. | Execute first, then use it as the norm-free GELU control for row `02`. |
| 2 | `delta_model_sandwich_activation_rational_v1` | activation_followup | no | screened | none | Starting from the norm-free sandwich control surface, replace GELU with the local version-A `5/4` GELU-initialized rational activation. | Execute after row `01`, then decide whether the rational row earns a benchmark-facing rerun. |

## Detailed Rows

### 1. `delta_model_sandwich_block_norm_none_v1`

- Dimension family: `model`
- Status: `screened`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Disable the internal sandwich block pre-norm modules while keeping the global `norm_type=layernorm` contract fixed on the TF-RD-010 multiclass benchmark surface.
- Rationale: Establish the norm-free GELU control row on the manifest-backed sandwich benchmark surface before interpreting any rational-activation change.
- Hypothesis: If internal sandwich block norms are partly limiting this family, removing them while keeping GELU should remain trainable enough on the TF-RD-010 medium contract to serve as the rational screen control.
- Upstream delta: This is a repo-local norm-free sandwich screen that uses the new `sandwich_block_norm=none` surface without reopening the rest of the benchmark contract.
- Anchor delta: Starting from the carried TF-RD-010 anchor, switch only the sandwich-internal block norm from `layernorm` to `none` on the CPU no-AMP screening surface.
- Expected effect: Removing the internal block norms may improve the sandwich family's core FF routing on this surface, but it also risks a training-stability regression.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3f3a61fbdd80abfd87d2c6da3e7101f30535887bd0babd2169ac57d0fa5cbc98`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_activation': 'gelu', 'sandwich_block_norm': 'none', 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Run this row first as the norm-free control for the activation screen.
  - Compare directly against the carried anchor on stability, runtime, and final validation metrics before giving the rational row any credit.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract on curated `tf_rd_010_dagzoo_medium_control_curated_v5`
  - CPU no-AMP screening surface inherited from the TF-RD-022 runtime-policy contract
  - internal sandwich block norm only; activation family, width, depth, optimizer, and corpus remain frozen
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `screened`
- Decision: `defer`
- Notes:
  - Train-only screen recorded as `sd_tf_rd_025_sandwich_rational_activation_screen_v1_01_delta_model_sandwich_block_norm_none_v1_v2`.
  - Recovered from the completed norm-free GELU control training artifacts after correcting TF-RD-025 to `screen_only`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_025_sandwich_rational_activation_screen_v1/delta_model_sandwich_block_norm_none_v1/result_card.md`
- Screen metrics:
  - Clipped-step fraction: `0.0000`
  - Final train-loss EMA: `1.4228`
- Benchmark metrics: pending

### 2. `delta_model_sandwich_activation_rational_v1`

- Dimension family: `model`
- Status: `screened`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Starting from the norm-free sandwich control surface, replace GELU with the local version-A `5/4` GELU-initialized rational activation.
- Rationale: Replace GELU with the local version-A rational activation only after the norm-free sandwich control is defined on the same CPU screening surface.
- Hypothesis: If the rational activation is a real win for sandwich core FF blocks, row `02` should match or beat the norm-free GELU control on final validation metrics without introducing optimizer or stability failures.
- Upstream delta: Reuses the repo-local rational activation module on the current TF-RD-010 multiclass sandwich contract rather than adding a separate dependency or reopening the wider architecture ladder.
- Anchor delta: Starting from row `01`, switch only `model.sandwich_activation` from `gelu` to `rational` while keeping `sandwich_block_norm=none`, the curated dagzoo corpus, and the CPU no-AMP runtime fixed.
- Expected effect: The rational activation may preserve more useful curvature than GELU on the norm-free sandwich surface, with the main risk being optimizer or stability regressions.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `775964a7de0772b6e402ceb561727f0667139fd78270f9f7611cf9ee4974a018`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_activation': 'rational', 'sandwich_block_norm': 'none', 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only after row `01` defines the norm-free GELU control on the same surface.
  - Compare directly against both row `01` and the carried anchor, and only recommend a later benchmark rerun if the rational row is clearly non-worse on quality while remaining stable.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract on curated `tf_rd_010_dagzoo_medium_control_curated_v5`
  - CPU no-AMP screening surface inherited from the TF-RD-022 runtime-policy contract
  - activation family only on top of `sandwich_block_norm=none`; width, depth, optimizer, and corpus remain frozen
- Execution policy: `screen_only`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `screened`
- Decision: `defer`
- Notes:
  - Train-only screen recorded as `sd_tf_rd_025_sandwich_rational_activation_screen_v1_02_delta_model_sandwich_activation_rational_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_025_sandwich_rational_activation_screen_v1/delta_model_sandwich_activation_rational_v1/result_card.md`
- Screen metrics:
  - Clipped-step fraction: `0.0000`
  - Final train-loss EMA: `1.4326`
- Benchmark metrics: pending
