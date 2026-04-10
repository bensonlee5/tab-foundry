# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_width_transfer_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_width_transfer_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_width_transfer_medium_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_009_anchor_replay_heads1_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_width_transfer_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `c9c54fabf0c0e969008b2561a35520d34a0d47dd78062058b7d1cf0c6a39381e`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1920`, final BPF `2.1920`, final log loss `0.6620`, final Brier score `0.4130`, best ROC AUC `0.5840`, final ROC AUC `0.6347`, final training time `8466.6s`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Local benchmark-manifest id `openml_classification_medium_v1` sourced from `nanotabpfn_openml_classification_medium` (242 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl48_v1` | classification_scaling_law | no | completed | none | Reduce TF-RD-009 classification sandwich width from `d_icl=60` to `48` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Keep as the losing lower-width contraction reference; do not carry this row forward into #255 except as evidence that shrinking below the anchor hurts the matched-regime-budget objective. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl96_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Carry `d_icl=96` into #255 as the explicit joint width-depth baseline because it improved the matched-regime-budget objective cleanly without the instability warnings seen at `d_icl=128`. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl128_v1` | classification_scaling_law | no | completed | none | Increase TF-RD-009 classification sandwich width from `d_icl=60` to `128` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed. | Keep as upper-width evidence only; do not use as the default #255 handoff because the raw log-loss win came with elevated instability signals. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl48_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reduce TF-RD-009 classification sandwich width from `d_icl=60` to `48` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl48_v1` against anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2` for sweep `tf_rd_009_width_transfer_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the first live scaling axis after the TF-RD-024 heads1 replay establishes the benchmark-registry-backed anchor.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl48_v1` against locked anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`.
- Expected effect: If the carried heads1 classification surface is width-overprovisioned on the closed medium contract, dropping to `d_icl=48` should preserve most benchmark quality at the matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3c810b2ef117d06453e1c486bf8fe130d54c17b720c856c68571414e9d310f4d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 48, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-009 heads1 replay anchor is registered and promoted.
  - Compare directly against the replayed `d_icl=60` anchor at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Interpret this contraction only in the context of the full four-width family `{48, 60(anchor), 96, 128}`.
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
  - Canonical rerun registered as `sd_tf_rd_009_width_transfer_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1`.
  - Width-transfer probe recorded against the TF-RD-009 heads1 anchor; interpret after the full width family completes.
  - Width-family closeout: `d_icl=48` underperformed the replay anchor and remains a contraction-only reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl48_v1/result_card.md`
- Registered run: `sd_tf_rd_009_width_transfer_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1` with final log loss `0.6939`, delta final log loss `+0.0318`, final Brier score `0.4328`, delta final brier score `+0.0198`, final ROC AUC `0.5942`, delta final roc auc `-0.0406`, final BPC (legacy feature-cell diagnostic) `2.1139`, delta final bpc (legacy feature-cell diagnostic) `-0.0781`, final BPF (legacy feature-cell diagnostic) `2.1139`, delta final bpf (legacy feature-cell diagnostic) `-0.0780`, best ROC AUC `0.4951`, delta final training time `-54.1s`

### 2. `delta_tf_rd_009_cls_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `96` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl96_v1` against anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2` for sweep `tf_rd_009_width_transfer_medium_v1`.
- Hypothesis: 
- Upstream delta: TF-RD-009 treats width as the clean first transfer axis after the heads1 replay anchor is benchmark-registry-backed.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl96_v1` against locked anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`.
- Expected effect: If the carried heads1 classification surface is still width-limited on the closed medium contract, increasing to `d_icl=96` should improve benchmark fit without reopening any other sandwich knob.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `c3a7e99b88ae55fc3546d8df3241f411768ce91bf05f474f2e7d8b8653b1dbe9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-009 heads1 replay anchor is registered and promoted.
  - Compare directly against the replayed `d_icl=60` anchor at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Use the result to decide whether width-only movement remains a live TF-RD-009 family before joint width-depth work opens.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1`.
  - Width-transfer probe recorded against the TF-RD-009 heads1 anchor; interpret after the full width family completes.
  - Width-family closeout: chosen #255 handoff baseline because it improved final log loss, Brier score, and ROC AUC with health verdict `ok`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl96_v1/result_card.md`
- Registered run: `sd_tf_rd_009_width_transfer_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1` with final log loss `0.6331`, delta final log loss `-0.0289`, final Brier score `0.3914`, delta final brier score `-0.0216`, final ROC AUC `0.6716`, delta final roc auc `+0.0369`, final BPC (legacy feature-cell diagnostic) `2.3481`, delta final bpc (legacy feature-cell diagnostic) `+0.1561`, final BPF (legacy feature-cell diagnostic) `2.3481`, delta final bpf (legacy feature-cell diagnostic) `+0.1561`, best ROC AUC `0.6109`, delta final training time `+63.2s`

### 3. `delta_tf_rd_009_cls_sandwich_dicl128_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Increase TF-RD-009 classification sandwich width from `d_icl=60` to `128` while keeping the carried `sandwich_heads=1` architecture and the TF-RD-022 runtime bundle fixed.
- Rationale: Contextualize `delta_tf_rd_009_cls_sandwich_dicl128_v1` against anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2` for sweep `tf_rd_009_width_transfer_medium_v1`.
- Hypothesis: 
- Upstream delta: This is the larger-width bracket for the first TF-RD-009 width-transfer family once the carried heads1 anchor is replayed and registered.
- Anchor delta: Delta description pending for `delta_tf_rd_009_cls_sandwich_dicl128_v1` against locked anchor `sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2`.
- Expected effect: If width-only improvement remains monotone beyond `d_icl=96`, expanding to `128` should either continue the gain or show that width-only transfer is flattening before joint width-depth scaling begins.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ee32eba36994c01501e8f45d77469d2c600762ca04306744fa9ab98713fd0a92`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-009 heads1 replay anchor is registered and promoted.
  - Compare directly against the replayed `d_icl=60` anchor at matched regime budget using `final_log_loss_at_matched_regime_budget` as the primary metric.
  - Treat this row as evidence about the upper edge of the first width-only family, not as permission to retune optimizer or open a compute-optimal frontier yet.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - width only; no layers, optimizer retune, curriculum, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Confounders:
  - health verdict `warn` in run inspection
  - max_grad_norm spiked to 54.68714918577819
  - legacy feature-cell diagnostics BPC/BPF diverged sharply despite improved matched-regime-budget objective
- Notes:
  - Canonical rerun registered as `sd_tf_rd_009_width_transfer_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`.
  - Width-transfer probe recorded against the TF-RD-009 heads1 anchor; interpret after the full width family completes.
  - Width-family closeout: objective-leading row but not the chosen #255 handoff because the run carried elevated stability warnings.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_width_transfer_medium_v1/delta_tf_rd_009_cls_sandwich_dicl128_v1/result_card.md`
- Registered run: `sd_tf_rd_009_width_transfer_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1` with final log loss `0.6225`, delta final log loss `-0.0395`, final Brier score `0.3848`, delta final brier score `-0.0282`, final ROC AUC `0.6719`, delta final roc auc `+0.0372`, final BPC (legacy feature-cell diagnostic) `5.6874`, delta final bpc (legacy feature-cell diagnostic) `+3.4954`, final BPF (legacy feature-cell diagnostic) `5.6871`, delta final bpf (legacy feature-cell diagnostic) `+3.4951`, best ROC AUC `0.5850`, delta final training time `+220.2s`
