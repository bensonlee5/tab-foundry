# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_024_classification_heads_prerow_followup_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_024_classification_heads_prerow_followup_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_024_classification_heads_prerow_followup_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_024_classification_knob_sweep_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_024_classification_heads_prerow_followup_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `f17ed01ad4d49b7eac5312bf06300f073427e3f565c58b100ac3790e4d5ae13a`

## Locked Surface

- Anchor run id: `sd_tf_rd_024_classification_knob_sweep_v1_anchor_compile_eager_dynamic_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_architecture_followup`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1107`, final BPF `2.1107`, final log loss `0.6820`, final Brier score `0.4226`, best ROC AUC `0.6091`, final ROC AUC `0.6091`, final training time `4657.1s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| attention partitioning | Transformer head count changes factorization without changing total width. | The settled compile-eager-dynamic anchor uses `sandwich_heads=4`, and the completed seven-row medium screen found `sandwich_heads=2` to be the best result so far. | `sandwich_heads=1` asks whether the heads=2 win reflects a broader low-head regime or a narrow local optimum. |
| row pre-mixer depth | The sandwich adds a row-wise feature mixer before the first latent read; TF-RD-021B showed that removing it is materially harmful. | The settled compile-eager-dynamic anchor uses `sandwich_pre_row_attention_layers=1`. | `sandwich_pre_row_attention_layers=2` asks whether the row pre-mixer is still capacity-limited before scaling begins. |
| benchmark contract | Not applicable. | Closed TF-RD-010 medium contract with `cls_benchmark_linear_multiclass_medium_v1` as the control baseline. | Medium-only is sufficient here because this follow-up exists to choose the pre-scaling family, not to reopen TF-RD-024 large validation. |
| training and runtime surface | No external paper fixes this exact prior-dump or runtime policy. | Training experiment `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1` with the kept TF-RD-022 compile-eager-dynamic runtime policy. | Hold optimizer, schedule, batching, corpus, and runtime policy fixed so these two rows isolate topology only. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_024_followup_cls_sandwich_heads1_v1` | architecture_followup | no | completed | none | Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface. | Carry `sandwich_heads=1` into TF-RD-009 as the pre-scaling family winner, and keep the remaining non-scaling knobs frozen unless the scaling study reopens them. |
| 2 | `delta_tf_rd_024_followup_cls_sandwich_prerow2_v1` | architecture_followup | no | completed | none | Extend the row pre-mixer family by increasing `sandwich_pre_row_attention_layers` from `1` to `2` on the inherited multiclass benchmark surface. | Keep `sandwich_pre_row_attention_layers=1` for TF-RD-009; this deeper pre-row mixer improved over the anchor but did not beat the `sandwich_heads=2` or `sandwich_heads=1` follow-up winners. |

## Detailed Rows

### 1. `delta_tf_rd_024_followup_cls_sandwich_heads1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface.
- Rationale: Follow the completed TF-RD-024 medium winner `sandwich_heads=2` with one lower head-partition bracket to test whether the gain reflects a broader low-head regime before TF-RD-009 freezes topology.
- Hypothesis: If the `sandwich_heads=2` win reflects excess head factorization rather than a narrow local optimum, reducing `sandwich_heads` from `4` to `1` may stay competitive or improve further on the closed medium contract.
- Upstream delta: Extends the TF-RD-024 attention-head family one bracket lower after `sandwich_heads=2` won the initial medium screen.
- Anchor delta: Changes `model.sandwich_heads` from `4` to `1` while holding `d_icl=60`, `sandwich_layers=2`, `sandwich_latents=24`, the compile-eager-dynamic runtime surface, and the closed TF-RD-010 medium benchmark contract fixed.
- Expected effect: If the medium gain from the lower-head bracket reflects excess head factorization rather than a narrow `2`-head sweet spot, `sandwich_heads=1` may remain competitive or improve further on the closed medium contract.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0b3be75f60d2d62aa06d1f7efcfa44229dc3a32efa606b43230e2c3b1a807252`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute as an independent follow-up against the fresh compile-eager-dynamic anchor instead of stacking on top of `sandwich_heads=2`.
  - Carry the lower head count into TF-RD-009 only if it beats the current best medium result from `sandwich_heads=2`.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime policy surface
  - attention partitioning only; no width, depth, batching, or optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical medium-only follow-up completed on the inherited compile-eager-dynamic runtime surface.
  - `sandwich_heads=1` finished at `final_log_loss=0.6603575333`, beating the fresh compile anchor (`0.6820309591`) and the prior best TF-RD-024 row `sandwich_heads=2` (`0.6762878243`).
  - Treat this row as the TF-RD-024 closeout winner and the direct scaling handoff surface for TF-RD-009.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_heads_prerow_followup_v1/delta_tf_rd_024_followup_cls_sandwich_heads1_v1/result_card.md`
- Benchmark metrics:
  - Best log loss: `0.7033` (step 1150.0)
  - Final log loss: `0.6604`
  - Final Brier score: `0.4112`
  - Final ROC AUC: `0.6362`
  - Drift (final − best): `-0.0429`
  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows.
  - Final BPC (legacy feature-cell diagnostic): `2.1937`
  - Final BPF (legacy feature-cell diagnostic): `2.1936`

### 2. `delta_tf_rd_024_followup_cls_sandwich_prerow2_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Extend the row pre-mixer family by increasing `sandwich_pre_row_attention_layers` from `1` to `2` on the inherited multiclass benchmark surface.
- Rationale: Probe the most plausible topology expansion suggested by the earlier sandwich ablations without reopening a broad search: deepen only the row pre-mixer that was clearly harmful to remove in TF-RD-021B.
- Hypothesis: If the row-wise pre-Perceiver mixer is still capacity-limited on the current medium contract, increasing `sandwich_pre_row_attention_layers` from `1` to `2` may improve benchmark quality enough to beat the current best `sandwich_heads=2` row before scaling.
- Upstream delta: Extends the historical TF-RD-021B row pre-mixer family upward after the `prerow=0` ablation showed that row-wise feature mixing is materially useful.
- Anchor delta: Changes `model.sandwich_pre_row_attention_layers` from `1` to `2` while holding `d_icl=60`, `sandwich_layers=2`, `sandwich_heads=4`, the compile-eager-dynamic runtime surface, and the closed TF-RD-010 medium benchmark contract fixed.
- Expected effect: If the row-wise pre-Perceiver mixer is still capacity-limited on the current multiclass contract, the deeper row pre-mixer bracket may improve medium benchmark fit enough to justify the scaling handoff.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `3622a8d310525faf142d32a42055dc79e98823a7e305a274adcde9f9add45eda`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 2, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute as an independent topology probe against the fresh compile-eager-dynamic anchor instead of stacking with `sandwich_heads=1`.
  - Carry the deeper row pre-mixer into TF-RD-009 only if it beats the current best medium result from `sandwich_heads=2`.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime policy surface
  - row pre-mixer depth only; no compounded head, width, or latent changes
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical medium-only follow-up completed on the inherited compile-eager-dynamic runtime surface.
  - `sandwich_pre_row_attention_layers=2` finished at `final_log_loss=0.6780725432`, improving over the fresh compile anchor (`0.6820309591`) but remaining worse than `sandwich_heads=2` (`0.6762878243`) and `sandwich_heads=1` (`0.6603575333`).
  - Keep the pre-row mixer at `1` for TF-RD-009 and treat this row as bounded positive-but-not-promoted evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_heads_prerow_followup_v1/delta_tf_rd_024_followup_cls_sandwich_prerow2_v1/result_card.md`
- Benchmark metrics:
  - Best log loss: `0.7084` (step 1575.0)
  - Final log loss: `0.6781`
  - Final Brier score: `0.4222`
  - Final ROC AUC: `0.6253`
  - Drift (final − best): `-0.0304`
  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows.
  - Final BPC (legacy feature-cell diagnostic): `2.1804`
  - Final BPF (legacy feature-cell diagnostic): `2.1804`
