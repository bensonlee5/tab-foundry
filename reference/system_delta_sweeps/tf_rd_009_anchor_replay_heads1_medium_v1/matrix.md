# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_anchor_replay_heads1_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_anchor_replay_heads1_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_anchor_replay_heads1_medium_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_024_classification_heads_prerow_followup_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_anchor_replay_heads1_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `6bed39234067f2b4b7dcaa7fb77fdc23e7c8930a9b9d502a3be2c77a6596d003`

## Locked Surface

- Anchor run id: `sd_tf_rd_024_classification_knob_sweep_v1_anchor_compile_eager_dynamic_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1107`, final BPF `2.1107`, final log loss `0.6820`, final Brier score `0.4226`, best ROC AUC `0.6091`, final ROC AUC `0.6091`, final training time `4657.1s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| attention partitioning | TF-RD-024 showed that the lower-head bracket was the strongest remaining topology family on the closed medium contract. | Replay `sandwich_heads=1` at `d_icl=60` while freezing `sandwich_layers=2`, `head_hidden_dim=96`, `sandwich_latents=24`, `sandwich_ff_expansion=2`, `sandwich_summary_tokens_per_axis=3`, `sandwich_self_attention_per_cross=4`, `sandwich_pre_row_attention_layers=1`, `sandwich_pre_column_attention_layers=1`, `sandwich_pre_column_inducing_tokens=16`, and `feature_type_conditioning=film`. | This replay is not a new topology search; it is a registry repair step that formalizes the carried TF-RD-009 anchor. |
| benchmark contract | Not applicable. | Closed TF-RD-010 curated multiclass medium manifest with `cls_benchmark_linear_multiclass_medium_v1` as the control baseline. | Keep the medium benchmark surface fixed so the replayed anchor is directly comparable to later TF-RD-009 width rows at matched regime budget. |
| training and runtime surface | Not applicable. | Inherited TF-RD-022 compile-eager-dynamic bundle with `task_batch_size=16`, `prior_dump_batch_size=64`, `mixed_precision=bf16`, `trace_activations=false`, `activation_checkpointing=true`, `grad_accum_steps=4`, `grad_clip=0.0`, `compile_model=true`, `compile_backend=eager`, `compile_dynamic=true`, `max_steps=2500`, `optimizer.min_lr=1e-5`, and linear warmup at `lr_max=1e-3`. | Optimizer and runtime settings stay frozen here; TF-RD-009 width transfer reads the inherited bundle first and treats any retuning as later work. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_024_followup_cls_sandwich_heads1_v1` | architecture_followup | no | ready | none | Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface. | Replay and promote this row, then bootstrap `tf_rd_009_width_transfer_medium_v1` with this registered run as the anchor. |

## Detailed Rows

### 1. `delta_tf_rd_024_followup_cls_sandwich_heads1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Extend the TF-RD-024 head-partition follow-up by reducing `sandwich_heads` from `4` to `1` on the inherited multiclass benchmark surface.
- Rationale: Replay the carried TF-RD-024 `sandwich_heads=1` winner on the closed medium classification contract so TF-RD-009 starts from a benchmark-registry-backed anchor instead of a historical queue-only result.
- Hypothesis: The carried `sandwich_heads=1` surface should replay cleanly on the inherited TF-RD-022 compile-eager-dynamic bundle and remain clearly better than the original `sandwich_heads=4` compile anchor at matched regime budget.
- Upstream delta: Extends the TF-RD-024 attention-head family one bracket lower after `sandwich_heads=2` won the initial medium screen.
- Anchor delta: Replay the carried `sandwich_heads=1` topology exactly against the locked `sd_tf_rd_024_classification_knob_sweep_v1_anchor_compile_eager_dynamic_v1` compile anchor while keeping `d_icl=60`, the non-scaling sandwich knobs, the curated medium corpus, and the runtime/optimizer bundle fixed.
- Expected effect: If the medium gain from the lower-head bracket reflects excess head factorization rather than a narrow `2`-head sweet spot, `sandwich_heads=1` may remain competitive or improve further on the closed medium contract.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0b3be75f60d2d62aa06d1f7efcfa44229dc3a32efa606b43230e2c3b1a807252`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute as a standalone replay package; do not mutate the historical TF-RD-024 follow-up sweep in place.
  - Register the resulting benchmark run locally, then promote this replayed row so `tf_rd_009_anchor_replay_heads1_medium_v1` becomes the formal TF-RD-009 anchor package.
  - Use the replayed run as the fixed `d_icl=60` anchor for the first TF-RD-009 width family `{48, 60(anchor), 96, 128}`.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime policy surface
  - attention partitioning only; no width, depth, batching, or optimizer reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row is required because the historical TF-RD-024 heads1 result is not benchmark-registry-backed on `main`.
  - Treat the resulting run as the formal TF-RD-009 anchor only after local registry import and in-package promotion complete.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_anchor_replay_heads1_medium_v1/delta_tf_rd_024_followup_cls_sandwich_heads1_v1/result_card.md`
- Benchmark metrics: pending
