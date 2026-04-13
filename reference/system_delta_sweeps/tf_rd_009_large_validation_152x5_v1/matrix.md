# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_large_validation_152x5_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_large_validation_152x5_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_large_validation_152x5_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_width_depth_medium_v1`
- Complexity level: `classification_lg`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_large_validation_152x5_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `0f98b7b63ecd7b00bb14f0aedb780be423e314e764b5fba62951a5ea8f356dd0`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_large_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_large_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.0860`, final BPF `2.0860`, final log loss `0.8974`, final Brier score `0.5465`, best ROC AUC `0.6324`, final ROC AUC `0.6324`, final training time `7449.8s`

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
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark manifest `data/manifests/bench/openml_classification_large_v1/manifest.parquet` sourced from `nanotabpfn_openml_classification_large` (3 tasks (missing values permitted)) with data surface label `tf_rd_010_dagzoo_medium_control_curated_v5`. | Manifest and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_cosine_warmup`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` | classification_scaling_law | no | completed | none | Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe. | Keep this corrected large-rung gate, maintain frozen hardware baseline `tf_rd_009_rtx8000_44gb_classification_medium_v1`, and treat any bracketed large-rung diagnosis as separate follow-on work outside issue 257. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the penultimate TF-RD-009 joint width-depth upper row at `d_icl=152`, `sandwich_layers=5`, continuing the dense diagonal toward the intended `rtx8000_44gb` ceiling probe.
- Rationale: Benchmark-only validate the current TF-RD-009 medium fixed-budget winner `delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1` against the closed TF-RD-010 large clean-control anchor before freezing any first hardware architecture baseline.
- Hypothesis: The retained medium winner `152x5` should transfer cleanly enough on the closed large benchmark rung to beat the carried clean-control anchor without reopening training, seeds, or neighboring bracket rows.
- Upstream delta: TF-RD-009 uses this row to extend the empirically bridged parameter ladder far enough to fit curvature and identify where hardware guardrails begin to dominate, without switching to a width-depth grid.
- Anchor delta: Reuse the completed medium `152x5` training artifact and benchmark it against locked anchor `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1` on the local large classification manifest; no retraining, no extra seeds, and no bracketed reruns inside issue 257.
- Expected effect: If the medium-rung joint law is still smooth at higher effective size, `152x5` should improve the matched-budget objective or expose the first clear bend in the runtime and stability guardrails.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 152, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 5, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1/train`
- Reuse training surface fingerprint: `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253`
- Parameter adequacy plan:
  - Confirm the reused `152x5` training surface fingerprint `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253` matches the preserved train artifact before execution so benchmark-only reuse cannot drift.
  - Benchmark only on the frozen local large bundle with task ids `[363685, 363699, 363707]` and rank by `final_log_loss_at_matched_regime_budget` against the carried large clean-control anchor `0.8974410961`.
  - Treat Brier, ROC AUC, runtime, and stability as advisory guardrails unless the run is invalid or clearly unstable.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Reuse completed run `sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` at `outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1/train`; do not retrain this row for `tf_rd_009_large_validation_152x5_v1`.
  - Preflight must confirm training-surface fingerprint `8fbebcbeb4951b28d1a1f26e007b427e0686d9fc58fb5b281107dca7c0f69253` before benchmark execution to block surface drift.
  - The large-rung success gate is `final_log_loss_at_matched_regime_budget < 0.8974410961` versus carried large clean-control anchor `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`.
  - Advisory guardrails are Brier, ROC AUC, benchmark/runtime timing, and stability; only invalid or clearly unstable behavior blocks a keep.
  - If this row passes, freeze medium hardware baseline `tf_rd_009_rtx8000_44gb_classification_medium_v1` from evidence rows `60x2`, `72x1`, `96x2`, `112x3`, `128x4`, `152x5`, and `176x6`; do not add the large validation row to the medium constraint model.
  - If this row fails, leave `hardware_architecture_baselines_v1.json` unfrozen and keep any `96x2` / `176x6` bracketed large-rung diagnosis as separate follow-on work.
  - Execute on the same CUDA / RTX 8000 class environment as the retained TF-RD-009 evidence; the current local workstation does not expose the required GPU surface.
  - Execution attempt `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` failed: [row 01] pinned reusable train artifact is missing or incomplete: outputs/staged_ladder/research/tf_rd_009_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/sd_tf_rd_009_width_depth_medium_v1_04_delta_tf_rd_009_cls_sa...
  - Partial-snapshot rerun `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v1` is superseded: telemetry listed checkpoints through step 2500, but the reusable artifact retained numbered `step_*.pt` files only through step 600 plus `latest.pt`, so the old artifact-selection path stopped on the last preserved numbered snapshot.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Canonical rerun registered as `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2`.
  - Corrected artifact resolution kept telemetry as the source of step and elapsed-time metadata, dropped telemetry-only missing late numbered checkpoints, and appended the retained terminal `latest.pt` checkpoint at `global_step=2500` while preserving `benchmark_checkpoint_selection: all`.
  - The corrected rerun completed `25/25` checkpoint comparisons with terminal `latest.pt` at step `2500`, finished at `final_log_loss=0.7436636568`, `final_brier_score=0.4288940`, and `final_roc_auc=0.7650940`, and beat the carried large clean-control anchor by `delta_final_log_loss=-0.1537774`.
  - Hardware baseline `tf_rd_009_rtx8000_44gb_classification_medium_v1` is now frozen from medium evidence rows only with preferred `152x5`, formal anchor `60x2`, and baseline `96x2`; the large validation row remains a gate and is not added to the medium constraint model.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_large_validation_152x5_v1/delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1/result_card.md`
- Registered run: `sd_tf_rd_009_large_validation_152x5_v1_01_delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1_v2` with final log loss `0.7437`, delta final log loss `-0.1538`, final Brier score `0.4289`, delta final brier score `-0.1177`, final ROC AUC `0.7651`, delta final roc auc `+0.1327`, final BPC (legacy feature-cell diagnostic) `4.5075`, delta final bpc (legacy feature-cell diagnostic) `+2.4214`, final BPF (legacy feature-cell diagnostic) `4.5075`, delta final bpf (legacy feature-cell diagnostic) `+2.4214`, best ROC AUC `0.5967`, delta final training time `+2673.7s`
