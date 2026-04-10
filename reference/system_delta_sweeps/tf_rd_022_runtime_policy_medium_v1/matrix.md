# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_022_runtime_policy_medium_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `6cb57e2f603d9ce9d2d7ba6e07c39148b1a9608859e0f3796d8efb1a11899cf5`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `classification_runtime_policy`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1136`, final BPF `2.1136`, final log loss `0.6812`, final Brier score `0.4229`, best ROC AUC `0.6094`, final ROC AUC `0.6094`, final training time `7449.8s`

## Anchor Comparison

Upstream reference: `PyTorch AMP` from `https://pytorch.org/docs/stable/amp.html`.

| Dimension | Upstream PyTorch AMP | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| precision policy | PyTorch AMP guidance treats bf16 as a low-risk accelerator-side runtime knob on supported CUDA hardware. | The closed TF-RD-010 contract was recorded with `mixed_precision='no'`. | Precision may change runtime and memory, but it only survives TF-RD-022 if benchmark quality is non-worse. |
| activation tracing | Activation tracing is a repo-local observability surface rather than a benchmark objective. | The control row keeps `trace_activations=false`. | Benchmark-facing activation tracing should stay off unless it is effectively free on the carried classification recipe. |
| activation checkpointing | Activation checkpointing is a standard memory-speed tradeoff, not a model change. | The control row keeps `activation_checkpointing=false`. | Checkpointing only survives if it is benchmark-safe and materially better on runtime or VRAM guardrails. |
| benchmark contract | Not applicable. | Closed TF-RD-010 medium contract on `data/manifests/bench/openml_classification_medium_v1/manifest.parquet`. | Medium is the screening rung only; any keep-worthy row still needs large-rung validation before it becomes the TF-RD-022 carried policy. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_022_cls_runtime_control_noamp_v1` | runtime_policy | no | completed | none | Replay the closed TF-RD-010 classification control recipe with no AMP, no activation trace, and no activation checkpointing. | Keep this row as the same-bundle benchmark-safe screening control while the checkpointing winner advances to the large validator. |
| 2 | `delta_tf_rd_022_cls_runtime_bf16_v1` | runtime_policy | no | completed | none | Switch only `mixed_precision` to `bf16` on the closed TF-RD-010 classification control recipe. | Keep this row as the simpler benchmark-safe bf16 reference, but defer it in favor of the lower-VRAM checkpointing winner. |
| 3 | `delta_tf_rd_022_cls_runtime_trace_v1` | runtime_policy | no | completed | none | Enable benchmark-facing activation tracing on top of the bf16 TF-RD-022 runtime candidate. | Defer this row as a benchmark-safe diagnostic that did not win the runtime tie-breakers against the kept checkpointing candidate. |
| 4 | `delta_tf_rd_022_cls_runtime_checkpoint_v1` | runtime_policy | no | completed | none | Enable activation checkpointing on top of the bf16 TF-RD-022 runtime candidate. | Promote this row into `tf_rd_022_runtime_policy_large_validation_v1` on the closed TF-RD-010 large contract before treating it as the closed TF-RD-022 carried runtime policy. |

## Detailed Rows

### 1. `delta_tf_rd_022_cls_runtime_control_noamp_v1`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replay the closed TF-RD-010 classification control recipe with no AMP, no activation trace, and no activation checkpointing.
- Rationale: Establish the same-bundle no-AMP control replay before reading any TF-RD-022 runtime-policy change.
- Hypothesis: The closed TF-RD-010 classification recipe should remain the benchmark-safe reference point when rerun without AMP, activation tracing, or activation checkpointing.
- Upstream delta: Not applicable; this is the no-AMP runtime-policy control for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Re-run the closed TF-RD-010 classification control recipe with `mixed_precision='no'`, `trace_activations=false`, and `activation_checkpointing=false` on the medium benchmark contract.
- Expected effect: Establish the benchmark-safe runtime and VRAM baseline before any low-risk runtime knob is considered.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8cfb97d6c9a54e9db704cdc7b104d1cfda9ed74404936242df6467b40ca04592`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'min_lr': 1e-05}, 'runtime': {'mixed_precision': 'no', 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Run this row first as the no-AMP control replay for the TF-RD-022 runtime ladder.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` baseline before interpreting any runtime-policy result as a keep or defer.
  - Rank rows first by `final_log_loss_at_matched_regime_budget`, then inspect runtime and VRAM only as guardrails and tie-breakers.
- Adequacy knobs to dimension explicitly:
  - closed TF-RD-010 medium benchmark contract on `tf_rd_010_dagzoo_medium_control_curated_v5`
  - fixed `task_batch_size=16`, `prior_dump_batch_size=64`, `grad_accum_steps=4`, `grad_clip=0.0`, `max_steps=2500`
  - runtime knobs only; architecture, corpus choice, optimizer family, and schedule family remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_022_runtime_policy_medium_v1_01_delta_tf_rd_022_cls_runtime_control_noamp_v1_v4`.
  - Recorded the canonical no-AMP control replay for the TF-RD-022 medium runtime ladder. Keep it as the benchmark-safe screening reference only because the checkpointing candidate remained non-worse on matched-budget log loss while materially reducing reserved VRAM.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_control_noamp_v1/result_card.md`
- Registered run: `sd_tf_rd_022_runtime_policy_medium_v1_01_delta_tf_rd_022_cls_runtime_control_noamp_v1_v4` with final log loss `0.6849`, delta final log loss `+0.0038`, final Brier score `0.4246`, delta final brier score `+0.0017`, final ROC AUC `0.6044`, delta final roc auc `-0.0050`, final BPC (legacy feature-cell diagnostic) `2.1154`, delta final bpc (legacy feature-cell diagnostic) `+0.0017`, final BPF (legacy feature-cell diagnostic) `2.1154`, delta final bpf (legacy feature-cell diagnostic) `+0.0017`, best ROC AUC `0.6044`, delta final training time `-1498.0s`

### 2. `delta_tf_rd_022_cls_runtime_bf16_v1`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Switch only `mixed_precision` to `bf16` on the closed TF-RD-010 classification control recipe.
- Rationale: Test whether bf16 alone is a benchmark-safe runtime and VRAM improvement on the carried classification recipe.
- Hypothesis: Switching only `mixed_precision` to `bf16` should preserve `final_log_loss_at_matched_regime_budget` while reducing runtime or reserved VRAM versus the no-AMP control.
- Upstream delta: Not applicable; this is the first low-risk AMP candidate for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 1, switch only `runtime.mixed_precision` from `"no"` to `bf16`.
- Expected effect: If AMP is benchmark-safe on the carried sandwich recipe, bf16 should preserve log loss while reducing runtime or VRAM cost versus the no-AMP control.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `28d50650fcfa4ddd57235a4824ee1a148527c3150ad6cadec72c6a9fa2d59d1d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'min_lr': 1e-05}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Compare directly against row 1 on `final_log_loss_at_matched_regime_budget`.
  - Promote to the large validator only if this row is non-worse on benchmark quality.
  - Use lower `peak_vram_reserved`, then higher `throughput_tokens_per_second`, then lower `non_train_overhead_seconds` only as tie-breakers among benchmark-safe rows.
- Adequacy knobs to dimension explicitly:
  - closed TF-RD-010 medium benchmark contract on `tf_rd_010_dagzoo_medium_control_curated_v5`
  - fixed `task_batch_size=16`, `prior_dump_batch_size=64`, `grad_accum_steps=4`, `grad_clip=0.0`, `max_steps=2500`
  - runtime knobs only; architecture, corpus choice, optimizer family, and schedule family remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_022_runtime_policy_medium_v1_02_delta_tf_rd_022_cls_runtime_bf16_v1_v1`.
  - Pure bf16 is benchmark-safe on the TF-RD-022 medium rung and improves over the no-AMP control, but it is not the kept medium winner because bf16 plus activation checkpointing preserved benchmark quality while lowering peak reserved VRAM further.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_bf16_v1/result_card.md`
- Registered run: `sd_tf_rd_022_runtime_policy_medium_v1_02_delta_tf_rd_022_cls_runtime_bf16_v1_v1` with final log loss `0.6818`, delta final log loss `+0.0007`, final Brier score `0.4226`, delta final brier score `-0.0003`, final ROC AUC `0.6087`, delta final roc auc `-0.0007`, final BPC (legacy feature-cell diagnostic) `2.1103`, delta final bpc (legacy feature-cell diagnostic) `-0.0034`, final BPF (legacy feature-cell diagnostic) `2.1103`, delta final bpf (legacy feature-cell diagnostic) `-0.0034`, best ROC AUC `0.6087`, delta final training time `-1611.3s`

### 3. `delta_tf_rd_022_cls_runtime_trace_v1`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Enable benchmark-facing activation tracing on top of the bf16 TF-RD-022 runtime candidate.
- Rationale: Measure the benchmark-facing activation-trace overhead on top of the simpler bf16 runtime candidate.
- Hypothesis: Activation tracing should stay a diagnostic-only cost on the carried classification recipe unless it is effectively free on both benchmark quality and runtime guardrails.
- Upstream delta: Not applicable; this is the bounded activation-trace diagnostic for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 2, switch only `runtime.trace_activations` from `false` to `true`.
- Expected effect: Benchmark-facing activation tracing should remain a measurable overhead unless it is effectively free on the carried classification recipe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `de1513184a77957bff60a4f9cfc5b0efa44a5e7277a9c34a65a607829c6ce3b2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': True, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'min_lr': 1e-05}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': True, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Compare this row against the simpler bf16 row first, not only against row 1.
  - Treat it as a diagnostic loser unless it is benchmark-safe and also beats the simpler bf16 row on the runtime tie-breakers.
  - Do not carry this row forward to large validation unless it wins the medium ladder under the benchmark-first keep bar.
- Adequacy knobs to dimension explicitly:
  - closed TF-RD-010 medium benchmark contract on `tf_rd_010_dagzoo_medium_control_curated_v5`
  - fixed `task_batch_size=16`, `prior_dump_batch_size=64`, `grad_accum_steps=4`, `grad_clip=0.0`, `max_steps=2500`
  - runtime knobs only; architecture, corpus choice, optimizer family, and schedule family remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_022_runtime_policy_medium_v1_03_delta_tf_rd_022_cls_runtime_trace_v1_v2`.
  - Benchmark-facing activation tracing stayed benchmark-safe on the medium rung, but it did not beat the kept candidate on the runtime tie-breakers: it reserved more VRAM than checkpointing and ran slower than the simpler bf16 row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_trace_v1/result_card.md`
- Registered run: `sd_tf_rd_022_runtime_policy_medium_v1_03_delta_tf_rd_022_cls_runtime_trace_v1_v2` with final log loss `0.6755`, delta final log loss `-0.0057`, final Brier score `0.4192`, delta final brier score `-0.0037`, final ROC AUC `0.6183`, delta final roc auc `+0.0089`, final BPC (legacy feature-cell diagnostic) `2.1165`, delta final bpc (legacy feature-cell diagnostic) `+0.0029`, final BPF (legacy feature-cell diagnostic) `2.1165`, delta final bpf (legacy feature-cell diagnostic) `+0.0029`, best ROC AUC `0.6183`, delta final training time `-1379.7s`

### 4. `delta_tf_rd_022_cls_runtime_checkpoint_v1`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Enable activation checkpointing on top of the bf16 TF-RD-022 runtime candidate.
- Rationale: Measure the activation-checkpointing tradeoff on top of the simpler bf16 runtime candidate.
- Hypothesis: Activation checkpointing may reduce reserved VRAM, but it should stay deferred unless that reduction is benchmark-safe and beats the simpler bf16 row on the runtime tie-breakers.
- Upstream delta: Not applicable; this is the bounded activation-checkpointing diagnostic for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 2, switch only `runtime.activation_checkpointing` from `false` to `true`.
- Expected effect: Activation checkpointing may reduce reserved VRAM at the cost of runtime; it should stay deferred unless that trade remains benchmark-safe and materially useful.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `4be8e8be7b0b1f97355c720a1eddd91bf47d0057984e072789a1c90138a55901`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Training overrides: `{'optimizer': {'min_lr': 1e-05}, 'runtime': {'mixed_precision': 'bf16', 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.001, 'lr_schedule': 'linear', 'warmup_ratio': 0.1}]}}`
- Parameter adequacy plan:
  - Compare this row against the simpler bf16 row first, not only against row 1.
  - Treat it as a diagnostic loser unless it is benchmark-safe and also beats the simpler bf16 row on the runtime tie-breakers.
  - Do not carry this row forward to large validation unless it wins the medium ladder under the benchmark-first keep bar.
- Adequacy knobs to dimension explicitly:
  - closed TF-RD-010 medium benchmark contract on `tf_rd_010_dagzoo_medium_control_curated_v5`
  - fixed `task_batch_size=16`, `prior_dump_batch_size=64`, `grad_accum_steps=4`, `grad_clip=0.0`, `max_steps=2500`
  - runtime knobs only; architecture, corpus choice, optimizer family, and schedule family remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2`.
  - bf16 plus activation checkpointing is the TF-RD-022 medium-rung winner: it stayed non-worse than the no-AMP control on matched-budget log loss while delivering the lowest peak reserved VRAM among benchmark-safe rows. Promote this row to the large validation gate before treating it as the closed TF-RD-022 carried policy.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_checkpoint_v1/result_card.md`
- Registered run: `sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2` with final log loss `0.6766`, delta final log loss `-0.0046`, final Brier score `0.4199`, delta final brier score `-0.0030`, final ROC AUC `0.6189`, delta final roc auc `+0.0095`, final BPC (legacy feature-cell diagnostic) `2.1138`, delta final bpc (legacy feature-cell diagnostic) `+0.0001`, final BPF (legacy feature-cell diagnostic) `2.1138`, delta final bpf (legacy feature-cell diagnostic) `+0.0001`, best ROC AUC `0.6189`, delta final training time `-1348.0s`
