# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_022_runtime_policy_medium_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_022_runtime_policy_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `96ea81483fe2b92da15b8cb7aed34f82601ddff5f57f5de0f6671e233e1af54e`

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
| 1 | `delta_tf_rd_022_cls_runtime_control_noamp_v1` | runtime_policy | no | ready | none | Replay the closed TF-RD-010 classification control recipe with no AMP, no activation trace, and no activation checkpointing. | Execute first, then use it as the same-bundle benchmark and runtime control for rows 2 through 4. |
| 2 | `delta_tf_rd_022_cls_runtime_bf16_v1` | runtime_policy | no | ready | none | Switch only `mixed_precision` to `bf16` on the closed TF-RD-010 classification control recipe. | Execute after row 1 and carry it forward only if bf16 is benchmark-safe on the medium rung. |
| 3 | `delta_tf_rd_022_cls_runtime_trace_v1` | runtime_policy | no | ready | none | Enable benchmark-facing activation tracing on top of the bf16 TF-RD-022 runtime candidate. | Execute after row 2, then defer it unless benchmark quality remains non-worse and tracing is runtime-competitive. |
| 4 | `delta_tf_rd_022_cls_runtime_checkpoint_v1` | runtime_policy | no | ready | none | Enable activation checkpointing on top of the bf16 TF-RD-022 runtime candidate. | Execute after row 2, then defer it unless checkpointing is benchmark-safe and materially better on VRAM or runtime guardrails. |

## Detailed Rows

### 1. `delta_tf_rd_022_cls_runtime_control_noamp_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replay the closed TF-RD-010 classification control recipe with no AMP, no activation trace, and no activation checkpointing.
- Rationale: Establish the same-bundle no-AMP control replay before reading any TF-RD-022 runtime-policy change.
- Hypothesis: The closed TF-RD-010 classification recipe should remain the benchmark-safe reference point when rerun without AMP, activation tracing, or activation checkpointing.
- Upstream delta: Not applicable; this is the no-AMP runtime-policy control for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Re-run the closed TF-RD-010 classification control recipe with `mixed_precision='no'`, `trace_activations=false`, and `activation_checkpointing=false` on the medium benchmark contract.
- Expected effect: Establish the benchmark-safe runtime and VRAM baseline before any low-risk runtime knob is considered.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `72b828ff3f0ff3295d3758e4a4d194a152569d0f83f8b1bec2d1da70d39be60e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_control_noamp_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_022_cls_runtime_bf16_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Switch only `mixed_precision` to `bf16` on the closed TF-RD-010 classification control recipe.
- Rationale: Test whether bf16 alone is a benchmark-safe runtime and VRAM improvement on the carried classification recipe.
- Hypothesis: Switching only `mixed_precision` to `bf16` should preserve `final_log_loss_at_matched_regime_budget` while reducing runtime or reserved VRAM versus the no-AMP control.
- Upstream delta: Not applicable; this is the first low-risk AMP candidate for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 1, switch only `runtime.mixed_precision` from `"no"` to `bf16`.
- Expected effect: If AMP is benchmark-safe on the carried sandwich recipe, bf16 should preserve log loss while reducing runtime or VRAM cost versus the no-AMP control.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `1bc54c4bf8e6287193c1fe9cb4cf8dbb6e9f20b3b1968971892ed82889740fb8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_bf16_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_022_cls_runtime_trace_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Enable benchmark-facing activation tracing on top of the bf16 TF-RD-022 runtime candidate.
- Rationale: Measure the benchmark-facing activation-trace overhead on top of the simpler bf16 runtime candidate.
- Hypothesis: Activation tracing should stay a diagnostic-only cost on the carried classification recipe unless it is effectively free on both benchmark quality and runtime guardrails.
- Upstream delta: Not applicable; this is the bounded activation-trace diagnostic for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 2, switch only `runtime.trace_activations` from `false` to `true`.
- Expected effect: Benchmark-facing activation tracing should remain a measurable overhead unless it is effectively free on the carried classification recipe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f083a4b5790d74b1635f3cfb3db6fa2d69ff75d68a7ba032c36fe1cee427971f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': True, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_trace_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_022_cls_runtime_checkpoint_v1`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Enable activation checkpointing on top of the bf16 TF-RD-022 runtime candidate.
- Rationale: Measure the activation-checkpointing tradeoff on top of the simpler bf16 runtime candidate.
- Hypothesis: Activation checkpointing may reduce reserved VRAM, but it should stay deferred unless that reduction is benchmark-safe and beats the simpler bf16 row on the runtime tie-breakers.
- Upstream delta: Not applicable; this is the bounded activation-checkpointing diagnostic for TF-RD-022 on the closed classification benchmark contract.
- Anchor delta: Starting from row 2, switch only `runtime.activation_checkpointing` from `false` to `true`.
- Expected effect: Activation checkpointing may reduce reserved VRAM at the cost of runtime; it should stay deferred unless that trade remains benchmark-safe and materially useful.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6df38c53aecd3b035440f141ec27669759bdd7bc68a8558606a1cfa965a34378`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_022_runtime_policy_medium_v1/delta_tf_rd_022_cls_runtime_checkpoint_v1/result_card.md`
- Benchmark metrics: pending
