# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/row_first_training_adequacy_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `row_first_training_adequacy_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_013_dagzoo_size_ladder_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_tf_rd_013_dagzoo_size_ladder_v1_03_delta_data_manifest_root_dagzoo_shape_aware_size_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `2.2604`, final Brier score `0.4912`, best ROC AUC `0.5711`, final ROC AUC `0.5625`, final training time `226.7s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block with test-self attention enabled. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Shifted grouped tokenizer. | Tokenizer changes reshape the effective table sequence and need their own adequacy commentary. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Row-CLS pooling path. | Row-pool changes alter how the table summary is extracted and should be isolated from context changes. |
| context encoder | None on the upstream direct path. | QASS context encoder. | QASS changes both compute-graph depth and label-context semantics and need explicit adequacy notes. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` with data surface label `tf_rd_013_dagzoo_shape_aware_size_medium` and corpus ref `tf_rd_013_dagzoo_shape_aware_size_medium_v1`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent-sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| task batching | No repo-local manifest task batching contract. | Manifest-backed singleton task updates with `training.task_batch_size=1`. | Manifest task batching is a first-class training-surface delta and must be read before optimizer or schedule follow-ons. |
| training recipe | No repo-local manifest training-surface contract. | Registered anchor training surface label `prior_linear_warmup_decay` with `schedulefree_adamw`, `max_steps=2500`, and `runtime.grad_accum_steps=1`. | Optimizer and schedule changes are later training-surface rows, not background recipe assumptions in this first ladder. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_task_batch4` | batch_size | yes | completed | none | Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to four at a time on the TF-RD-013 medium corpus surface. | Lock `task_batch_size=4` as the preferred TF-RD-018 batch rung and rebase issues `#137`, `#138`, and `#139` onto this manifest-batched surface instead of reopening singleton updates. |
| 2 | `delta_training_task_batch8` | batch_size | yes | completed | none | Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to eight at a time on the TF-RD-013 medium corpus surface. | Stop the ladder here because this row kept singleton fallback at `0.0%` but finished in `1109.3s`, so `task_batch_size=16` and `32` stay blocked and issues `#137`, `#138`, and `#139` should rebase onto the kept four-task rung. |
| 3 | `delta_training_task_batch16` | batch_size | yes | blocked | none | Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to sixteen at a time on the TF-RD-013 medium corpus surface. | Do not run this row in the current ladder; row 2 finished in `1109.3s` and failed the `<=900s` promotion gate, so issues `#137`, `#138`, and `#139` should continue from the kept `task_batch_size=4` surface instead. |
| 4 | `delta_training_task_batch32` | batch_size | yes | blocked | none | Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to thirty-two at a time on the TF-RD-013 medium corpus surface. | Keep this rung blocked because the current ladder stopped when row 2 missed the row-3 promotion gate; only reopen larger task batches if later runtime work changes the economics. |

## Detailed Rows

### 1. `delta_training_task_batch4`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to four at a time on the TF-RD-013 medium corpus surface.
- Rationale: Start TF-RD-018 with the lowest-risk manifest task-batch rung on the settled TF-RD-013 medium corpus surface.
- Hypothesis: Exact-shape task batching at four tasks should improve wall-clock efficiency without reopening architecture, preprocessing, or optimizer-family questions.
- Upstream delta: Not applicable; this is a repo-local manifest task-batching adequacy rung on the settled row-first anchor.
- Anchor delta: Keep the settled `row_cls + qass + no tfcol` anchor, preprocessing, and warmup-decay family fixed, but replace singleton manifest updates with `training.task_batch_size=4`.
- Expected effect: A four-task manifest batch should improve dataset-throughput efficiency with lower packing and memory risk than the larger rungs.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.2624`); context (grad `0.1001`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Keep the model, data surface, preprocessing, optimizer family, and schedule shape fixed so this row isolates manifest task batching only.
  - Read final log loss, final ROC AUC, train elapsed seconds, and singleton-fallback fraction together before preferring a larger rung.
- Adequacy knobs to dimension explicitly:
  - training.task_batch_size
  - task_batch_singleton_fallback_fraction
  - batched_update_count
  - train_elapsed_seconds
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical rerun registered as `sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Locked as the preferred rung because it finished in `699.4s`, avoided recorded OOM mitigation, and kept singleton fallback at `0.0%` while the eight-task rung missed the `<=900s` gate.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_task_batch4/result_card.md`
- Registered run: `sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1` with final log loss `4.4473`, delta final log loss `+2.1869`, final Brier score `0.6465`, delta final Brier score `+0.1553`, best ROC AUC `0.5958`, final ROC AUC `0.5746`, final-minus-best `-0.0213`, delta final ROC AUC `+0.0120`, delta drift `-0.0127`, delta final training time `+472.7s`

### 2. `delta_training_task_batch8`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to eight at a time on the TF-RD-013 medium corpus surface.
- Rationale: Probe the highest unconditional manifest task-batch rung before TF-RD-018 opens optimizer or schedule-family follow-ups.
- Hypothesis: Exact-shape task batching at eight tasks may further improve throughput while keeping singleton fallback and memory use low enough for iterative reads.
- Upstream delta: Not applicable; this is a repo-local manifest task-batching adequacy rung on the settled row-first anchor.
- Anchor delta: Keep the settled `row_cls + qass + no tfcol` anchor, preprocessing, and warmup-decay family fixed, but replace singleton manifest updates with `training.task_batch_size=8`.
- Expected effect: An eight-task manifest batch may improve wall-clock efficiency further if the medium corpus still spends too much time on singleton task dispatch.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0082`); context (grad `0.0041`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Keep the model, data surface, preprocessing, optimizer family, and schedule shape fixed so this row isolates manifest task batching only.
  - Use this row as the gatekeeper for larger task-batch rungs before reopening optimizer, LR, clipping, or budget questions.
- Adequacy knobs to dimension explicitly:
  - training.task_batch_size
  - task_batch_singleton_fallback_fraction
  - batched_update_count
  - train_elapsed_seconds
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_row_first_training_adequacy_v1_02_delta_training_task_batch8_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Reused the saved nanoTabPFN curve from row 1 rather than running a fresh external helper benchmark.
  - This row missed the `<=900s` promotion gate at `1109.3s` even though singleton fallback stayed at `0.0%` and no recorded OOM mitigation was needed.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_task_batch8/result_card.md`
- Registered run: `sd_row_first_training_adequacy_v1_02_delta_training_task_batch8_v1` with final log loss `5.0365`, delta final log loss `+2.7761`, final Brier score `0.6119`, delta final Brier score `+0.1207`, best ROC AUC `0.6246`, final ROC AUC `0.5508`, final-minus-best `-0.0738`, delta final ROC AUC `-0.0117`, delta drift `-0.0653`, delta final training time `+882.6s`

### 3. `delta_training_task_batch16`

- Dimension family: `training`
- Status: `blocked`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to sixteen at a time on the TF-RD-013 medium corpus surface.
- Rationale: Keep the larger manifest task-batch ladder explicit, but do not spend the extra runtime or memory budget until the eight-task rung proves it is still on the clean batched path.
- Hypothesis: Exact-shape task batching at sixteen tasks may still improve throughput, but only if the eight-task rung stays fast, avoids OOM, and rarely falls back to singleton dispatch.
- Upstream delta: Not applicable; this is a repo-local manifest task-batching adequacy rung on the settled row-first anchor.
- Anchor delta: Keep the settled `row_cls + qass + no tfcol` anchor, preprocessing, and warmup-decay family fixed, but replace singleton manifest updates with `training.task_batch_size=16`.
- Expected effect: A sixteen-task manifest batch may improve throughput further, but it carries materially higher singleton-fallback and memory-pressure risk than the opening rungs.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Leave blocked until the `task_batch_size=8` row finishes in `<=900s`, avoids OOM, and keeps singleton fallback at `<=10%` of updates.
  - If unblocked, compare against the eight-task rung first and stop the ladder if runtime exceeds `1800s` or singleton fallback rises above `10%`.
- Adequacy knobs to dimension explicitly:
  - training.task_batch_size
  - task_batch_singleton_fallback_fraction
  - batched_update_count
  - train_elapsed_seconds
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row is intentionally blocked pending the row-2 gate.
  - Row 2 completed in `1109.3s` with `0.0%` singleton fallback and no recorded OOM mitigation, so the ladder stops before this rung.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_task_batch16/result_card.md`
- Benchmark metrics: pending

### 4. `delta_training_task_batch32`

- Dimension family: `training`
- Status: `blocked`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the settled row-first model, preprocessing, and warmup-decay family fixed, but batch exact-shape manifest tasks up to thirty-two at a time on the TF-RD-013 medium corpus surface.
- Rationale: Keep the highest manifest task-batch rung visible in the queue, but leave it dormant unless the sixteen-task rung remains fast and mostly batched.
- Hypothesis: Exact-shape task batching at thirty-two tasks will only be useful if the sixteen-task rung still avoids OOM and singleton fallback on the settled medium corpus.
- Upstream delta: Not applicable; this is a repo-local manifest task-batching adequacy rung on the settled row-first anchor.
- Anchor delta: Keep the settled `row_cls + qass + no tfcol` anchor, preprocessing, and warmup-decay family fixed, but replace singleton manifest updates with `training.task_batch_size=32`.
- Expected effect: A thirty-two-task manifest batch would maximize batching pressure on this medium corpus, but it is most likely to trip OOM or singleton-fallback gates before quality can be read cleanly.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Leave blocked until the `task_batch_size=16` row finishes in `<=900s`, avoids OOM, and keeps singleton fallback at `<=10%` of updates.
  - If unblocked, compare against the sixteen-task rung first and stop the ladder if runtime exceeds `1800s` or singleton fallback rises above `10%`.
- Adequacy knobs to dimension explicitly:
  - training.task_batch_size
  - task_batch_singleton_fallback_fraction
  - batched_update_count
  - train_elapsed_seconds
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row is intentionally blocked pending the row-3 gate.
  - Row 3 never unblocked because the eight-task rung already failed the runtime promotion gate.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_task_batch32/result_card.md`
- Benchmark metrics: pending
