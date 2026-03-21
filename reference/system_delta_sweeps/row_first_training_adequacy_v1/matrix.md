# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/row_first_training_adequacy_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `row_first_training_adequacy_v1`
- Sweep status: `draft`
- Parent sweep id: `qass_tfcol_large_missing_validation_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4215`, final Brier score `0.2644`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final training time `2550.1s`

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
| context encoder | None on the upstream direct path. | QASS context encoder. | QASS changes both compute graph depth and label-context semantics and needs explicit adequacy notes. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_large` (12 tasks (missing values permitted)) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_linear_warmup_decay`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_linear_decay` | schedule | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces but replace the exact-prior constant LR with single-stage linear decay. | Run first as the no-warmup schedule comparator on the settled row-first anchor. |
| 2 | `delta_training_adamw` | optimizer | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule. | Run after the schedule-family comparison so optimizer-family effects are read on the settled schedule. |
| 3 | `delta_training_muon` | optimizer | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule. | Run after the AdamW comparator so the two non-schedulefree families can be read against the same schedule. |
| 4 | `delta_training_batch64_sqrt` | batch_size | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but raise prior-dump batch size to 64 with sqrt LR scaling on the same warmup-decay family. | Run after the schedule and optimizer-family rows so batch-size effects are not mixed with family changes. |
| 5 | `delta_training_clip05` | regularization | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but tighten gradient clipping from 1.0 to 0.5 on the same warmup-decay family. | Run after the schedule and optimizer-family rows so clip changes are not read against the wrong baseline family. |
| 6 | `delta_training_budget_5k` | budget | yes | ready | none | Keep the settled row-first anchor family fixed but double the training budget to 5000 steps under the same warmup-decay schedule. | Leave last in the ladder and run only after the shorter-budget adequacy rows are understood. |

## Detailed Rows

### 1. `delta_training_linear_decay`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but replace the exact-prior constant LR with single-stage linear decay.
- Rationale: Test whether the settled row-first anchor still needs the current 5 percent warmup once the model family itself is no longer moving.
- Hypothesis: Removing warmup may keep final log loss competitive while simplifying the baseline training recipe and exposing whether warmup is carrying unnecessary caution.
- Upstream delta: Not applicable; this is a repo-local exact-prior training recipe change.
- Anchor delta: Starting from the settled no-TFCol row-first anchor, set warmup to zero while keeping the schedule family, optimizer family, bundle, and model surface fixed.
- Expected effect: Better late-curve retention or stability if the constant exact-prior LR is too aggressive for the anchored surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.0}]}}`
- Parameter adequacy plan:
  - Compare final log loss first against the settled anchor, then read Brier, ROC AUC, and drift as supporting diagnostics.
  - Treat this as schedule-family evidence only and do not infer anything about the row-first architecture from the result.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
  - optimizer.min_lr
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_linear_decay/result_card.md`
- Benchmark metrics: pending

### 2. `delta_training_adamw`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule.
- Rationale: Compare plain AdamW against the settled schedulefree baseline on the same row-first anchor and warmup-decay schedule.
- Hypothesis: AdamW may trade some convenience for a cleaner or easier-to-read optimization path on the settled anchor.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the settled warmup-decay schedule and row-first model surface fixed, but replace schedulefree AdamW with plain AdamW.
- Expected effect: AdamW may provide a cleaner or more stable optimization path if schedulefree dynamics are masking the anchor's real quality ceiling.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Read this row only as an optimizer-family comparator on the settled warmup-decay schedule.
  - Prefer it only if it is competitive on the primary final metric and materially cleaner on stability or training-time diagnostics.
- Adequacy knobs to dimension explicitly:
  - optimizer.name
  - optimizer.weight_decay
  - optimizer.betas
  - optimizer.min_lr
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_adamw/result_card.md`
- Benchmark metrics: pending

### 3. `delta_training_muon`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule.
- Rationale: Give Muon one clean read on the settled row-first anchor before later sweeps keep treating it as background optimizer lore.
- Hypothesis: Muon may improve convergence or late retention on the settled anchor, but any win should be read strictly as training-surface evidence.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the settled warmup-decay schedule and row-first model surface fixed, but replace schedulefree AdamW with Muon using the bounded default Muon settings.
- Expected effect: Muon may improve convergence or late retention on the settled row-first surface, but any read must stay separate from model-surface expansion.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this as a bounded optimizer-family comparison and defer rather than reject if the first Muon read is mixed.
  - Compare final log loss first, then use Brier, ROC AUC, gradient norms, and clip fraction as supporting diagnostics.
- Adequacy knobs to dimension explicitly:
  - optimizer.name
  - optimizer.weight_decay
  - optimizer.betas
  - optimizer.min_lr
  - optimizer.muon_per_parameter_lr
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_muon/result_card.md`
- Benchmark metrics: pending

### 4. `delta_training_batch64_sqrt`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but raise prior-dump batch size to 64 with sqrt LR scaling on the same warmup-decay family.
- Rationale: Check whether the settled row-first anchor prefers a larger effective batch once the model family is fixed and the schedule family is known.
- Hypothesis: Batch 64 with sqrt LR scaling may improve stability or runtime efficiency without changing the primary final metric materially.
- Upstream delta: Not applicable; this is a repo-local batch-size and LR-coupling comparison on the settled row-first anchor.
- Anchor delta: Keep the settled row-first model and schedule family fixed, but increase prior-dump batch size to 64 with sqrt LR scaling from a batch-32 reference.
- Expected effect: A larger scaled batch may improve stability or runtime efficiency if the settled anchor is not already on its preferred batch surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Keep model, preprocessing, bundle, and optimizer family fixed so the row isolates batch-size coupling only.
  - Read final log loss first and treat training-time or clip-fraction changes as secondary support, not the promotion rule.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - training.prior_dump_batch_reference_size
  - schedule.stages[0].lr_max
  - optimizer.min_lr
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_batch64_sqrt/result_card.md`
- Benchmark metrics: pending

### 5. `delta_training_clip05`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but tighten gradient clipping from 1.0 to 0.5 on the same warmup-decay family.
- Rationale: Check whether the settled row-first anchor still wants the current `grad_clip=1.0`, or whether a tighter clip improves late stability.
- Hypothesis: A tighter 0.5 clip may reduce rare update spikes without materially harming the final metric.
- Upstream delta: Not applicable; this is a repo-local clip-policy adequacy probe on the settled row-first anchor.
- Anchor delta: Keep the settled row-first model, bundle, and warmup-decay family fixed, but tighten gradient clipping from 1.0 to 0.5.
- Expected effect: A tighter clip may reduce rare update spikes, though it can also overconstrain useful row-first updates.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_clip': 0.5, 'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this row only as clip-policy evidence on the settled training family.
  - Prefer it only if quality remains competitive while clip or gradient diagnostics improve materially.
- Adequacy knobs to dimension explicitly:
  - runtime.grad_clip
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_clip05/result_card.md`
- Benchmark metrics: pending

### 6. `delta_training_budget_5k`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the settled row-first anchor family fixed but double the training budget to 5000 steps under the same warmup-decay schedule.
- Rationale: Check whether the settled row-first anchor is still compute-limited after the first shorter-budget adequacy rows settle.
- Hypothesis: If the preferred 2500-step recipe still leaves clear headroom, a 5000-step rerun should improve the primary final metric without changing the mechanism family.
- Upstream delta: Not applicable; this is a repo-local training-budget adequacy probe on the settled row-first anchor.
- Anchor delta: Keep the settled row-first model and preferred warmup-decay family fixed, but extend the training budget from 2500 to 5000 steps.
- Expected effect: If the anchor is still compute-limited after the earlier schedule, optimizer, batch, and clip reads, a 5000-step budget should improve the primary final metric without changing the mechanism family.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'max_steps': 5000, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Run only after the shorter-budget schedule and optimizer-family rows identify one preferred 2500-step recipe worth extending.
  - Interpret any gain as budget adequacy evidence only, not as architecture evidence.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/row_first_training_adequacy_v1/delta_training_budget_5k/result_card.md`
- Benchmark metrics: pending
