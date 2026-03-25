# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_018_optimizer_family_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_018_optimizer_family_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_020_harder_dagzoo_ladder_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `4.4473`, final Brier score `0.6465`, best ROC AUC `0.5958`, final ROC AUC `0.5746`, final training time `699.4s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| promoted anchor | TabICLv2 motivates synthetic pretraining but does not define this repo-local row-first promoted-anchor contract. | The kept `row_cls + qass + no tfcol` anchor trained on the settled `task_batch_size=4` manifest-batched surface from TF-RD-018 issue `#109`. | TF-RD-018 issue `#137` must keep the promoted row-first model surface fixed while reading optimizer-family evidence. |
| harder carry-forward surface | No upstream reference defines this exact dagzoo harder-surface handoff. | The locked medium-surface anchor remains `tf_rd_013_dagzoo_shape_aware_size_medium_v1`, while row `01` replays schedulefree on `tf_rd_020_shift_noise_drift_v1`. | The TF-RD-020 noise-drift winner is evidence for which data surface to carry forward, not the optimizer anchor itself. |
| optimizer family | Not applicable. | Schedulefree AdamW on the settled `2500`-step `task_batch_size=4` recipe is the replay baseline for this sweep. | Compare `schedulefree_adamw`, `adamw`, and `muon` on the same harder-surface recipe before reopening LR, clipping, or budget work. |
| fallback scope | Not applicable. | `tf_rd_020_noise_mixture_v1` is excluded from this v1 sweep. | If noise drift is too confounded, stop with an explicit defer note and open fallback-surface work separately rather than broadening |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_020_shift_noise_drift` | shift | yes | ready | none | Point training at the TF-RD-020 noise-drift harder-front manifest while keeping the settled four-task row-first recipe fixed. | Run first under issue `#137` with `--promote-first-executed-row-to-anchor`, then compare rows `02` and `03` against the promoted harder-surface replay. |
| 2 | `delta_training_adamw` | optimizer | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule. | Run after row `01` is promoted so plain AdamW is read against the harder-surface replay rather than the medium-surface anchor alone. |
| 3 | `delta_training_muon` | optimizer | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule. | Run after row `01` is promoted; if the result is close, retain at most one fallback optimizer family for issue `#138`. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_020_shift_noise_drift`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 noise-drift harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Open issue `#137` by replaying the settled schedulefree baseline on the documented TF-RD-020 noise-drift carry-forward surface before comparing any non-schedulefree optimizer family.
- Hypothesis: Replaying the kept TF-RD-018 four-task schedulefree recipe on `tf_rd_020_shift_noise_drift_v1` should provide the cleanest optimizer baseline and remove the TF-RD-020 harmonized `400`-step recipe as a confounder.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, benchmark bundle, and `task_batch_size=4` recipe fixed, but replace the carried medium control corpus with `tf_rd_020_shift_noise_drift_v1` and replay the schedulefree baseline at the settled `2500`-step adequacy recipe.
- Expected effect: Moderate variance drift may create a harder front if the current overfitting problem is partly a mismatch in stochasticity rather than structure.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the TF-RD-020 noise-drift effective config artifacts before reading benchmark output.
  - Treat this row as the replayed schedulefree harder-surface baseline, not as the final optimizer-family decision by itself.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final drift fourth while treating NaNs, OOM, clipped-step fraction, and runtime as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit `shift.mode=noise_drift` resolution with `variance_scale=0.5` across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_optimizer_family_v1/delta_data_manifest_root_tf_rd_020_shift_noise_drift/result_card.md`
- Benchmark metrics: pending

### 2. `delta_training_adamw`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule.
- Rationale: Compare plain AdamW against the replayed schedulefree baseline on the same harder noise-drift surface so issue `#137` can separate optimizer family effects from TF-RD-020 data-surface choice.
- Hypothesis: Plain AdamW may improve late retention or reduce drift on the harder surface if schedulefree dynamics are masking the best benchmark-facing read.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the replayed `tf_rd_020_shift_noise_drift_v1` data surface, preprocessing surface, benchmark bundle, and settled `task_batch_size=4` runtime fixed, but replace `schedulefree_adamw` with plain `adamw`.
- Expected effect: AdamW may provide a cleaner or more stable optimization path if schedulefree dynamics are masking the anchor's real quality ceiling.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this as a bounded optimizer-family comparison on the replayed harder-surface baseline.
  - Compare final log loss first, then final Brier score, final ROC AUC, and best-to-final drift against rows `01` and `03`.
  - Use NaNs, OOM, clipped-step fraction, and obvious LR-transfer artifacts as guardrails before making a reject call.
- Adequacy knobs to dimension explicitly:
  - optimizer.name
  - optimizer.weight_decay
  - optimizer.betas
  - optimizer.min_lr
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_optimizer_family_v1/delta_training_adamw/result_card.md`
- Benchmark metrics: pending

### 3. `delta_training_muon`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule.
- Rationale: Compare Muon against the replayed schedulefree baseline on the same harder noise-drift surface so issue `#137` can read whether a more specialized optimizer changes the adequacy story.
- Hypothesis: Muon may improve convergence or late retention on the harder surface, but a weak result should only matter if it is not obviously an LR-transfer problem.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the replayed `tf_rd_020_shift_noise_drift_v1` data surface, preprocessing surface, benchmark bundle, and settled `task_batch_size=4` runtime fixed, but replace `schedulefree_adamw` with `muon`.
- Expected effect: Muon may improve convergence or late retention on the settled row-first surface, but any read must stay separate from model-surface expansion.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'grad_accum_steps': 1, 'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this as a bounded optimizer-family comparison only on the replayed harder-surface baseline.
  - Compare final log loss first, then final Brier score, final ROC AUC, and best-to-final drift against rows `01` and `02`.
  - If weak, defer rather than reject unless the result is clearly worse without an offsetting stability or runtime benefit and does not look like pure LR-transfer misspecification.
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
- Result card path: `outputs/staged_ladder/research/tf_rd_018_optimizer_family_v1/delta_training_muon/result_card.md`
- Benchmark metrics: pending
