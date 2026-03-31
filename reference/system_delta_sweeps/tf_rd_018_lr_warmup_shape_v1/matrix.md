# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_018_lr_warmup_shape_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_018_lr_warmup_shape_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_018_optimizer_family_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2`
- Benchmark bundle: `src/tab_foundry/bench/openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.5501`, final Brier score `0.3740`, best ROC AUC `0.5945`, final ROC AUC `0.5880`, final training time `1228.1s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| locked anchor | TabICLv2 motivates synthetic pretraining but does not define this repo-local row-first promoted-anchor contract. | The kept `row_cls + qass + no tfcol` TF-RD-020 noise-drift winner trained on the uncapped `task_batch_size=1`, `grad_accum_steps=4`, `400`-step runtime. | TF-RD-018 issue `#138` must keep the promoted row-first model surface fixed while reading LR and warmup evidence. |
| harder carry-forward surface | No upstream reference defines this exact dagzoo harder-surface handoff. | The locked harder-surface anchor is `tf_rd_020_shift_noise_drift_v1`, inherited directly from TF-RD-020 row `06`. | The TF-RD-020 noise-drift winner remains the carried training-data surface throughout this sweep. |
| optimizer family | Not applicable. | Completed issue `#137` kept `schedulefree_adamw` on the inherited TF-RD-020 runtime. | LR and warmup rows should not reopen optimizer-family choice. |
| LR and warmup shape | Adam-style scaling papers motivate treating LR ceiling, LR floor, and warmup as coupled knobs rather than independent universal rules. | The carried baseline uses `lr_max=0.004`, `optimizer.min_lr=0.0004`, and `warmup_ratio=0.05` on a corrected `400`-step schedule horizon layered onto the inherited runtime. | Rank rows by final log loss first, final Brier score second, final ROC AUC third, and best-to-final drift fourth while treating runtime and clipped-step fraction as guardrails only; use row `02` as the `warm0` challenger and row `05` as the `warm20` challenger around the corrected short-run baseline. |
| fallback scope | Not applicable. | `tf_rd_020_noise_mixture_v1` is not active in this sweep. | If this sweep stays ambiguous, defer cleanly or retain exactly one fallback schedule variant; do not broaden back to the alternate harder surface here. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_tf_rd_018_linear_warmup_decay` | schedule | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup. | Run first as the corrected short-run LR/warmup baseline replay, then compare the four bounded variants against it and the locked anchor. |
| 2 | `delta_training_tf_rd_018_linear_warmup_decay_warm0` | schedule | yes | ready | none | Keep the carried short-run warmup-decay surface but remove warmup entirely. | Run second as the no-warmup discriminator against the carried baseline. |
| 3 | `delta_training_tf_rd_018_linear_warmup_decay_lr3e3` | schedule | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces but lower the peak LR while keeping the carried warmup-decay floor fixed. | Run after rows `01` and `02` as the lower-ceiling LR probe. |
| 4 | `delta_training_tf_rd_018_linear_warmup_decay_minlr1e4` | schedule | yes | ready | none | Keep the anchor model, data, and preprocessing surfaces but lower the LR floor while keeping the carried peak LR and warmup fixed. | Run after the carried baseline as the lower-floor LR probe. |
| 5 | `delta_training_tf_rd_018_linear_warmup_decay_warm20` | schedule | yes | ready | none | Keep the carried short-run warmup-decay surface but lengthen warmup materially. | Run after the carried baseline as the materially longer-warmup probe. |

## Detailed Rows

### 1. `delta_training_tf_rd_018_linear_warmup_decay`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup.
- Rationale: Replay the carried schedulefree warmup-decay baseline on a corrected short-run `400`-step schedule horizon so issue `#138` has one explicit local reference row before reading the schedule variants.
- Hypothesis: The corrected `0.05` warmup replay should remain competitive and provide the local comparison point for warmup-zero, lower-ceiling, lower-floor, and materially longer-warmup probes.
- Upstream delta: Not applicable; this is a repo-local corrected short-run warmup-decay baseline on the inherited harder surface.
- Anchor delta: Keep the inherited `tf_rd_020_shift_noise_drift_v1` data surface, preprocessing surface, `schedulefree_adamw` optimizer family, and harmonized `task_batch_size=1` with `grad_accum_steps=4` `400`-step runtime fixed, then replay the carried linear-warmup-decay recipe on a matching `400`-step schedule horizon.
- Expected effect: Reduced early instability versus constant LR or plain linear decay, with uncertain final quality impact.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 400, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Use this row as the explicit local reference for rows `02` through `05`, while still comparing every row against the locked sweep anchor.
  - Rank rows by final log loss first, final Brier score second, and final ROC AUC third.
  - Treat drift, clipped-step fraction, and runtime as guardrails only.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
  - optimizer.min_lr
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Keep `tf_rd_020_noise_mixture_v1` inactive while reading this row.
  - Align `schedule.stages[0].steps` to `runtime.max_steps=400` so warmup and floor reads are interpreted on the actual short-run horizon rather than on the inherited `2500`-step parent setting.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_lr_warmup_shape_v1/delta_training_tf_rd_018_linear_warmup_decay/result_card.md`
- Benchmark metrics: pending

### 2. `delta_training_tf_rd_018_linear_warmup_decay_warm0`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the carried short-run warmup-decay surface but remove warmup entirely.
- Rationale: Remove warmup while keeping the corrected short-run LR pair fixed so issue `#138` can test whether the inherited `0.05` warmup is still necessary on the locked noise-drift runtime.
- Hypothesis: If the corrected `400`-step runtime is already stable enough, no warmup may preserve or improve final quality without materially reopening early instability.
- Upstream delta: Not applicable; this is a repo-local warmup-zero follow-up on the corrected short-run warmup-decay surface.
- Anchor delta: Keep the inherited harder-surface runtime fixed and change only `warmup_ratio` from `0.05` to `0.0` relative to row `01`.
- Expected effect: No warmup may preserve quality if the corrected `400`-step runtime is already stable enough without early protection.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 400, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.0}]}}`
- Parameter adequacy plan:
  - Compare directly against row `01` and the locked anchor on final log loss, then supporting Brier and ROC AUC.
  - Use early clipped-step behavior only to decide whether the no-warmup recipe reopened obvious instability.
  - Do not promote this row on cleaner runtime alone.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_lr_warmup_shape_v1/delta_training_tf_rd_018_linear_warmup_decay_warm0/result_card.md`
- Benchmark metrics: pending

### 3. `delta_training_tf_rd_018_linear_warmup_decay_lr3e3`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but lower the peak LR while keeping the carried warmup-decay floor fixed.
- Rationale: Lower the LR ceiling while keeping the carried floor and corrected warmup horizon fixed so issue `#138` can test whether the inherited `0.004` peak is still too aggressive on the locked noise-drift runtime.
- Hypothesis: A slightly lower peak LR may reduce residual clipping and improve final retention without needing a different optimizer family or harder-surface branch.
- Upstream delta: Not applicable; this is a repo-local LR-shape follow-up on the settled warmup-decay surface.
- Anchor delta: Keep the inherited harder-surface runtime fixed and change only `lr_max` from `0.004` to `0.003` relative to row `01`.
- Expected effect: Lower peak LR may reduce clipping and improve final retention if the carried `0.004` ceiling is still too aggressive on the inherited harder surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 400, 'lr_max': 0.003, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against row `01` and the locked anchor on final log loss, then supporting Brier and ROC AUC.
  - Prefer this row only if the quality win is not merely a guardrail-only stability trade.
  - Use the result to bound the LR search downward before issue `#139` opens.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_lr_warmup_shape_v1/delta_training_tf_rd_018_linear_warmup_decay_lr3e3/result_card.md`
- Benchmark metrics: pending

### 4. `delta_training_tf_rd_018_linear_warmup_decay_minlr1e4`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but lower the LR floor while keeping the carried peak LR and warmup fixed.
- Rationale: Lower the LR floor while keeping the carried peak LR and corrected warmup horizon fixed so issue `#138` can test whether the inherited floor is decaying too conservatively on the locked noise-drift runtime.
- Hypothesis: On a corrected `400`-step schedule horizon, a lower LR floor may preserve late plasticity and improve final retention without changing the early-step behavior materially.
- Upstream delta: Not applicable; this is a repo-local LR-floor follow-up on the settled warmup-decay surface.
- Anchor delta: Keep the inherited harder-surface runtime fixed and change only `optimizer.min_lr` from `0.0004` to `0.0001` relative to row `01`.
- Expected effect: A lower floor may preserve later plasticity if the carried `0.0004` floor is decaying too conservatively on the inherited harder surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0001, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 400, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against row `01` and the locked anchor on final log loss, then supporting Brier and ROC AUC.
  - Prefer this row only if it yields a real late-curve gain rather than a small telemetry-only change.
  - Treat worse drift or clipping without a quality win as negative evidence for reopening the LR floor.
- Adequacy knobs to dimension explicitly:
  - optimizer.min_lr
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_lr_warmup_shape_v1/delta_training_tf_rd_018_linear_warmup_decay_minlr1e4/result_card.md`
- Benchmark metrics: pending

### 5. `delta_training_tf_rd_018_linear_warmup_decay_warm20`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the carried short-run warmup-decay surface but lengthen warmup materially.
- Rationale: Materially lengthen warmup while keeping the carried LR pair fixed so issue `#138` can test whether the inherited runtime wants more early protection before issue `#139` opens clip or budget work.
- Hypothesis: A `0.20` warmup may reduce early clipping further on the corrected short-run horizon, though it risks underusing the fixed-step budget and hurting final quality.
- Upstream delta: Not applicable; this is a repo-local warmup-length follow-up on the corrected short-run warmup-decay surface.
- Anchor delta: Keep the inherited harder-surface runtime fixed and change only `warmup_ratio` from `0.05` to `0.20` relative to row `01`.
- Expected effect: A `0.20` warmup may reduce early clipping further on the corrected `400`-step runtime, though it risks underusing the fixed-step budget.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 400, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.2}]}}`
- Parameter adequacy plan:
  - Compare directly against row `01` and the locked anchor on final log loss, then supporting Brier and ROC AUC.
  - Prefer this row only if better early stability comes with a real benchmark quality win rather than a budget-underuse regression.
  - Treat this row as the upper warmup bracket; do not promote it solely on cleaner clipped-step behavior.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_lr_warmup_shape_v1/delta_training_tf_rd_018_linear_warmup_decay_warm20/result_card.md`
- Benchmark metrics: pending
