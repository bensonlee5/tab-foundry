# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_018_optimizer_family_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_018_optimizer_family_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_020_harder_dagzoo_ladder_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
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
| locked anchor | TabICLv2 motivates synthetic pretraining but does not define this repo-local row-first promoted-anchor contract. | The kept `row_cls + qass + no tfcol` TF-RD-020 noise-drift winner trained on the uncapped `task_batch_size=1`, `grad_accum_steps=4`, `400`-step runtime. | TF-RD-018 issue `#137` must keep the promoted row-first model surface fixed while reading optimizer-family evidence. |
| harder carry-forward surface | No upstream reference defines this exact dagzoo harder-surface handoff. | The locked harder-surface anchor is `tf_rd_020_shift_noise_drift_v1`, inherited directly from TF-RD-020 row `06`. | The TF-RD-020 noise-drift winner is already the carried optimizer anchor for this sweep rather than a replay target inside the queue. |
| optimizer family | Not applicable. | Schedulefree AdamW on the inherited TF-RD-020 row-`06` runtime is the locked comparison baseline for this sweep. | This completed read keeps `schedulefree_adamw` as the carried optimizer family before TF-RD-018 reopens LR, clipping, or budget work. |
| fallback scope | Not applicable. | `tf_rd_020_noise_mixture_v1` is excluded from this v1 sweep. | This completed read was not close or unstable enough to activate the fallback harder surface, so `tf_rd_020_shift_noise_drift_v1` stays carried into the LR-shape follow-up. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_adamw` | optimizer | yes | completed | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule. | Leave deferred and carry the locked `schedulefree_adamw` anchor into TF-RD-018 issue `#138` on the same `tf_rd_020_shift_noise_drift_v1` runtime. |
| 2 | `delta_training_muon` | optimizer | yes | completed | none | Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule. | Leave deferred, do not activate the `tf_rd_020_noise_mixture_v1` fallback surface, and carry the locked `schedulefree_adamw` anchor into TF-RD-018 issue `#138`. |

## Detailed Rows

### 1. `delta_training_adamw`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with plain AdamW on the same linear-warmup-decay schedule.
- Rationale: Compare plain AdamW against the locked TF-RD-020 noise-drift winner so issue `#137` can isolate optimizer family after the uncapped harder-front ladder removed dataset row caps.
- Hypothesis: Plain AdamW may improve late retention or reduce drift on the harder surface if schedulefree dynamics are masking the best benchmark-facing read.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the inherited `tf_rd_020_shift_noise_drift_v1` data surface, preprocessing surface, benchmark bundle, and harmonized `task_batch_size=1` with `grad_accum_steps=4` `400`-step runtime fixed, but replace `schedulefree_adamw` with plain `adamw`.
- Expected effect: AdamW may provide a cleaner or more stable optimization path if schedulefree dynamics are masking the anchor's real quality ceiling.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0169`); context (grad `0.0286`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this as a bounded optimizer-family comparison on the locked TF-RD-020 harder-surface anchor.
  - Compare final log loss first, then final Brier score, final ROC AUC, and best-to-final drift against the locked anchor and row `02`.
  - Use NaNs, OOM, clipped-step fraction, and obvious LR-transfer artifacts as guardrails before making a reject call.
- Adequacy knobs to dimension explicitly:
  - optimizer.name
  - optimizer.weight_decay
  - optimizer.betas
  - optimizer.min_lr
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_018_optimizer_family_v1_01_delta_training_adamw_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - This row underperformed the locked anchor on final log loss, final Brier score, and final ROC AUC, so it does not remain as an optimizer fallback.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_optimizer_family_v1/delta_training_adamw/result_card.md`
- Registered run: `sd_tf_rd_018_optimizer_family_v1_01_delta_training_adamw_v1` with final log loss `0.5971`, delta final log loss `+0.0470`, final Brier score `0.4114`, delta final Brier score `+0.0374`, best ROC AUC `0.5515`, final ROC AUC `0.5614`, final-minus-best `+0.0099`, delta final ROC AUC `-0.0266`, delta drift `+0.0164`, delta final training time `+25.7s`

### 2. `delta_training_muon`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces fixed but replace schedulefree AdamW with Muon on the same linear-warmup-decay schedule.
- Rationale: Compare Muon against the locked TF-RD-020 noise-drift winner so issue `#137` can read whether a more specialized optimizer changes the adequacy story on the inherited harder surface.
- Hypothesis: Muon may improve convergence or late retention on the harder surface, but a weak result should only matter if it is not obviously an LR-transfer problem.
- Upstream delta: Not applicable; this is a repo-local optimizer-family comparison on the settled row-first anchor.
- Anchor delta: Keep the inherited `tf_rd_020_shift_noise_drift_v1` data surface, preprocessing surface, benchmark bundle, and harmonized `task_batch_size=1` with `grad_accum_steps=4` `400`-step runtime fixed, but replace `schedulefree_adamw` with `muon`.
- Expected effect: Muon may improve convergence or late retention on the settled row-first surface, but any read must stay separate from model-surface expansion.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.1624`); context (grad `0.1858`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'muon', 'require_requested': True, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'min_lr': 1e-06, 'muon_per_parameter_lr': True, 'muon_lr_scale_base': 0.2, 'muon_partition_non2d': True}, 'runtime': {'grad_accum_steps': 4, 'max_steps': 400, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this as a bounded optimizer-family comparison only on the locked TF-RD-020 harder-surface anchor.
  - Compare final log loss first, then final Brier score, final ROC AUC, and best-to-final drift against the locked anchor and row `01`.
  - If weak, defer rather than reject unless the result is clearly worse without an offsetting stability or runtime benefit and does not look like pure LR-transfer misspecification.
- Adequacy knobs to dimension explicitly:
  - optimizer.name
  - optimizer.weight_decay
  - optimizer.betas
  - optimizer.min_lr
  - optimizer.muon_per_parameter_lr
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_018_optimizer_family_v1_02_delta_training_muon_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - This row regressed final log loss and final Brier badly enough that TF-RD-018 does not retain any non-schedulefree optimizer fallback after closing issue `#137`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_018_optimizer_family_v1/delta_training_muon/result_card.md`
- Registered run: `sd_tf_rd_018_optimizer_family_v1_02_delta_training_muon_v1` with final log loss `1.0768`, delta final log loss `+0.5267`, final Brier score `0.4742`, delta final Brier score `+0.1002`, best ROC AUC `0.6596`, final ROC AUC `0.6725`, final-minus-best `+0.0129`, delta final ROC AUC `+0.0845`, delta drift `+0.0194`, delta final training time `+37.3s`
