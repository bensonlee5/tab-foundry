# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/input_norm_followup/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `input_norm_followup`
- Sweep status: `completed`
- Parent sweep id: `stability_followup`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3972`, final Brier score `0.1307`, best ROC AUC `0.7634`, final ROC AUC `0.7634`, final training time `257.5s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| bridge architecture | Old bridge rows established that prenorm plus row-cls cls2 can train cleanly with linear warmup decay. | Keep `nano_exact`, `table_block_style=prenorm`, `row_pool=row_cls`, `tfrow_cls_tokens=2`, and `post_encoder_norm=none` fixed. | Changes in this sweep should be attributed to preprocessing or batch-size effects, not to a reopened architecture axis. |
| input normalization | The bridge baseline still uses `train_zscore_clip`. | Keep `train_zscore_clip` on the promoted batch64 sqrt-scaled bridge surface. | Rows 1-5 showed the normalized variants were tied on batch32, row 9 only tied row 7, and the later no-norm follow-up underperformed, so clipping stays in the anchor by default. |
| batch-size scaling | Exact-prior training previously treated batch size as a hidden function argument rather than a tracked surface knob. | Use prior-dump batch size 64 with sqrt LR scaling relative to the reference batch size of 32. | Batch-size rows were the only source of clear lift in this sweep; row 7 is the promoted systems surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_input_norm_anchor_replay` | baseline_replay | yes | completed | none | Replay the promoted bridge baseline exactly on this machine before comparing normalization or batch-size variants. | Treat this run as the batch32 reference only; row 7 (`dpnb_input_norm_anchor_replay_batch64_sqrt`) is the promoted canonical anchor. |
| 2 | `dpnb_input_norm_zscore` | input_normalization | yes | completed | none | Remove external clipping and use plain train-only z-score normalization on the promoted bridge baseline. | Interpret rows 2-5 as a normalization-family wash on the weaker batch32 surface; keep row 7 as the canonical anchor. |
| 3 | `dpnb_input_norm_winsorize_zscore` | input_normalization | yes | completed | none | Winsorize at the 1st and 99th train percentiles before z-scoring on the promoted bridge baseline. | Keep this as bounded outlier-defense evidence only; row 7 stays canonical because the batch64 surface moved the metric while rows 1-3 did not. |
| 4 | `dpnb_input_norm_zscore_tanh` | input_normalization | yes | completed | none | Apply train-only z-score normalization followed by a smooth tanh tail squash on the promoted bridge baseline. | Treat this as the smooth-tail comparator only; row 9 later tied the promoted winner, so row 7 stays canonical because it reaches the same result with less normalization. |
| 5 | `dpnb_input_norm_robust_tanh` | input_normalization | yes | completed | none | Apply robust median/IQR scaling followed by a smooth tanh tail squash on the promoted bridge baseline. | Keep this as negative evidence for the robust smooth-tail branch; it did not separate from the batch32 baseline and did not justify replacing row 7. |
| 6 | `dpnb_input_norm_anchor_replay_batch16_sqrt` | batch_size | yes | completed | none | Replay the promoted bridge baseline at prior-dump batch size 16 with sqrt LR scaling. | Keep this as negative batch-size evidence only; the smaller batch is not competitive with the promoted row-7 anchor. |
| 7 | `dpnb_input_norm_anchor_replay_batch64_sqrt` | batch_size | yes | completed | none | Replay the promoted bridge baseline at prior-dump batch size 64 with sqrt LR scaling. | Use this run as the canonical anchor for future binary_md follow-ups; it matched the best result in the sweep with the least additional normalization. |
| 8 | `dpnb_input_norm_zscore_tanh_batch16_sqrt` | batch_size | yes | completed | none | Combine the favored smooth-tail zscore_tanh normalization with prior-dump batch size 16 and sqrt LR scaling. | Treat this as negative interaction evidence; smaller batch size hurts even when paired with the smooth-tail normalization. |
| 9 | `dpnb_input_norm_zscore_tanh_batch64_sqrt` | batch_size | yes | completed | none | Combine the favored smooth-tail zscore_tanh normalization with prior-dump batch size 64 and sqrt LR scaling. | Keep this as tie evidence only; row 7 remains the promoted anchor because it matched this result with less invasive preprocessing. |

## Detailed Rows

### 1. `dpnb_input_norm_anchor_replay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Replay the promoted bridge baseline exactly on this machine before comparing normalization or batch-size variants.
- Rationale: Reproduce the promoted bridge baseline locally before interpreting any normalization or batch-size movement.
- Hypothesis: The row-cls bridge winner should replay cleanly on this machine and provide the correct comparator for the rest of the queue.
- Upstream delta: Not applicable; this is a repo-local metadata-only bridge anchor replay.
- Anchor delta: Replays `dpnb_row_cls_cls2_linear_warmup_decay` exactly, keeping `train_zscore_clip`, batch size 32, and the promoted linear-warmup-decay schedule.
- Expected effect: Establish a local comparator for all later rows while the canonical winning artifacts remain on another machine.
- Effective labels: model=`dpnb_row_cls_cls2_linear_warmup_decay`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Use this row as the batch32 reference for the normalization-only rows and the batch-size sidecars.
  - Report benchmark ROC, best-to-final drift, clipped-step fraction, and activation windows.
  - If a later row wins cleanly, preserve this replay as evidence rather than forcing it to remain the canonical anchor.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - optimizer.min_lr
  - schedule.stages[0].lr_max
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row is required because the promoted bridge baseline artifacts were produced on another machine.
  - The earlier local MPS replay is archived under `train_pre_cpu_fallback_20260317T2239` and is not canonical for this sweep.
  - The accepted local replay used the exact prior-train CPU fallback for requested `mps` multi-layer `row_cls` runs.
  - Supersedes historical queue run `sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v1`; that registry entry is retained as history only.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2`.
  - CUDA rerun established the same-machine batch32 reference, but row 7 later displaced it as the canonical anchor.
  - Row 7 improved final ROC AUC by 0.0068 on the same clipped normalization family, so this row remains evidence rather than the live anchor.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay/result_card.md`
- Registered run: `sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2` with best ROC AUC `0.7634`, final ROC AUC `0.7567`, final-minus-best `-0.0068`, delta final ROC AUC `-0.0068`, delta drift `-0.0068`, delta final training time `-114.5s`

### 2. `dpnb_input_norm_zscore`

- Dimension family: `preprocessing`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove external clipping and use plain train-only z-score normalization on the promoted bridge baseline.
- Rationale: Test the simplest unclipped alternative before trying smoother or more robust tails.
- Hypothesis: Removing hard clipping may preserve more useful scale information now that the bridge baseline is already stabilized.
- Upstream delta: Upstream nanoTabPFN keeps internal z-score-plus-clip behavior on the exact path.
- Anchor delta: Changes only `model.input_normalization` from `train_zscore_clip` to `train_zscore`.
- Expected effect: Clarifies whether clipping is still helping on the stabilized row-cls bridge surface.
- Effective labels: model=`dpnb_input_norm_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare directly against row 1 before reading the smoother-tail rows.
  - Use benchmark quality first; treat telemetry changes as explanatory evidence only.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - training_surface_record.json confirms this row actually resolved to input_normalization=train_zscore with benchmark profile dpnb_input_norm_zscore.
  - Best and final ROC AUC matched the row-1 anchor to checked precision on the locked bundle, so plain z-score looks viable but not clearly superior.
  - Clean telemetry means the tie is informative rather than a masked instability result.
  - Supersedes historical queue run `sd_input_norm_followup_02_dpnb_input_norm_zscore_v1`; that registry entry is retained as history only.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_02_dpnb_input_norm_zscore_v2`.
  - CUDA rerun recorded the canonical batch32 comparison, but batch-size evidence later dominated the sweep so this row remains supporting evidence only.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_zscore/result_card.md`
- Registered run: `sd_input_norm_followup_02_dpnb_input_norm_zscore_v2` with best ROC AUC `0.7634`, final ROC AUC `0.7567`, final-minus-best `-0.0068`, delta final ROC AUC `-0.0068`, delta drift `-0.0068`, delta final training time `-116.3s`

### 3. `dpnb_input_norm_winsorize_zscore`

- Dimension family: `preprocessing`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Winsorize at the 1st and 99th train percentiles before z-scoring on the promoted bridge baseline.
- Rationale: Keep explicit outlier defense while removing the hard post-z-score clip.
- Hypothesis: Percentile winsorization may retain robustness without the bluntness of fixed clipping.
- Upstream delta: Upstream nanoTabPFN does not expose percentile winsorization as a first-class normalization mode.
- Anchor delta: Changes `model.input_normalization` from `train_zscore_clip` to `train_winsorize_zscore`.
- Expected effect: May preserve outlier robustness while relaxing the hard post-z-score clip.
- Effective labels: model=`dpnb_input_norm_winsorize_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare against both row 1 and row 2 before concluding that outlier defense is helping or hurting.
  - Treat this row as a bounded outlier-robust alternative, not as the main smooth-tail candidate.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - training_surface_record.json confirms this row actually resolved to input_normalization=train_winsorize_zscore with benchmark profile dpnb_input_norm_winsorize_zscore.
  - Best and final ROC AUC again matched rows 1 and 2 to checked precision, so percentile winsorization is viable but not measurably useful on this bundle.
  - The row was slower than the clipped anchor and plain z-score while delivering no benchmark lift, which lowers its priority relative to the upcoming smooth-tail rows.
  - Supersedes historical queue run `sd_input_norm_followup_03_dpnb_input_norm_winsorize_zscore_v1`; that registry entry is retained as history only.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_03_dpnb_input_norm_winsorize_zscore_v2`.
  - CUDA rerun recorded the canonical batch32 comparison, but the row remains supporting evidence only after row 7 established the better systems surface.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_winsorize_zscore/result_card.md`
- Registered run: `sd_input_norm_followup_03_dpnb_input_norm_winsorize_zscore_v2` with best ROC AUC `0.7634`, final ROC AUC `0.7567`, final-minus-best `-0.0068`, delta final ROC AUC `-0.0068`, delta drift `-0.0068`, delta final training time `-116.2s`

### 4. `dpnb_input_norm_zscore_tanh`

- Dimension family: `preprocessing`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Apply train-only z-score normalization followed by a smooth tanh tail squash on the promoted bridge baseline.
- Rationale: Probe a smoother tail treatment that still stays near-linear around zero.
- Hypothesis: `train_zscore_tanh` may outperform both clipped z-score and plain z-score by avoiding hard saturation while still bounding outliers softly.
- Upstream delta: Upstream nanoTabPFN does not expose a smooth-tail normalization family.
- Anchor delta: Changes `model.input_normalization` from `train_zscore_clip` to `train_zscore_tanh`.
- Expected effect: Preserves near-linear scale around zero while soft-bounding outliers without hard clipping.
- Effective labels: model=`dpnb_input_norm_zscore_tanh`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare against rows 1-3 and use this as the favored smooth-tail candidate if it wins locally.
  - Promote batch-size interaction checks only if this row is at least competitive with the anchor replay.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical CUDA rerun registered as `sd_input_norm_followup_04_dpnb_input_norm_zscore_tanh_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - This row tied the batch32 clipped baseline but did not create its own lift; the only winning smooth-tail result required the same batch64 systems change as row 7.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_zscore_tanh/result_card.md`
- Registered run: `sd_input_norm_followup_04_dpnb_input_norm_zscore_tanh_v1` with best ROC AUC `0.7634`, final ROC AUC `0.7567`, final-minus-best `-0.0068`, delta final ROC AUC `-0.0068`, delta drift `-0.0068`, delta final training time `-116.1s`

### 5. `dpnb_input_norm_robust_tanh`

- Dimension family: `preprocessing`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Apply robust median/IQR scaling followed by a smooth tanh tail squash on the promoted bridge baseline.
- Rationale: Compare a more outlier-resistant center/scale estimate within the same smooth-tail family.
- Hypothesis: Robust scaling plus tanh may help if the remaining harmful feature variation is still driven by extreme values rather than by clipping itself.
- Upstream delta: Upstream nanoTabPFN does not expose this robust smooth-tail normalization family.
- Anchor delta: Changes `model.input_normalization` from `train_zscore_clip` to `train_robust_tanh`.
- Expected effect: Tests whether a more outlier-resistant center/scale estimate helps more than plain z-score plus smooth tails.
- Effective labels: model=`dpnb_input_norm_robust_tanh`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare primarily against row 4 so the smooth-tail family is separated cleanly from the batch-size sidecars.
  - Prefer the simpler z-score family unless this row shows clear benchmark upside.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical CUDA rerun registered as `sd_input_norm_followup_05_dpnb_input_norm_robust_tanh_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - Robust smooth-tail preprocessing produced no measurable upside on the locked bundle, so it stays below the promoted clipped batch64 surface.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_robust_tanh/result_card.md`
- Registered run: `sd_input_norm_followup_05_dpnb_input_norm_robust_tanh_v1` with best ROC AUC `0.7634`, final ROC AUC `0.7567`, final-minus-best `-0.0068`, delta final ROC AUC `-0.0068`, delta drift `-0.0068`, delta final training time `-112.2s`

### 6. `dpnb_input_norm_anchor_replay_batch16_sqrt`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Replay the promoted bridge baseline at prior-dump batch size 16 with sqrt LR scaling.
- Rationale: Test whether a smaller prior-dump batch improves the bridge baseline once LR is scaled conservatively.
- Hypothesis: Batch size 16 may reduce effective optimization noise enough to help the baseline replay even after sqrt LR scaling.
- Upstream delta: Not applicable; this is a repo-local exact-prior systems probe.
- Anchor delta: Keeps the baseline normalization fixed at `train_zscore_clip` but changes `training.prior_dump_batch_size` from 32 to 16 with sqrt LR scaling.
- Expected effect: Tests whether smaller batches improve the bridge baseline once LR is scaled conservatively.
- Effective labels: model=`dpnb_input_norm_anchor_replay_batch16_sqrt`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against row 1 before combining batch-size changes with normalization changes.
  - Treat the effective LR change as part of the row definition, not as a separate confound.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - schedule.stages[0].lr_max
  - optimizer.min_lr
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Effective LR scaling factor is approximately 0.7071, so effective lr_max is about 0.002828 and effective min_lr is about 0.000283.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_06_dpnb_input_norm_anchor_replay_batch16_sqrt_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - Batch size 16 hurt final ROC AUC relative to both the batch32 reference and the promoted batch64 surface.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay_batch16_sqrt/result_card.md`
- Registered run: `sd_input_norm_followup_06_dpnb_input_norm_anchor_replay_batch16_sqrt_v1` with best ROC AUC `0.7595`, final ROC AUC `0.7557`, final-minus-best `-0.0039`, delta final ROC AUC `-0.0078`, delta drift `-0.0039`, delta final training time `-149.8s`

### 7. `dpnb_input_norm_anchor_replay_batch64_sqrt`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Replay the promoted bridge baseline at prior-dump batch size 64 with sqrt LR scaling.
- Rationale: Test whether a larger prior-dump batch improves the bridge baseline once LR is scaled conservatively.
- Hypothesis: Batch size 64 may improve the bridge baseline if the current recipe still prefers a larger effective batch than 32.
- Upstream delta: Not applicable; this is a repo-local exact-prior systems probe.
- Anchor delta: Keeps the baseline normalization fixed at `train_zscore_clip` but changes `training.prior_dump_batch_size` from 32 to 64 with sqrt LR scaling.
- Expected effect: Tests whether the stabilized bridge surface prefers a larger effective batch when LR is scaled up conservatively.
- Effective labels: model=`dpnb_input_norm_anchor_replay_batch64_sqrt`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against row 1 before combining batch-size changes with normalization changes.
  - Treat the effective LR change as part of the row definition, not as a separate confound.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - schedule.stages[0].lr_max
  - optimizer.min_lr
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Effective LR scaling factor is approximately 1.4142, so effective lr_max is about 0.005657 and effective min_lr is about 0.000566.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - This row is promoted as the canonical anchor because it matched row 9 while keeping `train_zscore_clip`, which is the less invasive preprocessing choice on this implementation.
  - Supersedes historical queue run `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay_batch64_sqrt/result_card.md`
- Registered run: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2` with final log loss `0.3972`, delta final log loss `+0.0000`, final Brier score `0.1307`, delta final Brier score `+0.0000`, best ROC AUC `0.7634`, final ROC AUC `0.7634`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0000`, delta drift `+0.0000`, delta final training time `+0.0s`

### 8. `dpnb_input_norm_zscore_tanh_batch16_sqrt`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Combine the favored smooth-tail zscore_tanh normalization with prior-dump batch size 16 and sqrt LR scaling.
- Rationale: Check whether the favored smooth-tail row benefits from a smaller scaled batch.
- Hypothesis: If `train_zscore_tanh` is the right preprocessing family, batch size 16 may amplify that benefit rather than confound it away.
- Upstream delta: Not applicable; this is a repo-local interaction probe between smooth-tail preprocessing and smaller batch size.
- Anchor delta: Changes both `model.input_normalization` to `train_zscore_tanh` and `training.prior_dump_batch_size` to 16 with sqrt LR scaling.
- Expected effect: Checks whether any zscore_tanh gains survive or improve at a smaller, scaled batch.
- Effective labels: model=`dpnb_input_norm_zscore_tanh_batch16_sqrt`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare first against row 4 and only second against row 1.
  - Use this row to decide whether batch size is part of the smooth-tail mechanism or merely a separate systems effect.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Effective LR scaling factor is approximately 0.7071, so effective lr_max is about 0.002828 and effective min_lr is about 0.000283.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_08_dpnb_input_norm_zscore_tanh_batch16_sqrt_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - The batch16 smooth-tail variant reproduced the same degradation as row 6, so the batch-size effect is not a tanh-specific mechanism.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_zscore_tanh_batch16_sqrt/result_card.md`
- Registered run: `sd_input_norm_followup_08_dpnb_input_norm_zscore_tanh_batch16_sqrt_v1` with best ROC AUC `0.7595`, final ROC AUC `0.7557`, final-minus-best `-0.0039`, delta final ROC AUC `-0.0078`, delta drift `-0.0039`, delta final training time `-149.6s`

### 9. `dpnb_input_norm_zscore_tanh_batch64_sqrt`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Combine the favored smooth-tail zscore_tanh normalization with prior-dump batch size 64 and sqrt LR scaling.
- Rationale: Check whether the favored smooth-tail row benefits from a larger scaled batch.
- Hypothesis: If `train_zscore_tanh` likes cleaner gradient averaging, batch size 64 may outperform both row 4 and row 7.
- Upstream delta: Not applicable; this is a repo-local interaction probe between smooth-tail preprocessing and larger batch size.
- Anchor delta: Changes both `model.input_normalization` to `train_zscore_tanh` and `training.prior_dump_batch_size` to 64 with sqrt LR scaling.
- Expected effect: Checks whether any zscore_tanh gains survive or improve at a larger, scaled batch.
- Effective labels: model=`dpnb_input_norm_zscore_tanh_batch64_sqrt`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare first against row 4 and only second against row 1.
  - Use this row to decide whether batch size is part of the smooth-tail mechanism or merely a separate systems effect.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - model.input_normalization
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Effective LR scaling factor is approximately 1.4142, so effective lr_max is about 0.005657 and effective min_lr is about 0.000566.
  - Canonical CUDA rerun registered as `sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1`.
  - CUDA rerun recorded the canonical benchmark comparison, but the row remains deferred pending full sweep interpretation.
  - Row 9 matched row 7 on final ROC AUC, but `train_zscore_tanh` is more interventionist than the effectively no-op `train_zscore_clip` bound at +/-100.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_zscore_tanh_batch64_sqrt/result_card.md`
- Registered run: `sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1` with best ROC AUC `0.7638`, final ROC AUC `0.7634`, final-minus-best `-0.0003`, delta final ROC AUC `+0.0000`, delta drift `-0.0003`, delta final training time `-0.3s`
